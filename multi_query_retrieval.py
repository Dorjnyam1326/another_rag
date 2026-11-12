"""Search OpenAI Vector Store with query variants and fuse ranks using RRF."""

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
import hashlib
import re

from query_normalizer import clean_text


def build_query_variants(original: str, canonical: str, alternatives: list[str]):
    """Return unique source-preserving search variants in a stable order."""
    candidates = []
    if clean_text(original).casefold() != clean_text(canonical).casefold():
        candidates.append(("original", original))
    candidates.append(("canonical", canonical))
    candidates.extend(
        (f"alternative_{index}", query)
        for index, query in enumerate(alternatives[:2], start=1)
    )

    unique = []
    seen = set()
    for label, query in candidates:
        cleaned = clean_text(query)
        key = cleaned.casefold()
        if key not in seen:
            seen.add(key)
            unique.append((label, cleaned))
    return unique


_EXPLICIT_TIME_RANGE = re.compile(
    r"\b\d{1,2}(?::\d{2})?\s*(?:–|—|-|аас|ээс|хүртэл|to)\s*"
    r"\d{1,2}(?::\d{2})?\b",
    re.I,
)


def _leave_search_query(request):
    """Broaden clock-ranged leave searches without changing answer context."""
    query = str(request.get("canonical_query", "")).strip()
    intent = str(request.get("intent", "")).casefold()
    is_leave = re.search(r"чөлөө|leave|vacation|амралт", intent)
    if not is_leave or not _EXPLICIT_TIME_RANGE.search(query):
        return query

    query = _EXPLICIT_TIME_RANGE.sub(" ", query)
    query = re.sub(r"\b(?:өдрийн\s+)?хүртэл\b", " ", query, flags=re.I)
    query = re.sub(r"\s+([,;:.?])", r"\1", query)
    query = re.sub(r"[,;:]+", " ", query)
    return " ".join(query.split()).strip(" ,;:.?")


def _text_content(match) -> str:
    parts = []
    for item in getattr(match, "content", []) or []:
        text = getattr(item, "text", None)
        if text is None and isinstance(item, dict):
            text = item.get("text")
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _attributes(match) -> dict:
    attributes = getattr(match, "attributes", None)
    if not isinstance(attributes, Mapping):
        return {}
    return {
        str(key)[:100]: value[:300] if isinstance(value, str) else value
        for key, value in attributes.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }


def _passage_id(file_id, text):
    """Stable within the same file and exact passage content."""
    return hashlib.sha256(f"{file_id}\0{text}".encode("utf-8")).hexdigest()


def _provider_score(match):
    score = getattr(match, "score", None)
    try:
        return float(score) if score is not None else None
    except (TypeError, ValueError):
        return None


def _search_one(client, vector_store_id: str, variant: tuple[str, str], per_query: int):
    label, query = variant
    page = client.vector_stores.search(
        vector_store_id=vector_store_id,
        query=query,
        max_num_results=per_query,
        rewrite_query=False,
    )
    return label, page.data


def merge_intent_results(initial, additional, intent_order, *, final_k):
    """Deduplicate, fuse per-intent ranks, and interleave intent results."""
    by_identity = {}
    for result in [*initial, *additional]:
        identity = (result.get("file_id"), result.get("text", ""))
        item = by_identity.get(identity)
        if item is None:
            item = {
                **result,
                "rrf_score": 0.0,
                "rrf_score_by_intent": {},
                "intents": [],
                "query_provenance": [],
            }
            by_identity[identity] = item
        old_scores = item["rrf_score_by_intent"]
        scores = result.get("rrf_score_by_intent") or {
            intent: result.get("rrf_score", 0.0)
            for intent in result.get("intents", [])
        }
        for intent, score in scores.items():
            old_scores[intent] = old_scores.get(intent, 0.0) + score
        item["intents"] = list(dict.fromkeys([*item["intents"], *result.get("intents", [])]))
        provenance = item["query_provenance"]
        seen_provenance = {
            (p.get("intent"), p.get("variant"), p.get("rank"))
            for p in provenance
        }
        for entry in result.get("query_provenance", []):
            key = (entry.get("intent"), entry.get("variant"), entry.get("rank"))
            if key not in seen_provenance:
                provenance.append(entry)
                seen_provenance.add(key)
        item["provider_scores"] = {
            **item.get("provider_scores", {}), **result.get("provider_scores", {})
        }
        item["rrf_score"] = sum(old_scores.values())

    ranked = {
        intent: sorted(
            (item for item in by_identity.values() if intent in item["intents"]),
            key=lambda item: (
                -item["rrf_score_by_intent"].get(intent, 0.0),
                min(item.get("ranks", {}).values(), default=999),
                item.get("filename", ""),
                item.get("text", ""),
            ),
        )
        for intent in intent_order
    }
    cursors = {intent: 0 for intent in intent_order}
    merged = []
    seen = set()
    while len(merged) < final_k:
        progressed = False
        for intent in intent_order:
            items = ranked[intent]
            while cursors[intent] < len(items):
                item = items[cursors[intent]]
                cursors[intent] += 1
                identity = (item.get("file_id"), item.get("text", ""))
                if identity in seen:
                    continue
                seen.add(identity)
                item["final_rank"] = len(merged) + 1
                merged.append(item)
                progressed = True
                break
            if len(merged) >= final_k:
                break
        if not progressed:
            break
    return merged


def build_intent_query_groups(
    original: str,
    requests: list[dict],
    *,
    multi_query_enabled: bool = True,
    strategy: str | None = None,
):
    """Build separate search groups without mixing unrelated request intents."""
    searchable = [
        request
        for request in requests
        if request.get("search_documents") and not request.get("is_ambiguous")
    ]
    if not searchable:
        return []

    strategy = str(strategy or "").upper()
    if strategy and strategy not in {"A", "B", "C"}:
        raise ValueError(f"Unknown query strategy: {strategy}")

    if len(requests) == 1 and len(searchable) == 1:
        request = searchable[0]
        if strategy == "C" or (not strategy and not multi_query_enabled):
            queries = [("canonical", request["canonical_query"])]
        elif strategy == "B":
            # Production B-v2: search a validated intent, not the raw source.
            # A leave clock range can overweight work-hours evidence, so omit
            # that range from search while preserving the canonical request.
            query = _leave_search_query(request)
            queries = [("canonical", query)]
        elif strategy == "A" or multi_query_enabled:
            queries = build_query_variants(
                original,
                request["canonical_query"],
                request.get("alternatives", []),
            )
        else:
            queries = [("canonical", request["canonical_query"])]
        return [{"intent": request["intent"], "queries": queries}]

    groups = [
        {
            "intent": request["intent"],
            "queries": [
                (
                    "canonical",
                    _leave_search_query(request)
                    if strategy == "B"
                    else request["canonical_query"],
                )
            ],
        }
        for request in searchable
    ]
    if strategy == "A" or (not strategy and multi_query_enabled):
        # Keep the current two-extra-search budget for compound requests and
        # distribute alternatives across intents instead of favoring the first.
        for alternative_index in range(2):
            for request, group in zip(searchable, groups):
                alternatives = request.get("alternatives", [])
                if alternative_index < len(alternatives):
                    group["queries"].append(
                        (f"alternative_{alternative_index + 1}", alternatives[alternative_index])
                    )
    return groups


def search_intent_groups(
    client,
    vector_store_id: str,
    groups: list[dict],
    *,
    per_query: int = 5,
    final_k: int = 8,
    rrf_k: int = 60,
    trace_capture: dict | None = None,
) -> list[dict]:
    """Search groups concurrently, fuse within each intent, then interleave."""
    if not groups:
        return []
    if not 1 <= per_query <= 50 or final_k < 1 or rrf_k < 1:
        raise ValueError("Invalid retrieval or RRF limits.")

    pending = [
        (group_index, group["intent"], label, query)
        for group_index, group in enumerate(groups)
        for label, query in group["queries"]
    ]
    if not pending:
        return []
    rrf_enabled = [len(group["queries"]) > 1 for group in groups]

    def search(item):
        group_index, _, label, query = item
        _, data = _search_one(
            client, vector_store_id, (label, query), per_query
        )
        return group_index, label, data

    with ThreadPoolExecutor(max_workers=min(len(pending), 4)) as executor:
        search_results = list(executor.map(search, pending))

    if trace_capture is not None:
        trace_capture["queries"] = []
        for item, (_, _, matches) in zip(pending, search_results):
            group_index, intent, label, query = item
            raw_results = []
            for rank, match in enumerate(matches, start=1):
                passage = _text_content(match)
                file_id = getattr(match, "file_id", "")
                raw_results.append(
                    {
                        "rank": rank,
                        "passage_id": _passage_id(file_id, passage) if passage else None,
                        "file_id": file_id,
                        "filename": getattr(match, "filename", "Unknown source"),
                        "text": passage,
                        "attributes": _attributes(match),
                        "provider_score": _provider_score(match),
                    }
                )
            trace_capture["queries"].append(
                {
                    "intent": intent,
                    "variant": label,
                    "query": query,
                    "raw_results": raw_results,
                }
            )

    merged_by_group = [dict() for _ in groups]
    for group_index, label, matches in search_results:
        for rank, match in enumerate(matches, start=1):
            passage = _text_content(match)
            file_id = getattr(match, "file_id", "")
            filename = getattr(match, "filename", "Unknown source")
            provider_score = _provider_score(match)
            if not passage:
                continue
            identity = (file_id, passage)
            candidate = merged_by_group[group_index].setdefault(
                identity,
                {
                    "passage_id": _passage_id(file_id, passage),
                    "file_id": file_id,
                    "filename": filename,
                    "text": passage,
                    "attributes": _attributes(match),
                    "attributes_status": (
                        "available"
                        if isinstance(getattr(match, "attributes", None), Mapping)
                        else "unavailable"
                    ),
                    "rrf_score": 0.0,
                    "ranks": {},
                    "provider_scores": {},
                    "query_provenance": [],
                    "intents": [groups[group_index]["intent"]],
                },
            )
            if rrf_enabled[group_index]:
                candidate["rrf_score"] += 1.0 / (rrf_k + rank)
            candidate["ranks"][label] = rank
            if provider_score is not None:
                score_key = label if groups[group_index]["intent"] == "default" else (
                    f"{groups[group_index]['intent']}::{label}"
                )
                candidate["provider_scores"][score_key] = provider_score
            candidate["query_provenance"].append({
                "intent": groups[group_index]["intent"],
                "variant": label,
                "rank": rank,
                "provider_score": provider_score,
            })

    candidates = [item for group in merged_by_group for item in group.values()]
    intent_order = [group["intent"] for group in groups]
    combined = merge_intent_results([], candidates, intent_order, final_k=final_k)
    if trace_capture is not None:
        trace_capture["fused_results"] = combined
        trace_capture["settings"] = {
            "per_query": per_query,
            "final_k": final_k,
            "rrf_k": rrf_k,
        }
    return combined


def format_evidence(results: list[dict]) -> str:
    """Format retrieved passages with filenames for the answer model."""
    if not results:
        return "No relevant passages were returned by the document search."
    blocks = []
    for index, result in enumerate(results, start=1):
        intents = result.get("intents", [])
        relevant_to = f"\nRelevant to: {', '.join(intents)}" if intents else ""
        applicability = "".join(
            f"\nApplicability: {note}" for note in result.get("applicability_notes", [])
        )
        blocks.append(
            f"[Source {index}: {result['filename']}]"
            f"{relevant_to}{applicability}\n{result['text']}"
        )
    return "\n\n".join(blocks)


def source_filenames(results: list[dict]) -> list[str]:
    """Return unique filenames for the evidence shown to the answer model."""
    return list(dict.fromkeys(item["filename"] for item in results))
