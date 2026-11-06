"""Search OpenAI Vector Store with query variants and fuse ranks using RRF."""

from concurrent.futures import ThreadPoolExecutor

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


def _text_content(match) -> str:
    parts = []
    for item in getattr(match, "content", []) or []:
        text = getattr(item, "text", None)
        if text is None and isinstance(item, dict):
            text = item.get("text")
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _search_one(client, vector_store_id: str, variant: tuple[str, str], per_query: int):
    label, query = variant
    page = client.vector_stores.search(
        vector_store_id=vector_store_id,
        query=query,
        max_num_results=per_query,
        rewrite_query=False,
    )
    return label, page.data


def search_and_fuse(
    client,
    vector_store_id: str,
    variants: list[tuple[str, str]],
    *,
    per_query: int = 5,
    final_k: int = 8,
    rrf_k: int = 60,
) -> list[dict]:
    """Search each query separately, deduplicate passages, and rank by RRF."""
    if not variants:
        return []
    if not 1 <= per_query <= 50 or final_k < 1 or rrf_k < 1:
        raise ValueError("Invalid retrieval or RRF limits.")

    # Search calls are independent; wait for all of them before doing local RRF.
    with ThreadPoolExecutor(max_workers=min(len(variants), 4)) as executor:
        search_results = list(
            executor.map(
                lambda variant: _search_one(
                    client, vector_store_id, variant, per_query
                ),
                variants,
            )
        )

    merged = {}
    for label, matches in search_results:
        for rank, match in enumerate(matches, start=1):
            passage = _text_content(match)
            file_id = getattr(match, "file_id", "")
            filename = getattr(match, "filename", "Unknown source")
            if not passage:
                continue
            identity = (file_id, passage)
            candidate = merged.setdefault(
                identity,
                {
                    "file_id": file_id,
                    "filename": filename,
                    "text": passage,
                    "rrf_score": 0.0,
                    "ranks": {},
                },
            )
            candidate["rrf_score"] += 1.0 / (rrf_k + rank)
            candidate["ranks"][label] = rank

    ranked = sorted(
        merged.values(),
        key=lambda item: (
            -item["rrf_score"],
            min(item["ranks"].values()),
            item["filename"],
            item["text"],
        ),
    )
    return ranked[:final_k]


def build_intent_query_groups(
    original: str,
    requests: list[dict],
    *,
    multi_query_enabled: bool = True,
):
    """Build separate search groups without mixing unrelated request intents."""
    searchable = [
        request
        for request in requests
        if request.get("search_documents") and not request.get("is_ambiguous")
    ]
    if not searchable:
        return []

    if len(requests) == 1 and len(searchable) == 1:
        request = searchable[0]
        if multi_query_enabled:
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
            "queries": [("canonical", request["canonical_query"])],
        }
        for request in searchable
    ]
    if multi_query_enabled:
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

    def search(item):
        group_index, _, label, query = item
        _, data = _search_one(
            client, vector_store_id, (label, query), per_query
        )
        return group_index, label, data

    with ThreadPoolExecutor(max_workers=min(len(pending), 4)) as executor:
        search_results = list(executor.map(search, pending))

    merged_by_group = [dict() for _ in groups]
    for group_index, label, matches in search_results:
        for rank, match in enumerate(matches, start=1):
            passage = _text_content(match)
            file_id = getattr(match, "file_id", "")
            filename = getattr(match, "filename", "Unknown source")
            if not passage:
                continue
            identity = (file_id, passage)
            candidate = merged_by_group[group_index].setdefault(
                identity,
                {
                    "file_id": file_id,
                    "filename": filename,
                    "text": passage,
                    "rrf_score": 0.0,
                    "ranks": {},
                    "intents": [groups[group_index]["intent"]],
                },
            )
            candidate["rrf_score"] += 1.0 / (rrf_k + rank)
            candidate["ranks"][label] = rank

    ranked_groups = [
        sorted(
            group.values(),
            key=lambda item: (
                -item["rrf_score"],
                min(item["ranks"].values()),
                item["filename"],
                item["text"],
            ),
        )
        for group in merged_by_group
    ]
    cursors = [0] * len(ranked_groups)
    combined = []
    by_identity = {}
    while len(combined) < final_k:
        made_progress = False
        for group_index, ranked in enumerate(ranked_groups):
            while cursors[group_index] < len(ranked):
                item = ranked[cursors[group_index]]
                cursors[group_index] += 1
                made_progress = True
                identity = (item["file_id"], item["text"])
                existing = by_identity.get(identity)
                if existing:
                    intent = groups[group_index]["intent"]
                    if intent not in existing["intents"]:
                        existing["intents"].append(intent)
                    continue
                by_identity[identity] = item
                combined.append(item)
                break
            if len(combined) >= final_k:
                break
        if not made_progress:
            break
    return combined


def format_evidence(results: list[dict]) -> str:
    """Format retrieved passages with filenames for the answer model."""
    if not results:
        return "No relevant passages were returned by the document search."
    blocks = []
    for index, result in enumerate(results, start=1):
        intents = result.get("intents", [])
        relevant_to = f"\nRelevant to: {', '.join(intents)}" if intents else ""
        blocks.append(
            f"[Source {index}: {result['filename']}]"
            f"{relevant_to}\n{result['text']}"
        )
    return "\n\n".join(blocks)


def source_filenames(results: list[dict]) -> list[str]:
    """Return unique filenames for the evidence shown to the answer model."""
    return list(dict.fromkeys(item["filename"] for item in results))
