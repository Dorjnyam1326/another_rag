"""Bounded, deterministic alternative search for uncovered request topics."""

from domain_rules import DOMAIN_RULES
from multi_query_retrieval import merge_intent_results
from query_faithfulness import process_groups, query_issues
from query_precheck import request_restriction_candidates


_SCOPES = DOMAIN_RULES["scopes"]


def _request_text(request):
    return f"{request.get('intent', '')} {request.get('canonical_query', '')}".casefold()


def _applicability_notes(passage, request):
    text = str(passage.get("text", "")).casefold()
    request_text = _request_text(request)
    notes = []
    for scope in _SCOPES:
        if any(marker in text for marker in scope["passage_markers"]) and not any(
            marker in request_text for marker in scope["request_markers"]
        ):
            notes.append(scope["note"] + " Хэрэглэгчийн хамрах бүлэг тодорхойгүй тул үүнийг ерөнхий журам гэж бүү тайлбарла.")
    return notes


def request_evidence_status(request, passages):
    """Check known policy domains and applicability without an LLM call."""
    topics = process_groups(_request_text(request))
    if not topics:
        return {"status": "undetermined", "missing_topics": []}

    matched_topics = set()
    for passage in passages:
        if passage.get("intents") and request.get("intent") not in passage["intents"]:
            continue
        if _applicability_notes(passage, request):
            continue
        matched_topics.update(process_groups(passage.get("text", "")))

    missing = sorted(topics - matched_topics)
    return {
        "status": "insufficient" if missing else "sufficient",
        "missing_topics": missing,
    }


def targeted_expansion_groups(
    requests,
    passages,
    *,
    round_index,
    enabled_restrictions,
    source_text="",
):
    """Return at most one safe alternative per unresolved request and round."""
    if round_index not in (0, 1):
        return []

    groups = []
    for request in requests:
        if (
            not request.get("search_documents")
            or request.get("is_ambiguous")
            or request.get("restriction_modules")
            or request_evidence_status(request, passages)["status"] != "insufficient"
        ):
            continue
        alternatives = request.get("alternatives", [])
        if round_index >= len(alternatives):
            continue
        query = alternatives[round_index]
        if request_restriction_candidates(query, enabled_restrictions):
            continue
        if query_issues(source_text, query):
            continue
        groups.append({
            "intent": request["intent"],
            "queries": [(f"alternative_{round_index + 1}", query)],
        })
    return groups


def annotate_applicability(passages, requests):
    by_intent = {request.get("intent"): request for request in requests}
    for passage in passages:
        notes = []
        for intent in passage.get("intents", []):
            request = by_intent.get(intent)
            if request:
                notes.extend(_applicability_notes(passage, request))
        passage["applicability_notes"] = list(dict.fromkeys(notes))
    return passages
