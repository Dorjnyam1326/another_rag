"""Paired Stage 6 evaluation using frozen router outputs."""

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "context_resolver_stage6.json"
sys.path.insert(0, str(ROOT))

from context_resolver import (  # noqa: E402
    ContextResolutionError,
    bound_user_sources,
    resolve_followup_queries,
)


def score_query(query, case):
    normalized = " ".join(query.casefold().split())
    has_expected = all(
        any(option.casefold() in normalized for option in (
            anchor if isinstance(anchor, list) else [anchor]
        ))
        for anchor in case["required_anchors"]
    )
    no_forbidden = not any(anchor.casefold() in normalized for anchor in case["forbidden_anchors"])
    standalone = (
        case["expected_relationship"] != "follow_up"
        or (
            len(normalized) > 30
            and not normalized.endswith(("тэр", "тэр үед", "хэн?"))
        )
    )
    return has_expected and no_forbidden and standalone


def _selected_context(case, route_fixture):
    selected = route_fixture["selected_message_indices"]
    if any(index < 0 or index >= len(case["history"]) for index in selected):
        raise ValueError(f"Invalid selected message index in fixture: {case['id']}")
    return [case["history"][index] for index in selected]


def _score_candidate(case, route_fixture, resolution):
    expected_resolution = case.get("expected_resolution", "resolved")
    if resolution is None:
        queries = [item["canonical_query"] for item in route_fixture["requests"]]
        passed = expected_resolution == "not_needed" and score_query(" ".join(queries), case)
        return passed, False, []
    if any(item["status"] == "clarification" for item in resolution.values()):
        return expected_resolution == "clarification", True, []
    if expected_resolution == "clarification":
        return False, False, []
    queries = [item["resolved_query"] for item in resolution.values()]
    used_indices = sorted({
        source_index
        for item in resolution.values()
        for source_index in item["source_message_indices"]
    })
    allowed_indices = set(case["source_indices"])
    source_indices_ok = set(used_indices).issubset(allowed_indices)
    return score_query(" ".join(queries), case) and source_indices_ok, False, used_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="Run a single fixture case by id")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    model = (
        os.getenv("QUERY_ROUTER_MODEL")
        or os.getenv("QUERY_REWRITER_MODEL")
        or os.getenv("QUERY_NORMALIZER_MODEL")
        or "gpt-6-luna"
    )
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not configured.")
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if args.case:
        cases = [case for case in cases if case["id"] == args.case]
        if not cases:
            raise SystemExit(f"Unknown fixture case: {args.case}")

    client = OpenAI(timeout=40.0, max_retries=1)
    rows = []
    try:
        for case in cases:
            route = case["router_fixture"]
            if route["relationship"] != case["expected_relationship"]:
                raise SystemExit(f"Fixture relationship mismatch: {case['id']}")
            context = _selected_context(case, route)
            allowed_indices = {
                index for index in route["selected_message_indices"]
                if case["history"][index].get("role") == "user"
            }
            if not set(case["source_indices"]).issubset(allowed_indices):
                raise SystemExit(f"Fixture fact source is outside selected context: {case['id']}")

            baseline_queries = [item["canonical_query"] for item in route["requests"]]
            baseline_pass = score_query(" ".join(baseline_queries), case)
            resolution = None
            status = "not_needed"
            input_tokens = output_tokens = resolver_latency_ms = 0
            rejected_output = False
            source_index_violation = False
            unsupported_anchor_rejection = False
            if route["relationship"] == "follow_up" and any(
                item.get("search_documents") and not item.get("is_ambiguous")
                for item in route["requests"]
            ):
                started = perf_counter()
                try:
                    result = resolve_followup_queries(
                        client,
                        current_question=case["question"],
                        requests=route["requests"],
                        selected_context=context,
                        user_sources=bound_user_sources(
                            case["history"], allowed_indices=allowed_indices
                        ),
                        allowed_source_indices=allowed_indices,
                        active_topic_state="",
                        model=model,
                    )
                    resolution = result["requests"]
                    status = (
                        "clarification"
                        if any(item["status"] == "clarification" for item in resolution.values())
                        else "resolved"
                    )
                    input_tokens = result["usage"]["input_tokens"]
                    output_tokens = result["usage"]["output_tokens"]
                except ContextResolutionError as exc:
                    status = f"validation_failed:{str(exc)}"
                    rejected_output = True
                    source_index_violation = "outside router-selected" in str(exc)
                    unsupported_anchor_rejection = "unsupported" in str(exc)
                except Exception as exc:
                    status = f"resolver_error:{type(exc).__name__}"
                    rejected_output = True
                resolver_latency_ms = round((perf_counter() - started) * 1000)

            candidate_pass, clarification, used_indices = _score_candidate(
                case, route, resolution
            )
            candidate_queries = (
                " ".join(
                    item.get("resolved_query", "")
                    or item.get("clarification_question", "")
                    for item in (resolution or {}).values()
                )
                if resolution is not None
                else " ".join(baseline_queries)
            )
            candidate_forbidden = any(
                anchor.casefold() in " ".join(
                    [candidate_queries]
                ).casefold()
                for anchor in case["forbidden_anchors"]
            )
            rows.append({
                "id": case["id"],
                "critical": case["critical"],
                "baseline_pass": baseline_pass,
                "candidate_pass": candidate_pass,
                "candidate_resolution": status,
                "clarification_returned": clarification,
                "used_source_indices": used_indices,
                "source_index_violation": source_index_violation,
                "unsupported_anchor_rejection": unsupported_anchor_rejection,
                "unsupported_added_fact": candidate_forbidden,
                "rejected_candidate_output": rejected_output,
                "resolver_latency_ms": resolver_latency_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            })
            print(
                f"{case['id']}: baseline={'pass' if baseline_pass else 'fail'} "
                f"candidate={'pass' if candidate_pass else 'fail'} ({status})"
            )
    finally:
        client.close()

    critical = [row for row in rows if row["critical"]]
    baseline_score = sum(row["baseline_pass"] for row in rows) / len(rows)
    candidate_score = sum(row["candidate_pass"] for row in rows) / len(rows)
    report = {
        "model": model,
        "cases": len(rows),
        "router_calls": 0,
        "vector_store_calls": 0,
        "answer_model_calls": 0,
        "baseline_passed": sum(row["baseline_pass"] for row in rows),
        "candidate_passed": sum(row["candidate_pass"] for row in rows),
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "critical_passed": sum(row["candidate_pass"] for row in critical),
        "critical_total": len(critical),
        "unsupported_added_facts": sum(row["unsupported_added_fact"] for row in rows),
        "source_index_violations": sum(row["source_index_violation"] for row in rows),
        "clarification_search_cases": 0,
        "rejected_candidate_outputs": sum(row["rejected_candidate_output"] for row in rows),
        "input_tokens_total": sum(row["input_tokens"] for row in rows),
        "output_tokens_total": sum(row["output_tokens"] for row in rows),
        "resolver_latency_ms_total": sum(row["resolver_latency_ms"] for row in rows),
        "candidate_adoptable": None if args.case else (
            all(row["candidate_pass"] for row in critical)
            and sum(row["unsupported_added_fact"] for row in rows) == 0
            and sum(row["source_index_violation"] for row in rows) == 0
            and candidate_score > baseline_score
        ),
        "results": rows,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
