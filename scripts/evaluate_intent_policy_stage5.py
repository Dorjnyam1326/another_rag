"""Run the approved router-only paired Stage 5 evaluation."""

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "intent_policy_stage5.json"
sys.path.insert(0, str(ROOT))

from query_precheck import precheck_query  # noqa: E402
from semantic_router import rewrite_queries  # noqa: E402


def _request_modules(request):
    modules = set(request.get("restriction_modules", []))
    restricted = precheck_query(request.get("canonical_query", ""))
    if restricted:
        modules.add(restricted[0])
    return modules


def _matches_expected(request, expected):
    text = " ".join(
        [request.get("intent", ""), request.get("canonical_query", "")]
    ).casefold()
    for term in expected["match_terms"]:
        alternatives = term if isinstance(term, list) else [term]
        if not any(value.casefold() in text for value in alternatives):
            return False
    return True


def score(route, case):
    requests = route["requests"]
    expected = case["expected"]
    if len(requests) != len(expected):
        return {"passed": False, "restricted_query_leak": False}
    unmatched = list(requests)
    allowed_queries = []
    classification_ok = True
    for item in expected:
        match = next(
            (request for request in unmatched if _matches_expected(request, item)),
            None,
        )
        if match is None:
            return {"passed": False, "restricted_query_leak": False}
        unmatched.remove(match)
        modules = _request_modules(match)
        if item["restriction"]:
            if item["restriction"] not in modules:
                classification_ok = False
        elif modules:
            classification_ok = False
        if not modules:
            allowed_queries.append(match.get("canonical_query", "").casefold())

    combined_allowed_query = " ".join(allowed_queries)
    leaked = any(
        term.casefold() in combined_allowed_query
        for term in case.get("restricted_terms", [])
    )
    return {
        "passed": classification_ok and not leaked,
        "restricted_query_leak": leaked,
    }


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
            row = {"id": case["id"], "critical": case["critical"]}
            for label, strategy in (
                ("baseline", "relationship_candidate"),
                ("candidate", "intent_policy_candidate"),
            ):
                started = perf_counter()
                try:
                    route = rewrite_queries(
                        client,
                        case["question"],
                        history=[],
                        model=model,
                        include_alternatives=True,
                        route_strategy=strategy,
                    )
                    case_score = score(route, case)
                    row[label] = {
                        "passed": case_score["passed"],
                        "restricted_query_leak": case_score["restricted_query_leak"],
                        "request_count": len(route["requests"]),
                        "input_tokens": route["usage"]["input_tokens"],
                        "output_tokens": route["usage"]["output_tokens"],
                        "latency_ms": round((perf_counter() - started) * 1000),
                    }
                except Exception as exc:
                    row[label] = {
                        "passed": False,
                        "error_type": type(exc).__name__,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "latency_ms": round((perf_counter() - started) * 1000),
                    }
            rows.append(row)
            print(
                f"{case['id']}: baseline={'pass' if row['baseline']['passed'] else 'fail'} "
                f"candidate={'pass' if row['candidate']['passed'] else 'fail'}"
            )
    finally:
        client.close()

    baseline = [row["baseline"] for row in rows]
    candidate = [row["candidate"] for row in rows]
    critical = [row for row in rows if row["critical"]]
    baseline_score = sum(item["passed"] for item in baseline) / len(rows)
    candidate_score = sum(item["passed"] for item in candidate) / len(rows)
    report = {
        "model": model,
        "cases": len(rows),
        "baseline_passed": sum(item["passed"] for item in baseline),
        "candidate_passed": sum(item["passed"] for item in candidate),
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "critical_passed": sum(row["candidate"]["passed"] for row in critical),
        "critical_total": len(critical),
        "restricted_query_leaks": sum(
            row["candidate"].get("restricted_query_leak", False)
            for row in rows
        ),
        "input_tokens_total": sum(
            item.get("input_tokens", 0) for item in [*baseline, *candidate]
        ),
        "output_tokens_total": sum(
            item.get("output_tokens", 0) for item in [*baseline, *candidate]
        ),
        "latency_ms_total": sum(
            item.get("latency_ms", 0) for item in [*baseline, *candidate]
        ),
        "candidate_adoptable": None if args.case else (
            all(row["candidate"]["passed"] for row in critical)
            and candidate_score > baseline_score
            and sum(
                row["candidate"].get("restricted_query_leak", False)
                for row in rows
            ) == 0
        ),
        "results": rows,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
