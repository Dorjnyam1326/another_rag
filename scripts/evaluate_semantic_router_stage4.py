"""Compare baseline and relationship-candidate routing on a fixed live set."""

import argparse
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "tests" / "fixtures" / "semantic_router_stage4.json"
sys.path.insert(0, str(ROOT))

from semantic_router import rewrite_queries  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="Run only a fixture case by id.")
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
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if args.case:
        cases = [case for case in cases if case["id"] == args.case]
        if not cases:
            raise SystemExit(f"Unknown fixture case: {args.case}")
    client = OpenAI(timeout=30.0, max_retries=1)
    results = []
    try:
        for case in cases:
            row = {"id": case["id"], "critical": case["critical"]}
            for label, strategy in (
                ("baseline", "baseline"),
                ("candidate", "relationship_candidate"),
            ):
                result = rewrite_queries(
                    client,
                    case["question"],
                    history=case["history"],
                    model=model,
                    include_alternatives=False,
                    route_strategy=strategy,
                    active_topic_state=case.get("active_topic_state", ""),
                )
                expected_relationship = case["relationship"]
                expected_context = case["context_indices"]
                # The baseline has only the old follow-up/topic-change flags.
                # Compare its effective behavior and also count its missing
                # distinction between independent and explicit topic changes.
                actual_relationship = result["relationship"]
                expected_behavior = (
                    "follow_up" if expected_relationship == "follow_up" else "independent"
                )
                actual_behavior = (
                    "follow_up" if result["is_follow_up"] else "independent"
                )
                row[label] = {
                    "relationship": actual_relationship,
                    "behavior_ok": actual_behavior == expected_behavior,
                    "relation_ok": actual_relationship == expected_relationship,
                    "context_indices": result["relevant_context_indices"],
                    "context_ok": (
                        result["relevant_context_indices"] == expected_context
                        and result.get("active_topic_state_used", False)
                        == case.get("use_active_topic_state", False)
                    ),
                    "input_tokens": result["usage"]["input_tokens"],
                    "output_tokens": result["usage"]["output_tokens"],
                }
            results.append(row)
            print(
                f"{case['id']}: baseline={row['baseline']['relationship']} "
                f"ctx={row['baseline']['context_indices']} | "
                f"candidate={row['candidate']['relationship']} "
                f"ctx={row['candidate']['context_indices']}"
            )
    finally:
        client.close()

    report = {}
    for label in ("baseline", "candidate"):
        rows = [row[label] for row in results]
        report[label] = {
            "behavior_correct": sum(item["behavior_ok"] for item in rows),
            "relationship_correct": sum(item["relation_ok"] for item in rows),
            "context_correct": sum(item["context_ok"] for item in rows),
            "combined_score": sum(
                item["relation_ok"] + item["context_ok"] for item in rows
            )
            / (2 * len(rows)),
            "input_tokens": sum(item["input_tokens"] for item in rows),
            "output_tokens": sum(item["output_tokens"] for item in rows),
            "critical_cases_passed": sum(
                row["critical"] and row[label]["relation_ok"] and row[label]["context_ok"]
                for row in results
            ),
            "critical_cases_total": sum(row["critical"] for row in results),
        }
    report["candidate_adoptable"] = None if args.case else (
        report["candidate"]["critical_cases_passed"]
        == report["candidate"]["critical_cases_total"]
        and report["candidate"]["combined_score"]
        > report["baseline"]["combined_score"]
    )
    report["model"] = model
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
