"""Live A/B/C Vector Store comparison; network access requires --live."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET = ROOT / "tests" / "eval" / "gold.json"
OUTPUT_DIR = ROOT / "eval_results"
TOP_K = 8


def load_dataset():
    payload = json.loads(DATASET.read_text(encoding="utf-8"))
    items = payload.get("items")
    if payload.get("schema_version") != 1 or not isinstance(items, list) or len(items) < 40:
        raise ValueError("gold.json must contain at least 40 items")
    ids = set()
    for item in items:
        required = {"id", "question", "language", "expected_filename", "expected_phrase", "kind"}
        if not required.issubset(item) or item["id"] in ids:
            raise ValueError(f"Invalid or duplicate item: {item.get('id')}")
        if item["language"] not in {"mn", "latin-mn", "en"}:
            raise ValueError(f"Invalid language on {item['id']}")
        if item["kind"] not in {"answerable", "unanswerable", "restricted", "followup"}:
            raise ValueError(f"Invalid kind on {item['id']}")
        ids.add(item["id"])
    return payload


def _is_relevant(result, expected_filename, expected_phrase):
    return (
        result.get("filename", "").casefold() == expected_filename.casefold()
        and expected_phrase.casefold() in result.get("text", "").casefold()
    )


def _reciprocal_rank(results, expected_filename, expected_phrase):
    if (
        not expected_filename or expected_filename.startswith("TODO:")
        or not expected_phrase or expected_phrase.startswith("TODO:")
    ):
        return None
    for rank, item in enumerate(results, 1):
        if _is_relevant(item, expected_filename, expected_phrase):
            return 1 / rank
    return 0.0


def _summarize(records):
    scored = [row for row in records if row["mrr"] is not None]
    metrics = {}
    for strategy in ("A", "B", "C"):
        for per_query in (5, 10, 15):
            rows = [r for r in scored if r["strategy"] == strategy and r["per_query"] == per_query]
            key = f"{strategy}/per_query={per_query}"
            metrics[key] = {
                "labeled_items": len(rows),
                "hit_at_5": sum(r["hit_at_5"] for r in rows) / len(rows) if rows else None,
                "hit_at_8": sum(r["hit_at_8"] for r in rows) / len(rows) if rows else None,
                "mrr": sum(r["mrr"] for r in rows) / len(rows) if rows else None,
                "vector_store_calls": sum(r["vector_store_calls"] for r in rows),
                "calls_per_question": sum(r["vector_store_calls"] for r in rows) / len(rows) if rows else None,
            }
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Call the configured Router and Vector Store")
    parser.add_argument("--validate-only", action="store_true", help="Validate the gold set without API access")
    parser.add_argument("--limit", type=int, help="Limit the live run to the first N items")
    args = parser.parse_args()
    payload = load_dataset()
    if args.validate_only:
        print(f"Validated {len(payload['items'])} items; no API calls made.")
        return
    if not args.live:
        parser.error("Pass --live to make paid Router and Vector Store calls, or --validate-only.")

    from dotenv import load_dotenv
    from openai import OpenAI
    from multi_query_retrieval import build_intent_query_groups, search_intent_groups
    from query_precheck import request_restriction_candidates
    from semantic_router import rewrite_queries

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    vector_store_id = os.getenv("OPENAI_VECTOR_STORE_ID")
    if not api_key or not vector_store_id:
        raise SystemExit("OPENAI_API_KEY and OPENAI_VECTOR_STORE_ID must be configured.")
    model = (
        os.getenv("QUERY_ROUTER_MODEL") or os.getenv("QUERY_REWRITER_MODEL")
        or os.getenv("QUERY_NORMALIZER_MODEL") or "gpt-6-luna"
    )
    items = payload["items"][:args.limit] if args.limit else payload["items"]
    client = OpenAI(api_key=api_key, timeout=30.0, max_retries=1)
    rows = []
    try:
        for index, item in enumerate(items, 1):
            route = rewrite_queries(
                client, item["question"], history=item.get("history", []), model=model,
                include_alternatives=True, route_strategy="stage7_candidate",
            )
            safe_requests = [
                request for request in route.get("requests", [])
                if not request.get("restriction_modules")
                and not request_restriction_candidates(request.get("canonical_query", ""))
            ]
            for strategy in ("A", "B", "C"):
                groups = build_intent_query_groups(
                    item["question"], safe_requests,
                    multi_query_enabled=True, strategy=strategy,
                )
                for per_query in (5, 10, 15):
                    capture = {}
                    results = search_intent_groups(
                        client, vector_store_id, groups, per_query=per_query,
                        final_k=TOP_K, trace_capture=capture,
                    )
                    expected = item["expected_filename"]
                    expected_phrase = item["expected_phrase"]
                    reciprocal_rank = _reciprocal_rank(results, expected, expected_phrase)
                    row = {
                        "id": item["id"], "strategy": strategy,
                        "per_query": per_query,
                        "expected_filename": expected,
                        "hit_at_5": None if reciprocal_rank is None else any(
                            _is_relevant(r, expected, expected_phrase) for r in results[:5]
                        ),
                        "hit_at_8": None if reciprocal_rank is None else any(
                            _is_relevant(r, expected, expected_phrase) for r in results[:8]
                        ),
                        "mrr": reciprocal_rank,
                        "vector_store_calls": len(capture.get("queries", [])),
                        "result_filenames": list(dict.fromkeys(r.get("filename") for r in results)),
                    }
                    rows.append(row)
            print(f"Completed {index}/{len(items)} items")
    finally:
        client.close()

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET.relative_to(ROOT)),
        "router_model": model,
        "gold_items": len(items),
        "gold_labels_todo": sum(
            item["expected_filename"].startswith("TODO:") or item["expected_phrase"].startswith("TODO:")
            for item in items
        ),
        "metrics": _summarize(rows),
        "rows": rows,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "retrieval.json"
    md_path = OUTPUT_DIR / "retrieval.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["| Strategy | per_query | Labeled | Hit@5 | Hit@8 | MRR | Calls/q |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key, values in summary["metrics"].items():
        def fmt(value): return "pending" if value is None else f"{value:.3f}"
        strategy, setting = key.split("/")
        lines.append(f"| {strategy} | {setting.split('=')[1]} | {values['labeled_items']} | {fmt(values['hit_at_5'])} | {fmt(values['hit_at_8'])} | {fmt(values['mrr'])} | {fmt(values['calls_per_question'])} |")
    lines.extend(["", f"Unverified gold labels: {summary['gold_labels_todo']}/{summary['gold_items']}."])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"JSON: {json_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
