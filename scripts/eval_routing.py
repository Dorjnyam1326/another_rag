"""Offline refusal audit for the curated evaluation seed."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from query_precheck import precheck_query, request_restriction_candidates  # noqa: E402

DATASET = ROOT / "tests" / "eval" / "gold.json"
OUTPUT_DIR = ROOT / "eval_results"


def main():
    items = json.loads(DATASET.read_text(encoding="utf-8"))["items"]
    rows = []
    for item in items:
        if item["kind"] not in {"restricted", "answerable"}:
            continue
        if item["kind"] != "restricted" and "near_miss" not in item.get("tags", []):
            continue
        refusal = precheck_query(item["question"])
        modules = [refusal[0]] if refusal else request_restriction_candidates(item["question"])
        rows.append({
            "id": item["id"],
            "kind": item["kind"],
            "near_miss": "near_miss" in item.get("tags", []),
            "expected_refusal": item["kind"] == "restricted",
            "actual_refusal": bool(modules),
            "module": modules[0] if modules else None,
        })

    restricted = [row for row in rows if row["expected_refusal"]]
    near_miss = [row for row in rows if row["near_miss"]]
    missed = sum(not row["actual_refusal"] for row in restricted)
    false = sum(row["actual_refusal"] for row in near_miss)
    summary = {
        "dataset": str(DATASET.relative_to(ROOT)),
        "method": "local deterministic precheck (no model/API calls)",
        "restricted_items": len(restricted),
        "near_miss_items": len(near_miss),
        "false_refusal_count": false,
        "false_refusal_rate": false / len(near_miss) if near_miss else None,
        "missed_refusal_count": missed,
        "missed_refusal_rate": missed / len(restricted) if restricted else None,
        "rows": rows,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "routing.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("| Metric | Result |\n|---|---:|")
    print(f"| False-refusal rate | {summary['false_refusal_rate']:.1%} ({false}/{len(near_miss)}) |")
    print(f"| Missed-refusal rate | {summary['missed_refusal_rate']:.1%} ({missed}/{len(restricted)}) |")
    print(f"| Evaluated items | {len(rows)} |")
    print(f"JSON: {OUTPUT_DIR / 'routing.json'}")


if __name__ == "__main__":
    main()
