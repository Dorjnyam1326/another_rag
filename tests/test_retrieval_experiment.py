import json
import unittest
from types import SimpleNamespace

from evaluation.retrieval_experiment.metrics import retrieval_metrics
from evaluation.retrieval_experiment.retrieval import (
    build_strategy_groups,
    compare_search_strategies,
    filter_searchable_requests,
)
from evaluation.retrieval_experiment.run import (
    load_dataset,
    score_reviewed_results,
    validate_dataset,
)


class FakeSearch:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data=self.results_by_query.get(kwargs["query"], []))


class RetrievalExperimentTests(unittest.TestCase):
    def test_pilot_dataset_preserves_session_turns_and_pending_labels(self):
        dataset = load_dataset()
        counts = validate_dataset(dataset)
        self.assertEqual(counts["case_count"], 14)
        self.assertEqual(counts["turn_count"], 20)
        self.assertEqual(counts["scenario_38_39_turn_count"], 6)
        self.assertEqual(counts["reviewed_turn_count"], 0)
        self.assertEqual(counts["live_operation_estimate"]["route_calls_max"], 20)
        self.assertEqual(
            counts["live_operation_estimate"]["normalization_calls_max"], 160
        )
        self.assertEqual(
            counts["live_operation_estimate"]["vector_store_search_calls_max"], 480
        )
        case39 = next(case for case in dataset["cases"] if case["source_case_id"] == 39)
        self.assertTrue(case39["turns"][2]["session_continuation"])
        self.assertFalse(case39["turns"][2]["expected_semantic_follow_up"])

    def test_query_strategies_and_mixed_restriction_keep_intents_separate(self):
        routes = [
            {
                "intent": "Хувийн цалингийн дүн",
                "topic": "payroll",
                "request_type": "amount",
                "data_scope": "personal",
                "canonical_query": "Энэ сарын миний цалин хэд вэ?",
                "search_documents": True,
                "is_ambiguous": False,
                "alternatives": [],
            },
            {
                "intent": "Хоцролтын бүртгэл",
                "topic": "attendance",
                "request_type": "procedure",
                "data_scope": "general",
                "canonical_query": "Нэг цаг хоцорсныг хэрхэн бүртгүүлэх вэ?",
                "search_documents": True,
                "is_ambiguous": False,
                "alternatives": ["Хоцролтын бүртгэлийн журам"],
            },
        ]
        original = "Нэг цаг хоцорсныг бүртгэх болон энэ сарын миний цалинг тооцох"
        safe_original, searchable, restricted, prepared = filter_searchable_requests(original, routes)
        self.assertEqual(len(restricted), 1)
        self.assertEqual([item["intent"] for item in searchable], ["Хоцролтын бүртгэл"])
        self.assertEqual(safe_original, routes[1]["canonical_query"])
        groups_b = build_strategy_groups(safe_original, prepared, "B")
        groups_c = build_strategy_groups(safe_original, prepared, "C")
        self.assertEqual(groups_b, groups_c)
        self.assertTrue(all("цалин" not in query.casefold() for group in groups_b for _, query in group["queries"]))

    def test_cached_searches_fuse_with_recomputable_attribution_and_c_is_direct(self):
        def match(file_id, text, score):
            return SimpleNamespace(
                file_id=file_id,
                filename=f"{file_id}.docx",
                attributes={"section": "policy"},
                score=score,
                content=[SimpleNamespace(text=text)],
            )

        fake = FakeSearch({
            "raw query": [match("f1", "passage A", 0.9), match("f1", "passage B", 0.8)],
            "resolved query": [match("f1", "passage B", 0.95), match("f2", "passage C", 0.7)],
            "alternative": [match("f3", "passage D", 0.6)],
        })
        client = SimpleNamespace(vector_stores=SimpleNamespace(search=fake.search))
        groups = {
            "A": [{"intent": "leave", "queries": [("original", "raw query"), ("canonical", "resolved query"), ("alternative_1", "alternative")]}],
            "B": [{"intent": "leave", "queries": [("original", "raw query"), ("canonical", "resolved query")]}],
            "C": [{"intent": "leave", "queries": [("canonical", "resolved query")]}],
        }
        cache = {}
        comparison = compare_search_strategies(
            client,
            "vs-not-printed",
            groups,
            vector_store_fingerprint="store-fingerprint",
            search_cache=cache,
        )
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(comparison["physical_search_operations"], 3)
        self.assertEqual(comparison["strategies"]["A"]["logical_query_group_searches"], 3)
        self.assertEqual(comparison["strategies"]["C"]["ranked_results"][0]["filename"], "f1.docx")
        self.assertIsNone(comparison["strategies"]["C"]["ranked_results"][0]["rrf_score"])
        fused = comparison["strategies"]["A"]["ranked_results"][0]
        self.assertEqual(fused["filename"], "f1.docx")
        self.assertEqual(len(fused["contributions"]), 2)
        self.assertAlmostEqual(fused["rrf_score"], 1 / 62 + 1 / 61)
        self.assertIn("attributes", fused)
        self.assertNotIn("_text", fused)
        self.assertIn("passage_id", comparison["strategies"]["A"]["queries"][0]["results"][0])

    def test_metrics_require_complete_human_judgments_and_compute_top_k(self):
        ranking = [f"p{i}" for i in range(1, 11)]
        pending = retrieval_metrics(ranking, {"status": "pending"})
        self.assertEqual(pending, {"status": "pending", "metrics": None})
        review = {
            "status": "reviewed",
            "relevant_passage_ids": ["p2", "p8"],
            "partial_passage_ids": ["p3"],
            "irrelevant_passage_ids": [f"p{i}" for i in (1, 4, 5, 6, 7, 9, 10)],
            "required_subintents": ["leave", "approval"],
            "relevant_by_subintent": {"leave": ["p2"], "approval": ["p8"]},
            "wrong_domain_passage_ids": ["p1"],
            "authority_status": "reviewed",
            "authoritative_passage_ids": ["p2"],
        }
        result = retrieval_metrics(ranking, review)
        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["metrics"]["hit_at_5"], 1)
        self.assertEqual(result["metrics"]["mrr_at_10"], 0.5)
        self.assertEqual(result["metrics"]["precision_at_5"], 0.2)
        self.assertEqual(result["metrics"]["subintent_coverage_at_5"], 0.5)

    def test_saved_candidate_reviews_can_be_scored_offline(self):
        ranking = [f"p{i}" for i in range(1, 11)]
        results = {
            "cases": [{
                "turns": [{
                    "review": {
                        "status": "reviewed",
                        "relevant_passage_ids": ["p2"],
                        "partial_passage_ids": [],
                        "irrelevant_passage_ids": [
                            f"p{i}" for i in (1, 3, 4, 5, 6, 7, 8, 9, 10)
                        ],
                        "required_subintents": [],
                        "relevant_by_subintent": {},
                        "wrong_domain_passage_ids": ["p1"],
                        "authority_status": "unknown",
                        "authoritative_passage_ids": [],
                    },
                    "strategies": {
                        key: {"ranked_results": [{"passage_id": item} for item in ranking]}
                        for key in ("A", "B", "C")
                    },
                }]
            }]
        }

        score_reviewed_results(results)

        for strategy in ("A", "B", "C"):
            metrics = results["cases"][0]["turns"][0]["strategies"][strategy]["metrics"]
            self.assertEqual(metrics["status"], "scored")
            self.assertEqual(metrics["metrics"]["hit_at_5"], 1)


if __name__ == "__main__":
    unittest.main()
