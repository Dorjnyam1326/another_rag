import json
import unittest
from types import SimpleNamespace

from evaluation.retrieval_experiment import stage10


class Stage10RetrievalTests(unittest.TestCase):
    def test_holdout_has_three_real_leave_cases_and_human_review_guides(self):
        dataset = stage10.load_holdout()
        self.assertEqual(len(dataset["cases"]), 3)
        self.assertTrue(all(case["expected_coverage"] for case in dataset["cases"]))
        self.assertTrue(all(case["process_focused_query"] for case in dataset["cases"]))
        self.assertEqual(
            next(case for case in dataset["cases"] if case["case_id"] == "leave-07")["clock_ablation_query"],
            "3 цагийн чөлөө авахад хэн зөвшөөрч, хэдэн цагийн өмнө мэдэгдэх вэ?",
        )
        self.assertTrue(all("known_answer_retrieval_failure" in case["tags"] for case in dataset["cases"]))
        self.assertTrue(all("not verified" in case["historical_candidate_status"] for case in dataset["cases"]))

    def test_variants_are_raw_canonical_and_process_focused(self):
        variants = stage10.build_query_variants(
            "raw question",
            "process query",
            [
                {"intent": "approval", "canonical_query": "Who approves?", "search_documents": True},
                {"intent": "submission", "canonical_query": "Where to submit?", "search_documents": True},
                {"intent": "restricted", "canonical_query": "Private salary", "search_documents": False},
            ],
        )
        self.assertEqual(
            variants,
            [
                ("raw", "raw question"),
                ("canonical", "Who approves?\nWhere to submit?"),
                ("process_focused", "process query"),
            ],
        )

    def test_restrictions_block_the_whole_case_before_search(self):
        variants = [("raw", "Хэлтсийн цалингийн дүн хэд вэ?")]
        reasons = stage10.restriction_reasons(
            variants[0][1],
            [{
                "intent": "Хэлтсийн цалингийн дүн",
                "canonical_query": "Хэлтсийн цалингийн дүн хэд вэ?",
                "restriction_modules": ["salary"],
            }],
            variants,
            ("salary",),
        )
        self.assertTrue(reasons)

    def test_each_distinct_variant_is_searched_once_and_top_five_are_ranked(self):
        class VectorStores:
            calls = []

            def search(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(data=[
                    SimpleNamespace(
                        file_id=f"file-{index}",
                        filename="leave-policy.docx",
                        score=1 / index,
                        attributes={"section": "10.4"},
                        content=[{"text": f"passage {index}"}],
                    )
                    for index in range(1, 7)
                ])

        stores = VectorStores()
        client = SimpleNamespace(vector_stores=stores)
        variants = [("raw", "same"), ("canonical", "same "), ("process_focused", "policy")]
        results, calls = stage10.search_variants(client, "vs-local-test", variants)

        self.assertEqual(calls, 2)
        self.assertEqual(len(stores.calls), 2)
        self.assertEqual(stores.calls[0]["max_num_results"], 5)
        self.assertFalse(stores.calls[0]["rewrite_query"])
        self.assertEqual([item["rank"] for item in results[0]["passages"]], [1, 2, 3, 4, 5])
        self.assertEqual(results[1]["status"], "duplicate_query")
        self.assertEqual(results[2]["variant"], "process_focused")


if __name__ == "__main__":
    unittest.main()
