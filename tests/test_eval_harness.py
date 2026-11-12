import unittest

from scripts.eval_retrieval import _is_relevant, _reciprocal_rank, _summarize, load_dataset


class EvalHarnessTests(unittest.TestCase):
    def test_seed_has_required_size_and_schema(self):
        payload = load_dataset()
        self.assertGreaterEqual(len(payload["items"]), 40)

    def test_relevance_requires_filename_and_gold_phrase(self):
        expected = "policy.docx"
        phrase = "ажлын 10 өдрөөс"
        self.assertTrue(_is_relevant(
            {"filename": expected, "text": "Амралтын хэсэг ажлын 10 өдрөөс багагүй."},
            expected, phrase,
        ))
        self.assertFalse(_is_relevant(
            {"filename": expected, "text": "Амралтын тухай ерөнхий мэдээлэл."},
            expected, phrase,
        ))
        self.assertIsNone(_reciprocal_rank(
            [{"filename": expected, "text": phrase}], "TODO: verify", phrase,
        ))

    def test_retrieval_summary_reports_hit_rates_and_calls_per_question(self):
        records = [
            {"strategy": "A", "per_query": 5, "mrr": 1.0, "hit_at_5": True,
             "hit_at_8": True, "vector_store_calls": 2},
            {"strategy": "A", "per_query": 5, "mrr": 0.0, "hit_at_5": False,
             "hit_at_8": False, "vector_store_calls": 1},
        ]
        result = _summarize(records)["A/per_query=5"]
        self.assertEqual(result["hit_at_5"], 0.5)
        self.assertEqual(result["hit_at_8"], 0.5)
        self.assertEqual(result["mrr"], 0.5)
        self.assertEqual(result["calls_per_question"], 1.5)


if __name__ == "__main__":
    unittest.main()
