import unittest
from unittest.mock import patch

from evaluation.retrieval_experiment.stage8 import _ndcg, _passage_records, _safe_resolve


class Stage8RetrievalTests(unittest.TestCase):
    @patch("evaluation.retrieval_experiment.stage8.resolve_followup_queries")
    def test_resolver_receives_the_live_client(self, resolver):
        client = object()
        resolver.return_value = {"requests": []}

        self.assertEqual(_safe_resolve(client, current_question="Хэн батлах вэ?"), {"requests": []})
        resolver.assert_called_once_with(client, current_question="Хэн батлах вэ?")

    def test_ndcg_rewards_relevant_passages_at_higher_ranks(self):
        self.assertEqual(_ndcg([2, 0]), 1.0)
        self.assertLess(_ndcg([0, 2]), 1.0)
        self.assertEqual(_ndcg([0, 0]), 0.0)

    def test_report_passage_record_keeps_provenance_but_drops_text(self):
        passage = {
            "passage_id": "a" * 64,
            "file_id": "file-123",
            "filename": "leave-policy.docx",
            "text": "This text must not be saved in the audit report.",
            "attributes": {"status": "active"},
            "attributes_status": "available",
            "provider_scores": {"canonical": 0.88},
            "ranks": {"canonical": 1},
            "rrf_score": 1 / 61,
            "final_rank": 1,
            "intents": ["leave approval"],
        }
        record = _passage_records([passage], {"a" * 64: 2})[0]

        self.assertNotIn("text", record)
        self.assertEqual(record["file_id"], "file-123")
        self.assertEqual(record["provider_scores"], {"canonical": 0.88})
        self.assertEqual(record["ranks"], {"canonical": 1})
        self.assertEqual(record["rrf_score"], 1 / 61)
        self.assertEqual(record["final_rank"], 1)
        self.assertEqual(record["relevance"], 2)


if __name__ == "__main__":
    unittest.main()
