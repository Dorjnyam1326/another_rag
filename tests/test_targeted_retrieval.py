import unittest

from targeted_retrieval import (
    annotate_applicability,
    merge_intent_results,
    request_evidence_status,
    targeted_expansion_groups,
)
from multi_query_retrieval import format_evidence


class TargetedRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.request = {
            "intent": "Чөлөө авах журам",
            "canonical_query": "Чөлөө авах журам ямар байдаг вэ?",
            "search_documents": True,
            "is_ambiguous": False,
            "restriction_modules": [],
            "alternatives": ["Ажилтны чөлөө авах журам", "Чөлөө хүсэхэд ямар журам баримтлах вэ?"],
        }

    def test_uncovered_leave_expands_twice_but_sufficient_does_not(self):
        irrelevant = [{"file_id": "faq", "text": "Компанийн үнэт зүйл болон Skyteams цэс"}]
        self.assertEqual(request_evidence_status(self.request, irrelevant)["status"], "insufficient")
        for index, expected in enumerate(self.request["alternatives"]):
            groups = targeted_expansion_groups(
                [self.request], irrelevant, round_index=index,
                enabled_restrictions=("salary",), source_text="Чөлөөний журам",
            )
            self.assertEqual(groups[0]["queries"], [(f"alternative_{index + 1}", expected)])
        self.assertEqual(targeted_expansion_groups(
            [self.request], irrelevant, round_index=2,
            enabled_restrictions=("salary",), source_text="Чөлөөний журам",
        ), [])
        sufficient = [{"file_id": "leave", "text": "Чөлөө авах журамд хүсэлт гаргах нөхцөл тусгасан."}]
        self.assertEqual(targeted_expansion_groups(
            [self.request], sufficient, round_index=0,
            enabled_restrictions=("salary",), source_text="Чөлөөний журам",
        ), [])

    def test_restricted_and_unscoped_role_queries_never_expand_or_count_scoped_evidence(self):
        restricted = {**self.request, "restriction_modules": ["salary"]}
        self.assertEqual(targeted_expansion_groups(
            [restricted], [], round_index=0,
            enabled_restrictions=("salary",), source_text="Миний цалин хэд вэ?",
        ), [])
        call_center = {"file_id": "policy", "filename": "policy.docx",
                       "intents": [self.request["intent"]],
                       "text": "Дуудлагын төвийн ХЗ, БЗ, МА ажилтанд чөлөө олгоно."}
        self.assertEqual(request_evidence_status(self.request, [call_center])["status"], "insufficient")
        annotate_applicability([call_center], [self.request])
        self.assertIn("дуудлагын төв", call_center["applicability_notes"][0])
        self.assertIn("Applicability:", format_evidence([call_center]))

    def test_expansion_rejects_alternative_that_crosses_restriction(self):
        request = {**self.request, "alternatives": ["Миний сарын үндсэн цалин хэд вэ?"]}
        self.assertEqual(targeted_expansion_groups(
            [request], [], round_index=0,
            enabled_restrictions=("salary",), source_text="Чөлөө авах журам",
        ), [])

    def test_merge_deduplicates_and_keeps_intent_order(self):
        initial = [{"file_id": "same", "text": "leave rule", "intents": ["leave"],
                    "rrf_score": 0.01, "rrf_score_by_intent": {"leave": 0.01},
                    "query_provenance": [{"intent": "leave", "variant": "canonical", "rank": 1}]}]
        additional = [
            {"file_id": "same", "text": "leave rule", "intents": ["leave"],
             "rrf_score": 0.01, "rrf_score_by_intent": {"leave": 0.01},
             "query_provenance": [{"intent": "leave", "variant": "alternative_1", "rank": 1}]},
            {"file_id": "second", "text": "second rule", "intents": ["leave"],
             "rrf_score": 0.01, "rrf_score_by_intent": {"leave": 0.01},
             "query_provenance": []},
        ]
        merged = merge_intent_results(initial, additional, ["leave"], final_k=2)
        self.assertEqual([item["file_id"] for item in merged], ["same", "second"])
        self.assertEqual(len(merged[0]["query_provenance"]), 2)


if __name__ == "__main__":
    unittest.main()
