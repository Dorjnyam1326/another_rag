import unittest
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

from evaluation.retrieval_experiment import stage9
from restriction_modules import RESTRICTION_MODULES


class Stage9Tests(unittest.TestCase):
    def test_fresh_holdout_has_required_coverage(self):
        data = stage9.load_holdout()
        self.assertEqual(len(data["turns"]), 14)
        tags = {tag for turn in data["turns"] for tag in turn["tags"]}
        self.assertTrue({"single_intent", "multi_intent", "latin_mongolian",
                         "corrected_followup", "restricted_plus_allowed"}.issubset(tags))
        correction = next(turn for turn in data["turns"] if "corrected_followup" in turn["tags"])
        self.assertEqual(correction["history"][0]["source_message_index"], 0)

    def test_corrected_followup_never_reuses_raw_query(self):
        requests = [{"intent": "3 цагийн чөлөө", "canonical_query": "3 цагийн чөлөө авах журам",
                     "search_documents": True, "is_ambiguous": False,
                     "restriction_modules": [], "alternatives": []}]
        groups = stage9.build_arm_groups("3 өдөр биш 3 цаг", requests, strategy="A", follow_up=True)
        self.assertEqual(groups[0]["queries"], [("canonical", "3 цагийн чөлөө авах журам")])

    def test_latin_mixed_payroll_and_leave_requests_pass_faithfulness_guard(self):
        source = "tsalin hezee buudag ve, bas tsalintai chuluu awah nuhtsul yu ve?"
        groups = [
            {"intent": "Цалин олгох өдөр", "queries": [("canonical", "Цалин хэзээ олгодог вэ?")]},
            {"intent": "Цалинтай чөлөө", "queries": [("canonical", "Цалинтай чөлөө авах нөхцөл юу вэ?")]},
        ]
        safe, rejected = stage9._guard_groups(groups, source, ("salary",))
        self.assertEqual(safe, groups)
        self.assertEqual(rejected, [])
        restricted, rejected = stage9._guard_groups(
            [{"intent": "Хувийн цалингийн дүн", "queries": [("canonical", "Миний сарын үндсэн цалин хэд вэ?")]}],
            source,
            ("salary",),
        )
        self.assertEqual(restricted, [])
        self.assertEqual(rejected[0]["restrictions"], ["salary"])

    def test_a_b_c_use_the_defined_single_intent_query_sets(self):
        request = {"intent": "Чөлөө", "canonical_query": "Чөлөө авах журам",
                   "search_documents": True, "is_ambiguous": False,
                   "restriction_modules": [], "alternatives": ["Хүсэлт батлах арга", "Хугацаа"]}
        a = stage9.build_arm_groups("leave request", [request], strategy="A")
        b = stage9.build_arm_groups("leave request", [request], strategy="B")
        c = stage9.build_arm_groups("leave request", [request], strategy="C")
        self.assertEqual([label for label, _ in a[0]["queries"]], ["original", "canonical", "alternative_1", "alternative_2"])
        self.assertEqual([label for label, _ in b[0]["queries"]], ["original", "canonical"])
        self.assertEqual([label for label, _ in c[0]["queries"]], ["canonical"])

    def test_answer_text_extracts_responses_api_text_blocks(self):
        self.assertEqual(stage9._answer_text([{"type": "text", "text": "Хариулт"},
                                              {"type": "file_citation", "filename": "ignored"}]), "Хариулт")

    def test_sufficiency_retries_output_limit_then_parses_the_decision(self):
        incomplete = SimpleNamespace(status="incomplete",
                                     incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                                     usage=SimpleNamespace(input_tokens=4, output_tokens=2), output_text="")
        complete = SimpleNamespace(status="completed", usage=SimpleNamespace(input_tokens=5, output_tokens=3),
                                   output_text='{"sufficient":true,"reason":"covered"}')
        client = SimpleNamespace(responses=SimpleNamespace(create=Mock(side_effect=[incomplete, complete])))
        result, usage = stage9._sufficiency(client, "judge", {"expected_coverage": ["журам"]},
                                            {"intent": "leave", "canonical_query": "Чөлөө"},
                                            [{"filename": "policy.pdf", "text": "passage"}])
        self.assertTrue(result["sufficient"])
        self.assertEqual(usage, {"input_tokens": 9, "output_tokens": 5, "calls": 2})
        self.assertEqual(client.responses.create.call_args_list[0].kwargs["max_output_tokens"], 200)
        self.assertEqual(client.responses.create.call_args_list[1].kwargs["max_output_tokens"], 400)

    def test_non_token_incomplete_sufficiency_fails_to_expansion(self):
        incomplete = SimpleNamespace(status="incomplete",
                                     incomplete_details=SimpleNamespace(reason="content_filter"),
                                     usage=SimpleNamespace(input_tokens=2, output_tokens=0), output_text="")
        client = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=incomplete)))
        result, usage = stage9._sufficiency(client, "judge", {"expected_coverage": ["журам"]},
                                            {"intent": "leave", "canonical_query": "Чөлөө"},
                                            [{"filename": "policy.pdf", "text": "passage"}])
        self.assertFalse(result["sufficient"])
        self.assertEqual(result["judge_error"], "content_filter")
        self.assertEqual(usage["calls"], 1)

    def test_restricted_query_part_is_removed_before_arm_queries(self):
        requests = [
            {"intent": "Чөлөөний журам", "canonical_query": "Нэг өдрийн чөлөөний журам",
             "search_documents": True, "is_ambiguous": False, "restriction_modules": [], "alternatives": []},
            {"intent": "Хувийн цалингийн дүн", "canonical_query": "Миний сарын үндсэн цалин хэдэн төгрөг вэ?",
             "search_documents": True, "is_ambiguous": False, "restriction_modules": ["salary"], "alternatives": []},
        ]
        prepared, allowed, _, restricted, original = stage9._restricted_parts(
            "Чөлөөний журам болон миний сарын үндсэн цалин хэд вэ?", requests, ("salary",)
        )
        self.assertEqual(len(restricted), 1)
        self.assertEqual(original, "")
        groups = stage9.build_arm_groups(original, prepared, strategy="A", restricted=True)
        self.assertEqual(len(allowed), 1)
        self.assertTrue(all("цалин" not in query for group in groups for _, query in group["queries"]))

    @patch.object(stage9, "_search", return_value=([], {"calls": 1, "latency_ms": 1}))
    def test_restricted_text_never_reaches_search_or_answer_generation(self, search):
        question = "Чөлөөний журам болон миний сарын үндсэн цалин хэд вэ?"
        requests = [
            {"intent": "Чөлөөний журам", "canonical_query": "Нэг өдрийн чөлөөний журам",
             "search_documents": True, "is_ambiguous": False, "restriction_modules": [], "alternatives": []},
            {"intent": "Хувийн цалингийн дүн", "canonical_query": "Миний сарын үндсэн цалин хэдэн төгрөг вэ?",
             "search_documents": True, "is_ambiguous": False, "restriction_modules": ["salary"], "alternatives": []},
        ]
        prepared, allowed, _, restricted, original = stage9._restricted_parts(question, requests, ("salary",))
        groups = stage9.build_arm_groups(original, prepared, strategy="A", restricted=True)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}), patch("langchain_openai.ChatOpenAI") as chat:
            chat.return_value.invoke.return_value = SimpleNamespace(content="Чөлөөний журмын хариу.", usage_metadata={})
            arm = stage9._run_arm(object(), "vs", {"expected_coverage": ["Чөлөө"], "restricted": restricted},
                                  allowed, groups, strategy="A", model="unused", judge_model="unused",
                                  source_text=question, enabled=("salary",))
        self.assertNotIn("цалин", str(search.call_args).casefold())
        self.assertNotIn("миний сарын үндсэн цалин хэдэн төгрөг", str(chat.return_value.invoke.call_args).casefold())
        self.assertIn(RESTRICTION_MODULES["salary"]["refusal"], arm["answer"])

    @patch.object(stage9, "_search")
    @patch.object(stage9, "_sufficiency", return_value=({"sufficient": False, "reason": "missing"}, {"input_tokens": 2, "output_tokens": 1}))
    def test_conditional_strategy_expands_only_after_insufficient_evidence(self, sufficiency, search):
        first = [{"passage_id": "canonical", "file_id": "f1", "text": "A", "filename": "a", "intents": ["leave"],
                  "query_provenance": [{"variant": "canonical", "rank": 1}]}]
        second = [{"passage_id": "alternative", "file_id": "f2", "text": "B", "filename": "b", "intents": ["leave"],
                   "query_provenance": [{"variant": "alternative_1", "rank": 1}]}]
        search.side_effect = [(first, {"calls": 1, "latency_ms": 4}), (second, {"calls": 1, "latency_ms": 5})]
        request = {"intent": "leave", "canonical_query": "Чөлөөний журам", "alternatives": ["Богино чөлөө хүсэх арга"]}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}), patch("langchain_openai.ChatOpenAI") as chat:
            chat.return_value.invoke.return_value = SimpleNamespace(content="ok", usage_metadata={})
            arm = stage9._run_arm(object(), "vs", {"expected_coverage": ["Хүсэлт"]}, [request],
                                  [{"intent": "leave", "queries": [("canonical", "Чөлөөний журам")]}],
                                  strategy="C", model="unused", judge_model="unused",
                                  source_text="Чөлөөний журам", enabled=())
        self.assertEqual(search.call_count, 2)
        self.assertEqual(arm["calls"], 2)
        self.assertEqual(arm["expanded_queries"][0]["variant"], "alternative_1")
        self.assertEqual({item["passage_id"] for item in arm["passages"]}, {"canonical", "alternative"})

    @patch.object(stage9, "_search", return_value=([{"passage_id": "canonical", "file_id": "f1", "text": "A",
                                                       "filename": "a", "intents": ["leave"]}],
                                                    {"calls": 1, "latency_ms": 4}))
    @patch.object(stage9, "_sufficiency", return_value=({"sufficient": True, "reason": "covered"},
                                                          {"input_tokens": 2, "output_tokens": 1}))
    def test_conditional_strategy_stops_when_canonical_evidence_is_sufficient(self, sufficiency, search):
        request = {"intent": "leave", "canonical_query": "Чөлөөний журам", "alternatives": ["Нэмэлт query"]}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}), patch("langchain_openai.ChatOpenAI") as chat:
            chat.return_value.invoke.return_value = SimpleNamespace(content="ok", usage_metadata={})
            arm = stage9._run_arm(object(), "vs", {"expected_coverage": ["Хүсэлт"]}, [request],
                                  [{"intent": "leave", "queries": [("canonical", "Чөлөөний журам")]}],
                                  strategy="C", model="unused", judge_model="unused",
                                  source_text="Чөлөөний журам", enabled=())
        search.assert_called_once()
        self.assertEqual(arm["calls"], 1)
        self.assertNotIn("expanded_queries", arm)

    @patch.object(stage9, "search_intent_groups", return_value=[])
    def test_search_records_logical_vector_calls_per_arm(self, search):
        _, metrics = stage9._search(object(), "vs", [{"intent": "one", "queries": [("canonical", "A"), ("alternative_1", "B")] }])
        self.assertEqual(metrics["calls"], 2)
        self.assertEqual(len(metrics["queries"]), 2)
        search.assert_called_once()

    def test_blinded_review_metrics_calculate_quality_and_latency(self):
        results = {"turns": [{"turn_id": "h1", "expected_coverage": ["a", "b"], "strategies": {
            name: {"passages": [{"passage_id": "p1"}], "calls": count, "total_ms": latency,
                   "logical_searches": count, "vector_store_calls": count,
                   "retrieval_ms": latency - 20, "sufficiency_ms": 5, "answer_ms": 15,
                   "sufficiency_usage": {"input_tokens": 1, "output_tokens": 1},
                   "answer_usage": {"input_tokens": 2, "output_tokens": 2}}
            for name, count, latency in (("A", 3, 90), ("B", 2, 70), ("C", 1, 80))
        }}]}
        review = {"entries": [{"review_id": "h1.X", "expected_coverage": ["a", "b"], "review": {
            "passage_labels": {"p1": "relevant"}, "coverage_supported": {"a": True, "b": False},
            "answer_correctness": 2, "grounding": 1,
        }}]}
        metrics = stage9.summarize(results, review, {"h1": {"X": "A"}})
        self.assertEqual(metrics["A"]["hit_at_5"], 1.0)
        self.assertEqual(metrics["A"]["final_context_coverage"], 0.5)
        self.assertEqual(metrics["A"]["answer_correctness"], 1.0)
        self.assertEqual(metrics["A"]["grounding"], 0.5)
        self.assertEqual(metrics["A"]["mean_search_calls"], 3)
        self.assertEqual(metrics["A"]["latency_p50_ms"], 90)


if __name__ == "__main__":
    unittest.main()
