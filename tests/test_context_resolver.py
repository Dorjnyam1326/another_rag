import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from context_resolver import (
    ContextResolutionError,
    _prepare_requests,
    _compose_resolved_query,
    _complete_correction_facts,
    bound_user_sources,
    resolve_followup_queries,
)
from query_utils import query_anchors_supported


class ContextResolverTests(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {"source_message_index": 0, "content": "Маргааш хувийн ажлаар чөлөө авна."},
            {"source_message_index": 2, "content": "Өглөө биш ээ, өдрийн 2-оос 6 хүртэл."},
        ]

    def test_bound_sources_include_only_user_messages_and_keep_original_indices(self):
        messages = [
            {"role": "assistant", "content": "Assistant invented 4 hours."},
            {"role": "user", "content": "Маргааш хувийн чөлөө авна."},
            {"role": "assistant", "content": "Ямар цагт?"},
            {"role": "user", "content": "Тийм." , "interaction_type": "acknowledgment"},
            {"role": "user", "content": "Өдрийн 2-оос 6 хүртэл."},
        ]
        self.assertEqual(
            bound_user_sources(messages),
            [
                {"source_message_index": 1, "content": "Маргааш хувийн чөлөө авна."},
                {"source_message_index": 4, "content": "Өдрийн 2-оос 6 хүртэл."},
            ],
        )

    def test_bound_sources_can_be_limited_to_router_selected_turns(self):
        messages = [
            {"role": "user", "content": "Маргааш чөлөө авна."},
            {"role": "assistant", "content": "Хэн батлах вэ?"},
            {"role": "user", "content": "Нууц баримтын тухай асууя."},
        ]
        self.assertEqual(
            bound_user_sources(messages, allowed_indices={0}),
            [{"source_message_index": 0, "content": "Маргааш чөлөө авна."}],
        )

    def test_daytime_range_and_duration_are_verified_from_user_quote(self):
        facts = [
            {"source_message_index": 0, "source_quote": "Маргааш хувийн ажлаар чөлөө авна."},
            {"source_message_index": 2, "source_quote": "Өглөө биш ээ, өдрийн 2-оос 6 хүртэл."},
        ]
        result = _prepare_requests(
            [{"request_index": 0, "status": "resolved",
              "clarification_question": "", "facts": facts}],
            [0], self.sources, "Тэр үед хэн батлах вэ?",
        )[0]
        self.assertEqual(result["source_message_indices"], [0, 2])
        self.assertIn("14:00–18:00, 4 цаг", result["resolved_query"])
        self.assertIn("Тэр үед хэн батлах вэ?", result["resolved_query"])

    def test_resolved_query_uses_router_canonical_and_not_historical_quote(self):
        canonical = "Маргааш хувийн чөлөөний хүсэлтийг хэн батлах вэ?"
        result = _prepare_requests(
            [{"request_index": 0, "status": "resolved",
              "clarification_question": "", "facts": [
                  {"source_message_index": 0,
                   "source_quote": "Маргааш хувийн ажлаар чөлөө авна."}
              ]}],
            [0], self.sources, "Тэгвэл хэн батлах вэ?",
            canonical_queries={0: canonical},
        )[0]

        self.assertEqual(result["resolved_query"], canonical)
        self.assertEqual(result["source_message_indices"], [0])

    def test_latin_duration_correction_replaces_old_value_in_canonical_query(self):
        old_request = "bi 3 udriin chuluu awna"
        current_correction = "ugui 3 udur bish 3 tsag"
        canonical = "3 өдрийн биш, 3 цагийн чөлөө авах гэсэн юм."
        result = _prepare_requests(
            [{"request_index": 0, "status": "resolved",
              "clarification_question": "", "facts": [
                  {"source_message_index": 0, "source_quote": old_request}
              ]}],
            [0], [{"source_message_index": 0, "content": old_request}],
            current_correction,
            canonical_queries={0: canonical},
        )[0]

        self.assertEqual(result["resolved_query"], "3 цагийн чөлөө авах гэсэн юм.")
        self.assertNotIn("3 өдрийн", result["resolved_query"])
        self.assertNotIn("3 udur", result["resolved_query"].casefold())

    def test_unverified_assistant_fact_is_rejected(self):
        with self.assertRaises(ContextResolutionError):
            _prepare_requests(
                [{"request_index": 0, "status": "resolved",
                  "clarification_question": "", "facts": [
                      {"source_message_index": 1, "source_quote": "4 цагийн чөлөө"}
                  ]}],
                [0], self.sources, "Хэн батлах вэ?",
            )

    def test_unsupported_query_number_is_rejected(self):
        self.assertFalse(
            query_anchors_supported(
                "Маргааш 14:00–18:00 буюу 5 цагийн чөлөө хэн батлах вэ?",
                "Хэн батлах вэ?",
                ["Өдрийн 2-оос 6 хүртэл"],
            )
        )

    def test_query_is_composed_only_from_quotes_question_and_safe_time_normalization(self):
        query = _compose_resolved_query(
            "Тэр үед хэн батлах вэ?",
            [{"source_message_index": 0,
              "source_quote": "Маргааш 2–6 цагийн хооронд чөлөө авна."}],
        )
        self.assertIn("Маргааш 2–6 цагийн хооронд чөлөө авна.", query)
        self.assertNotIn("14:00", query)
        self.assertNotIn("4 цаг", query)
        self.assertIn("Тэр үед хэн батлах вэ?", query)

    def test_explicit_latest_correction_keeps_topic_and_drops_superseded_time(self):
        sources = {
            0: "Маргааш 09:00–12:00 цагт чөлөө авна.",
            2: "Үгүй, 13:00–17:00 цагт.",
        }
        facts = _complete_correction_facts(
            [{"source_message_index": 2, "source_quote": sources[2]}],
            sources,
        )
        query = _compose_resolved_query("Тэр хүсэлтийг хэн батлах вэ?", facts)
        self.assertIn("Маргааш чөлөө авна.", query)
        self.assertIn("13:00–17:00", query)
        self.assertNotIn("09:00", query)
        self.assertNotIn("12:00", query)

    def test_unsourced_assistant_fact_cannot_be_selected(self):
        with self.assertRaises(ContextResolutionError):
            _prepare_requests(
                [{"request_index": 0, "status": "resolved",
                  "clarification_question": "", "facts": [
                      {"source_message_index": 1,
                       "source_quote": "Жилд 500,000 төгрөгийн дэмжлэгтэй."}
                  ]}],
                [0], self.sources, "Хэн батлах вэ?",
            )

    def test_unresolved_result_does_not_need_source_facts(self):
        result = _prepare_requests(
            [{"request_index": 0, "status": "clarification", "resolved_query": "",
              "clarification_question": "Ямар өдрийг хэлж байна вэ?", "facts": []}],
            [0], self.sources, "Тэр өдөр хэн батлах вэ?",
        )[0]
        self.assertEqual(result["status"], "clarification")

    def test_resolver_sends_selected_context_and_returns_validated_result(self):
        payload = {"requests": [{"request_index": 0, "status": "resolved",
            "clarification_question": "", "facts": [
                {"source_message_index": 0, "source_quote": "Маргааш хувийн ажлаар чөлөө авна."}
            ]}]}
        call = {}
        class Responses:
            def create(self, **kwargs):
                call.update(kwargs)
                return SimpleNamespace(output_text=json.dumps(payload, ensure_ascii=False),
                    status="completed", usage=SimpleNamespace(input_tokens=12, output_tokens=6))
        response = resolve_followup_queries(
            SimpleNamespace(responses=Responses()), current_question="Хэн батлах вэ?",
            requests=[{"intent":"чөлөө", "canonical_query":"Хэн батлах вэ?"}],
            selected_context=[{"role":"user", "content":"Маргааш хувийн ажлаар чөлөө авна."}],
            user_sources=self.sources, model="gpt-6-luna",
        )
        self.assertEqual(response["usage"], {"input_tokens":12, "output_tokens":6})
        self.assertEqual(response["requests"][0]["source_message_indices"], [0])
        self.assertEqual(response["requests"][0]["resolved_query"], "Хэн батлах вэ?")
        self.assertEqual(response["requests"][0]["source_message_indices"], [0])
        self.assertFalse(call["store"])

    def test_resolver_excludes_user_facts_outside_router_selected_indices(self):
        payload = {"requests": [{"request_index": 0, "status": "resolved",
            "clarification_question": "", "facts": [
                {"source_message_index": 0, "source_quote": "Маргааш хувийн ажлаар чөлөө авна."}
            ]}]}
        call = {}

        class Responses:
            def create(self, **kwargs):
                call.update(kwargs)
                return SimpleNamespace(output_text=json.dumps(payload, ensure_ascii=False),
                    status="completed", usage=SimpleNamespace(input_tokens=12, output_tokens=6))

        with self.assertRaises(ContextResolutionError):
            resolve_followup_queries(
                SimpleNamespace(responses=Responses()),
                current_question="Хэн батлах вэ?",
                requests=[{"intent": "чөлөө", "canonical_query": "Хэн батлах вэ?"}],
                selected_context=[{"role": "user", "content": "Нууц баримтын тухай асууя."}],
                user_sources=self.sources,
                allowed_source_indices={2},
                model="gpt-6-luna",
            )
        sent_payload = json.loads(call["input"])
        self.assertEqual(sent_payload["user_sources"], [self.sources[1]])

    def test_fixture_has_12_source_mapped_cases(self):
        fixture = Path(__file__).parent / "fixtures" / "context_resolver_stage6.json"
        cases = json.loads(fixture.read_text(encoding="utf-8"))
        self.assertEqual(len(cases), 12)
        self.assertTrue(all("source_indices" in case for case in cases))


if __name__ == "__main__":
    unittest.main()
