import json
import unittest
from types import SimpleNamespace

from restriction_modules import RESTRICTION_MODULES
from semantic_router import (
    IncompleteSemanticRouteError,
    MAX_RECENT_CONTEXT_CHARS,
    MAX_SUMMARY_CHARS,
    MAX_UNSUMMARIZED_CONTEXT_CHARS,
    bound_messages,
    build_active_topic_state,
    context_messages_for_answer,
    is_explicit_topic_change,
    matching_restrictions,
    rewrite_queries,
)


class SemanticRouterTests(unittest.TestCase):
    def test_router_retries_incomplete_structured_response_with_more_output_tokens(self):
        route = {
            "interaction_type": "question",
            "is_follow_up": False,
            "updated_summary": "",
            "requests": [
                {
                    "intent": intent,
                    "topic": topic,
                    "request_type": request_type,
                    "data_scope": scope,
                    "request_context": "workplace",
                    "canonical_query": query,
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": [],
                }
                for intent, topic, request_type, scope, query in (
                    ("Дөрвөн цагийн чөлөө авах хүсэлт", "leave", "procedure", "general", "Дөрвөн цагийн чөлөөг хэрхэн хүсэх вэ?"),
                    ("Дөрвөн цагийн чөлөө цалинтай эсэх", "leave", "paid_leave", "general", "Дөрвөн цагийн чөлөө цалинтай юу?"),
                    ("Хувийн үндсэн цалингийн дүн", "payroll", "amount", "personal", "Миний үндсэн цалин хэд вэ?"),
                )
            ],
        }

        class Responses:
            calls = []

            @classmethod
            def create(cls, **kwargs):
                cls.calls.append(kwargs)
                if len(cls.calls) == 1:
                    return SimpleNamespace(
                        status="incomplete",
                        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                        output_text='{"requests":[{"intent":"truncated',
                        usage=SimpleNamespace(input_tokens=10, output_tokens=1400),
                    )
                return SimpleNamespace(
                    status="completed",
                    output_text=json.dumps(route, ensure_ascii=False),
                    usage=SimpleNamespace(input_tokens=12, output_tokens=500),
                )

        class Client:
            responses = Responses

        trace = {}
        result = rewrite_queries(
            Client(),
            "bi 4 tsagiin chuluu awmaar baina, yaj huselt uguh ve, tsalintai yu, bas minii undsen tsalin hed ve?",
            trace_capture=trace,
        )

        self.assertEqual(len(Responses.calls), 2)
        self.assertEqual(
            [call["max_output_tokens"] for call in Responses.calls], [1400, 1800]
        )
        self.assertEqual(len(result["requests"]), 3)
        self.assertEqual(result["matched_restrictions"], ["salary"])
        self.assertEqual(result["usage"], {"input_tokens": 22, "output_tokens": 1900})
        self.assertEqual([item["failure_kind"] for item in trace["attempts"]], ["incomplete", None])
        self.assertFalse(trace["routing_failed"])

    def test_router_reports_incomplete_parse_and_api_failures_separately(self):
        scenarios = (
            (
                SimpleNamespace(
                    status="completed",
                    output_text="{not json",
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                ),
                "parse_error",
            ),
            (RuntimeError("provider unavailable"), "api_error"),
        )
        for outcome, expected_failure_kind in scenarios:
            with self.subTest(failure_kind=expected_failure_kind):
                class Responses:
                    calls = 0

                    @classmethod
                    def create(cls, **kwargs):
                        cls.calls += 1
                        if isinstance(outcome, Exception):
                            raise outcome
                        return outcome

                class Client:
                    responses = Responses

                trace = {}
                with self.assertRaises(Exception):
                    rewrite_queries(Client(), "Ээлжийн амралтаа яаж хүсэх вэ?", trace_capture=trace)
                self.assertEqual(trace["failure_kind"], expected_failure_kind)
                self.assertTrue(trace["routing_failed"])
                self.assertEqual(Responses.calls, 1)

        class IncompleteResponses:
            @staticmethod
            def create(**kwargs):
                return SimpleNamespace(
                    status="incomplete",
                    incomplete_details={"reason": "max_output_tokens"},
                    output_text='{"requests":',
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1400),
                )

        class IncompleteClient:
            responses = IncompleteResponses

        trace = {}
        with self.assertRaises(IncompleteSemanticRouteError):
            rewrite_queries(IncompleteClient(), "Ээлжийн амралтаа яаж хүсэх вэ?", trace_capture=trace)
        self.assertEqual(trace["failure_kind"], "incomplete")
        self.assertTrue(trace["routing_failed"])
        self.assertEqual(len(trace["attempts"]), 2)

    def test_relationship_candidate_selects_only_the_router_named_context(self):
        route = {
            "interaction_type": "reply",
            "updated_summary": "",
            "relationship": "follow_up",
            "relevant_context_indices": [0, 1],
            "use_active_topic_state": True,
            "requests": [
                {
                    "intent": "Чөлөөний хүсэлтийг батлах журам",
                    "topic": "leave",
                    "request_type": "approval",
                    "data_scope": "general",
                    "request_context": "workplace",
                    "canonical_query": "Менежер эзгүй үед чөлөөний хүсэлтийг хэн батлах вэ?",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": [],
                }
            ],
        }

        class Client:
            class responses:
                calls = []

                @classmethod
                def create(cls, **kwargs):
                    cls.calls.append(kwargs)
                    return SimpleNamespace(
                        status="completed",
                        output_text=json.dumps(route, ensure_ascii=False),
                        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                    )

        previous = [
            {"role": "user", "content": "Маргааш 2–6 цагт чөлөө авна."},
            {"role": "assistant", "content": "Чөлөөний журмыг шалгалаа."},
        ]
        result = rewrite_queries(
            Client(),
            "Тэгвэл менежер байхгүй бол хэн батлах вэ?",
            history=previous,
            active_topic_state="Маргааш 2–6 цагийн чөлөө.",
            route_strategy="relationship_candidate",
        )

        self.assertEqual(result["relationship"], "follow_up")
        self.assertEqual(result["relevant_context_indices"], [0, 1])
        context_for_answer = context_messages_for_answer(result)
        self.assertIn("чөлөө", context_for_answer[0]["content"])
        self.assertEqual(context_for_answer[1], previous[1])
        payload = json.loads(Client.responses.calls[-1]["input"])
        self.assertEqual(payload["recent_conversation"], previous)
        self.assertEqual(payload["active_topic_state"], "Маргааш 2–6 цагийн чөлөө.")

    def test_complete_new_leave_duration_does_not_inherit_followup_context(self):
        route = {
            "interaction_type": "question",
            "updated_summary": "",
            "relationship": "follow_up",
            "relevant_context_indices": [0, 1],
            "use_active_topic_state": True,
            "requests": [{
                "intent": "Гурван өдрийн чөлөө авах журам",
                "topic": "leave",
                "request_type": "procedure",
                "data_scope": "general",
                "request_context": "workplace",
                "canonical_query": "3 өдрийн чөлөө авахын тулд ямар журам баримтлах вэ?",
                "search_documents": True,
                "is_ambiguous": False,
                "alternatives": [],
            }],
        }

        class Client:
            class responses:
                @staticmethod
                def create(**kwargs):
                    return SimpleNamespace(
                        status="completed",
                        output_text=json.dumps(route, ensure_ascii=False),
                        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                    )

        result = rewrite_queries(
            Client(),
            "bi 3 udriin chuluu awna",
            history=[
                {"role": "user", "content": "bi 4 tsagiin chuluu yaj awah ve?"},
                {"role": "assistant", "content": "Чөлөөний журам."},
            ],
            active_topic_state="4 цагийн чөлөө",
            route_strategy="stage7_candidate",
        )

        self.assertEqual(result["relationship"], "new_topic")
        self.assertFalse(result["is_follow_up"])
        self.assertEqual(result["relevant_context_indices"], [])
        self.assertEqual(result["context_messages"], [])
        self.assertEqual(result["context_resolution_decision"], "self_contained_leave_request")
        self.assertEqual(
            result["requests"][0]["canonical_query"],
            "3 өдрийн чөлөө авахын тулд ямар журам баримтлах вэ?",
        )

    def test_self_contained_leave_time_range_drops_previous_salary_review_context(self):
        route = {
            "interaction_type": "reply",
            "updated_summary": "",
            "relationship": "follow_up",
            "relevant_context_indices": [0, 1],
            "use_active_topic_state": True,
            "requests": [{
                "intent": "Гурван цагийн чөлөө авах журам",
                "topic": "leave",
                "request_type": "procedure",
                "data_scope": "general",
                "request_context": "workplace",
                "canonical_query": "Өдрийн 15:00–18:00 цагийн хооронд 3 цагийн чөлөө авах журам",
                "search_documents": True,
                "is_ambiguous": False,
                "alternatives": [],
            }],
        }

        class Client:
            class responses:
                calls = []

                @classmethod
                def create(cls, **kwargs):
                    cls.calls.append(kwargs)
                    return SimpleNamespace(
                        status="completed",
                        output_text=json.dumps(route, ensure_ascii=False),
                        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                    )

        result = rewrite_queries(
            Client(),
            "Өдрийн 3-аас 6 хүртэл чөлөө авах",
            history=[
                {"role": "user", "content": "Цалин нэмүүлэх хүсэлтээ хаана гаргах вэ?"},
                {"role": "assistant", "content": "Цалин нэмүүлэх хүсэлтийн хариу."},
            ],
            active_topic_state="Цалин нэмүүлэх хүсэлтээ хаана гаргах вэ?",
            route_strategy="stage7_candidate",
        )

        route_payload = json.loads(Client.responses.calls[-1]["input"])
        self.assertEqual(route_payload["recent_conversation"], [])
        self.assertEqual(route_payload["active_topic_state"], "")
        self.assertEqual(result["relationship"], "new_topic")
        self.assertFalse(result["is_follow_up"])
        self.assertEqual(result["context_messages"], [])
        self.assertIn("15:00–18:00", result["requests"][0]["canonical_query"])
        self.assertIn("3 цаг", result["requests"][0]["canonical_query"])
        self.assertNotIn("хаана гаргах", result["requests"][0]["canonical_query"])

    def test_duration_only_correction_still_uses_followup_context(self):
        route = {
            "interaction_type": "reply",
            "updated_summary": "",
            "relationship": "follow_up",
            "relevant_context_indices": [0, 1],
            "use_active_topic_state": True,
            "requests": [{
                "intent": "Гурван цагийн чөлөө авах журам",
                "topic": "leave",
                "request_type": "procedure",
                "data_scope": "general",
                "request_context": "workplace",
                "canonical_query": "3 цагийн чөлөө авах журам",
                "search_documents": True,
                "is_ambiguous": False,
                "alternatives": [],
            }],
        }

        class Client:
            class responses:
                @staticmethod
                def create(**kwargs):
                    return SimpleNamespace(
                        status="completed",
                        output_text=json.dumps(route, ensure_ascii=False),
                        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                    )

        result = rewrite_queries(
            Client(),
            "ugui 3 udur bish 3 tsag",
            history=[
                {"role": "user", "content": "bi 3 udriin chuluu awna"},
                {"role": "assistant", "content": "Чөлөөний журам."},
            ],
            active_topic_state="3 өдрийн чөлөө",
            route_strategy="stage7_candidate",
        )

        self.assertEqual(result["relationship"], "follow_up")
        self.assertTrue(result["is_follow_up"])
        self.assertEqual(
            result["context_resolution_decision"],
            "explicit_duration_correction_current_value",
        )
        self.assertEqual(result["context_messages"], [])
        self.assertEqual(result["requests"][0]["canonical_query"], "3 цагийн чөлөө авах журам")
        self.assertNotIn("3 өдрийн", result["requests"][0]["canonical_query"])

    def test_followup_time_range_correction_inherits_daypart_and_skips_resolver(self):
        route = {
            "interaction_type": "reply",
            "updated_summary": "",
            "relationship": "follow_up",
            "relevant_context_indices": [0, 1],
            "use_active_topic_state": True,
            "requests": [{
                "intent": "Гурван цагийн чөлөөний журам",
                "topic": "leave",
                "request_type": "procedure",
                "data_scope": "general",
                "request_context": "workplace",
                "canonical_query": "Өдрийн 15:00–18:00 цагийн хооронд 3 цагийн чөлөө авах журам",
                "search_documents": True,
                "is_ambiguous": False,
                "alternatives": [],
            }],
        }

        class Client:
            class responses:
                @staticmethod
                def create(**kwargs):
                    return SimpleNamespace(
                        status="completed",
                        output_text=json.dumps(route, ensure_ascii=False),
                        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                    )

        result = rewrite_queries(
            Client(),
            "Үгүй, 3-аас 6 хүртэл",
            history=[
                {"role": "user", "content": "Өдрийн 2-оос 6 хүртэл чөлөө авна"},
                {"role": "assistant", "content": "Чөлөөний журмыг шалгалаа."},
            ],
            active_topic_state="Өдрийн 2-оос 6 хүртэл, 4 цагийн чөлөө",
            route_strategy="stage7_candidate",
        )

        request = result["requests"][0]
        self.assertIn("15:00–18:00", request["canonical_query"])
        self.assertIn("3 цаг", request["canonical_query"])
        self.assertNotIn("query_guard_rejections", request)
        self.assertTrue(request["current_message_conditions_resolved"])
        self.assertEqual(result["context_resolution_decision"], "explicit_time_range_current_message")
        self.assertEqual(result["context_messages"], [])

    def test_independent_topic_cannot_pass_selected_or_implicit_context(self):
        self.assertEqual(
            context_messages_for_answer(
                {
                    "relationship": "new_topic",
                    "is_follow_up": False,
                    "context_messages": [
                        {"role": "user", "content": "leave"},
                    ],
                }
            ),
            [],
        )
        self.assertEqual(
            context_messages_for_answer(
                {
                    "relationship": "follow_up",
                    "is_follow_up": True,
                    "context_messages": [
                        {"role": "system", "content": "invalid role"},
                    ],
                }
            ),
            [],
        )

    def test_candidate_normalizes_implicit_change_and_rejects_invalid_indices(self):
        route = {
            "interaction_type": "question",
            "updated_summary": "",
            "relationship": "topic_change",
            "relevant_context_indices": [0, 1],
            "use_active_topic_state": True,
            "requests": [
                {
                    "intent": "Нууц баримт хуваалцах журам",
                    "topic": "security",
                    "request_type": "sharing",
                    "data_scope": "general",
                    "request_context": "workplace",
                    "canonical_query": "Нууцын зэрэглэлтэй баримт бичгийг хэнтэй хуваалцаж болох вэ?",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": [],
                }
            ],
        }

        class Client:
            class responses:
                @staticmethod
                def create(**kwargs):
                    return SimpleNamespace(
                        status="completed",
                        output_text=json.dumps(route, ensure_ascii=False),
                        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                    )

        previous = [
            {"role": "user", "content": "Тэгвэл менежер байхгүй бол хэн батлах вэ?"},
            {"role": "assistant", "content": "Чөлөөний хүсэлтийн журмыг шалгалаа."},
        ]
        result = rewrite_queries(
            Client(),
            "Нууцын зэрэглэлтэй баримт бичгийг хэнтэй хуваалцаж болох вэ?",
            history=previous,
            active_topic_state="Чөлөөний хүсэлт.",
            route_strategy="relationship_candidate",
        )
        self.assertEqual(result["relationship"], "new_topic")
        self.assertEqual(result["relevant_context_indices"], [])
        self.assertFalse(result["active_topic_state_used"])
        self.assertEqual(context_messages_for_answer(result), [])

        route["relationship"] = "follow_up"
        route["relevant_context_indices"] = [9]
        route["use_active_topic_state"] = False
        result = rewrite_queries(
            Client(),
            "Тэр өдөр менежер байхгүй бол хэн батлах вэ?",
            history=previous,
            route_strategy="relationship_candidate",
        )
        self.assertEqual(result["relationship"], "new_topic")
        self.assertEqual(result["relevant_context_indices"], [])
        self.assertFalse(result["is_follow_up"])

    def test_restrictions_are_registry_driven_and_independently_toggleable(self):
        programming = {
            "topic": "programming",
            "request_context": "standalone",
            "request_type": "concept",
        }
        self.assertEqual(matching_restrictions([programming]), ["programming"])
        self.assertEqual(matching_restrictions([programming], {"salary", "math"}), [])
        self.assertEqual(
            matching_restrictions(
                [{**programming, "request_context": "document"}]
            ),
            [],
        )

    def test_salary_restriction_requires_personal_amount_scope(self):
        personal_amount = {
            "topic": "payroll",
            "request_type": "amount",
            "data_scope": "personal",
        }
        general_deduction = {
            "topic": "payroll",
            "request_type": "deduction",
            "data_scope": "general",
        }
        self.assertEqual(matching_restrictions([personal_amount]), ["salary"])
        self.assertEqual(matching_restrictions([general_deduction]), [])

    def test_new_registry_module_appears_in_router_and_matches_without_router_changes(self):
        module_id = "test_policy"
        RESTRICTION_MODULES[module_id] = {
            "label": "Test policy",
            "router_description": "Block internal test-policy requests.",
            "semantic_match": {"topic": ["test_policy"]},
            "prompt_instruction": "Do not answer test-policy requests.",
            "refusal": "Restricted.",
        }
        try:
            match = {"topic": "test_policy", "request_context": "workplace"}
            self.assertEqual(matching_restrictions([match]), ["test_policy"])
            self.assertEqual(matching_restrictions([match], set()), [])
        finally:
            del RESTRICTION_MODULES[module_id]

    def test_router_builds_existing_plan_shape_and_only_applies_enabled_modules(self):
        route = {
            "interaction_type": "question",
            "is_follow_up": False,
            "updated_summary": "",
            "requests": [
                {
                    "intent": "Хувийн цалингийн дүн",
                    "topic": "payroll",
                    "request_type": "amount",
                    "data_scope": "personal",
                    "request_context": "workplace",
                    "canonical_query": "Миний цалин хэд вэ?",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": ["Цалингийн хэмжээ хэд вэ?"],
                }
            ],
        }

        class FakeResponses:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    status="completed",
                    output_text=json.dumps(route, ensure_ascii=False),
                    usage=SimpleNamespace(input_tokens=25, output_tokens=8),
                )

        class FakeClient:
            responses = FakeResponses()

        result = rewrite_queries(FakeClient(), "Миний цалин хэд вэ?", active_restrictions={"salary"})
        self.assertEqual(result["matched_restrictions"], ["salary"])
        self.assertEqual(result["requests"][0]["canonical_query"], "Миний цалин хэд вэ?")
        self.assertEqual(result["requests"][0]["alternatives"], ["Цалингийн хэмжээ хэд вэ?"])
        self.assertEqual(result["requests"][0]["restriction_modules"], ["salary"])
        self.assertEqual(result["usage"], {"input_tokens": 25, "output_tokens": 8})
        self.assertNotIn("matched_restrictions", result["requests"][0])

    def test_disabling_multiple_query_alternatives_keeps_only_canonical_query(self):
        route = {
            "interaction_type": "question",
            "is_follow_up": False,
            "updated_summary": "",
            "requests": [
                {
                    "intent": "Цалин олгох хуваарь",
                    "topic": "payroll",
                    "request_type": "schedule",
                    "data_scope": "general",
                    "request_context": "workplace",
                    "canonical_query": "Цалин хэзээ буух вэ?",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": ["Цалин олгох өдөр"],
                }
            ],
        }

        class Client:
            class responses:
                @staticmethod
                def create(**kwargs):
                    return SimpleNamespace(
                        status="completed",
                        output_text=json.dumps(route, ensure_ascii=False),
                        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                    )

        result = rewrite_queries(Client(), "Цалин хэзээ буух вэ?", include_alternatives=False)
        self.assertEqual(result["matched_restrictions"], [])
        self.assertEqual(result["requests"][0]["alternatives"], [])

    def test_source_followup_is_attached_to_the_immediately_previous_topic(self):
        route = {
            "interaction_type": "question",
            "is_follow_up": False,
            "updated_summary": "",
            "requests": [
                {
                    "intent": "Ердийн чөлөөний журам",
                    "topic": "leave",
                    "request_type": "source_location",
                    "data_scope": "general",
                    "request_context": "workplace",
                    "canonical_query": "Ердийн чөлөөний журмын эх сурвалжийг хаанаас унших вэ?",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": [],
                }
            ],
        }

        class Client:
            class responses:
                @staticmethod
                def create(**kwargs):
                    return SimpleNamespace(
                        status="completed",
                        output_text=json.dumps(route, ensure_ascii=False),
                        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                    )

        result = rewrite_queries(
            Client(),
            "ойлголоо, харин үүнийг хаанаас унших вэ, эх сурвалжийн заалтыг нь харуулж өгөөч",
            history=[
                {"role": "user", "content": "Ердийн чөлөө авахад ямар процесс байдаг вэ?"},
                {"role": "assistant", "content": "Ердийн чөлөөний хүсэлт гаргах журам."},
            ],
        )
        self.assertTrue(result["is_follow_up"])

    def test_reference_to_that_day_is_a_follow_up_when_model_misses_it(self):
        route = {
            "interaction_type": "question",
            "is_follow_up": False,
            "updated_summary": "",
            "requests": [
                {
                    "intent": "Хагас өдрийн чөлөөний батлах журам",
                    "topic": "leave",
                    "request_type": "approval",
                    "data_scope": "general",
                    "request_context": "workplace",
                    "canonical_query": "Маргааш 14:00–18:00 цагт авах хувийн чөлөөг хэн батлах вэ?",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": [],
                }
            ],
        }

        class Client:
            class responses:
                @staticmethod
                def create(**kwargs):
                    return SimpleNamespace(
                        status="completed",
                        output_text=json.dumps(route, ensure_ascii=False),
                        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                    )

        result = rewrite_queries(
            Client(),
            "тэр өдөр manager маань байхгүй юм байна лээ, тэгэхээр хэн батлах вэ?",
            history=[
                {"role": "user", "content": "Маргааш хувийн ажлаар хагас өдөр чөлөө авч болох уу?"},
                {"role": "assistant", "content": "Хагас өдрийн чөлөөний журмыг шалгая."},
            ],
        )
        self.assertTrue(result["is_follow_up"])
        self.assertFalse(result["topic_change"])
        self.assertIn("хувийн чөлөөг хэн батлах", result["requests"][0]["canonical_query"])

    def test_explicitly_abandoning_topic_overrides_model_follow_up_flag(self):
        route = {
            "interaction_type": "question",
            "is_follow_up": True,
            "updated_summary": "",
            "requests": [
                {
                    "intent": "Дотоод тайланг гадна хуваалцах зөвшөөрөл",
                    "topic": "security",
                    "request_type": "approval",
                    "data_scope": "general",
                    "request_context": "workplace",
                    "canonical_query": "Энгийн дотоод тайланг гадны байгууллагад явуулахад хэний зөвшөөрөл хэрэгтэй вэ?",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": [],
                }
            ],
        }

        class Client:
            class responses:
                @staticmethod
                def create(**kwargs):
                    return SimpleNamespace(
                        status="completed",
                        output_text=json.dumps(route, ensure_ascii=False),
                        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                    )

        result = rewrite_queries(
            Client(),
            "за тэрийг больё, харин энгийн дотоод тайланг гадны байгууллага руу явуулж болох уу?",
            history=[
                {"role": "user", "content": "Нууцын зэрэглэлтэй баримтыг хэнтэй хуваалцаж болох вэ?"},
                {"role": "assistant", "content": "Нууц баримтын журмыг шалгая."},
            ],
        )
        self.assertFalse(result["is_follow_up"])
        self.assertTrue(result["topic_change"])
        self.assertIn("дотоод тайланг", result["requests"][0]["canonical_query"])

    def test_active_topic_state_keeps_correction_details_and_replaces_abandoned_topic(self):
        first_state = build_active_topic_state(
            "Маргааш хувийн ажлаар хагас өдөр чөлөө авч болох уу?",
            [{"canonical_query": "Маргааш хувийн чөлөө авах журам", "search_documents": True}],
        )
        corrected_state = build_active_topic_state(
            "Өглөө биш ээ, өдрийн 2-оос 6 хүртэл.",
            [{"canonical_query": "Маргааш 14:00–18:00 цагийн хувийн чөлөө", "search_documents": True}],
            previous_state=first_state,
            is_follow_up=True,
        )
        self.assertIn("хувийн", corrected_state.casefold())
        self.assertIn("маргааш", corrected_state.casefold())
        self.assertIn("14:00–18:00", corrected_state)
        self.assertLessEqual(len(corrected_state), MAX_SUMMARY_CHARS)

        switched_state = build_active_topic_state(
            "за тэрийг больё, харин энгийн дотоод тайланг гадны байгууллага руу явуулж болох уу?",
            [{"canonical_query": "Энгийн дотоод тайланг гадны байгууллагад явуулах журам", "search_documents": True}],
            previous_state=corrected_state,
            is_follow_up=False,
            topic_change=True,
        )
        self.assertIn("дотоод тайлан", switched_state.casefold())
        self.assertNotIn("14:00", switched_state)
        self.assertNotIn("хувийн чөлөө", switched_state.casefold())

    def test_followup_state_is_sent_but_explicit_topic_switch_clears_route_context(self):
        route = {
            "interaction_type": "question",
            "is_follow_up": True,
            "updated_summary": "",
            "requests": [
                {
                    "intent": "Чөлөөний хүсэлт",
                    "topic": "leave",
                    "request_type": "approval",
                    "data_scope": "general",
                    "request_context": "workplace",
                    "canonical_query": "Маргааш 14:00–18:00 цагийн хувийн чөлөөг хэн батлах вэ?",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": [],
                }
            ],
        }

        class Client:
            class responses:
                calls = []

                @classmethod
                def create(cls, **kwargs):
                    cls.calls.append(kwargs)
                    return SimpleNamespace(
                        status="completed",
                        output_text=json.dumps(route, ensure_ascii=False),
                        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                    )

        prior = [
            {"role": "user", "content": "Маргааш хувийн ажлаар хагас өдөр чөлөө авч болох уу?"},
            {"role": "assistant", "content": "Хүсэлтийн журмыг шалгая."},
        ]
        state = "Маргааш хувийн чөлөө; өдрийн 14:00–18:00 цагийн хуваарь."
        followup = rewrite_queries(
            Client(), "тэр өдөр manager байхгүй, хэн батлах вэ?", history=prior,
            active_topic_state=state,
        )
        followup_payload = json.loads(Client.responses.calls[-1]["input"])
        self.assertEqual(followup_payload["active_topic_state"], state)
        self.assertTrue(followup["active_topic_state_used"])

        switched = rewrite_queries(
            Client(),
            "за тэрийг больё, харин энгийн дотоод тайланг гадна явуулж болох уу?",
            history=prior,
            active_topic_state=state,
        )
        switched_payload = json.loads(Client.responses.calls[-1]["input"])
        self.assertTrue(is_explicit_topic_change("за тэрийг больё, харин өөр асуулт"))
        self.assertEqual(switched_payload["recent_conversation"], [])
        self.assertEqual(switched_payload["active_topic_state"], "")
        self.assertTrue(switched["topic_change"])
        self.assertFalse(switched["active_topic_state_used"])

    def test_router_uses_only_the_immediate_pair_and_drops_old_summaries(self):
        route = {
            "interaction_type": "question",
            "is_follow_up": False,
            "updated_summary": "Ө" * (MAX_SUMMARY_CHARS + 500),
            "requests": [
                {
                    "intent": "Амралтын журам",
                    "topic": "leave",
                    "request_type": "procedure",
                    "data_scope": "general",
                    "request_context": "workplace",
                    "canonical_query": "Ээлжийн амралтаа хэрхэн хүсэх вэ?",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": [],
                }
            ],
        }

        class Client:
            class responses:
                @staticmethod
                def create(**kwargs):
                    Client.payload = json.loads(kwargs["input"])
                    Client.instructions = kwargs["instructions"]
                    return SimpleNamespace(
                        status="completed",
                        output_text=json.dumps(route, ensure_ascii=False),
                        usage=SimpleNamespace(input_tokens=10, output_tokens=20),
                    )

        history = [{"role": "user", "content": "x" * 2000} for _ in range(10)]
        older = [{"role": "assistant", "content": "y" * 2000} for _ in range(20)]
        result = rewrite_queries(
            Client(),
            "Ээлжийн амралтаа хэрхэн хүсэх вэ?",
            history=history,
            conversation_summary="s" * (MAX_SUMMARY_CHARS + 50),
            unsummarized_history=older,
            active_restrictions=set(),
        )
        payload = Client.payload
        self.assertEqual(payload["conversation_summary"], "")
        self.assertLessEqual(
            sum(len(item["content"]) for item in payload["recent_conversation"]),
            MAX_RECENT_CONTEXT_CHARS,
        )
        self.assertEqual(payload["conversation_to_summarize"], [])
        self.assertEqual(result["conversation_summary"], "")
        self.assertFalse(result["summary_updated"])


if __name__ == "__main__":
    unittest.main()
