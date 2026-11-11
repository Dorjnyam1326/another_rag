import json
import unittest
from types import SimpleNamespace

from restriction_modules import RESTRICTION_MODULES
from semantic_router import (
    MAX_RECENT_CONTEXT_CHARS,
    MAX_SUMMARY_CHARS,
    MAX_UNSUMMARIZED_CONTEXT_CHARS,
    bound_messages,
    build_active_topic_state,
    is_explicit_topic_change,
    matching_restrictions,
    rewrite_queries,
)


class SemanticRouterTests(unittest.TestCase):
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
