import json
import unittest
from types import SimpleNamespace

from query_rewriter import latest_completed_turn, rewrite_queries


class FakeResponses:
    def __init__(self, output):
        self.output = output
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            status="completed",
            output_text=json.dumps(self.output, ensure_ascii=False),
            usage=SimpleNamespace(input_tokens=40, output_tokens=22),
        )


class QueryRewriterTests(unittest.TestCase):
    def test_rewrites_negated_question_and_keeps_two_variants(self):
        response = FakeResponses(
            {
                "interaction_type": "question",
                "is_follow_up": False,
                "requests": [
                    {
                        "intent": "Ажлын байранд хориглосон үйлдлүүд",
                        "canonical_query": "Ажил дээр юу хийхийг хориглодог вэ?",
                        "search_documents": True,
                        "is_ambiguous": False,
                        "alternatives": [
                            "Ажлын байранд хийхийг хориглосон үйлдлүүд юу вэ?",
                            "Ажил дээр юу хийхийг хориглодог вэ?",
                            "Ажлын байрны хориглох дүрэм юу вэ?",
                        ],
                    }
                ],
            }
        )
        result = rewrite_queries(
            SimpleNamespace(responses=response),
            "ajil deer yu yu hiij boldoggui we",
        )

        request = result["requests"][0]
        self.assertEqual(request["canonical_query"], "Ажил дээр юу хийхийг хориглодог вэ?")
        self.assertEqual(len(request["alternatives"]), 2)
        self.assertIn("negation", response.kwargs["instructions"])
        self.assertFalse(json.loads(response.kwargs["input"])["recent_conversation"])
        self.assertFalse(response.kwargs["store"])
        self.assertEqual(result["usage"]["input_tokens"], 40)

    def test_splits_independent_requests_and_limits_extra_variants_to_two(self):
        response = FakeResponses(
            {
                "interaction_type": "question",
                "is_follow_up": False,
                "requests": [
                    {
                        "intent": "Чөлөөний хүсэлт гаргах",
                        "canonical_query": "Чөлөө авах хүсэлтээ хэрхэн гаргах вэ?",
                        "search_documents": True,
                        "is_ambiguous": False,
                        "alternatives": [
                            "Чөлөөний хүсэлтийг хаана, яаж өгөх вэ?",
                            "Чөлөөний хүсэлтийг хэрхэн илгээх вэ?",
                        ],
                    },
                    {
                        "intent": "Улаанбаатарын маргаашийн цаг агаар",
                        "canonical_query": "Маргааш Улаанбаатарт цаг агаар ямар байх вэ?",
                        "search_documents": False,
                        "is_ambiguous": False,
                        "alternatives": [],
                    },
                    {
                        "intent": "Чөлөөний хүсэлтийг батлах хүн",
                        "canonical_query": "Чөлөөний хүсэлтийг хэн батлах вэ?",
                        "search_documents": True,
                        "is_ambiguous": False,
                        "alternatives": ["Чөлөө авах хүсэлтийг хэн зөвшөөрөх вэ?"],
                    },
                ],
            }
        )
        result = rewrite_queries(
            SimpleNamespace(responses=response),
            "leave request yaj ogoh ve bas UB-d weather margaash yah bol?",
        )

        self.assertEqual(len(result["requests"]), 3)
        self.assertFalse(result["requests"][1]["search_documents"])
        self.assertEqual(
            sum(len(item["alternatives"]) for item in result["requests"]),
            2,
        )
        self.assertTrue(all("гэрээний өргөдөл" not in item["canonical_query"] for item in result["requests"]))

    def test_marks_ambiguous_request_as_no_search(self):
        response = FakeResponses(
            {
                "interaction_type": "question",
                "is_follow_up": False,
                "requests": [
                    {
                        "intent": "Тодруулах шаардлагатай хүсэлт",
                        "canonical_query": "Тодорхойгүй нэр томьёоны хүсэлтийн талаар мэдээлэл",
                        "search_documents": True,
                        "is_ambiguous": True,
                        "alternatives": ["Өөр утгатай хайлтын хувилбар"],
                    }
                ],
            }
        )
        result = rewrite_queries(SimpleNamespace(responses=response), "ene yu ve")

        request = result["requests"][0]
        self.assertTrue(request["is_ambiguous"])
        self.assertFalse(request["search_documents"])
        self.assertEqual(request["alternatives"], [])

    def test_vague_followup_cannot_switch_remote_work_to_annual_leave(self):
        response = FakeResponses(
            {
                "interaction_type": "question",
                "is_follow_up": True,
                "requests": [
                    {
                        "intent": "Ээлжийн амралтын хүсэлт",
                        "canonical_query": "Ээлжийн амралтаар хоёр өдөр дараалан амарч болох уу?",
                        "search_documents": True,
                        "is_ambiguous": False,
                        "alternatives": [],
                    }
                ],
            }
        )
        result = rewrite_queries(
            SimpleNamespace(responses=response),
            "Хоёр өдөр дараалан хүсэлт гаргаж болох уу?",
            history=[
                {
                    "role": "user",
                    "content": "Гэрээсээ ажиллахын тулд ямар зөвшөөрөл авах хэрэгтэй вэ?",
                },
                {
                    "role": "assistant",
                    "content": "Гэрээсээ ажиллах зөвшөөрлийн талаар баримтаас мэдээлэл олдсонгүй.",
                },
            ],
        )

        request = result["requests"][0]
        self.assertTrue(request["is_ambiguous"])
        self.assertFalse(request["search_documents"])
        self.assertIn("өмнөх сэдэвтэй зөрүүтэй", request["intent"])
        self.assertIn("remote-work days", response.kwargs["instructions"])

    def test_company_leave_request_is_searched_even_if_model_marks_general(self):
        response = FakeResponses(
            {
                "interaction_type": "question",
                "is_follow_up": False,
                "requests": [
                    {
                        "intent": "Жилийн ээлжийн амралт авах",
                        "canonical_query": "Жилийн ээлжийн амралтаа хэрхэн авах вэ?",
                        "search_documents": False,
                        "is_ambiguous": False,
                        "alternatives": [],
                    }
                ],
            }
        )
        result = rewrite_queries(
            SimpleNamespace(responses=response),
            "jiliin eeljiin amrltaa yaj awh we",
        )

        self.assertTrue(result["requests"][0]["search_documents"])

    def test_mixed_workplace_and_weather_requests_route_separately(self):
        response = FakeResponses(
            {
                "interaction_type": "question",
                "is_follow_up": False,
                "requests": [
                    {
                        "intent": "Амралтын хүсэлт өгөх",
                        "canonical_query": "Амралтын хүсэлтийг яаж өгөх вэ?",
                        "search_documents": False,
                        "is_ambiguous": False,
                        "alternatives": [],
                    },
                    {
                        "intent": "Улаанбаатарын цаг агаар",
                        "canonical_query": "Улаанбаатарт маргааш цаг агаар ямар байх вэ?",
                        "search_documents": True,
                        "is_ambiguous": False,
                        "alternatives": [],
                    },
                ],
            }
        )
        result = rewrite_queries(
            SimpleNamespace(responses=response),
            "leave request yaj ogoh ve bas UB-d weather margaash yah bol?",
        )

        self.assertEqual(
            [item["search_documents"] for item in result["requests"]],
            [True, False],
        )

    def test_duplicate_intents_are_merged_without_extra_searches(self):
        response = FakeResponses(
            {
                "interaction_type": "question",
                "is_follow_up": False,
                "requests": [
                    {
                        "intent": "Улаанбаатарын цаг агаар",
                        "canonical_query": "Маргааш Улаанбаатарт цаг агаар ямар байх вэ?",
                        "search_documents": True,
                        "is_ambiguous": False,
                        "alternatives": [],
                    },
                    {
                        "intent": "Улаанбаатарын цаг агаар",
                        "canonical_query": "Улаанбаатар хотын маргаашийн цаг агаарын төлөв",
                        "search_documents": True,
                        "is_ambiguous": False,
                        "alternatives": [],
                    },
                ],
            }
        )
        result = rewrite_queries(
            SimpleNamespace(responses=response),
            "margaash UB-d weather yamar bol?",
        )

        self.assertEqual(len(result["requests"]), 1)
        request = result["requests"][0]
        self.assertFalse(request["search_documents"])
        self.assertEqual(request["alternatives"], [])

    def test_followup_inherits_company_document_routing(self):
        response = FakeResponses(
            {
                "interaction_type": "question",
                "is_follow_up": True,
                "requests": [
                    {
                        "intent": "Амралтын хүсэлтийг батлах хүн",
                        "canonical_query": "Жилийн ээлжийн амралтыг хэн батлах вэ?",
                        "search_documents": False,
                        "is_ambiguous": False,
                        "alternatives": [],
                    }
                ],
            }
        )
        history = [
            {"role": "user", "content": "Жилийн ээлжийн амралтаа яаж авах вэ?"},
            {"role": "assistant", "content": "Skyteams-д хүсэлт өгнө."},
        ]
        result = rewrite_queries(
            SimpleNamespace(responses=response),
            "who approve hiideg yum?",
            history=history,
        )

        self.assertTrue(result["requests"][0]["search_documents"])

    def test_drops_alternatives_that_invent_approver_roles(self):
        response = FakeResponses(
            {
                "interaction_type": "question",
                "is_follow_up": True,
                "requests": [
                    {
                        "intent": "Жилийн амралтын хүсэлтийг батлах хүн",
                        "canonical_query": "Жилийн ээлжийн амралтын хүсэлтийг хэн батлах вэ?",
                        "search_documents": True,
                        "is_ambiguous": False,
                        "alternatives": [
                            "Амралтын хүсэлтийг хэн хянах вэ?",
                            "Кадр, хүний нөөцийн хэлтэс батлах уу?",
                        ],
                    }
                ],
            }
        )
        result = rewrite_queries(
            SimpleNamespace(responses=response),
            "who approve hiideg yum?",
            history=[
                {"role": "user", "content": "jiliin eeljiin amrltaa yaj awh we"},
                {
                    "role": "assistant",
                    "content": "Хүсэлтээ ДМС-д оруулна.",
                },
            ],
        )

        self.assertEqual(
            result["requests"][0]["alternatives"],
            ["Амралтын хүсэлтийг хэн хянах вэ?"],
        )

    def test_single_followup_does_not_inherit_extra_request_from_history(self):
        response = FakeResponses(
            {
                "interaction_type": "question",
                "is_follow_up": True,
                "requests": [
                    {
                        "intent": "Ээлжийн амралтын хүсэлтийг батлах",
                        "canonical_query": "Жилийн ээлжийн амралтын хүсэлтийг хэн батлах вэ?",
                        "search_documents": True,
                        "is_ambiguous": False,
                        "alternatives": [],
                    },
                    {
                        "intent": "Ээлжийн амралтын процесс",
                        "canonical_query": "Жилийн ээлжийн амралтын процесс гэж юу вэ?",
                        "search_documents": True,
                        "is_ambiguous": False,
                        "alternatives": [],
                    },
                ],
            }
        )
        result = rewrite_queries(
            SimpleNamespace(responses=response),
            "who approve hiideg yum?",
            history=[
                {"role": "user", "content": "jiliin eeljiin amrltaa yaj awh we"},
                {"role": "assistant", "content": "Хүсэлтээ ДМС-д оруулна."},
            ],
        )

        self.assertEqual(len(result["requests"]), 1)
        self.assertIn("хэн батлах", result["requests"][0]["canonical_query"].casefold())

    def test_explicitly_compound_followup_keeps_separate_requests(self):
        response = FakeResponses(
            {
                "interaction_type": "question",
                "is_follow_up": True,
                "requests": [
                    {
                        "intent": "Ээлжийн амралтын хүсэлтийг батлах",
                        "canonical_query": "Жилийн ээлжийн амралтын хүсэлтийг хэн батлах вэ?",
                        "search_documents": True,
                        "is_ambiguous": False,
                        "alternatives": [],
                    },
                    {
                        "intent": "Цалин олгох өдөр",
                        "canonical_query": "Цалин хэзээ олгох вэ?",
                        "search_documents": True,
                        "is_ambiguous": False,
                        "alternatives": [],
                    },
                ],
            }
        )
        result = rewrite_queries(
            SimpleNamespace(responses=response),
            "who approve hiideg yum bas tsalin hezee buuh we?",
            history=[
                {"role": "user", "content": "jiliin eeljiin amrltaa yaj awh we"},
                {"role": "assistant", "content": "Хүсэлтээ ДМС-д оруулна."},
            ],
        )

        self.assertEqual(len(result["requests"]), 2)

    def test_can_normalize_without_generating_alternatives(self):
        response = FakeResponses(
            {
                "interaction_type": "question",
                "is_follow_up": False,
                "requests": [
                    {
                        "intent": "Захирлуудын хурлын цаг",
                        "canonical_query": "Захирлуудын хурал хэзээ болдог вэ?",
                        "search_documents": True,
                        "is_ambiguous": False,
                        "alternatives": ["Альтернатив асуулт"],
                    }
                ],
            }
        )
        result = rewrite_queries(
            SimpleNamespace(responses=response),
            "zahiraluudiin hural hezee boldog we",
            include_alternatives=False,
        )

        self.assertEqual(len(result["requests"]), 1)
        self.assertEqual(result["requests"][0]["alternatives"], [])
        self.assertEqual(response.kwargs["max_output_tokens"], 350)

    def test_rejects_unexpected_writing_system_in_canonical_query(self):
        client = SimpleNamespace(
            responses=FakeResponses(
                {
                    "interaction_type": "question",
                    "is_follow_up": False,
                    "requests": [
                        {
                            "intent": "Шалгалтын төлбөр",
                            "canonical_query": "Шалгалтын费用 төлбөр хэд вэ?",
                            "search_documents": True,
                            "is_ambiguous": False,
                            "alternatives": [],
                        }
                    ],
                }
            )
        )
        with self.assertRaises(ValueError):
            rewrite_queries(client, "certification exam fee hed ve")

    def test_preserves_latin_proper_names_from_input(self):
        response = FakeResponses(
            {
                "interaction_type": "question",
                "is_follow_up": False,
                "requests": [
                    {
                        "intent": "Skyteams дэх ажилтны дугаар",
                        "canonical_query": "Skyteams-ийн Ажилтан цэснээс дугаараа хэрхэн харах вэ?",
                        "search_documents": True,
                        "is_ambiguous": False,
                        "alternatives": [],
                    }
                ],
            }
        )
        result = rewrite_queries(
            SimpleNamespace(responses=response),
            "Skyteams ajiltnii dugaar haana baina ve?",
        )
        self.assertIn("Skyteams", result["requests"][0]["canonical_query"])

    def test_non_cyrillic_intent_label_does_not_discard_valid_query(self):
        response = FakeResponses(
            {
                "interaction_type": "question",
                "is_follow_up": False,
                "requests": [
                    {
                        "intent": "Annual leave request form location",
                        "canonical_query": "Жилийн ээлжийн амралтын хүсэлтийн маягт хаана байна вэ?",
                        "search_documents": True,
                        "is_ambiguous": False,
                        "alternatives": [],
                    }
                ],
            }
        )
        result = rewrite_queries(
            SimpleNamespace(responses=response),
            "leave request form ni хаана бга вэ? link өгөөч",
        )

        self.assertEqual(
            result["requests"][0]["intent"],
            result["requests"][0]["canonical_query"],
        )
        self.assertTrue(result["requests"][0]["search_documents"])

    def test_reaction_has_no_requests(self):
        response = FakeResponses(
            {
                "interaction_type": "reaction",
                "is_follow_up": True,
                "requests": [],
            }
        )
        result = rewrite_queries(
            SimpleNamespace(responses=response),
            "hhhha",
            history=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Сайн байна уу?"},
            ],
        )

        self.assertEqual(result["interaction_type"], "reaction")
        self.assertEqual(result["requests"], [])

    def test_resolves_elliptical_followup_from_previous_turn(self):
        response = FakeResponses(
            {
                "interaction_type": "reply",
                "is_follow_up": True,
                "requests": [
                    {
                        "intent": "Ажилтны дугаарыг олох журам",
                        "canonical_query": "Ажилтны дугаараа Skyteams-ийн «Ажилтан» цэснээс хэрхэн харах вэ?",
                        "search_documents": True,
                        "is_ambiguous": False,
                        "alternatives": [],
                    }
                ],
            }
        )
        history = [
            {"role": "user", "content": "ajiltnii dugaar herhen awah ve"},
            {
                "role": "assistant",
                "content": "Skyteams-ийн «Ажилтан» цэсээс ажилтны дугаараа харна.",
            },
        ]
        result = rewrite_queries(
            SimpleNamespace(responses=response),
            "holbogdoh juramiig ni tailbarlaad uguuch",
            history=history,
        )

        self.assertIn("Ажилтны дугаараа", result["requests"][0]["canonical_query"])
        conversation = json.loads(response.kwargs["input"])
        self.assertEqual(conversation["current_question"], "holbogdoh juramiig ni tailbarlaad uguuch")
        self.assertEqual(conversation["recent_conversation"], history)

    def test_router_receives_bounded_history_and_updates_summary_once(self):
        response = FakeResponses(
            {
                "interaction_type": "question",
                "is_follow_up": True,
                "matched_restrictions": ["programming", "salary"],
                "updated_summary": "Эхэнд автобус, дараа нь хөгжим ярьсан.",
                "requests": [
                    {
                        "intent": "Өмнөх ярианы эхний сэдэв",
                        "canonical_query": "Өмнөх ярианы эхний сэдэв юу байсан бэ?",
                        "search_documents": False,
                        "is_ambiguous": False,
                        "alternatives": [],
                    }
                ],
            }
        )
        history = [
            {"role": "user", "content": f"turn {index}"}
            for index in range(12)
        ]
        older = [{"role": "user", "content": "Анх автобусны талаар ярьсан."}]

        result = rewrite_queries(
            SimpleNamespace(responses=response),
            "ehnii ni yu bilee",
            history=history,
            conversation_summary="Өмнөх сэдвүүдийн товч санамж.",
            unsummarized_history=older,
            active_restrictions={"salary"},
        )

        payload = json.loads(response.kwargs["input"])
        self.assertEqual(payload["recent_conversation"], history[-10:])
        self.assertEqual(payload["conversation_to_summarize"], older)
        self.assertEqual(payload["conversation_summary"], "Өмнөх сэдвүүдийн товч санамж.")
        self.assertEqual(
            [item["id"] for item in payload["active_restrictions"]],
            ["salary"],
        )
        self.assertEqual(result["matched_restrictions"], ["salary"])
        self.assertEqual(
            result["conversation_summary"],
            "Эхэнд автобус, дараа нь хөгжим ярьсан.",
        )
        self.assertTrue(result["summary_updated"])

    def test_latest_completed_turn_ignores_older_topics_and_unanswered_prompt(self):
        messages = [
            {"role": "user", "content": "employee rights"},
            {"role": "assistant", "content": "rights answer"},
            {"role": "user", "content": "meeting schedule"},
            {"role": "assistant", "content": "meeting answer"},
            {"role": "user", "content": "follow-up"},
        ]
        self.assertEqual(latest_completed_turn(messages), messages[2:4])


if __name__ == "__main__":
    unittest.main()
