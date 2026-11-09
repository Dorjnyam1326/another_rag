import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class FakeVectorStores:
    calls = 0
    queries = []

    def search(self, **kwargs):
        type(self).calls += 1
        type(self).queries.append(kwargs["query"])
        if FakeOpenAI.turn == 1:
            data = [
                SimpleNamespace(
                    file_id="file-1",
                    filename="meeting.docx",
                    content=[
                        SimpleNamespace(
                            text="The directors meeting is weekly on Monday at 10:00."
                        )
                    ],
                )
            ]
        else:
            data = []
        return SimpleNamespace(data=data)


class FakeResponses:
    calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        FakeOpenAI.turn += 1
        payload = json.loads(kwargs["input"])
        current_question = payload["current_question"]
        is_reaction = current_question.casefold() in {"hhhha", "хмм"}
        is_follow_up = bool(payload["recent_conversation"])
        active_ids = {item["id"] for item in payload["active_restrictions"]}
        matched_restrictions = []
        is_salary_schedule = current_question.casefold() == "tsalin hezee buudag we"
        is_salary_holiday_rule = (
            current_question == "Цалин буух өдөр амралтын өдөр таарвал яадаг вэ?"
        )
        is_new_year_leave = (
            current_question.casefold() == "ene shine jileer amrah shinj bnu"
        )
        is_salary_process = (
            current_question.casefold() == "tsalingaa hezee yaj nemuuldeg ym bol"
        )
        if current_question == "salary router restriction" and "salary" in active_ids:
            matched_restrictions = ["salary"]
        elif (is_salary_schedule or is_salary_holiday_rule) and "salary" in active_ids:
            # Simulate the model's false positive; the deterministic app guard
            # must still allow document retrieval for a payday question.
            matched_restrictions = ["salary"]
        elif is_new_year_leave and "salary" in active_ids:
            # Context can cause the router to carry the previous salary topic forward.
            matched_restrictions = ["salary"]
        elif is_salary_process and "salary" in active_ids:
            matched_restrictions = ["salary"]
        elif current_question.casefold() == "what is a pointer in c++?" and "programming" in active_ids:
            matched_restrictions = ["programming"]
        updated_summary = (
            "Эхэнд автобусны тухай, дараа нь хурлын тухай ярьсан."
            if payload["conversation_to_summarize"]
            else ""
        )
        interaction_type = (
            "reaction" if is_reaction else "reply" if is_follow_up else "question"
        )
        if is_salary_holiday_rule:
            query = "Цалин олгох өдөр амралтын өдөртэй давхцвал ямар журамтай вэ?"
            intent = "Амралтын өдөртэй давхцсан цалин олгох хуваарь"
        elif is_salary_schedule:
            query = "Цалин хэзээ буудаг вэ?"
            intent = "Цалин олгох хуваарь"
        elif is_salary_process:
            query = "Цалин нэмүүлэх хүсэлтийг ямар журмаар гаргадаг вэ?"
            intent = "Цалин нэмүүлэх хүсэлтийн журам"
        elif is_follow_up:
            query = (
                "\u0417\u0430\u0445\u0438\u0440\u043b\u0443\u0443\u0434\u044b\u043d "
                "\u0445\u0443\u0440\u043b\u044b\u043d \u0436\u0443\u0440\u0430\u043c\u0434 "
                "\u04e9\u04e9\u0440 \u044e\u0443 \u0434\u0443\u0440\u0434\u0441\u0430\u043d "
                "\u0431\u0430\u0439\u0434\u0430\u0433 \u0432\u044d?"
            )
            intent = "Захирлуудын хурлын мэдээлэл"
        else:
            query = (
                "\u0417\u0430\u0445\u0438\u0440\u043b\u0443\u0443\u0434\u044b\u043d "
                "\u0445\u0443\u0440\u0430\u043b \u0445\u044d\u0437\u044d\u044d "
                "\u0431\u043e\u043b\u0434\u043e\u0433 \u0432\u044d?"
            )
            intent = "Захирлуудын хурлын мэдээлэл"
        requests = []
        if interaction_type != "reaction":
            search_documents = current_question.casefold() != "ehnii ni"
            requests = [
                {
                    "intent": intent,
                    "canonical_query": query,
                    "search_documents": search_documents,
                    "is_ambiguous": False,
                    "alternatives": [],
                }
            ]
        output = json.dumps(
            {
                "interaction_type": interaction_type,
                "is_follow_up": is_follow_up,
                "matched_restrictions": matched_restrictions,
                "updated_summary": updated_summary,
                "requests": requests,
            },
            ensure_ascii=False,
        )
        return SimpleNamespace(
            status="completed",
            output_text=output,
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )


class FakeOpenAI:
    turn = 0

    def __init__(self, **kwargs):
        self.responses = FakeResponses()
        self.vector_stores = FakeVectorStores()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeChatOpenAI:
    calls = []
    system_prompts = []

    def __init__(self, **kwargs):
        pass

    def invoke(self, messages):
        self.calls.append(messages[-1]["content"])
        self.system_prompts.append(messages[0][1])
        return SimpleNamespace(
            content="meeting.docx: previous answer context",
            usage_metadata={"input_tokens": 20, "output_tokens": 5},
        )


class AppFollowupTests(unittest.TestCase):
    def test_dashboard_recent_questions_include_tokens_and_latency_totals(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeResponses.calls = []
        FakeChatOpenAI.calls = []
        env = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_VECTOR_STORE_ID": "vs_test",
        }

        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.toggle[1].set_value(False).run()
            app.chat_input[0].set_value("zahiraluudiin hural hezee boldog we").run()
            self.assertFalse(app.exception)
            app.segmented_control[0].set_value("Dashboard").run()
            self.assertFalse(app.exception)

        table = app.dataframe[0].value
        self.assertIn("Оролтын token", table.columns)
        self.assertIn("Нийт token", table.columns)
        self.assertIn("Rewrite оролтын token", table.columns)
        self.assertIn("Нийт хугацаа (ms)", table.columns)
        self.assertIn("Vector Store хайлт (ms)", table.columns)
        self.assertIn("Model (ms)", table.columns)
        self.assertEqual(table.iloc[0]["Асуулт"], "zahiraluudiin hural hezee boldog we")
        self.assertEqual(table.iloc[-1]["Асуулт"], "Нийт (сүүлийн 1 асуулт)")
        self.assertEqual(table.iloc[-1]["Нийт token"], table.iloc[0]["Нийт token"])
        elapsed_ms = int(str(table.iloc[0]["Нийт хугацаа (ms)"]).replace(",", ""))
        self.assertGreaterEqual(elapsed_ms, 0)

    def test_structured_rewrite_failure_uses_normalized_query_fallback(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeVectorStores.queries = []
        FakeResponses.calls = []
        FakeChatOpenAI.calls = []
        normalized_query = "Жилийн ээлжийн амралтын хүсэлтийн маягт хаана байна вэ? Холбоос өгөөч?"
        env = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_VECTOR_STORE_ID": "vs_test",
        }

        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
            patch("query_rewriter.rewrite_queries", side_effect=ValueError("invalid schema")),
            patch(
                "query_normalizer.normalize_query",
                return_value={
                    "normalized_query": normalized_query,
                    "usage": {"input_tokens": 31, "output_tokens": 14},
                },
            ),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.toggle[1].set_value(False).run()
            app.chat_input[0].set_value(
                "leave request form ni хаана бга вэ? link өгөөч"
            ).run()
            self.assertFalse(app.exception)

        assistant = app.session_state["messages"][-1]
        self.assertEqual(FakeVectorStores.queries, [normalized_query])
        self.assertEqual(assistant["requests"][0]["canonical_query"], normalized_query)
        self.assertEqual(assistant["usage"]["rewrite_input"], 31)
        self.assertEqual(assistant["usage"]["rewrite_output"], 14)
        self.assertEqual(assistant["rewrite_failure"], "ValueError")
        timings = assistant["timing_ms"]
        self.assertEqual(
            set(timings), {"rewrite_ms", "retrieval_ms", "model_ms", "total_ms"}
        )
        self.assertTrue(all(value >= 0 for value in timings.values()))
        self.assertTrue(
            any(
                "Хугацаа: нийт" in item.value
                and "Vector Store хайлт" in item.value
                and "model" in item.value
                for item in app.caption
            )
        )
        self.assertTrue(
            any(
                "хэвийнжүүлж Vector Store-оос хайлаа" in item.value
                for item in app.warning
            )
        )

    def test_followup_reuses_last_turn_and_its_retrieved_evidence(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeResponses.calls = []
        FakeChatOpenAI.calls = []
        env = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_VECTOR_STORE_ID": "vs_test",
        }

        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.toggle[1].set_value(False).run()
            app.chat_input[0].set_value("zahiraluudiin hural hezee boldog we").run()
            self.assertFalse(app.exception)
            first_answer = app.session_state["messages"][-1]
            self.assertEqual(len(first_answer["retrieval_queries"]), 1)
            self.assertTrue(
                any(
                    "\u0400" <= char <= "\u04ff"
                    for char in first_answer["retrieval_queries"][0]
                )
            )
            self.assertEqual(first_answer["evidence"][0]["filename"], "meeting.docx")

            app.chat_input[0].set_value(
                "holbogdoh durem juramd uur yu durdsan we"
            ).run()
            self.assertFalse(app.exception)

        rewrite_input = json.loads(FakeResponses.calls[1]["input"])
        self.assertEqual(
            rewrite_input["recent_conversation"][0]["content"],
            "zahiraluudiin hural hezee boldog we",
        )
        self.assertIn(
            "Previous-turn evidence for this follow-up",
            FakeChatOpenAI.calls[1],
        )
        self.assertIn(
            "The directors meeting is weekly on Monday at 10:00.",
            FakeChatOpenAI.calls[1],
        )
        self.assertEqual(FakeVectorStores.calls, 2)

    def test_reaction_is_not_sent_to_document_search_as_a_new_question(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeResponses.calls = []
        FakeChatOpenAI.calls = []
        env = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_VECTOR_STORE_ID": "vs_test",
        }

        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.toggle[1].set_value(False).run()
            app.chat_input[0].set_value("zahiraluudiin hural hezee boldog we").run()
            self.assertFalse(app.exception)
            app.chat_input[0].set_value("hhhha").run()
            self.assertFalse(app.exception)

        reaction = app.session_state["messages"][-1]
        self.assertEqual(reaction["interaction_type"], "reaction")
        self.assertEqual(reaction["shortcut"], "local_reaction")
        self.assertEqual(reaction["retrieval_queries"], [])
        self.assertEqual(FakeVectorStores.calls, 1)
        self.assertEqual(len(FakeResponses.calls), 1)
        self.assertEqual(len(FakeChatOpenAI.calls), 1)

    def test_contextual_reply_can_use_previous_evidence_without_a_new_search(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeResponses.calls = []
        FakeChatOpenAI.calls = []
        env = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_VECTOR_STORE_ID": "vs_test",
        }

        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.toggle[1].set_value(False).run()
            app.chat_input[0].set_value("zahiraluudiin hural hezee boldog we").run()
            self.assertFalse(app.exception)
            app.chat_input[0].set_value("za oilgoloo").run()
            self.assertFalse(app.exception)
            app.chat_input[0].set_value("ehnii ni").run()
            self.assertFalse(app.exception)

        reply = app.session_state["messages"][-1]
        self.assertEqual(reply["interaction_type"], "reply")
        self.assertEqual(reply["retrieval_queries"], [])
        acknowledgment = next(
            message
            for message in app.session_state["messages"]
            if message["role"] == "user"
            and message.get("interaction_type") == "acknowledgment"
        )
        self.assertEqual(acknowledgment["content"], "za oilgoloo")
        self.assertEqual(FakeVectorStores.calls, 1)
        self.assertEqual(len(FakeResponses.calls), 2)
        self.assertIn(
            "The directors meeting is weekly on Monday at 10:00.",
            FakeChatOpenAI.calls[-1],
        )

    def test_router_classified_reaction_keeps_the_current_message_for_answering(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeResponses.calls = []
        FakeChatOpenAI.calls = []
        env = {"OPENAI_API_KEY": "test-key", "OPENAI_VECTOR_STORE_ID": "vs_test"}

        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.toggle[1].set_value(False).run()
            app.chat_input[0].set_value("zahiraluudiin hural hezee boldog we").run()
            self.assertFalse(app.exception)
            app.chat_input[0].set_value("хмм").run()
            self.assertFalse(app.exception)

        reaction = app.session_state["messages"][-1]
        self.assertEqual(reaction["interaction_type"], "reaction")
        self.assertEqual(reaction["retrieval_queries"], [])
        self.assertEqual(FakeVectorStores.calls, 1)
        self.assertEqual(FakeChatOpenAI.calls[-1], "хмм")

    def test_simple_greeting_skips_rewriter_retrieval_and_answer_model(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeResponses.calls = []
        FakeChatOpenAI.calls = []

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "", "OPENAI_VECTOR_STORE_ID": ""}),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.chat_input[0].set_value("snuu??").run()
            self.assertFalse(app.exception)

        assistant = app.session_state["messages"][-1]
        self.assertEqual(assistant["shortcut"], "local_greeting")
        self.assertEqual(FakeResponses.calls, [])
        self.assertEqual(FakeVectorStores.calls, 0)
        self.assertEqual(FakeChatOpenAI.calls, [])

    def test_standalone_acknowledgment_does_not_get_an_echo_or_call_models(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeResponses.calls = []
        FakeChatOpenAI.calls = []

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "", "OPENAI_VECTOR_STORE_ID": ""}),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.chat_input[0].set_value("za oilgoloo").run()
            self.assertFalse(app.exception)

        self.assertEqual(len(app.session_state["messages"]), 2)
        acknowledgment = app.session_state["messages"][0]
        self.assertEqual(acknowledgment["interaction_type"], "acknowledgment")
        self.assertEqual(acknowledgment["shortcut"], "local_acknowledgment")
        local_reply = app.session_state["messages"][1]
        self.assertEqual(
            local_reply["interaction_type"], "local_acknowledgment_response"
        )
        self.assertEqual(
            local_reply["content"], "За. Хэрэгтэй бол үргэлжлүүлээд асуугаарай."
        )
        self.assertEqual(FakeResponses.calls, [])
        self.assertEqual(FakeVectorStores.calls, 0)
        self.assertEqual(FakeChatOpenAI.calls, [])

    def test_router_salary_false_positive_does_not_block_payday_search(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeResponses.calls = []
        FakeChatOpenAI.calls = []
        env = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_VECTOR_STORE_ID": "vs_test",
        }

        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.chat_input[0].set_value("tsalin hezee buudag we").run()
            self.assertFalse(app.exception)

        assistant = app.session_state["messages"][-1]
        self.assertNotIn("restriction_module", assistant)
        self.assertNotIn("restriction_status", assistant)
        self.assertTrue(assistant["retrieval_performed"])
        self.assertGreater(FakeVectorStores.calls, 0)

    def test_router_salary_false_positive_does_not_block_holiday_schedule_search(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeResponses.calls = []
        FakeChatOpenAI.calls = []
        env = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_VECTOR_STORE_ID": "vs_test",
        }

        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.chat_input[0].set_value(
                "Цалин буух өдөр амралтын өдөр таарвал яадаг вэ?"
            ).run()
            self.assertFalse(app.exception)

        assistant = app.session_state["messages"][-1]
        self.assertNotIn("restriction_module", assistant)
        self.assertNotIn("restriction_status", assistant)
        self.assertTrue(assistant["retrieval_performed"])
        self.assertGreater(FakeVectorStores.calls, 0)

    def test_previous_salary_topic_does_not_block_new_year_leave_question(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeVectorStores.queries = []
        FakeResponses.calls = []
        FakeChatOpenAI.calls = []
        env = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_VECTOR_STORE_ID": "vs_test",
        }

        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.chat_input[0].set_value("tsalin hezee buudag we").run()
            app.chat_input[0].set_value("ene shine jileer amrah shinj bnu").run()
            self.assertFalse(app.exception)

        assistant = app.session_state["messages"][-1]
        self.assertNotIn("restriction_module", assistant)
        self.assertNotIn("restriction_status", assistant)
        self.assertTrue(assistant["retrieval_performed"])
        self.assertGreater(FakeVectorStores.calls, 0)

    def test_router_salary_false_positive_does_not_block_raise_process_search(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeResponses.calls = []
        FakeChatOpenAI.calls = []
        env = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_VECTOR_STORE_ID": "vs_test",
        }

        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.chat_input[0].set_value("tsalingaa hezee yaj nemuuldeg ym bol").run()
            self.assertFalse(app.exception)

        assistant = app.session_state["messages"][-1]
        self.assertNotIn("restriction_module", assistant)
        self.assertNotIn("restriction_status", assistant)
        self.assertTrue(assistant["retrieval_performed"])
        self.assertGreater(FakeVectorStores.calls, 0)

    def test_restriction_switch_changes_precheck_and_router_configuration(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeResponses.calls = []
        FakeChatOpenAI.calls = []
        env = {"OPENAI_API_KEY": "test-key", "OPENAI_VECTOR_STORE_ID": "vs_test"}

        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.toggle[1].set_value(False).run()
            app.checkbox(key="restriction_programming").uncheck().run()
            app.chat_input[0].set_value("What is a pointer in C++?").run()
            self.assertFalse(app.exception)
            allowed = app.session_state["messages"][-1]
            self.assertNotIn("restriction_module", allowed)
            self.assertNotIn(
                "Do not answer standalone programming questions",
                FakeChatOpenAI.system_prompts[-1],
            )
            active = json.loads(FakeResponses.calls[-1]["input"])["active_restrictions"]
            self.assertNotIn("programming", {item["id"] for item in active})

            app.checkbox(key="restriction_programming").check().run()
            app.chat_input[0].set_value("Can you write Java code to sum odd numbers?").run()
            self.assertFalse(app.exception)

        restricted = app.session_state["messages"][-1]
        self.assertEqual(restricted["restriction_module"], "programming")
        self.assertIn("Хязгаарласан", restricted["restriction_status"])
        self.assertEqual(FakeVectorStores.calls, 1)

    def test_router_restriction_shows_status_without_search_or_answer_call(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeResponses.calls = []
        FakeChatOpenAI.calls = []
        env = {"OPENAI_API_KEY": "test-key", "OPENAI_VECTOR_STORE_ID": "vs_test"}

        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.toggle[1].set_value(False).run()
            app.chat_input[0].set_value("salary router restriction").run()
            self.assertFalse(app.exception)

        restricted = app.session_state["messages"][-1]
        self.assertEqual(restricted["restriction_module"], "salary")
        self.assertIn("нууцын зэрэглэлтэй", restricted["content"])
        self.assertEqual(FakeVectorStores.calls, 0)
        self.assertEqual(FakeChatOpenAI.calls, [])

    def test_standalone_programming_concept_is_restricted_by_contextual_router(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeResponses.calls = []
        FakeChatOpenAI.calls = []
        env = {"OPENAI_API_KEY": "test-key", "OPENAI_VECTOR_STORE_ID": "vs_test"}

        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.toggle[1].set_value(False).run()
            app.chat_input[0].set_value("What is a pointer in C++?").run()
            self.assertFalse(app.exception)

        restricted = app.session_state["messages"][-1]
        self.assertEqual(restricted["restriction_module"], "programming")
        self.assertEqual(FakeVectorStores.calls, 0)
        self.assertEqual(FakeChatOpenAI.calls, [])

    def test_router_runs_in_general_chat_without_vector_store_access(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeResponses.calls = []
        FakeChatOpenAI.calls = []

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_VECTOR_STORE_ID": ""}),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.toggle[0].set_value(False).run()
            app.chat_input[0].set_value("How do I plant tomatoes?").run()
            self.assertFalse(app.exception)

        self.assertEqual(len(FakeResponses.calls), 1)
        self.assertEqual(FakeVectorStores.calls, 0)
        self.assertEqual(len(FakeChatOpenAI.calls), 1)

    def test_clear_conversation_resets_summary_but_keeps_restriction_switches(self):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
        app.checkbox(key="restriction_programming").uncheck().run()
        app.session_state["conversation_summary"] = "Өмнөх сэдвүүд."
        app.session_state["summary_message_count"] = 8
        app.button(key="clear_conversation").click().run()

        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["conversation_summary"], "")
        self.assertEqual(app.session_state["summary_message_count"], 0)
        self.assertFalse(app.session_state["restriction_programming"])

    def test_older_session_topic_is_summarized_and_available_to_followup(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeResponses.calls = []
        FakeChatOpenAI.calls = []
        FakeChatOpenAI.system_prompts = []
        env = {"OPENAI_API_KEY": "test-key", "OPENAI_VECTOR_STORE_ID": "vs_test"}

        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.toggle[1].set_value(False).run()
            for index in range(7):
                app.chat_input[0].set_value(f"Хурлын мэдээлэл {index}").run()
                self.assertFalse(app.exception)

            self.assertIn("автобусны тухай", app.session_state["conversation_summary"])
            seventh_payload = json.loads(FakeResponses.calls[6]["input"])
            self.assertEqual(len(seventh_payload["conversation_to_summarize"]), 2)
            self.assertEqual(len(seventh_payload["recent_conversation"]), 10)

            app.chat_input[0].set_value("ehnii ni").run()
            self.assertFalse(app.exception)

        eighth_payload = json.loads(FakeResponses.calls[7]["input"])
        self.assertTrue(eighth_payload["conversation_summary"])
        self.assertIn("## Older conversation context", FakeChatOpenAI.system_prompts[-1])
        self.assertEqual(FakeVectorStores.calls, 7)


if __name__ == "__main__":
    unittest.main()
