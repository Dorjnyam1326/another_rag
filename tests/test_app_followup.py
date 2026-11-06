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
        is_reaction = current_question.casefold() == "hhhha"
        is_follow_up = bool(payload["recent_conversation"])
        interaction_type = (
            "reaction" if is_reaction else "reply" if is_follow_up else "question"
        )
        query = (
            "\u0417\u0430\u0445\u0438\u0440\u043b\u0443\u0443\u0434\u044b\u043d "
            "\u0445\u0443\u0440\u043b\u044b\u043d \u0436\u0443\u0440\u0430\u043c\u0434 "
            "\u04e9\u04e9\u0440 \u044e\u0443 \u0434\u0443\u0440\u0434\u0441\u0430\u043d "
            "\u0431\u0430\u0439\u0434\u0430\u0433 \u0432\u044d?"
            if is_follow_up
            else "\u0417\u0430\u0445\u0438\u0440\u043b\u0443\u0443\u0434\u044b\u043d "
            "\u0445\u0443\u0440\u0430\u043b \u0445\u044d\u0437\u044d\u044d "
            "\u0431\u043e\u043b\u0434\u043e\u0433 \u0432\u044d?"
        )
        requests = []
        if interaction_type != "reaction":
            search_documents = current_question.casefold() != "ehnii ni"
            requests = [
                {
                    "intent": "Захирлуудын хурлын мэдээлэл",
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

    def __init__(self, **kwargs):
        pass

    def invoke(self, messages):
        self.calls.append(messages[-1]["content"])
        return SimpleNamespace(
            content="meeting.docx: previous answer context",
            usage_metadata={"input_tokens": 20, "output_tokens": 5},
        )


class AppFollowupTests(unittest.TestCase):
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
        self.assertEqual(reaction["retrieval_queries"], [])
        self.assertEqual(FakeVectorStores.calls, 1)
        self.assertEqual(FakeChatOpenAI.calls[-1], "hhhha")

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
            app.chat_input[0].set_value("ehnii ni").run()
            self.assertFalse(app.exception)

        reply = app.session_state["messages"][-1]
        self.assertEqual(reply["interaction_type"], "reply")
        self.assertEqual(reply["retrieval_queries"], [])
        self.assertEqual(FakeVectorStores.calls, 1)
        self.assertIn("The directors meeting is weekly on Monday at 10:00.", FakeChatOpenAI.calls[-1])

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


if __name__ == "__main__":
    unittest.main()
