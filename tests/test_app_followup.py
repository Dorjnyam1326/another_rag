import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from evaluation.conversation_regression import run as conversation_regression


class FakeVectorStores:
    calls = 0
    queries = []
    evidence_turns = {1}

    def search(self, **kwargs):
        type(self).calls += 1
        type(self).queries.append(kwargs["query"])
        if FakeOpenAI.turn in type(self).evidence_turns:
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
        is_new_leave_topic = current_question == "Ээлжийн амралтаа хэрхэн хүсэх вэ?"
        is_follow_up = bool(payload["recent_conversation"]) and not is_new_leave_topic
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
        is_leave_start = current_question.startswith("Маргааш хувийн ажлаар хагас өдөр")
        is_leave_correction = current_question.startswith("Өглөө биш ээ")
        is_leave_approval = current_question.startswith("тэр өдөр manager")
        is_story_leave_turn = is_leave_start or is_leave_correction or is_leave_approval
        is_confidential_start = current_question.startswith("Нууцын зэрэглэлтэй баримт")
        is_confidential_email = current_question.startswith("тэрийг имэйлээр")
        is_external_report = current_question.startswith("за тэрийг больё")
        is_education_benefit = (
            current_question.casefold()
            == "huuhediin surgaltiin tulbur deer company demjleg baidag uu?"
        )
        is_salary_restriction = current_question == "salary router restriction"
        is_mixed_salary_attendance = "энэ сарын миний цалин" in current_question
        is_pointer_question = current_question.casefold() == "what is a pointer in c++?"
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
        elif is_leave_start:
            query = "Маргааш хувийн ажлаар хагас өдөр чөлөө авах журам юу вэ?"
            intent = "Хагас өдрийн хувийн чөлөөний журам"
        elif is_leave_correction:
            query = "Маргааш 14:00–18:00 цагт хагас өдрийн хувийн чөлөө авах журам юу вэ?"
            intent = "Хувийн чөлөөний хугацааг өөрчлөх"
        elif is_leave_approval:
            query = "Маргааш 14:00–18:00 цагт хувийн чөлөө авахад хэн батлах вэ?"
            intent = "Хувийн чөлөөний батлах журам"
        elif is_confidential_start:
            query = "Нууцын зэрэглэлтэй баримт бичгийг хэнтэй хуваалцаж болох вэ?"
            intent = "Нууц баримт хуваалцах зөвшөөрөл"
        elif is_confidential_email:
            query = "Нууцын зэрэглэлтэй баримтыг имэйлээр явуулах нөхцөл юу вэ?"
            intent = "Нууц баримтыг имэйлээр илгээх журам"
        elif is_external_report:
            query = "Энгийн дотоод тайланг гадны байгууллагад явуулахад хэний зөвшөөрөл хэрэгтэй вэ?"
            intent = "Дотоод тайланг гадна хуваалцах зөвшөөрөл"
        elif is_new_leave_topic:
            query = "Ээлжийн амралтаа хэрхэн хүсэх вэ?"
            intent = "Ээлжийн амралтын хүсэлт"
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
        if is_salary_restriction:
            topic, request_type, request_context = "payroll", "amount", "workplace"
        elif is_salary_schedule or is_salary_holiday_rule:
            topic, request_type, request_context = "payroll", "schedule", "workplace"
        elif is_salary_process:
            topic, request_type, request_context = "payroll", "procedure", "workplace"
        elif is_education_benefit:
            topic, request_type, request_context = "benefits", "eligibility", "workplace"
        elif is_story_leave_turn:
            topic, request_type, request_context = "leave", "approval", "workplace"
        elif is_confidential_start or is_confidential_email or is_external_report:
            topic, request_type, request_context = "security", "approval", "workplace"
        elif is_new_leave_topic:
            topic, request_type, request_context = "leave", "procedure", "workplace"
        elif is_new_year_leave:
            topic, request_type, request_context = "leave", "procedure", "workplace"
        elif is_pointer_question or "write java code" in current_question.casefold():
            topic, request_type, request_context = "programming", "concept", "standalone"
            query = "C++ заагч гэж юу вэ?"
            intent = "C++ заагчийн тухай ойлголт"
        elif current_question.casefold() == "how do i plant tomatoes?":
            topic, request_type, request_context = "other", "procedure", "standalone"
        else:
            topic, request_type, request_context = "meeting", "procedure", "workplace"
        requests = []
        if interaction_type != "reaction":
            search_documents = (
                request_context == "workplace"
                and current_question.casefold() != "ehnii ni"
            )
            requests = [
                {
                    "intent": intent,
                    "topic": topic,
                    "request_type": request_type,
                    "data_scope": "personal" if is_salary_restriction else "general",
                    "request_context": request_context,
                    "canonical_query": query,
                    "search_documents": search_documents,
                    "is_ambiguous": False,
                    "alternatives": [],
                }
            ]
            if is_mixed_salary_attendance:
                requests = [
                    {
                        "intent": "Хувийн цалингийн дүн",
                        "topic": "payroll",
                        "request_type": "amount",
                        "data_scope": "personal",
                        "request_context": "workplace",
                        "canonical_query": "Энэ сарын миний цалин хэд болох вэ?",
                        "search_documents": True,
                        "is_ambiguous": False,
                        "alternatives": [],
                    },
                    {
                        "intent": "Хоцролтын бүртгэл",
                        "topic": "attendance",
                        "request_type": "procedure",
                        "data_scope": "general",
                        "request_context": "workplace",
                        "canonical_query": "Нэг цаг хоцорсныг хэрхэн бүртгүүлэх вэ?",
                        "search_documents": True,
                        "is_ambiguous": False,
                        "alternatives": [],
                    },
                ]
        output = json.dumps(
            {
                "interaction_type": interaction_type,
                "is_follow_up": is_follow_up,
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
    message_fields = []

    def __init__(self, **kwargs):
        pass

    def invoke(self, messages):
        self.calls.append(messages[-1]["content"])
        self.system_prompts.append(messages[0][1])
        self.message_fields.append(
            [set(item) for item in messages if isinstance(item, dict)]
        )
        return SimpleNamespace(
            content="meeting.docx: previous answer context",
            usage_metadata={"input_tokens": 20, "output_tokens": 5},
        )


class AppFollowupTests(unittest.TestCase):
    def test_targeted_runner_serializes_trace_and_answer_without_live_provider_calls(self):
        _, stories = conversation_regression.load_datasets()
        selected_stories = {"cases": [stories["cases"][38]]}
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeVectorStores.queries = []
        FakeResponses.calls = []
        FakeChatOpenAI.calls = []
        env = {"OPENAI_API_KEY": "test-key", "OPENAI_VECTOR_STORE_ID": "vs_test"}

        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        ):
            latest = conversation_regression._run_live(
                {"turns": []}, selected_stories, scenario_ids={39}
            )

        rows = latest["results"]["stories"]
        self.assertEqual(set(rows), {"39.1", "39.2", "39.3"})
        for row in rows.values():
            self.assertEqual(row["schema_version"], conversation_regression.SCHEMA_VERSION)
            self.assertEqual(row["evaluation_trace_status"], "valid")
            self.assertTrue(row["answer"])
            self.assertEqual(row["answer_outcome"], "answer_recorded")
            self.assertIn("original_question", row)
        self.assertEqual(rows["39.2"]["route_trace"]["follow_up"], True)
        self.assertEqual(rows["39.2"]["context_trace"]["effective_prior_turn_ids"], ["39.1"])
        self.assertTrue(rows["39.2"]["retrieval_trace"]["queries"])
        self.assertTrue(rows["39.2"]["retrieval_trace"]["evidence"][0]["file_id"])
        self.assertIn("attributes", rows["39.2"]["retrieval_trace"]["evidence"][0])
        self.assertTrue(rows["39.2"]["citations"])
        self.assertIn("evidence_ids", rows["39.2"]["citations"][0])
        self.assertTrue(rows["39.3"]["route_trace"]["topic_change"])
        self.assertEqual(rows["39.3"]["context_trace"]["effective_prior_turn_ids"], [])

    def test_evaluation_trace_records_context_evidence_and_citation_ids(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeVectorStores.queries = []
        FakeResponses.calls = []
        FakeChatOpenAI.calls = []
        FakeChatOpenAI.message_fields = []
        env = {"OPENAI_API_KEY": "test-key", "OPENAI_VECTOR_STORE_ID": "vs_test"}

        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.toggle[1].set_value(False).run()
            app.session_state["_evaluation_turn"] = {
                "suite": "stories",
                "session_id": "story-39",
                "turn_id": "39.1",
                "turn_index": 1,
            }
            app.chat_input[0].set_value("zahiraluudiin hural hezee boldog we").run()
            self.assertFalse(app.exception)
            app.session_state["_evaluation_turn"] = {
                "suite": "stories",
                "session_id": "story-39",
                "turn_id": "39.2",
                "turn_index": 2,
            }
            app.chat_input[0].set_value("holbogdoh durem juramd uur yu durdsan we").run()
            self.assertFalse(app.exception)

        messages = app.session_state["messages"]
        current_user = messages[-2]
        self.assertEqual(current_user["evaluation"]["turn_id"], "39.2")
        assistant = messages[-1]
        trace = assistant["evaluation_trace"]
        self.assertEqual(
            {item["turn_id"] for item in trace["context"]["selected_messages"]},
            {"39.1"},
        )
        self.assertFalse(trace["context"]["summary_used"])
        self.assertEqual(trace["route"]["follow_up"], True)
        self.assertEqual(trace["retrieval"]["evidence"][0]["file_id"], "file-1")
        self.assertEqual(trace["retrieval"]["evidence"][0]["origin"], "previous_turn")
        self.assertEqual(trace["citations"][0]["evidence_ids"], ["evidence-01"])
        self.assertNotIn("text", trace["retrieval"]["evidence"][0])
        self.assertTrue(
            all(fields <= {"role", "content"} for fields in FakeChatOpenAI.message_fields[-1])
        )

    def test_story_38_resolves_time_and_approval_from_latest_pair_repeatedly(self):
        env = {"OPENAI_API_KEY": "test-key", "OPENAI_VECTOR_STORE_ID": "vs_test"}
        turns = (
            "Маргааш хувийн ажлаар хагас өдөр чөлөө авч болох уу?",
            "Өглөө биш ээ, өдрийн 2-оос 6 хүртэл.",
            "тэр өдөр manager маань байхгүй юм байна лээ, тэгэхээр хэн батлах вэ?",
        )

        for _ in range(3):
            FakeOpenAI.turn = 0
            FakeVectorStores.calls = 0
            FakeVectorStores.queries = []
            FakeResponses.calls = []
            FakeChatOpenAI.calls = []
            with (
                patch.dict(os.environ, env),
                patch("openai.OpenAI", FakeOpenAI),
                patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
            ):
                app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
                app.segmented_control[0].set_value("Chat").run()
                app.toggle[1].set_value(False).run()
                turn_traces = []
                for index, query in enumerate(turns, start=1):
                    app.session_state["_evaluation_turn"] = {
                        "suite": "stories",
                        "session_id": "story-38",
                        "turn_id": f"38.{index}",
                        "turn_index": index,
                    }
                    app.chat_input[0].set_value(query).run()
                    self.assertFalse(app.exception)
                    turn_traces.append(app.session_state["messages"][-1]["evaluation_trace"])

            trace = app.session_state["messages"][-1]["evaluation_trace"]
            self.assertEqual([item["schema_version"] for item in turn_traces], [2, 2, 2])
            self.assertEqual(turn_traces[0]["route"]["follow_up"], False)
            self.assertEqual(turn_traces[1]["route"]["follow_up"], True)
            self.assertEqual(
                turn_traces[1]["context"]["effective_prior_turn_ids"], ["38.1"]
            )
            self.assertTrue(trace["route"]["follow_up"])
            self.assertFalse(trace["route"]["topic_change"])
            self.assertEqual(
                {item["turn_id"] for item in trace["context"]["selected_messages"]},
                {"38.2"},
            )
            self.assertIn("14:00–18:00", trace["route"]["requests"][0]["resolved_query"])
            self.assertIn("хувийн чөлөө", trace["route"]["requests"][0]["resolved_query"])
            self.assertEqual(trace["context"]["effective_context_status"], "used")
            self.assertEqual(trace["context"]["effective_prior_turn_ids"], ["38.2"])
            queries = trace["retrieval"]["queries"]
            self.assertTrue(queries)
            self.assertTrue(all("цалин" not in query.casefold() for query in queries))
            self.assertIn("14:00–18:00", " ".join(queries))
            self.assertIn("хувийн чөлөө", " ".join(queries).casefold())
            third_route_payload = json.loads(FakeResponses.calls[-1]["input"])
            self.assertIn("хувийн чөлөө", third_route_payload["active_topic_state"].casefold())
            self.assertIn("14:00–18:00", third_route_payload["active_topic_state"])

    def test_explicit_topic_switch_drops_previous_turn_evidence(self):
        env = {"OPENAI_API_KEY": "test-key", "OPENAI_VECTOR_STORE_ID": "vs_test"}
        turns = (
            "Нууцын зэрэглэлтэй баримтыг хэнтэй хуваалцаж болох вэ?",
            "тэрийг имэйлээр явуулж болох уу?",
            "за тэрийг больё, харин энгийн дотоод тайланг гадны байгууллага руу явуулж болох уу?",
        )

        for _ in range(3):
            FakeOpenAI.turn = 0
            FakeVectorStores.calls = 0
            FakeVectorStores.queries = []
            FakeResponses.calls = []
            FakeChatOpenAI.calls = []
            with (
                patch.dict(os.environ, env),
                patch.object(FakeVectorStores, "evidence_turns", {1, 2}),
                patch("openai.OpenAI", FakeOpenAI),
                patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
            ):
                app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
                app.segmented_control[0].set_value("Chat").run()
                app.toggle[1].set_value(False).run()
                turn_traces = []
                for index, query in enumerate(turns, start=1):
                    app.session_state["_evaluation_turn"] = {
                        "suite": "stories",
                        "session_id": "story-39",
                        "turn_id": f"39.{index}",
                        "turn_index": index,
                    }
                    app.chat_input[0].set_value(query).run()
                    self.assertFalse(app.exception)
                    turn_traces.append(app.session_state["messages"][-1]["evaluation_trace"])

            assistant = app.session_state["messages"][-1]
            trace = assistant["evaluation_trace"]
            self.assertFalse(turn_traces[0]["route"]["follow_up"])
            self.assertTrue(turn_traces[1]["route"]["follow_up"])
            self.assertEqual(
                turn_traces[1]["context"]["effective_prior_turn_ids"], ["39.1"]
            )
            self.assertEqual(
                [item["origin"] for item in turn_traces[1]["retrieval"]["evidence"]],
                ["current_retrieval", "previous_turn"],
            )
            self.assertEqual(
                turn_traces[1]["context"]["effective_evidence_ids"], ["evidence-02"]
            )
            self.assertFalse(trace["route"]["follow_up"])
            self.assertTrue(trace["route"]["topic_change"])
            self.assertEqual(trace["context"]["effective_context_status"], "not_used")
            self.assertEqual(trace["context"]["effective_prior_turn_ids"], [])
            self.assertEqual(trace["context"]["effective_evidence_ids"], [])
            query_groups = trace["retrieval"]["query_groups"]
            self.assertEqual(len(query_groups), 1)
            self.assertTrue(
                all("дотоод тайлан" in query.casefold() for query in query_groups[0]["queries"])
            )
            self.assertTrue(
                all("нууц" not in query.casefold() for query in trace["retrieval"]["queries"])
            )
            self.assertEqual(trace["retrieval"]["evidence"], [])
            switch_payload = json.loads(FakeResponses.calls[-1]["input"])
            self.assertEqual(switch_payload["recent_conversation"], [])
            self.assertEqual(switch_payload["active_topic_state"], "")
            self.assertNotIn("Previous-turn evidence for this follow-up", FakeChatOpenAI.calls[-1])
            self.assertNotIn("The directors meeting is weekly", FakeChatOpenAI.calls[-1])

    def test_mixed_salary_and_attendance_searches_only_attendance(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeVectorStores.queries = []
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
            app.session_state["_evaluation_turn"] = {
                "suite": "stories",
                "session_id": "mixed-request",
                "turn_id": "mixed.1",
                "turn_index": 1,
            }
            app.chat_input[0].set_value(
                "Нэг цаг хоцорсныг яаж бүртгүүлэх вэ, бас энэ сарын миний цалин хэд болох вэ?"
            ).run()
            self.assertFalse(app.exception)

        self.assertTrue(FakeVectorStores.queries)
        self.assertTrue(all("цалин" not in query.casefold() for query in FakeVectorStores.queries))
        self.assertNotIn("цалин", FakeChatOpenAI.calls[-1].casefold())
        self.assertIn("Хоцролт", app.session_state["messages"][-1]["requests"][1]["intent"])
        trace = app.session_state["messages"][-1]["evaluation_trace"]
        restricted = [
            item for item in trace["route"]["requests"]
            if item["restriction_status"] == "matched"
        ]
        allowed = [
            item for item in trace["route"]["requests"]
            if item["restriction_status"] == "not_matched"
        ]
        self.assertEqual(len(restricted), 1)
        self.assertIn("цалин", restricted[0]["resolved_query"].casefold())
        self.assertEqual(len(allowed), 1)
        self.assertIn("хоцор", allowed[0]["resolved_query"].casefold())
        self.assertTrue(trace["retrieval"]["queries"])
        self.assertTrue(
            all("цалин" not in query.casefold() for query in trace["retrieval"]["queries"])
        )
        self.assertEqual(
            [group["intent"] for group in trace["retrieval"]["query_groups"]],
            ["Хоцролтын бүртгэл"],
        )

    def test_unrelated_new_question_does_not_use_previous_answer_context(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeVectorStores.queries = []
        FakeResponses.calls = []
        FakeChatOpenAI.calls = []
        env = {"OPENAI_API_KEY": "test-key", "OPENAI_VECTOR_STORE_ID": "vs_test"}
        questions = (
            "zahiraluudiin hural hezee boldog we",
            "Ээлжийн амралтаа хэрхэн хүсэх вэ?",
        )

        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.toggle[1].set_value(False).run()
            for index, question in enumerate(questions, start=1):
                app.session_state["_evaluation_turn"] = {
                    "suite": "stories",
                    "session_id": "unrelated-topic-switch",
                    "turn_id": f"unrelated.{index}",
                    "turn_index": index,
                }
                app.chat_input[0].set_value(question).run()
                self.assertFalse(app.exception)

        trace = app.session_state["messages"][-1]["evaluation_trace"]
        route_payload = json.loads(FakeResponses.calls[-1]["input"])
        self.assertEqual(route_payload["active_topic_state"], "")
        self.assertEqual(trace["context"]["selected_messages"][0]["turn_id"], "unrelated.1")
        self.assertFalse(trace["route"]["follow_up"])
        self.assertTrue(trace["route"]["topic_change"])
        self.assertEqual(trace["context"]["effective_context_status"], "not_used")
        self.assertEqual(trace["context"]["effective_prior_turn_ids"], [])
        self.assertTrue(trace["retrieval"]["query_groups"])
        query_text = " ".join(trace["retrieval"]["queries"]).casefold()
        self.assertIn("ээлжийн амралт", query_text)
        self.assertNotIn("захирлуудын хурал", query_text)
        self.assertNotIn("The directors meeting", FakeChatOpenAI.calls[-1])

    def test_unsplit_mixed_request_fails_closed_without_document_search(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeVectorStores.queries = []
        FakeResponses.calls = []
        FakeChatOpenAI.calls = []
        mixed_route = {
            "original_query": "Хоцролтыг бүртгүүлэх болон хувийн цалингийн дүн",
            "requests": [
                {
                    "intent": "Хоцролт ба хувийн цалин",
                    "canonical_query": "Хоцролтыг бүртгүүлэх болон хувийн цалингийн дүн хэд вэ?",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": [],
                    "restriction_modules": [],
                }
            ],
            "interaction_type": "question",
            "is_follow_up": False,
            "topic_change": False,
            "matched_restrictions": [],
            "summary_updated": False,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        env = {"OPENAI_API_KEY": "test-key", "OPENAI_VECTOR_STORE_ID": "vs_test"}

        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
            patch("semantic_router.rewrite_queries", return_value=mixed_route),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.toggle[1].set_value(False).run()
            app.chat_input[0].set_value(
                "Нэг цаг хоцорсныг яаж бүртгүүлэх вэ, бас энэ сарын миний цалин хэд болох вэ?"
            ).run()
            self.assertFalse(app.exception)

        assistant = app.session_state["messages"][-1]
        self.assertEqual(assistant["restriction_module"], "salary")
        self.assertFalse(assistant["retrieval_performed"])
        self.assertEqual(FakeVectorStores.calls, 0)
        self.assertEqual(FakeChatOpenAI.calls, [])

    def test_half_day_leave_keeps_its_own_query_and_scope_instruction(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeVectorStores.queries = []
        FakeResponses.calls = []
        FakeChatOpenAI.calls = []
        leave_route = {
            "original_query": "Маргааш хувийн ажлаар хагас өдөр чөлөө авч болох уу?",
            "requests": [
                {
                    "intent": "Хагас өдрийн хувийн чөлөө",
                    "canonical_query": "Хагас өдрийн хувийн чөлөө авах журам юу вэ?",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": [],
                    "restriction_modules": [],
                }
            ],
            "interaction_type": "question",
            "is_follow_up": False,
            "topic_change": False,
            "matched_restrictions": [],
            "summary_updated": False,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        env = {"OPENAI_API_KEY": "test-key", "OPENAI_VECTOR_STORE_ID": "vs_test"}

        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
            patch("semantic_router.rewrite_queries", return_value=leave_route),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.toggle[1].set_value(False).run()
            app.chat_input[0].set_value(
                "Маргааш хувийн ажлаар хагас өдөр чөлөө авч болох уу?"
            ).run()
            self.assertFalse(app.exception)

        self.assertEqual(FakeVectorStores.queries, ["Хагас өдрийн хувийн чөлөө авах журам юу вэ?"])
        self.assertIn(
            "Do not transfer a condition from annual leave to personal or half-day leave",
            FakeChatOpenAI.system_prompts[-1],
        )

    def test_half_day_leave_prompt_scopes_annual_leave_conditions(self):
        prompt = (
            Path(__file__).resolve().parents[1] / "prompts" / "rag_agent.md"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "Do not transfer a condition from annual leave to personal or half-day leave",
            prompt,
        )

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
            patch("semantic_router.rewrite_queries", side_effect=ValueError("invalid schema")),
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

    def test_salary_false_positive_does_not_block_child_education_benefit_search(self):
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
            app.chat_input[0].set_value(
                "Huuhediin surgaltiin tulbur deer company demjleg baidag uu?"
            ).run()
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
            app.chat_input[0].set_value("What is a pointer in C++?").run()
            self.assertFalse(app.exception)
            active = json.loads(FakeResponses.calls[-1]["input"])["active_restrictions"]
            self.assertIn("programming", {item["id"] for item in active})

        restricted = app.session_state["messages"][-1]
        self.assertEqual(restricted["restriction_module"], "programming")
        self.assertIn("Хязгаарласан", restricted["restriction_status"])
        self.assertEqual(FakeVectorStores.calls, 0)

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

    def test_older_session_summary_is_not_sent_into_later_prompts(self):
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

            seventh_payload = json.loads(FakeResponses.calls[6]["input"])
            self.assertEqual(seventh_payload["conversation_to_summarize"], [])
            self.assertLessEqual(len(seventh_payload["recent_conversation"]), 2)

            app.chat_input[0].set_value("ehnii ni").run()
            self.assertFalse(app.exception)

        eighth_payload = json.loads(FakeResponses.calls[7]["input"])
        self.assertEqual(eighth_payload["conversation_summary"], "")
        self.assertNotIn("## Older conversation context", FakeChatOpenAI.system_prompts[-1])
        self.assertEqual(FakeVectorStores.calls, 7)

    def test_restricted_personal_salary_does_not_block_allowed_attendance_intent(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeVectorStores.queries = []
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
            app.chat_input[0].set_value(
                "Өнөөдөр ажилдаа 1 цаг хоцорсон, үүнийг яаж бүртгүүлэх вэ, "
                "бас энэ сарын миний цалин яг хэд болж орохыг тооцоод өгөөч?"
            ).run()
            self.assertFalse(app.exception)

        assistant = app.session_state["messages"][-1]
        self.assertGreater(FakeVectorStores.calls, 0)
        self.assertTrue(
            all("цалин" not in query.casefold() for query in FakeVectorStores.queries)
        )
        self.assertEqual(len(FakeChatOpenAI.calls), 1)
        self.assertNotIn("энэ сарын миний цалин", FakeChatOpenAI.calls[0])
        self.assertTrue(assistant["retrieval_performed"])
        self.assertIn("meeting.docx: previous answer context", assistant["content"])
        self.assertIn("нууцын зэрэглэлтэй", assistant["content"])
        self.assertEqual(assistant["restricted_modules"], ["salary"])


if __name__ == "__main__":
    unittest.main()
