import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from conversation_export import conversation_json
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
    resolver_calls = []
    force_resolver_clarification = False

    def create(self, **kwargs):
        payload = json.loads(kwargs["input"])
        if "user_sources" in payload:
            type(self).resolver_calls.append(kwargs)
            sources = payload["user_sources"]
            results = []
            for request in payload["requests"]:
                current_question = payload["current_question"].casefold()
                missing_leave_source = (
                    "хэн батлах" in current_question
                    and not any("чөлөө" in source["content"].casefold() for source in sources)
                )
                if (
                    sources
                    and not missing_leave_source
                    and not type(self).force_resolver_clarification
                ):
                    results.append({
                        "request_index": request["request_index"],
                        "status": "resolved",
                        "clarification_question": "",
                        "facts": [
                            {
                                "source_message_index": source["source_message_index"],
                                "source_quote": source["content"],
                            }
                            for source in sources
                        ],
                    })
                else:
                    results.append({
                        "request_index": request["request_index"],
                        "status": "clarification",
                        "clarification_question": "Ямар хүсэлтийн талаар асууж байна вэ?",
                        "facts": [],
                    })
            return SimpleNamespace(
                status="completed",
                output_text=json.dumps({"requests": results}, ensure_ascii=False),
                usage=SimpleNamespace(input_tokens=3, output_tokens=2),
            )
        self.calls.append(kwargs)
        FakeOpenAI.turn += 1
        current_question = payload["current_question"]
        is_reaction = current_question.casefold() in {"hhhha", "хмм"}
        is_new_leave_topic = current_question == "Ээлжийн амралтаа хэрхэн хүсэх вэ?"
        is_confidential_start = current_question.startswith("Нууцын зэрэглэлтэй баримт")
        is_follow_up = (
            bool(payload["recent_conversation"])
            and not is_new_leave_topic
            and not is_confidential_start
            and not current_question.casefold()
            == "ene shine jileer amrah shinj bnu"
        )
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
        relationship = (
            "topic_change"
            if is_external_report
            else "follow_up"
            if is_follow_up
            else "new_topic"
        )
        relevant_context_indices = (
            list(range(len(payload["recent_conversation"])))
            if is_follow_up
            else []
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
                "relationship": relationship,
                "relevant_context_indices": relevant_context_indices,
                "use_active_topic_state": bool(
                    is_follow_up and payload["active_topic_state"]
                ),
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


class FailingChatOpenAI:
    def __init__(self, **kwargs):
        pass

    def invoke(self, messages):
        raise RuntimeError("answer model unavailable")


class AppFollowupTests(unittest.TestCase):
    def setUp(self):
        FakeResponses.resolver_calls = []
        FakeResponses.force_resolver_clarification = False

    def test_pipeline_trace_is_complete_session_only_and_collapsed_rerender_makes_no_calls(self):
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
            app.chat_input[0].set_value("Баримтад амралтын өдөр юу гэж байна вэ?").run()
            self.assertFalse(app.exception)

            assistant = app.session_state["messages"][-1]
            trace = assistant["pipeline_trace"]
            self.assertEqual(trace["schema_version"], 1)
            self.assertEqual(len(trace["steps"]), 9)
            self.assertEqual(trace["steps"][2]["status"], "complete")
            self.assertEqual(trace["steps"][6]["status"], "complete")
            self.assertTrue(trace["steps"][6]["output"]["per_query_results"])
            self.assertTrue(trace["steps"][7]["input"]["messages"])
            self.assertNotIn("test-key", str(trace))
            self.assertNotIn("vs_test", str(trace))
            self.assertNotIn("pipeline_trace", json.loads(conversation_json(app.session_state["messages"]))["messages"][-1])

            counts = (
                len(FakeResponses.calls),
                FakeVectorStores.calls,
                len(FakeChatOpenAI.calls),
            )
            trace_key = f"pipeline-trace-{len(app.session_state['messages']) - 1}"
            app.session_state[trace_key] = True
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(any("1. Хэрэглэгчийн оролт" in item.value for item in app.markdown))
            self.assertTrue(app.code)
            app.session_state[trace_key] = False
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(any("1. Хэрэглэгчийн оролт" in item.value for item in app.markdown))
            self.assertFalse(app.exception)
            self.assertEqual(
                counts,
                (len(FakeResponses.calls), FakeVectorStores.calls, len(FakeChatOpenAI.calls)),
            )

    def test_model_failure_is_attached_to_user_turn_with_later_steps_marked(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeVectorStores.queries = []
        FakeResponses.calls = []
        env = {"OPENAI_API_KEY": "test-key", "OPENAI_VECTOR_STORE_ID": "vs_test"}
        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FailingChatOpenAI),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.chat_input[0].set_value("Баримтад амралтын өдөр юу гэж байна вэ?").run()
            self.assertFalse(app.exception)

        user_turn = app.session_state["messages"][-1]
        trace = user_turn["pipeline_trace"]
        steps = {step["id"]: step for step in trace["steps"]}
        self.assertEqual(steps["retrieval"]["status"], "complete")
        self.assertEqual(steps["answer"]["status"], "error")
        self.assertEqual(steps["session_output"]["status"], "error")

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
        self.assertEqual(rows["39.2"]["citations"], [])
        self.assertTrue(rows["39.3"]["route_trace"]["topic_change"])
        self.assertEqual(rows["39.3"]["context_trace"]["effective_prior_turn_ids"], [])

    def test_evaluation_trace_does_not_infer_citations_from_filename_text(self):
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
        self.assertEqual(trace["citations"], [])
        self.assertEqual(assistant["sources"], [])
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
                turn_traces[1]["context"]["effective_prior_turn_ids"], []
            )
            self.assertTrue(turn_traces[1]["retrieval"]["performed"])
            self.assertTrue(
                any("14:00–18:00" in query for query in turn_traces[1]["retrieval"]["queries"])
            )
            self.assertEqual(
                turn_traces[1]["route"]["requests"][0]["resolution_status"],
                "not_needed",
            )
            self.assertTrue(trace["route"]["follow_up"])
            self.assertFalse(trace["route"]["topic_change"])
            self.assertEqual(
                {item["turn_id"] for item in trace["context"]["selected_messages"]},
                {"38.2"},
            )
            self.assertEqual(
                trace["route"]["requests"][0]["resolution_status"],
                "clarification",
            )
            self.assertEqual(trace["context"]["effective_context_status"], "not_used")
            self.assertEqual(trace["context"]["effective_prior_turn_ids"], [])
            queries = trace["retrieval"]["queries"]
            self.assertFalse(queries)
            self.assertFalse(trace["retrieval"]["performed"])
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
        for personal_salary_detail in ("үндсэн цалин", "хэдэн төгрөг"):
            self.assertTrue(
                all(personal_salary_detail not in query.casefold() for query in FakeVectorStores.queries)
            )
            self.assertNotIn(personal_salary_detail, FakeChatOpenAI.calls[-1].casefold())
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
        pipeline = app.session_state["messages"][-1]["pipeline_trace"]
        policy = next(item for item in pipeline["steps"] if item["id"] == "policy")
        strategy = next(item for item in pipeline["steps"] if item["id"] == "strategy")
        self.assertEqual(policy["status"], "complete")
        self.assertTrue(any(item["decision"] == "restricted" for item in policy["output"]["decisions"]))
        self.assertTrue(strategy["output"]["retrieval_queries"])
        self.assertNotIn("цалин", json.dumps(strategy["output"]["retrieval_queries"], ensure_ascii=False).casefold())
        self.assertTrue(
            all("цалин" not in query.casefold() for query in trace["retrieval"]["queries"])
        )
        self.assertEqual(
            [group["intent"] for group in trace["retrieval"]["query_groups"]],
            ["Хоцролтын бүртгэл"],
        )

    def test_mixed_programming_and_math_restrictions_preserve_allowed_document_requests(self):
        scenarios = (
            {
                "question": "Тайлангийн томьёог тайлбарла. Бас Python дээр код бич.",
                "allowed": {
                    "intent": "Тайлангийн томьёоны тайлбар",
                    "canonical_query": "Оруулсан тайлангийн томьёог тайлбарла.",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": [],
                    "restriction_modules": [],
                },
                "restricted": {
                    "intent": "Python код бичих",
                    "topic": "programming",
                    "request_type": "code_creation",
                    "data_scope": "general",
                    "request_context": "standalone",
                    "canonical_query": "Python дээр код бичих.",
                    "search_documents": False,
                    "is_ambiguous": False,
                    "alternatives": [],
                    "restriction_modules": [],
                },
                "blocked_terms": ("python", "код"),
                "refusal": "код бичих/засах",
            },
            {
                "question": "Оруулсан тайлан дахь зардлын томьёог тайлбарла. Мөн 240000-ийг 12-т хуваа.",
                "allowed": {
                    "intent": "Тайлан дахь зардлын томьёоны тайлбар",
                    "canonical_query": "Тайлан дахь зардлын томьёог тайлбарла.",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": [],
                    "restriction_modules": [],
                },
                "restricted": {
                    "intent": "Бие даасан математик тооцоо",
                    "topic": "math",
                    "request_type": "calculation",
                    "data_scope": "general",
                    "request_context": "standalone",
                    "canonical_query": "240000-ийг 12-т хуваа.",
                    "search_documents": False,
                    "is_ambiguous": False,
                    "alternatives": [],
                    "restriction_modules": [],
                },
                "blocked_terms": ("240000", "12-т"),
                "refusal": "математикийн бодлого",
            },
        )
        env = {"OPENAI_API_KEY": "test-key", "OPENAI_VECTOR_STORE_ID": "vs_test"}
        for scenario in scenarios:
            with self.subTest(question=scenario["question"]):
                FakeOpenAI.turn = 0
                FakeVectorStores.calls = 0
                FakeVectorStores.queries = []
                FakeResponses.calls = []
                FakeChatOpenAI.calls = []
                route = {
                    "original_query": scenario["question"],
                    "requests": [scenario["allowed"], scenario["restricted"]],
                    "interaction_type": "question",
                    "is_follow_up": False,
                    "topic_change": False,
                    "matched_restrictions": [],
                    "summary_updated": False,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
                with (
                    patch.dict(os.environ, env),
                    patch("openai.OpenAI", FakeOpenAI),
                    patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
                    patch("semantic_router.rewrite_queries", return_value=route),
                ):
                    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
                    app.segmented_control[0].set_value("Chat").run()
                    app.toggle[1].set_value(False).run()
                    app.chat_input[0].set_value(scenario["question"]).run()
                    self.assertFalse(app.exception)

                self.assertTrue(FakeVectorStores.queries)
                self.assertTrue(
                    all(
                        term not in query.casefold()
                        for term in scenario["blocked_terms"]
                        for query in FakeVectorStores.queries
                    )
                )
                self.assertTrue(
                    all(
                        term not in FakeChatOpenAI.calls[-1].casefold()
                        for term in scenario["blocked_terms"]
                    )
                )
                self.assertIn(scenario["refusal"], app.session_state["messages"][-1]["content"])

    def test_mixed_restriction_router_and_normalizer_failures_never_search_raw_question(self):
        question = (
            "Ирцийн хоцролтоо яаж засуулах вэ? Бас энэ сарын миний цалин хэд болохыг тооц."
        )
        normalized = {
            "normalized_query": "Ирцийн хоцролтыг засуулах болон энэ сарын цалинг тооцох.",
            "usage": {"input_tokens": 2, "output_tokens": 1},
        }
        env = {"OPENAI_API_KEY": "test-key", "OPENAI_VECTOR_STORE_ID": "vs_test"}
        for normalizer_result in (normalized, ValueError("normalizer unavailable")):
            with self.subTest(normalizer_success=not isinstance(normalizer_result, Exception)):
                FakeOpenAI.turn = 0
                FakeVectorStores.calls = 0
                FakeVectorStores.queries = []
                FakeResponses.calls = []
                FakeChatOpenAI.calls = []
                with (
                    patch.dict(os.environ, env),
                    patch("openai.OpenAI", FakeOpenAI),
                    patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
                    patch("semantic_router.rewrite_queries", side_effect=ValueError("route unavailable")),
                    patch(
                        "query_normalizer.normalize_query",
                        return_value=normalizer_result
                        if not isinstance(normalizer_result, Exception)
                        else None,
                        side_effect=normalizer_result
                        if isinstance(normalizer_result, Exception)
                        else None,
                    ),
                ):
                    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
                    app.segmented_control[0].set_value("Chat").run()
                    app.toggle[1].set_value(False).run()
                    app.session_state["_evaluation_turn"] = {
                        "suite": "stories",
                        "session_id": "mixed-router-failure",
                        "turn_id": "c01.1",
                        "turn_index": 1,
                    }
                    app.chat_input[0].set_value(question).run()
                    self.assertFalse(app.exception)

                assistant = app.session_state["messages"][-1]
                self.assertFalse(assistant["retrieval_performed"])
                self.assertEqual(FakeVectorStores.calls, 0)
                self.assertEqual(FakeChatOpenAI.calls, [])
                self.assertIn("цалингийн дүн", assistant["content"])
                self.assertEqual(assistant["requests"], [])
                self.assertTrue(assistant["routing_failed"])
                self.assertTrue(assistant["evaluation_trace"]["route"]["routing_failed"])
                router_step = next(
                    item for item in assistant["pipeline_trace"]["steps"]
                    if item["id"] == "router"
                )
                self.assertTrue(router_step["output"]["routing_failed"])

    def test_incomplete_router_retries_and_answers_two_allowed_requests(self):
        question = (
            "bi 4 tsagiin chuluu awmaar baina, yaj huselt uguh ve, tsalintai yu, "
            "bas minii undsen tsalin hed ve?"
        )
        route = {
            "interaction_type": "question",
            "relationship": "new_topic",
            "relevant_context_indices": [],
            "use_active_topic_state": False,
            "updated_summary": "",
            "requests": [
                {
                    "intent": "Дөрвөн цагийн чөлөө авах хүсэлт",
                    "topic": "leave",
                    "request_type": "procedure",
                    "data_scope": "general",
                    "request_context": "workplace",
                    "canonical_query": "Дөрвөн цагийн чөлөө авах хүсэлтийг хэрхэн гаргах вэ?",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": [],
                },
                {
                    "intent": "Дөрвөн цагийн чөлөө цалинтай эсэх",
                    "topic": "leave",
                    "request_type": "paid_leave",
                    "data_scope": "general",
                    "request_context": "workplace",
                    "canonical_query": "Дөрвөн цагийн чөлөө ямар нөхцөлд цалинтай вэ?",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": [],
                },
                {
                    "intent": "Хувийн үндсэн цалингийн дүн",
                    "topic": "payroll",
                    "request_type": "amount",
                    "data_scope": "personal",
                    "request_context": "workplace",
                    "canonical_query": "Миний үндсэн цалин хэд вэ?",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": [],
                },
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

        env = {"OPENAI_API_KEY": "test-key", "OPENAI_VECTOR_STORE_ID": "vs_test"}
        FakeVectorStores.calls = 0
        FakeVectorStores.queries = []
        FakeChatOpenAI.calls = []
        Responses.calls = []
        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
            patch.object(FakeResponses, "create", Responses.create),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.chat_input[0].set_value(question).run()
            self.assertFalse(app.exception)

        assistant = app.session_state["messages"][-1]
        self.assertEqual([call["max_output_tokens"] for call in Responses.calls], [1400, 1800])
        self.assertEqual(len(assistant["requests"]), 3)
        self.assertEqual(assistant["restricted_modules"], ["salary"])
        self.assertTrue(assistant["retrieval_performed"])
        self.assertTrue(FakeVectorStores.queries)
        self.assertTrue(all("үндсэн цалин" not in query.casefold() for query in FakeVectorStores.queries))
        self.assertNotIn("үндсэн цалин", FakeChatOpenAI.calls[-1].casefold())
        self.assertIn("нууцын зэрэглэлтэй", assistant["content"])
        pipeline = {item["id"]: item for item in assistant["pipeline_trace"]["steps"]}
        decisions = pipeline["policy"]["output"]["decisions"]
        self.assertEqual([item["decision"] for item in decisions], ["allowed", "allowed", "restricted"])
        self.assertEqual(len(pipeline["router"]["output"]["routing_attempts"]), 2)

    def test_mixed_latin_payday_redacts_restricted_text_but_keeps_safe_followup_context(self):
        mixed_query = "tsalin hedend buudag bas minii tsalin hed ve?"
        followup_query = "tegwel ter udur ni amraltiin udur bwal?"
        routes = {
            mixed_query: {
                "interaction_type": "question",
                "relationship": "new_topic",
                "relevant_context_indices": [],
                "use_active_topic_state": False,
                "updated_summary": "",
                "requests": [
                    {
                        "intent": "Цалин олгох хуваарь",
                        "topic": "payroll",
                        "request_type": "schedule",
                        "data_scope": "general",
                        "request_context": "workplace",
                        "canonical_query": "Цалин хэдний өдөр олгодог вэ?",
                        "search_documents": True,
                        "is_ambiguous": False,
                        "alternatives": [],
                    },
                    {
                        "intent": "Хувийн цалингийн дүн",
                        "topic": "payroll",
                        "request_type": "amount",
                        "data_scope": "personal",
                        "request_context": "workplace",
                        "canonical_query": "Миний цалин хэд вэ?",
                        "search_documents": True,
                        "is_ambiguous": False,
                        "alternatives": [],
                    },
                ],
            },
            followup_query: {
                "interaction_type": "reply",
                "relationship": "follow_up",
                "relevant_context_indices": [0, 1],
                "use_active_topic_state": True,
                "updated_summary": "",
                "requests": [{
                    "intent": "Амралтын өдөртэй давхцсан цалин олгох хуваарь",
                    "topic": "payroll",
                    "request_type": "schedule",
                    "data_scope": "general",
                    "request_context": "workplace",
                    "canonical_query": "Цалин олгох өдөр амралтын өдөртэй давхцвал ямар журамтай вэ?",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": [],
                }],
            },
        }

        class RouterClient:
            def __init__(self, **kwargs):
                self.responses = SimpleNamespace(create=self._route)
                self.vector_stores = FakeVectorStores()

            def _route(self, **kwargs):
                payload = json.loads(kwargs["input"])
                if "user_sources" in payload:
                    request = payload["requests"][0]
                    source = payload["user_sources"][0]
                    outcome = {
                        "requests": [{
                            "request_index": request["request_index"],
                            "status": "resolved",
                            "clarification_question": "",
                            "facts": [{
                                "source_message_index": source["source_message_index"],
                                "source_quote": source["content"],
                            }],
                        }]
                    }
                    return SimpleNamespace(
                        status="completed",
                        output_text=json.dumps(outcome, ensure_ascii=False),
                        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                    )
                route = routes[payload["current_question"]]
                FakeResponses.calls.append(kwargs)
                return SimpleNamespace(
                    status="completed",
                    output_text=json.dumps(route, ensure_ascii=False),
                    usage=SimpleNamespace(input_tokens=2, output_tokens=1),
                )

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        FakeResponses.calls = []
        FakeVectorStores.calls = 0
        FakeVectorStores.queries = []
        FakeChatOpenAI.calls = []
        env = {"OPENAI_API_KEY": "test-key", "OPENAI_VECTOR_STORE_ID": "vs_test"}
        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", RouterClient),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.chat_input[0].set_value(mixed_query).run()
            self.assertFalse(app.exception)
            first_assistant = app.session_state["messages"][-1]
            self.assertEqual(
                [request["canonical_query"] for request in first_assistant["requests"]],
                ["Цалин хэдний өдөр олгодог вэ?", "Миний цалин хэд вэ?"],
            )
            self.assertEqual(first_assistant["restricted_modules"], ["salary"])
            self.assertEqual(
                app.session_state["messages"][-2]["context_content"],
                "Цалин хэдний өдөр олгодог вэ?",
            )

            app.chat_input[0].set_value(followup_query).run()
            self.assertFalse(app.exception)

        second_assistant = app.session_state["messages"][-1]
        second_route_payload = json.loads(FakeResponses.calls[-1]["input"])
        safe_history = json.dumps(second_route_payload["recent_conversation"], ensure_ascii=False)
        self.assertIn("Цалин хэдний өдөр олгодог вэ?", safe_history)
        self.assertNotIn("minii tsalin hed ve", safe_history.casefold())
        self.assertEqual(
            FakeVectorStores.queries,
            [
                "Цалин хэдний өдөр олгодог вэ?",
                "Цалин олгох өдөр амралтын өдөртэй давхцвал ямар журамтай вэ?",
            ],
        )
        self.assertTrue(second_assistant["retrieval_performed"])
        self.assertEqual(second_assistant["requests"][0]["resolution_status"], "resolved")

    def test_salary_review_full_and_plus_mixed_requests_keep_safe_followup_context(self):
        initial = "tsalingaa nemuuleh huselt gargah gesen ym"
        full_mixed = "tsalin nemuuleh huseltiig hen batlah bas manai heltsiin tsalin hed ve"
        mixed = "tsalin nemuuleh huseltiig hen batlah + heltsiin tsalinii dun"
        followup = "ter huselt haana gargadag ve"
        allowed_query = "Цалин нэмүүлэх хүсэлтийг хэн батлах вэ?"
        routes = {
            initial: {
                "interaction_type": "question",
                "relationship": "new_topic",
                "relevant_context_indices": [],
                "use_active_topic_state": False,
                "updated_summary": "",
                "requests": [{
                    "intent": "Цалин нэмүүлэх хүсэлтийн журам",
                    "topic": "payroll",
                    "request_type": "procedure",
                    "data_scope": "general",
                    "request_context": "workplace",
                    "canonical_query": "Цалин нэмүүлэх хүсэлтийг хэрхэн гаргах вэ?",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": [],
                }],
            },
            mixed: {
                "interaction_type": "question",
                "relationship": "new_topic",
                "relevant_context_indices": [],
                "use_active_topic_state": False,
                "updated_summary": "",
                "requests": [
                    {
                        "intent": "Цалин нэмүүлэх хүсэлтийг хэн батлах журам",
                        "topic": "payroll",
                        "request_type": "approval",
                        "data_scope": "general",
                        "request_context": "workplace",
                        "canonical_query": allowed_query,
                        "search_documents": True,
                        "is_ambiguous": False,
                        "alternatives": [],
                    },
                    {
                        "intent": "Хэлтсийн ажилтнуудын цалингийн дүн",
                        "topic": "payroll",
                        "request_type": "amount",
                        "data_scope": "personal",
                        "request_context": "workplace",
                        "canonical_query": "Хэлтсийн ажилтнуудын цалингийн дүн хэд вэ?",
                        "search_documents": True,
                        "is_ambiguous": False,
                        "alternatives": [],
                    },
                ],
            },
            followup: {
                "interaction_type": "reply",
                "relationship": "follow_up",
                "relevant_context_indices": [0, 1],
                "use_active_topic_state": True,
                "updated_summary": "",
                "requests": [{
                    "intent": "Цалин нэмүүлэх хүсэлт гаргах газар",
                    "topic": "payroll",
                    "request_type": "procedure",
                    "data_scope": "general",
                    "request_context": "workplace",
                    "canonical_query": "Цалин нэмүүлэх хүсэлтийг хаана гаргаж болох вэ?",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": [],
                }],
            },
        }
        routes[full_mixed] = routes[mixed]

        class RouterClient:
            def __init__(self, **kwargs):
                self.responses = SimpleNamespace(create=self._route)
                self.vector_stores = FakeVectorStores()

            def _route(self, **kwargs):
                payload = json.loads(kwargs["input"])
                if "user_sources" in payload:
                    request = payload["requests"][0]
                    source = payload["user_sources"][0]
                    result = {"requests": [{
                        "request_index": request["request_index"],
                        "status": "resolved",
                        "clarification_question": "",
                        "facts": [{
                            "source_message_index": source["source_message_index"],
                            "source_quote": source["content"],
                        }],
                    }]}
                    return SimpleNamespace(
                        status="completed",
                        output_text=json.dumps(result, ensure_ascii=False),
                        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                    )
                FakeResponses.calls.append(kwargs)
                return SimpleNamespace(
                    status="completed",
                    output_text=json.dumps(routes[payload["current_question"]], ensure_ascii=False),
                    usage=SimpleNamespace(input_tokens=2, output_tokens=1),
                )

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        FakeResponses.calls = []
        FakeVectorStores.calls = 0
        FakeVectorStores.queries = []
        FakeChatOpenAI.calls = []
        env = {"OPENAI_API_KEY": "test-key", "OPENAI_VECTOR_STORE_ID": "vs_test"}
        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", RouterClient),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.chat_input[0].set_value(initial).run()
            self.assertFalse(app.exception)
            app.chat_input[0].set_value(full_mixed).run()
            self.assertFalse(app.exception)
            full_assistant = app.session_state["messages"][-1]
            self.assertEqual(full_assistant["restricted_modules"], ["salary"])

            app.chat_input[0].set_value(mixed).run()
            self.assertFalse(app.exception)
            second_assistant = app.session_state["messages"][-1]
            self.assertEqual(second_assistant["restricted_modules"], ["salary"])
            self.assertEqual(
                app.session_state["messages"][-2]["context_content"], allowed_query
            )
            app.chat_input[0].set_value(followup).run()
            self.assertFalse(app.exception)

        third_route = json.loads(FakeResponses.calls[-1]["input"])
        safe_context = json.dumps(third_route["recent_conversation"], ensure_ascii=False).casefold()
        self.assertIn(allowed_query.casefold(), safe_context)
        self.assertNotIn("heltsiin tsalinii dun", safe_context)
        self.assertEqual(
            app.session_state["messages"][-1]["requests"][0]["resolution_status"],
            "resolved",
        )
        self.assertEqual(
            FakeVectorStores.queries,
            [
                "Цалин нэмүүлэх хүсэлтийг хэрхэн гаргах вэ?",
                allowed_query,
                allowed_query,
                "Цалин нэмүүлэх хүсэлтийг хаана гаргаж болох вэ?",
            ],
        )

    def test_pure_restricted_turn_does_not_erase_prior_allowed_followup_context(self):
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
            app.chat_input[0].set_value("tsalingaa hezee yaj nemuuldeg ym bol").run()
            self.assertFalse(app.exception)
            allowed_topic = app.session_state["active_topic_state"]
            self.assertIn("Цалин нэмүүлэх", allowed_topic)

            app.chat_input[0].set_value("minii undsen tsalin yag hed ve?").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["active_topic_state"], allowed_topic)

            app.chat_input[0].set_value("ter huselt haana gargadag ve").run()
            self.assertFalse(app.exception)

        followup_payload = json.loads(FakeResponses.calls[-1]["input"])
        recent_context = json.dumps(followup_payload["recent_conversation"], ensure_ascii=False).casefold()
        self.assertIn("tsalingaa hezee yaj nemuuldeg ym bol", recent_context)
        self.assertNotIn("minii undsen tsalin yag hed ve", recent_context)

    def test_corrected_followup_searches_only_the_router_canonical_query(self):
        old_request = "bi 3 udriin chuluu awna"
        assistant_history = "Чөлөөний хүсэлтийг ойлголоо."
        correction = "ugui 3 udur bish 3 tsag"
        canonical = "3 өдрийн биш, 3 цагийн чөлөө авах гэсэн юм."
        route = {
            "original_query": correction,
            "requests": [{
                "intent": "Гурван цагийн чөлөө авах журам",
                "canonical_query": canonical,
                "search_documents": True,
                "is_ambiguous": False,
                "alternatives": [],
                "restriction_modules": [],
            }],
            "interaction_type": "reply",
            "is_follow_up": True,
            "topic_change": False,
            "relationship": "follow_up",
            "relevant_context_indices": [0, 1],
            "context_messages": [
                {"role": "user", "content": old_request, "source_message_index": 0},
                {"role": "assistant", "content": assistant_history, "source_message_index": 1},
            ],
            "selected_active_topic_state": "",
            "active_topic_state": canonical,
            "matched_restrictions": [],
            "summary_updated": False,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        resolver_result = {
            "requests": {
                0: {
                    "status": "resolved",
                    "resolved_query": "3 цагийн чөлөө авах гэсэн юм.",
                    "source_message_indices": [0],
                }
            },
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        env = {"OPENAI_API_KEY": "test-key", "OPENAI_VECTOR_STORE_ID": "vs_test"}
        FakeVectorStores.calls = 0
        FakeVectorStores.queries = []
        FakeChatOpenAI.calls = []

        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
            patch("semantic_router.rewrite_queries", return_value=route),
            patch("context_resolver.resolve_followup_queries", return_value=resolver_result),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.session_state["messages"] = [
                {"role": "user", "content": old_request},
                {"role": "assistant", "content": assistant_history},
            ]
            app.chat_input[0].set_value(correction).run()
            self.assertFalse(app.exception)

        assistant = app.session_state["messages"][-1]
        self.assertEqual(FakeVectorStores.queries, ["3 цагийн чөлөө авах гэсэн юм."])
        self.assertNotIn("3 өдрийн", FakeVectorStores.queries[0])
        self.assertNotIn("udriin", FakeVectorStores.queries[0].casefold())
        self.assertTrue(assistant["retrieval_performed"])
        strategy = next(
            item for item in assistant["pipeline_trace"]["steps"]
            if item["id"] == "strategy"
        )
        self.assertEqual(strategy["output"]["strategy"], "B")

    def test_affirmative_confirmation_resumes_pending_query_without_router(self):
        FakeOpenAI.turn = 0
        FakeVectorStores.calls = 0
        FakeVectorStores.queries = []
        FakeResponses.calls = []
        FakeChatOpenAI.calls = []
        env = {"OPENAI_API_KEY": "test-key", "OPENAI_VECTOR_STORE_ID": "vs_test"}
        pending_query = "3 цагийн чөлөө авах журмыг тайлбарлана уу?"

        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", FakeOpenAI),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
            patch("context_resolver.resolve_followup_queries") as resolver,
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.session_state["messages"] = [
                {"role": "user", "content": "3 цагийн чөлөөний нөхцөл?"},
                {
                    "role": "assistant",
                    "content": "Та 3 цагийн чөлөөний журмыг асууж байна уу?",
                    "pending_requests": [{
                        "intent": "3 цагийн чөлөө авах журам",
                        "pending_query": pending_query,
                        "restriction_modules": [],
                    }],
                },
            ]
            app.chat_input[0].set_value("yes").run()
            self.assertFalse(app.exception)

        self.assertEqual(FakeVectorStores.queries, [pending_query])
        self.assertEqual(FakeResponses.calls, [])
        resolver.assert_not_called()
        assistant = app.session_state["messages"][-1]
        self.assertEqual(assistant["requests"][0]["canonical_query"], pending_query)
        router_trace = next(
            item for item in assistant["pipeline_trace"]["steps"]
            if item["id"] == "router"
        )
        self.assertTrue(router_trace["output"]["confirmation_resume"])
        self.assertNotIn("yes", FakeChatOpenAI.calls[-1].casefold())

    def test_new_three_day_leave_request_after_four_hour_request_skips_resolver(self):
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

        class RouteClient:
            def __init__(self, **kwargs):
                self.responses = SimpleNamespace(create=self._route)
                self.vector_stores = FakeVectorStores()

            def _route(self, **kwargs):
                return SimpleNamespace(
                    status="completed",
                    output_text=json.dumps(route, ensure_ascii=False),
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                )

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        FakeOpenAI.turn = 1
        FakeVectorStores.calls = 0
        FakeVectorStores.queries = []
        FakeChatOpenAI.calls = []
        env = {"OPENAI_API_KEY": "test-key", "OPENAI_VECTOR_STORE_ID": "vs_test"}
        with (
            patch.dict(os.environ, env),
            patch("openai.OpenAI", RouteClient),
            patch("langchain_openai.ChatOpenAI", FakeChatOpenAI),
            patch("context_resolver.resolve_followup_queries") as resolver,
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.session_state["messages"] = [
                {"role": "user", "content": "bi 4 tsagiin chuluu yaj awah ve?"},
                {"role": "assistant", "content": "Өмнөх чөлөөний хариулт."},
            ]
            app.chat_input[0].set_value("bi 3 udriin chuluu awna").run()
            self.assertFalse(app.exception)

        resolver.assert_not_called()
        assistant = app.session_state["messages"][-1]
        self.assertEqual(FakeVectorStores.queries, [
            "3 өдрийн чөлөө авахын тулд ямар журам баримтлах вэ?"
        ])
        self.assertTrue(assistant["retrieval_performed"])
        self.assertFalse(assistant["follow_up_context"])
        router = next(
            item for item in assistant["pipeline_trace"]["steps"]
            if item["id"] == "router"
        )
        self.assertEqual(
            router["output"]["normalized_route"]["context_resolution_decision"],
            "self_contained_leave_request",
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
        self.assertFalse(trace["route"]["topic_change"])
        self.assertEqual(trace["route"]["relationship"], "new_topic")
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

    def test_half_day_leave_guidance_comes_from_domain_config(self):
        from domain_rules import load_domain_rules

        prompt = (
            Path(__file__).resolve().parents[1] / "prompts" / "rag_agent.md"
        ).read_text(encoding="utf-8")
        self.assertNotIn("Do not transfer a condition from annual leave", prompt)
        self.assertIn(
            "Do not transfer a condition from annual leave",
            " ".join(load_domain_rules()["answer_instructions"]),
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
            set(timings), {
                "rewrite_ms", "context_resolution_ms", "retrieval_ms",
                "model_ms", "total_ms",
            }
        )
        self.assertTrue(all(value is None or value >= 0 for value in timings.values()))
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
        self.assertEqual(len(FakeResponses.resolver_calls), 1)
        resolver_payload = json.loads(FakeResponses.resolver_calls[0]["input"])
        self.assertEqual(
            [item["role"] for item in resolver_payload["selected_context"]],
            ["user", "assistant"],
        )
        self.assertEqual(
            [item["source_message_index"] for item in resolver_payload["user_sources"]],
            [0],
        )
        followup = app.session_state["messages"][-1]
        self.assertEqual(followup["requests"][0]["source_message_indices"], [0])
        pipeline = followup["pipeline_trace"]
        resolver_step = next(item for item in pipeline["steps"] if item["id"] == "resolver")
        self.assertIn("source_quote", json.dumps(resolver_step, ensure_ascii=False))
        self.assertIn("zahiraluudiin hural hezee boldog we", json.dumps(resolver_step, ensure_ascii=False))
        self.assertEqual(
            followup["retrieval_queries"],
            ["Захирлуудын хурлын журамд өөр юу дурдсан байдаг вэ?"],
        )
        self.assertNotIn("zahiraluudiin", followup["retrieval_queries"][0].casefold())
        self.assertNotIn("holbogdoh durem", followup["retrieval_queries"][0].casefold())
        self.assertEqual(FakeVectorStores.calls, 2)

    def test_unresolved_followup_asks_clarification_without_search_or_answer_call(self):
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
            app.chat_input[0].set_value("zahiraluudiin hural hezee boldog we").run()
            self.assertFalse(app.exception)
            FakeResponses.force_resolver_clarification = True
            app.chat_input[0].set_value("holbogdoh durem juramd uur yu durdsan we").run()
            self.assertFalse(app.exception)

        assistant = app.session_state["messages"][-1]
        self.assertIn("Ямар хүсэлтийн талаар асууж байна вэ?", assistant["content"])
        self.assertEqual(FakeVectorStores.calls, 1)
        self.assertEqual(len(FakeChatOpenAI.calls), 1)
        self.assertEqual(len(FakeResponses.resolver_calls), 1)
        self.assertEqual(assistant["retrieval_queries"], [])
        steps = {step["id"]: step for step in assistant["pipeline_trace"]["steps"]}
        self.assertEqual(steps["resolver"]["status"], "complete")
        self.assertEqual(steps["retrieval"]["status"], "skipped")
        self.assertEqual(steps["answer"]["status"], "skipped")

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
        steps = {step["id"]: step for step in app.session_state["messages"][-1]["pipeline_trace"]["steps"]}
        self.assertEqual(steps["fast_check"]["status"], "complete")
        self.assertTrue(all(steps[name]["status"] == "skipped" for name in ("router", "resolver", "retrieval", "answer")))

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

    def test_mixed_transliterated_leave_questions_search_without_personal_salary(self):
        question = (
            "bi margash 1 udriin chuluu awmaar baina. manager baihgui bol hen batalah ve? "
            "ene chuluu tsalintai yu? bas minii sariin undsen tsalin yag heden tugrug baidgiig heleed uguuch?"
        )
        route = {
            "original_query": question,
            "requests": [
                {
                    "intent": "Нэг өдрийн чөлөөний батлах журам",
                    "canonical_query": "Менежер эзгүй үед нэг өдрийн чөлөөг хэн батлах вэ?",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": [],
                    "restriction_modules": [],
                },
                {
                    "intent": "Нэг өдрийн чөлөө цалинтай эсэх",
                    "canonical_query": "Нэг өдрийн чөлөө ямар нөхцөлд цалинтай вэ?",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": [],
                    "restriction_modules": [],
                },
                {
                    "intent": "Хувийн сарын үндсэн цалингийн дүн",
                    "canonical_query": "Миний сарын үндсэн цалин яг хэдэн төгрөг вэ?",
                    "search_documents": True,
                    "is_ambiguous": False,
                    "alternatives": [],
                    "restriction_modules": [],
                },
            ],
            "interaction_type": "question",
            "is_follow_up": False,
            "topic_change": False,
            "matched_restrictions": [],
            "summary_updated": False,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
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
            patch("semantic_router.rewrite_queries", return_value=route),
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            app.segmented_control[0].set_value("Chat").run()
            app.chat_input[0].set_value(question).run()
            self.assertFalse(app.exception)

        assistant = app.session_state["messages"][-1]
        self.assertGreater(FakeVectorStores.calls, 0)
        for personal_salary_detail in ("үндсэн цалин", "хэдэн төгрөг"):
            self.assertTrue(
                all(personal_salary_detail not in query.casefold() for query in FakeVectorStores.queries)
            )
            self.assertNotIn(personal_salary_detail, FakeChatOpenAI.calls[-1].casefold())
        self.assertIn("нууцын зэрэглэлтэй", assistant["content"])
        self.assertEqual(assistant["restricted_modules"], ["salary"])


if __name__ == "__main__":
    unittest.main()
