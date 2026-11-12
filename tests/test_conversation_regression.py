import json
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.conversation_regression import run


class ConversationRegressionTests(unittest.TestCase):
    def test_real_employee_question_review_seed_is_complete_but_unverified(self):
        dataset_path = Path(__file__).parent / "fixtures" / "real_employee_questions_review_seed.json"
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        cases = dataset["cases"]

        self.assertEqual(dataset["status"], "review_seed_pending")
        self.assertEqual(len(cases), 11)
        self.assertEqual([case["id"] for case in cases], [f"real-{index:02d}" for index in range(1, 12)])
        self.assertTrue(all(case["question"].strip() for case in cases))
        self.assertTrue(all(case["tags"] and case["expected_behavior"] for case in cases))
        self.assertTrue(all(case["review"]["status"] == "pending" for case in cases))
        self.assertTrue(all(case["review"]["source_document"] is None for case in cases))
        self.assertTrue(all(case["review"]["source_section"] is None for case in cases))
        self.assertTrue(all(case["review"]["retrieved_rank"] is None for case in cases))

        sessions = {}
        for case in cases:
            sessions.setdefault(case["session_id"], []).append(case)
        for session in sessions.values():
            self.assertEqual(
                [case["turn_index"] for case in session],
                list(range(1, len(session) + 1)),
            )
        for case_id in ("real-06", "real-08", "real-10"):
            case = next(item for item in cases if item["id"] == case_id)
            self.assertIn("parent_case_id", case)
            self.assertEqual(case["history_capture_status"], "prior assistant answer not supplied")

    def test_story_report_joins_questions_and_coverage_from_fixture(self):
        pdf, stories = run.load_datasets()
        latest = {
            "schema_version": run.SCHEMA_VERSION,
            "created_at": "2026-11-09T00:00:00+00:00",
            "elapsed_seconds": 1,
            "results": {"pdf": {}, "stories": {}},
        }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with (
                patch.object(run, "RESULTS", output),
                patch.object(run, "CHART", output / "latest.svg"),
                patch.object(run, "REPORT", output / "latest.md"),
            ):
                run._write_report(pdf, stories, None, latest)
            report = (output / "latest.md").read_text(encoding="utf-8")

        story_questions = [
            turn["query"]
            for case in stories["cases"]
            for turn in case["turns"]
        ]
        story_coverage = [case["expected_coverage"] for case in stories["cases"]]
        self.assertEqual(report.count("#### Scenario "), 48)
        self.assertEqual(report.count("- Question: "), 96)
        self.assertNotIn("- Question: \n", report)
        self.assertNotIn("- Expected coverage (review guide): \n", report)
        for value in story_questions + story_coverage:
            self.assertIn(value, report)
        self.assertIn("story-38 / turn 3", report)
        self.assertIn("| Input tokens total | unavailable |", report)
        self.assertNotIn("| Input tokens total | 0.0 |", report)
        self.assertIn("Selected context: unavailable (legacy trace not recorded)", report)
        self.assertIn("Effective answer context: unavailable (legacy trace not recorded)", report)
        self.assertIn("Evidence metadata: unavailable (legacy trace not recorded)", report)
        self.assertIn("Tokens: unavailable input", report)
        self.assertIn("Retrieval query groups: unavailable (legacy trace did not record query groups)", report)

    def test_session_continuation_and_semantic_followup_labels_are_distinct(self):
        pdf, stories = run.load_datasets()
        pdf_semantic = sum(turn["expected_semantic_follow_up"] for turn in pdf["turns"])
        pdf_continuation = sum(turn["session_continuation"] for turn in pdf["turns"])
        story_turns = [turn for case in stories["cases"] for turn in case["turns"]]
        story_semantic = sum(turn["expected_semantic_follow_up"] for turn in story_turns)
        story_continuation = sum(turn["session_continuation"] for turn in story_turns)

        self.assertEqual((pdf_continuation, pdf_semantic), (47, 7))
        self.assertEqual((story_continuation, story_semantic), (8, 7))
        case39_turn3 = stories["cases"][38]["turns"][2]
        self.assertTrue(case39_turn3["session_continuation"])
        self.assertFalse(case39_turn3["expected_semantic_follow_up"])
        self.assertFalse(pdf["turns"][44]["expected_semantic_follow_up"])

    def test_legacy_missing_metrics_and_trace_fields_render_as_unavailable(self):
        pdf, stories = run.load_datasets()
        legacy = {
            "created_at": "2026-11-09T00:00:00+00:00",
            "elapsed_seconds": 1,
            "results": {
                "pdf": {
                    "1": {
                        "status": "error",
                        "answer": "",
                        "sources": [],
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_ms": 0,
                    }
                },
                "stories": {},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run._write_report(pdf, stories, None, legacy, output_dir=output)
            report = (output / "latest.md").read_text(encoding="utf-8")

        self.assertIn("| Input tokens total | unavailable |", report)
        self.assertIn("| Total latency (seconds) | unavailable |", report)
        self.assertIn("Execution mode: unavailable (legacy result metadata)", report)
        self.assertIn("Trace schema: unavailable", report)
        self.assertIn("Tokens: unavailable input · unavailable output", report)
        self.assertIn("Latency: unavailable ms", report)
        self.assertIn("Selected context: unavailable (legacy trace not recorded)", report)
        self.assertIn("Retrieval queries: unavailable (legacy retrieval query list not recorded)", report)

    def test_trace_schema_accepts_current_and_legacy_but_marks_malformed(self):
        current = {
            "schema_version": run.TRACE_SCHEMA_VERSION,
            "context": {
                "selected_messages": [],
                "effective_answer_messages": [],
                "effective_prior_turn_ids": [],
                "effective_evidence_ids": [],
                "active_topic_state_used": False,
                "active_topic_state_characters": 0,
            },
            "route": {"follow_up": False, "topic_change": False, "requests": []},
            "retrieval": {"queries": [], "query_groups": [], "evidence": []},
            "citations": [],
        }
        legacy = {
            "context": {},
            "route": {},
            "retrieval": {},
            "citations": [],
        }

        self.assertEqual(run._trace_schema_status(current), "valid")
        version_one = {**current, "schema_version": 1}
        version_one["context"] = {
            key: value for key, value in current["context"].items()
            if key not in {"active_topic_state_used", "active_topic_state_characters"}
        }
        self.assertEqual(run._trace_schema_status(version_one), "legacy")
        self.assertEqual(run._trace_schema_status(legacy), "legacy")
        self.assertEqual(run._trace_schema_status({"context": []}), "invalid")
        self.assertEqual(run._trace_schema_status(None), "unavailable")

    def test_scenarios_38_39_write_only_to_isolated_result_directory(self):
        pdf, stories = run.load_datasets()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            baseline = output / "baseline.json"
            latest = output / "latest.json"
            baseline.write_text("baseline sentinel", encoding="utf-8")
            latest.write_text("latest sentinel", encoding="utf-8")
            selected = {}

            def fake_run_live(selected_pdf, selected_stories, scenario_ids):
                selected["ids"] = scenario_ids
                selected["turn_count"] = sum(
                    len(case["turns"]) for case in selected_stories["cases"]
                )
                rows = {
                    f"{case['id']:02d}.{turn['turn_index']}": {
                        "status": "ok",
                        "answer": "offline fixture answer",
                        "sources": [],
                    }
                    for case in selected_stories["cases"]
                    for turn in case["turns"]
                }
                return {
                    "schema_version": run.SCHEMA_VERSION,
                    "created_at": "2026-11-09T00:00:00+00:00",
                    "elapsed_seconds": 0,
                    "results": {"stories": rows},
                }

            with (
                patch.object(run, "RESULTS", output),
                patch.object(run, "_run_live", side_effect=fake_run_live),
                patch.object(sys, "argv", ["run.py", "--scenarios", "38,39"]),
            ):
                self.assertEqual(run.main(), 0)

            targeted = json.loads(
                (output / "scenarios_38_39" / "latest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(selected, {"ids": {38, 39}, "turn_count": 6})
            self.assertEqual(len(targeted["results"]["stories"]), 6)
            self.assertEqual(baseline.read_text(encoding="utf-8"), "baseline sentinel")
            self.assertEqual(latest.read_text(encoding="utf-8"), "latest sentinel")

    def test_story_sessions_and_turn_order_are_preserved(self):
        _, stories = run.load_datasets()
        sessions = run._app_sessions({"turns": []}, stories)
        story_sessions = [item for item in sessions if item[0] == "stories"]

        self.assertEqual(len(story_sessions), 40)
        for case, (_, session_id, turns) in zip(stories["cases"], story_sessions):
            self.assertEqual(session_id, case["session_id"])
            self.assertEqual(
                [turn["turn_index"] for turn in turns],
                list(range(1, len(case["turns"]) + 1)),
            )
            self.assertTrue(all(turn["expected_coverage"] for turn in turns))
            self.assertEqual(
                sum(turn["expected_semantic_follow_up"] for turn in turns),
                case["follow_up_turns"],
            )


if __name__ == "__main__":
    unittest.main()
