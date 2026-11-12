import unittest

from pipeline_trace import create_pipeline_trace, public_trace, skip_steps, update_step


class PipelineTraceTests(unittest.TestCase):
    def test_trace_has_nine_stages_and_redacts_credentials_and_vector_store_id(self):
        trace = create_pipeline_trace(
            "асуулт",
            [{"role": "user", "content": "өмнөх асуулт"}],
            secret_values=("sk-test-secret", "vs-private-id"),
        )
        update_step(
            trace,
            "router",
            status="complete",
            input={"request": {"api_key": "sk-test-secret", "input": "vs-private-id"}},
            processing="routed",
            output={"raw": "value sk-test-secret"},
        )
        skip_steps(trace, ("resolver",), "new topic")

        public = public_trace(trace)

        self.assertEqual(public["schema_version"], 1)
        self.assertEqual(len(public["steps"]), 9)
        self.assertEqual(public["steps"][2]["status"], "complete")
        self.assertEqual(public["steps"][4]["status"], "skipped")
        self.assertNotIn("sk-test-secret", str(public))
        self.assertNotIn("vs-private-id", str(public))
        self.assertEqual(public["steps"][2]["input"]["request"]["api_key"], "[redacted]")


if __name__ == "__main__":
    unittest.main()
