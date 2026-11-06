import json
import unittest
from pathlib import Path


class QueryRewriteEvaluationFixtureTests(unittest.TestCase):
    def test_fixture_has_twenty_cases_with_expected_intents(self):
        path = Path(__file__).parent / "fixtures" / "query_rewrite_eval.json"
        cases = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(len(cases), 20)
        self.assertEqual([case["id"] for case in cases], list(range(1, 21)))
        self.assertTrue(all(case["query"] and "expected_intents" in case for case in cases))
        self.assertTrue(
            all(
                case["must_preserve"] or case["kind"] == "greeting"
                for case in cases
            )
        )
        self.assertEqual(cases[17]["kind"], "mixed")
        self.assertEqual(cases[19]["kind"], "followup")


if __name__ == "__main__":
    unittest.main()
