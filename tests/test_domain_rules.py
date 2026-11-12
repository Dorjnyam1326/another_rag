import json
import tempfile
import unittest
from pathlib import Path

from domain_rules import load_domain_rules


class DomainRulesTests(unittest.TestCase):
    def test_empty_config_keeps_the_pipeline_generic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "domain_rules.json"
            path.write_text(json.dumps({}), encoding="utf-8")
            rules = load_domain_rules(path)

        self.assertEqual(rules["process_groups"], {})
        self.assertEqual(rules["role_groups"], {})
        self.assertEqual(rules["scopes"], [])
        self.assertEqual(rules["answer_instructions"], [])
        self.assertIsNone(rules["new_year_leave_pattern"])
        self.assertIsNone(rules["education_benefit_pattern"])

    def test_current_domain_config_loads_existing_markers_and_patterns(self):
        rules = load_domain_rules()
        self.assertIn("attendance", rules["process_groups"])
        self.assertIn("manager", rules["role_groups"])
        self.assertTrue(rules["scopes"])
        self.assertTrue(rules["answer_instructions"])
        self.assertIsNotNone(rules["new_year_leave_pattern"])
        self.assertIsNotNone(rules["education_benefit_pattern"])


if __name__ == "__main__":
    unittest.main()
