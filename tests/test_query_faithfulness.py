import json
import tempfile
import unittest
from pathlib import Path

from query_faithfulness import (
    _process_drift_status,
    active_query_configuration,
    canonicalize_duration_correction,
    current_time_range_supported,
    filter_search_queries,
    has_explicit_duration_correction,
    normalize_current_time_range,
    promote_query_strategy,
    query_issues,
)


class QueryFaithfulnessTests(unittest.TestCase):
    def test_rejects_a_rewrite_that_changes_40_minutes_to_two_hours(self):
        issues = query_issues(
            "Ажилдаа 40 минут хоцорсон, хэнд мэдэгдэх вэ?",
            "Ажилдаа хоёр цаг хоцорсон үед хэнд мэдэгдэх вэ?",
        )
        self.assertIn("unsupported_duration", issues)

    def test_rejects_a_different_business_process_and_new_role(self):
        issues = query_issues(
            "Шууд удирдлага эзгүй үед түр чөлөөний хүсэлтийг хэн батлах вэ?",
            "Захирал эзгүй үед гэрээнд хэн гарын үсэг зурах вэ?",
        )
        self.assertIn("business_process_drift", issues)
        self.assertIn("unsupported_role", issues)

    def test_h10_latin_payroll_and_leave_queries_are_faithful(self):
        source = (
            "tsalin hezee buudag ve, bas tsalintai chuluu awah nuhtsul yu ve?"
        )
        self.assertEqual(
            _process_drift_status(source, "Цалин хэзээ олгодог вэ?"),
            "no_drift",
        )
        self.assertEqual(
            _process_drift_status(source, "Цалинтай чөлөө авах нөхцөл юу вэ?"),
            "no_drift",
        )
        self.assertEqual(query_issues(source, "Цалин хэзээ олгодог вэ?"), [])
        self.assertEqual(
            query_issues(source, "Цалинтай чөлөө авах нөхцөл юу вэ?"), []
        )

    def test_latin_process_markers_match_words_not_substrings(self):
        from query_faithfulness import process_groups

        self.assertNotIn("attendance", process_groups("calculate a value"))
        self.assertNotIn("attendance", process_groups("translate this text"))
        self.assertNotIn("attendance", process_groups("template configuration"))
        self.assertIn("attendance", process_groups("late for work"))

    def test_equivalent_english_to_cyrillic_negation_is_not_drift(self):
        self.assertNotIn(
            "negation_or_obligation_drift",
            query_issues(
                "Employees must not share confidential files.",
                "Ажилтнууд нууц файлыг хуваалцаж болохгүй.",
            ),
        )

    def test_unknown_process_classification_is_not_confirmed_drift(self):
        self.assertEqual(
            _process_drift_status("latin text not in our process lexicon", "Цалин хэзээ вэ?"),
            "undetermined",
        )
        self.assertNotIn(
            "business_process_drift",
            query_issues("latin text not in our process lexicon", "Цалин хэзээ вэ?"),
        )

    def test_allows_a_supported_leave_approval_paraphrase(self):
        self.assertEqual(
            query_issues(
                "Шууд удирдлага эзгүй үед түр чөлөөний хүсэлтийг хэн батлах вэ?",
                "Менежер байхгүй үед чөлөөг орлон батлах эрх хэнд байдаг вэ?",
            ),
            [],
        )

    def test_allows_a_cyrillic_canonical_query_for_latin_duration_correction(self):
        self.assertEqual(
            query_issues(
                "bi 3 udriin chuluu awna ugui 3 udur bish 3 tsag",
                "3 цагийн чөлөө авах гэсэн юм.",
            ),
            [],
        )

    def test_canonicalizes_explicit_duration_correction_to_accepted_value(self):
        source = "ugui 3 udur bish 3 tsag"
        self.assertEqual(
            canonicalize_duration_correction(
                "3 өдрийн биш, 3 цагийн чөлөө авах журам", source
            ),
            "3 цагийн чөлөө авах журам",
        )

    def test_does_not_treat_uncertain_duration_options_as_a_correction(self):
        self.assertFalse(
            has_explicit_duration_correction(
                "3 tsag uu 3 udur uu medehgui"
            )
        )

    def test_accepts_daytime_range_and_derived_duration_without_false_clock_duration(self):
        source = "Өдрийн 2-оос 6 хүртэл чөлөө авна"
        query = normalize_current_time_range(
            "Өдрийн 2-оос 6 хүртэл чөлөө авах журам", source
        )
        self.assertIn("14:00–18:00", query)
        self.assertIn("4 цаг", query)
        self.assertEqual(query_issues(source, query), [])
        self.assertTrue(current_time_range_supported(query, source))

    def test_time_range_supersedes_old_duration_but_does_not_infer_unstated_daypart(self):
        source = "Өглөө биш ээ, өдрийн 2-оос 6 хүртэл."
        normalized = normalize_current_time_range(
            "Маргааш өдрийн 2-оос 6 хүртэл чөлөө авах журам", source
        )
        combined_source = "Маргааш 3 цагийн чөлөө хүссэн. " + source
        self.assertEqual(query_issues(combined_source, normalized), [])
        self.assertTrue(current_time_range_supported(normalized, source))
        self.assertFalse(
            current_time_range_supported(
                "14:00–18:00 4 цаг", "2-оос 6 хүртэл чөлөө авна"
            )
        )
        self.assertTrue(
            current_time_range_supported(
                "2-оос 6 хүртэл 4 цаг", "2-оос 6 хүртэл чөлөө авна"
            )
        )
        correction = "Үгүй, 3-аас 6 хүртэл"
        prior = "Өдрийн 2-оос 6 хүртэл чөлөө авна"
        corrected = normalize_current_time_range(
            "Өдрийн 15:00–18:00 цагийн хооронд 3 цагийн чөлөө авах журам",
            correction,
            prior,
        )
        self.assertIn("15:00–18:00", corrected)
        self.assertTrue(current_time_range_supported(corrected, correction, prior))
        self.assertIsNone(
            canonicalize_duration_correction(
                "3 цаг эсвэл 3 өдрийн чөлөө", "3 tsag uu 3 udur uu medehgui"
            )
        )

    def test_filters_unsafe_alternatives_and_falls_back_to_user_wording(self):
        request = {
            "canonical_query": "Хоцролт 2 цаг давсан үед ямар журамтай вэ?",
            "alternatives": [
                "40 минут хоцорсныг хэнд мэдэгдэх вэ?",
                "Хоёр цаг хоцорсны тайлбар бичих журам",
            ],
        }
        filter_search_queries(
            "Ажилдаа 40 минут хоцорлоо, хэнд мэдэгдэх вэ?",
            request,
            fallback_query="Ажилдаа 40 минут хоцорлоо, хэнд мэдэгдэх вэ?",
        )
        self.assertEqual(
            request["canonical_query"],
            "Ажилдаа 40 минут хоцорлоо, хэнд мэдэгдэх вэ?",
        )
        self.assertEqual(request["alternatives"], ["40 минут хоцорсныг хэнд мэдэгдэх вэ?"])
        self.assertTrue(request["query_guard_rejections"])

    def test_strategy_configuration_defaults_and_promotes_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query_strategy.json"
            self.assertEqual(
                active_query_configuration(path),
                {"enabled": False, "strategy": "A"},
            )
            promote_query_strategy("B", path)
            self.assertEqual(
                active_query_configuration(path),
                {"enabled": True, "strategy": "B"},
            )
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["stage"], 7)


if __name__ == "__main__":
    unittest.main()
