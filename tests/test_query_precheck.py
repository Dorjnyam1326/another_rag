import unittest

from query_precheck import precheck_query


class QueryPrecheckTests(unittest.TestCase):
    def test_blocks_salary_amount_but_allows_payday_schedule(self):
        self.assertEqual(precheck_query("системийн инженерийн цалин хэд вэ?")[0], "salary")
        self.assertEqual(precheck_query("How much does a systems engineer earn?")[0], "salary")
        self.assertEqual(precheck_query("minii tsalin hed we?")[0], "salary")
        self.assertIsNone(precheck_query("Цалин хэзээ буух вэ?"))
        self.assertIsNone(precheck_query("When is payday this month?"))
        self.assertIsNone(precheck_query("энэ баримтад ямар жимс дурдсан бэ?"))

    def test_prechecks_code_creation_and_math_but_defers_concepts_to_router(self):
        # The contextual router classifies programming concepts so references
        # to document material or the assistant itself are not blocked by keywords.
        self.assertIsNone(precheck_query("What is a pointer in C++?"))
        self.assertEqual(
            precheck_query("Can you write Java code to sum odd numbers?")[0],
            "programming",
        )
        self.assertEqual(precheck_query("Solve 2x + 3 = 7")[0], "math")
        self.assertEqual(
            precheck_query("what is sum of n and how do I calculate it")[0],
            "math",
        )

    def test_allows_explanation_of_source_material(self):
        self.assertIsNone(
            precheck_query("Explain the C++ code in the uploaded document.")
        )
        self.assertIsNone(
            precheck_query("Тайлан дахь томьёог тайлбарла.")
        )

    def test_allows_unrelated_general_questions(self):
        self.assertIsNone(precheck_query("How do I plant tomatoes?"))
        self.assertIsNone(precheck_query("What is a fruit?"))

    def test_each_restriction_can_be_disabled_independently(self):
        self.assertIsNone(
            precheck_query("системийн инженерийн цалин хэд вэ?", {"programming", "math"})
        )
        self.assertEqual(
            precheck_query(
                "Can you write Java code to sum odd numbers?", {"programming"}
            )[0],
            "programming",
        )
        self.assertIsNone(
            precheck_query("Solve 2x + 3 = 7", {"salary", "programming"})
        )
        self.assertIsNone(
            precheck_query(
                "Can you write Java code to sum odd numbers?", {"salary", "math"}
            )
        )

    def test_allows_meta_question_about_the_assistant_even_when_code_is_enabled(self):
        self.assertIsNone(
            precheck_query("код талаасаа чи өөрийгөө яаж засах байсан бэ?")
        )


if __name__ == "__main__":
    unittest.main()
