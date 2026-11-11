import unittest

from query_precheck import (
    is_education_benefit_question,
    is_leave_policy_question,
    is_salary_process_question,
    is_salary_schedule_question,
    composite_restrictions,
    precheck_query,
)


class QueryPrecheckTests(unittest.TestCase):
    def test_mixed_salary_and_attendance_request_reaches_semantic_router(self):
        query = (
            "Өнөөдөр ажилдаа 1 цаг хоцорсон, үүнийг яаж бүртгүүлэх вэ, "
            "бас энэ сарын миний цалин яг хэд болж орохыг тооцоод өгөөч?"
        )
        self.assertIsNone(precheck_query(query))
        self.assertEqual(composite_restrictions(query), ["salary"])

    def test_blocks_salary_amount_but_allows_payday_schedule(self):
        self.assertEqual(precheck_query("системийн инженерийн цалин хэд вэ?")[0], "salary")
        self.assertEqual(precheck_query("How much does a systems engineer earn?")[0], "salary")
        self.assertEqual(precheck_query("minii tsalin hed we?")[0], "salary")
        self.assertIsNone(precheck_query("Цалин хэзээ буух вэ?"))
        self.assertIsNone(precheck_query("tsalin hezee buudag we"))
        self.assertIsNone(precheck_query("энэ сарын цалин хэдэнд буух вэ?"))
        self.assertIsNone(precheck_query("When is payday this month?"))
        self.assertEqual(precheck_query("Цалин хэдэн төгрөг вэ?")[0], "salary")
        self.assertIsNone(precheck_query("энэ баримтад ямар жимс дурдсан бэ?"))

    def test_salary_schedule_override_does_not_allow_amount_questions(self):
        self.assertTrue(is_salary_schedule_question("tsalin hezee buudag we"))
        self.assertTrue(is_salary_schedule_question("энэ сарын цалин хэдэнд буух вэ?"))
        self.assertTrue(
            is_salary_schedule_question("Цалин буух өдөр амралтын өдөр таарвал яадаг вэ?")
        )
        self.assertTrue(
            is_salary_schedule_question(
                "tsalin buuh udur amraltiin udur taarval yah ve"
            )
        )
        self.assertTrue(is_salary_schedule_question("When is payday this month?"))
        self.assertTrue(
            is_salary_schedule_question("What happens if payday falls on a holiday?")
        )
        self.assertFalse(is_salary_schedule_question("миний цалин хэд вэ?"))
        self.assertFalse(
            is_salary_schedule_question("Цалин хэдэн төгрөг вэ, энэ сард хэдэнд буух вэ?")
        )

    def test_salary_raise_procedure_is_allowed_but_amounts_are_not(self):
        for query in (
            "tsalingaa hezee yaj nemuuldeg ym bol",
            "цалингаа яаж нэмүүлэх вэ?",
            "How do I request a salary increase?",
        ):
            with self.subTest(query=query):
                self.assertTrue(is_salary_process_question(query))
                self.assertIsNone(precheck_query(query))
        self.assertFalse(
            is_salary_process_question("цалингаа 10 хувиар яаж нэмүүлэх вэ?")
        )
        self.assertEqual(
            precheck_query("цалингаа 10 хувиар яаж нэмүүлэх вэ?")[0], "salary"
        )

    def test_new_year_leave_question_is_distinguished_from_salary_restriction(self):
        for query in (
            "ene shine jileer amrah shinj bnu",
            "Энэ шинэ жилээр амрах шинж байна уу?",
        ):
            with self.subTest(query=query):
                self.assertTrue(is_leave_policy_question(query))
                self.assertIsNone(precheck_query(query))

    def test_education_benefits_do_not_match_salary_amount_restriction(self):
        for query in (
            "Huuhediin surgaltiin tulbur deer company demjleg baidag uu?",
            "Does the company reimburse professional certification fees?",
            "Does the company reimburse certification fees up to 2,000,000 төгрөг?",
            "Хүүхдийн сургалтын төлбөрийг компани даадаг уу?",
        ):
            with self.subTest(query=query):
                self.assertTrue(is_education_benefit_question(query))
                self.assertIsNone(precheck_query(query))

        self.assertFalse(
            is_education_benefit_question("Does the company reimburse tuition and my salary amount?")
        )

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
