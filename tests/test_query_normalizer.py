import json
import unittest
from types import SimpleNamespace

from query_normalizer import QueryNormalizationError, normalize_query


class FakeResponses:
    def __init__(self, output):
        self.output = output
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            output_text=self.output,
            usage=SimpleNamespace(input_tokens=31, output_tokens=14),
        )


class QueryNormalizerTests(unittest.TestCase):
    def test_normalizes_mixed_input_using_recent_topic(self):
        responses = FakeResponses(
            "Жилийн ээлжийн амралтын хүсэлтийн маягт хаана байна вэ? Холбоос өгөөч?"
        )
        history = [
            {"role": "user", "content": "jiliin eeljiin amrltaa yaj awh we"},
            {"role": "assistant", "content": "ДМС-д хүсэлт оруулна."},
        ]

        result = normalize_query(
            SimpleNamespace(responses=responses),
            "leave request form ni хаана бга вэ? link өгөөч",
            history=history,
        )

        payload = json.loads(responses.kwargs["input"])
        self.assertEqual(
            payload["current_question"],
            "leave request form ni хаана бга вэ? link өгөөч",
        )
        self.assertEqual(payload["recent_conversation"], history)
        self.assertIn("Жилийн ээлжийн амралтын хүсэлтийн маягт", result["normalized_query"])
        self.assertEqual(result["usage"]["input_tokens"], 31)

    def test_rejects_unexpected_writing_system(self):
        responses = FakeResponses("Шалгалтын费用 төлбөр хэд вэ?")
        with self.assertRaises(QueryNormalizationError):
            normalize_query(
                SimpleNamespace(responses=responses),
                "certification fee hed ve",
            )

    def test_cyrillic_without_context_skips_model_call(self):
        responses = FakeResponses("should not be called")
        result = normalize_query(
            SimpleNamespace(responses=responses),
            "Цалин хэзээ буух вэ?",
        )

        self.assertEqual(result["normalized_query"], "Цалин хэзээ буух вэ?")
        self.assertIsNone(responses.kwargs)
        self.assertEqual(result["usage"]["input_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
