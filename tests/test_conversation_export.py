import json
import unittest

from conversation_export import conversation_json


class ConversationExportTests(unittest.TestCase):
    def test_exports_transcript_and_sources_without_internal_evidence(self):
        messages = [
            {"role": "user", "content": "цалин хэзээ буух вэ?"},
            {
                "role": "assistant",
                "content": "Сарын 15-нд болон сүүлийн өдөр.",
                "sources": ["policy.docx"],
                "usage": {"input": 10, "output": 8},
                "evidence": [{"text": "private retrieval context"}],
            },
            {"role": "system", "content": "omit this"},
        ]

        exported = json.loads(conversation_json(messages))

        self.assertEqual(len(exported["messages"]), 2)
        self.assertEqual(exported["messages"][0]["content"], "цалин хэзээ буух вэ?")
        self.assertEqual(exported["messages"][1]["sources"], ["policy.docx"])
        self.assertEqual(exported["messages"][1]["usage"], {"input": 10, "output": 8})
        self.assertNotIn("evidence", exported["messages"][1])


if __name__ == "__main__":
    unittest.main()
