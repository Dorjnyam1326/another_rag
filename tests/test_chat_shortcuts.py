import unittest

from chat_shortcuts import GREETING_RESPONSE, simple_greeting_response


class ChatShortcutTests(unittest.TestCase):
    def test_known_standalone_greetings_are_local(self):
        for message in ("snuu??", "Сайн уу!", "hello", "sain baina uu"):
            with self.subTest(message=message):
                self.assertEqual(simple_greeting_response(message), GREETING_RESPONSE)

    def test_mixed_messages_and_non_greetings_do_not_match(self):
        for message in (
            "snuu tsalin hezee buuh ve",
            "hi, how do I request annual leave?",
            "what is the weather tomorrow?",
        ):
            with self.subTest(message=message):
                self.assertIsNone(simple_greeting_response(message))


if __name__ == "__main__":
    unittest.main()
