import unittest

from chat_shortcuts import (
    GREETING_RESPONSE,
    REACTION_RESPONSE,
    ACKNOWLEDGMENT_RESPONSE,
    simple_greeting_response,
    simple_acknowledgment_response,
    simple_reaction_response,
)


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

    def test_only_standalone_laughter_uses_reaction_fast_path(self):
        for message in ("hhhha", "hahaha!", "хаха", "хэхэ"):
            with self.subTest(message=message):
                self.assertEqual(simple_reaction_response(message), REACTION_RESPONSE)
        for message in ("тийм ээ", "үгүй ээ", "хаха, харин тийм үү?"):
            with self.subTest(message=message):
                self.assertIsNone(simple_reaction_response(message))

    def test_only_standalone_acknowledgments_are_consumed_locally(self):
        for message in (
            "за ойлголоо",
            "ОЙЛГОЛОО.",
            "za oilgoloo!",
            "oilgoj awlaa",
        ):
            with self.subTest(message=message):
                self.assertEqual(
                    simple_acknowledgment_response(message), ACKNOWLEDGMENT_RESPONSE
                )
        for message in (
            "тийм ээ",
            "үгүй ээ",
            "за ойлголоо, гэхдээ нэг асуулт байна",
        ):
            with self.subTest(message=message):
                self.assertIsNone(simple_acknowledgment_response(message))


if __name__ == "__main__":
    unittest.main()
