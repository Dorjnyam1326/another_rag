"""Small local responses that do not need model or retrieval calls."""

import re
import unicodedata

_GREETING_TEXT = re.compile(r"[!?.,\s]+$")
_GREETINGS = {
    "snuu",
    "sain uu",
    "sain baina uu",
    "hi",
    "hello",
    "сайн уу",
    "сайн байна уу",
}
GREETING_RESPONSE = "Сайн уу. Танд юугаар туслах вэ?"
_REACTIONS = {"hhhha", "haha", "hahaha", "хаха", "хахаха", "hehe", "хэхэ", "лол"}
REACTION_RESPONSE = "Хаха."
_ACKNOWLEDGMENTS = {
    "за ойлголоо",
    "ойлголоо",
    "ойлгож авлаа",
    "za oilgoloo",
    "oilgoloo",
    "oilgoj awlaa",
    "ok oilgoloo",
    "ок ойлголоо",
}
ACKNOWLEDGMENT_RESPONSE = "За. Хэрэгтэй бол үргэлжлүүлээд асуугаарай."


def simple_greeting_response(message: str) -> str | None:
    """Return a local reply only when the whole message is a known greeting."""
    normalized = unicodedata.normalize("NFC", " ".join(message.split())).casefold()
    normalized = _GREETING_TEXT.sub("", normalized).strip()
    return GREETING_RESPONSE if normalized in _GREETINGS else None


def simple_reaction_response(message: str) -> str | None:
    """Handle unmistakable standalone laughter without classifying it as a query."""
    normalized = unicodedata.normalize("NFC", " ".join(message.split())).casefold()
    normalized = _GREETING_TEXT.sub("", normalized).strip()
    return REACTION_RESPONSE if normalized in _REACTIONS else None


def simple_acknowledgment_response(message: str) -> str | None:
    """Return a short local close for a standalone acknowledgment."""
    normalized = unicodedata.normalize("NFC", " ".join(message.split())).casefold()
    normalized = _GREETING_TEXT.sub("", normalized).strip()
    return ACKNOWLEDGMENT_RESPONSE if normalized in _ACKNOWLEDGMENTS else None
