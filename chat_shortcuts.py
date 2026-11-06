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


def simple_greeting_response(message: str) -> str | None:
    """Return a local reply only when the whole message is a known greeting."""
    normalized = unicodedata.normalize("NFC", " ".join(message.split())).casefold()
    normalized = _GREETING_TEXT.sub("", normalized).strip()
    return GREETING_RESPONSE if normalized in _GREETINGS else None
