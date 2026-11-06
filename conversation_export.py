"""Serialize the current chat transcript without internal retrieval context."""

import json


_TRANSCRIPT_FIELDS = ("role", "content", "sources", "usage", "interaction_type")


def conversation_json(messages: list[dict]) -> str:
    transcript = [
        {field: message[field] for field in _TRANSCRIPT_FIELDS if field in message}
        for message in messages
        if message.get("role") in {"user", "assistant"}
    ]
    return json.dumps({"messages": transcript}, ensure_ascii=False, indent=2)
