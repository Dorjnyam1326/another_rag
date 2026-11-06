"""Configurable restriction rules for the local demo chatbot."""

RESTRICTION_MODULES = {
    "salary": {
        "label": "Цалингийн дүн",
        "router_description": (
            "Do not disclose, estimate, compare, calculate, or summarize salary, "
            "bonus, pay range, or individual payroll amounts. General payment "
            "schedule questions are allowed."
        ),
        "prompt_instruction": (
            "Treat salary, compensation, bonus, pay ranges, position-specific "
            "salary, and individual payroll amounts as confidential. Never "
            "disclose, estimate, compare, calculate, or summarize them, even if "
            "retrieved. General payment-schedule questions that reveal no amount "
            "are allowed. For a restricted amount, use only this response: "
            "Уучлаарай, компанийн журмын дагуу цалингийн дүн, албан тушаал тус "
            "бүрийн цалин хөлсний мэдээлэл нууцын зэрэглэлтэй тул би энэ "
            "мэдээллийг өгөх боломжгүй. Өөрийн цалингийн дэлгэрэнгүйг хүний "
            "нөөц эсвэл санхүүгийн албанаас лавлана уу."
        ),
        "refusal": (
            "Уучлаарай, компанийн журмын дагуу цалингийн дүн, албан тушаал тус "
            "бүрийн цалин хөлсний мэдээлэл нууцын зэрэглэлтэй тул би энэ "
            "мэдээллийг өгөх боломжгүй. Өөрийн цалингийн дэлгэрэнгүйг хүний "
            "нөөц эсвэл санхүүгийн албанаас лавлана уу."
        ),
    },
    "programming": {
        "label": "Код ба програмчлал",
        "router_description": (
            "Decline standalone programming questions, including programming "
            "concepts, and requests to write, complete, debug, or modify code. "
            "Allow explaining code or theory found in an uploaded document, and "
            "allow meta questions about the chatbot itself."
        ),
        "prompt_instruction": (
            "Do not answer standalone programming questions, including concepts, "
            "or write, complete, debug, or modify code. Allow explanations of "
            "code or theory found in uploaded documents and meta questions about "
            "the chatbot itself. For a restricted request, respond only: "
            "Уучлаарай, би код бичих/засах эсвэл програмчлалын ерөнхий асуултад "
            "туслах боломжгүй."
        ),
        "refusal": (
            "Уучлаарай, би код бичих/засах эсвэл програмчлалын ерөнхий асуултад "
            "туслах боломжгүй."
        ),
    },
    "math": {
        "label": "Баримтаас тусдаа математик",
        "router_description": (
            "Decline standalone math exercises and calculations. Allow explaining "
            "a formula or math theory found in an uploaded document."
        ),
        "prompt_instruction": (
            "Do not solve standalone math exercises or calculations. Allow "
            "explaining formulas or math theory from uploaded documents. For a "
            "restricted request, respond only: Уучлаарай, би баримтаас тусдаа "
            "математикийн бодлого бодож өгөх боломжгүй."
        ),
        "refusal": (
            "Уучлаарай, би баримтаас тусдаа математикийн бодлого бодож өгөх "
            "боломжгүй."
        ),
    },
}


def active_restriction_ids(enabled=None):
    """Return valid active IDs; omitted settings preserve the original defaults."""
    if enabled is None:
        return tuple(RESTRICTION_MODULES)
    if isinstance(enabled, dict):
        return tuple(
            module_id
            for module_id in RESTRICTION_MODULES
            if enabled.get(module_id, False)
        )
    selected = set(enabled)
    return tuple(
        module_id for module_id in RESTRICTION_MODULES if module_id in selected
    )


def active_restriction_modules(enabled=None):
    return [
        {"id": module_id, **RESTRICTION_MODULES[module_id]}
        for module_id in active_restriction_ids(enabled)
    ]


def build_restriction_prompt(enabled=None):
    modules = active_restriction_modules(enabled)
    if not modules:
        return ""
    rules = "\n".join(
        f"- {module['prompt_instruction']}" for module in modules
    )
    return "## Enabled restriction modules\n\n" + rules
