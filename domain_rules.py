"""Load optional domain-specific query markers without hardwiring HR rules."""

import json
import re
from pathlib import Path

DOMAIN_RULES_PATH = Path(__file__).resolve().parent / "config" / "domain_rules.json"


def load_domain_rules(path=DOMAIN_RULES_PATH):
    """Load the configured domain rules; a missing or empty file is generic."""
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        config = {}
    if not isinstance(config, dict):
        config = {}
    return {
        "process_groups": config.get("process_groups", {}),
        "role_groups": config.get("role_groups", {}),
        "scopes": config.get("scopes", []),
        "answer_instructions": config.get("answer_instructions", []),
        "new_year_leave_pattern": _compile(config.get("new_year_leave_pattern")),
        "education_benefit_pattern": _compile(config.get("education_benefit_pattern")),
    }


def _compile(pattern):
    if not isinstance(pattern, str) or not pattern:
        return None
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return None


DOMAIN_RULES = load_domain_rules()
