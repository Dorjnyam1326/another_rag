"""Deterministic checks for unsupported facts and process drift in search queries."""

import re
import json
import os
import tempfile
import unicodedata
from pathlib import Path

from domain_rules import DOMAIN_RULES
from query_utils import (
    DURATION as _DURATION,
    TIME_RANGE as _TIME_RANGE,
    duration_value as _duration_value,
    duration_values,
    query_anchors_supported,
    time_ranges,
)

_PROCESS_GROUPS = DOMAIN_RULES["process_groups"]
_ROLE_GROUPS = DOMAIN_RULES["role_groups"]
_NEGATION_PATTERNS = (
    re.compile(r"\b(?:болохгүй|хоригло\w*|эсэхгүй|заавал|зөвхөн)\b", re.I),
    re.compile(r"\b(?:not|no|never|cannot|can't|must not|only)\b", re.I),
)
_CORRECTION_CONNECTOR = re.compile(
    r"\b(?:биш|үгүй|угүй|bish|ugui|not|instead|rather|харин)\b", re.I
)
QUERY_CONFIGURATION_PATH = Path(__file__).resolve().parent / "query_strategy.json"


def active_query_configuration(path=QUERY_CONFIGURATION_PATH):
    """Return the promoted strategy, defaulting safely to the existing A path."""
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Invalid query strategy configuration")
        strategy = str(config.get("strategy", "A")).upper()
        enabled = config.get("enabled") is True
        if strategy not in {"A", "B", "C"}:
            raise ValueError("Invalid query strategy")
        return {"enabled": enabled, "strategy": strategy}
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {"enabled": False, "strategy": "A"}


def promote_query_strategy(strategy, path=QUERY_CONFIGURATION_PATH):
    """Atomically persist a Stage 7 strategy after every promotion gate passes."""
    strategy = str(strategy or "").upper()
    if strategy not in {"A", "B", "C"}:
        raise ValueError("Unknown query strategy")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"enabled": True, "strategy": strategy, "stage": 7},
        ensure_ascii=False,
        indent=2,
    ) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def has_explicit_duration_correction(text):
    """Whether a correction connector separates two different durations."""
    durations = list(_DURATION.finditer(str(text or "").casefold()))
    for earlier, later in zip(durations, durations[1:]):
        if (
            _duration_value(earlier) != _duration_value(later)
            and _CORRECTION_CONNECTOR.search(text[earlier.end():later.start()])
        ):
            return True
    return False


def accepted_correction_duration(text):
    if not has_explicit_duration_correction(text):
        return None
    source = str(text or "").casefold()
    connectors = list(_CORRECTION_CONNECTOR.finditer(source))
    durations = list(_DURATION.finditer(source))
    if not connectors:
        return None
    connector = connectors[-1]
    return next(
        (_duration_value(item) for item in durations if item.start() >= connector.end()),
        None,
    )


def canonicalize_duration_correction(query, source_text):
    """Return a corrected Cyrillic query only when it retains the accepted value."""
    accepted = accepted_correction_duration(source_text)
    if accepted is None:
        return None
    candidate = remove_rejected_duration(query, source_text)
    return candidate if duration_values(candidate) == {accepted} else None


def remove_rejected_duration(query, source_text):
    """Keep only the corrected duration in a Router query for an explicit correction."""
    if not has_explicit_duration_correction(source_text):
        return " ".join(str(query or "").split())

    accepted = accepted_correction_duration(source_text)
    if accepted is None:
        return " ".join(str(query or "").split())

    value = str(query or "")
    matches = list(_DURATION.finditer(value.casefold()))
    for match in reversed(matches):
        if _duration_value(match) != accepted:
            value = value[:match.start()] + value[match.end():]
    value = _CORRECTION_CONNECTOR.sub(" ", value)
    value = re.sub(r"\s*[,;]\s*", " ", value)
    return " ".join(value.split()).strip(" ,;.")


def has_explicit_time_range(text):
    return bool(_TIME_RANGE.search(str(text or "")))


def normalize_current_time_range(query, source_text, context_text=""):
    """Keep the latest user-supplied range, normalizing only explicit daytime ranges."""
    source = str(source_text or "")
    source_match = _TIME_RANGE.search(source)
    if not source_match:
        return " ".join(str(query or "").split())

    context = str(context_text or "")
    inherit_daypart = bool(
        _CORRECTION_CONNECTOR.search(source)
        and re.search(r"өдрийн|өдөр", context, re.I)
    )
    range_source = f"{context}\n{source}" if inherit_daypart else source
    ranges = time_ranges(range_source)
    if not ranges:
        return " ".join(str(query or "").split())
    start, end = ranges[-1]
    daypart_is_explicit = bool(
        re.search(r"өдрийн|өдөр", source, re.I) or inherit_daypart
    )
    if daypart_is_explicit:
        start_text = f"{start // 60:02d}:{start % 60:02d}"
        end_text = f"{end // 60:02d}:{end % 60:02d}"
        canonical_range = f"{start_text}–{end_text}"
    else:
        canonical_range = source_match.group(0)

    candidate = str(query or "").strip()
    query_match = _TIME_RANGE.search(candidate)
    if query_match:
        candidate = (
            candidate[:query_match.start()]
            + canonical_range
            + candidate[query_match.end():]
        )
    else:
        candidate = f"{candidate} {canonical_range}".strip()

    duration = (end - start) / 60
    if daypart_is_explicit and duration > 0 and not duration_values(candidate):
        duration_text = str(int(duration)) if duration.is_integer() else str(duration)
        candidate = f"{candidate} {duration_text} цаг"
    return " ".join(candidate.split())


def current_time_range_supported(query, source_text, context_text=""):
    """Require a rewrite to preserve the latest explicit range and its duration."""
    source = str(source_text or "")
    context = str(context_text or "")
    inherit_daypart = bool(
        _CORRECTION_CONNECTOR.search(source)
        and re.search(r"өдрийн|өдөр", context, re.I)
    )
    range_source = f"{context}\n{source}" if inherit_daypart else source
    source_ranges = time_ranges(range_source)
    query_ranges = time_ranges(str(query or ""))
    if not source_ranges or source_ranges[-1] not in query_ranges:
        return False
    start, end = source_ranges[-1]
    if end < start:
        return False
    allowed_durations = {round(end - start, 2)}
    allowed_durations.update(duration_values(source))
    return duration_values(query).issubset(allowed_durations)


def process_groups(text):
    lowered = text.casefold()
    return {
        group
        for group, markers in _PROCESS_GROUPS.items()
        if any(_matches_marker(lowered, marker) for marker in markers)
    }


_LATIN_STEMS = {"chuluu", "amralt", "tsalin", "hotsor", "zuvshuur", "surgalt", "tetgemj"}


def _matches_marker(text, marker):
    if not marker.isascii():
        return marker in text
    suffix = r"[a-z]*" if marker in _LATIN_STEMS else ""
    return bool(re.search(
        rf"(?<![a-z0-9]){re.escape(marker)}{suffix}(?![a-z0-9])",
        text,
    ))


def _script_family(text):
    cyrillic = any(
        "CYRILLIC" in unicodedata.name(char, "")
        for char in text
        if char.isalpha()
    )
    latin = any(
        "LATIN" in unicodedata.name(char, "")
        for char in text
        if char.isalpha()
    )
    if cyrillic and not latin:
        return "cyrillic"
    if latin and not cyrillic:
        return "latin"
    return "mixed"


def _process_drift_status(source, candidate):
    source_groups = process_groups(source)
    candidate_groups = process_groups(candidate)
    if not source_groups or not candidate_groups:
        return "undetermined"
    if candidate_groups.issubset(source_groups):
        return "no_drift"
    if source_groups.isdisjoint(candidate_groups):
        return "confirmed_drift"
    return "undetermined"


def _marker_groups(text, groups):
    lowered = text.casefold()
    return {
        group
        for group, markers in groups.items()
        if any(marker in lowered for marker in markers)
    }


def query_issues(source_text, query):
    """Return stable issue codes when a rewrite adds facts or changes process."""
    source = " ".join(str(source_text or "").split())
    candidate = " ".join(str(query or "").split())
    if not candidate:
        return ["empty_query"]

    issues = []
    if not query_anchors_supported(candidate, source, []):
        issues.append("unsupported_numeric_or_date_anchor")

    source_durations = duration_values(source)
    candidate_durations = duration_values(candidate)
    if not candidate_durations.issubset(source_durations):
        # Explicit daytime ranges may deterministically imply a duration.
        source_ranges = time_ranges(source)
        permitted = set(source_durations)
        permitted.update(round(end - start, 2) for start, end in source_ranges if end >= start)
        if not candidate_durations.issubset(permitted):
            issues.append("unsupported_duration")

    if _script_family(source) == _script_family(candidate):
        for pattern in _NEGATION_PATTERNS:
            if bool(pattern.search(source)) != bool(pattern.search(candidate)):
                issues.append("negation_or_obligation_drift")
                break

    if _process_drift_status(source, candidate) == "confirmed_drift":
        issues.append("business_process_drift")
    if _marker_groups(candidate, _ROLE_GROUPS) - _marker_groups(source, _ROLE_GROUPS):
        issues.append("unsupported_role")
    return issues


def filter_search_queries(source_text, request, *, fallback_query=""):
    """Guard a request in place; preserve source wording if canonical text drifts."""
    canonical = str(request.get("canonical_query", ""))
    issues = query_issues(source_text, canonical)
    rejections = []
    if issues:
        rejections.append({"kind": "canonical", "issues": issues})
        request["canonical_query"] = str(fallback_query or "").strip()

    accepted = []
    for alternative in request.get("alternatives", []):
        issues = query_issues(source_text, alternative)
        if issues:
            rejections.append({"kind": "alternative", "issues": issues})
        elif alternative not in accepted:
            accepted.append(alternative)
    request["alternatives"] = accepted[:2]
    if rejections:
        request["query_guard_rejections"] = rejections
    return request
