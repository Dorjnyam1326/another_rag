"""Shared, script-neutral helpers for query anchors and time ranges."""

import re


TIME_RANGE = re.compile(
    r"(?P<hour1>\d{1,2})(?::(?P<minute1>\d{2}))?\s*"
    r"(?:–|—|-аас|-оос|-өөс|аас|ээс|\bto\b|-)\s*"
    r"(?P<hour2>\d{1,2})(?::(?P<minute2>\d{2}))?",
    re.IGNORECASE,
)
EXPLICIT_CLOCK = re.compile(r"(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d(?!\d)")
DURATION = re.compile(
    r"(?<![\d:])(?P<number>\d+(?:[.,]\d+)?|арван тав|\w+)\s*"
    r"(?P<unit>өдрийн|өдөрт|өдөр|хоног(?:ийн)?|day(?:s)?|udur|udriin|"
    r"цаг(?:ийн|т|аар)?|tsag(?:iin)?|hour(?:s)?|"
    r"минут(?:ын|анд|аар)?|minute(?:s)?|min(?:ute)?s?)\b",
    re.IGNORECASE,
)
DATE_WORDS = re.compile(
    r"\b(?:өнөөдөр|маргааш|өчигдөр|энэ долоо хоногт|дараа долоо хоногт)\b",
    re.IGNORECASE,
)

WORD_NUMBERS = {
    "нэг": 1, "хоёр": 2, "гурван": 3, "гурав": 3, "дөрвөн": 4,
    "дөрөв": 4, "таван": 5, "тав": 5, "зургаан": 6, "зургаа": 6,
    "долоон": 7, "долоо": 7, "найман": 8, "найм": 8, "есөн": 9,
    "ес": 9, "арван": 10, "арван тав": 15, "хорин": 20, "гучин": 30,
    "дөчин": 40, "тавин": 50, "жаран": 60, "наян": 80, "зуун": 100,
}


def time_ranges(value):
    text = " ".join(str(value or "").split()).casefold()
    day_time = bool(re.search(r"өдрийн|өдөр", text))
    ranges = []
    range_matches = list(TIME_RANGE.finditer(text))
    for match in range_matches:
        start_hour = int(match.group("hour1"))
        end_hour = int(match.group("hour2"))
        if day_time:
            start_hour = start_hour + 12 if start_hour < 12 else start_hour
            end_hour = end_hour + 12 if end_hour < 12 else end_hour
        ranges.append((
            start_hour * 60 + int(match.group("minute1") or 0),
            end_hour * 60 + int(match.group("minute2") or 0),
        ))
    for match in EXPLICIT_CLOCK.finditer(text):
        if any(item.start() <= match.start() < item.end() for item in range_matches):
            continue
        minutes = int(match.group(0)[:2]) * 60 + int(match.group(0)[3:])
        ranges.append((minutes, minutes))
    return ranges


def duration_value(match):
    raw = match.group("number").replace(",", ".").casefold()
    try:
        amount = float(raw)
    except ValueError:
        amount = WORD_NUMBERS.get(raw)
    if amount is None:
        return None
    unit = match.group("unit").casefold()
    multiplier = 1440 if unit.startswith(("өдөр", "өдр", "хоног", "day", "udur", "udriin")) else (
        1 if unit.startswith(("минут", "minute", "min")) else 60
    )
    return round(amount * multiplier, 2)


def duration_values(value):
    return {
        amount for match in DURATION.finditer(str(value or ""))
        if (amount := duration_value(match)) is not None
    }


def numeric_values(value):
    text = TIME_RANGE.sub(" ", " ".join(str(value or "").split()))
    text = EXPLICIT_CLOCK.sub(" ", text)
    text = DURATION.sub(" ", text)
    return set(re.findall(r"(?<![\w])\d+(?![\w])", text))


def query_anchors_supported(query, current_question, verified_facts):
    supported_text = " ".join([current_question, *verified_facts])
    query_ranges = set(time_ranges(query))
    supported_ranges = set(time_ranges(supported_text))
    if not query_ranges.issubset(supported_ranges):
        return False
    if not numeric_values(query).issubset(numeric_values(supported_text)):
        return False
    supported_durations = duration_values(supported_text)
    supported_durations.update(
        round(end - start, 2)
        for start, end in supported_ranges
        if end >= start
    )
    if not duration_values(query).issubset(supported_durations):
        return False
    query_dates = {item.casefold() for item in DATE_WORDS.findall(query)}
    supported_dates = {item.casefold() for item in DATE_WORDS.findall(supported_text)}
    return query_dates.issubset(supported_dates)
