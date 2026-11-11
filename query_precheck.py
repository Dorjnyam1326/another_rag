"""Fast, local precheck for request types this document assistant declines."""

import re
import unicodedata

from restriction_modules import RESTRICTION_MODULES, active_restriction_ids

SALARY_AMOUNT = re.compile(
    r"(?:"
    r"\b(?:salary (?:of|for|amount)|pay (?:range|band|scale)|"
    r"compensation (?:of|for|amount)|(?:how much|what (?:is|'s|does))"
    r".{0,50}(?:salary|compensation|earn|make|get paid))\b"
    r"|\b(?:salary|compensation|pay)\b.{0,35}\b(?:how much|amount|range|band|of|for)\b"
    r"|\b(?:цалин(?:гийн|гаа|ы)?|цалин)\s*.{0,30}(?:хэд\b|дүн\b|хэмжээ\b|шатлал\b|түвшин\b|төгрөг\w*)"
    r"|\b(?:how much|salary|pay)\b.{0,35}\b(?:engineer|manager|position|employee|role)\b"
    r"|\b(?:tsalin(?:giin|gaa)?|tsalinii)\b.{0,35}\b(?:hed\b|hemjee\b|dun\b|tugrug\w*)"
    r")",
    re.IGNORECASE,
)
SALARY_SCHEDULE = re.compile(
    r"(?:"
    r"\b(?:payday|pay day)\b"
    r"|\bwhen\b.{0,35}\b(?:salary|pay|wage|payroll)\b"
    r"|\b(?:salary|pay|wage|payroll)\b.{0,35}\b(?:when|paid|deposit(?:ed)?)\b"
    r"|(?:цалин\w*|цалинг\w*).{0,50}(?:хэзээ|хэдэнд).{0,40}(?:буух|бууд|олгогд|авах|орох)"
    r"|(?:tsalin\w*|tsalinii).{0,50}(?:hezee|hedend).{0,40}(?:buuh|buud|olgo|avah|oroh)"
    r"|(?:цалин\w*|цалинг\w*).{0,70}(?:амралтын өдөр|баярын өдөр)"
    r".{0,40}(?:таар|болбол|тохиолд|буувал|олговол)"
    r"|(?:tsalin\w*|tsalinii).{0,70}(?:amraltiin udur|bayriin udur)"
    r".{0,40}(?:taar|bolbol|tohiold|buubal|olgovol)"
    r"|\bwhat happens if\b.{0,35}\b(?:payday|pay date|salary date)\b"
    r".{0,40}\b(?:holiday|weekend)\b"
    r"|\b(?:payday|pay date|salary date)\b.{0,40}\b(?:holiday|weekend)\b"
    r")",
    re.IGNORECASE,
)
SALARY_PROCESS = re.compile(
    r"(?:"
    r"\b(?:how|when)\b.{0,50}\b(?:request|ask for|get|apply for)\b"
    r".{0,40}\b(?:salary|pay|compensation)\b.{0,40}\b(?:increase|raise|review)\b"
    r"|\b(?:salary|pay|compensation)\b.{0,40}\b(?:increase|raise|review)\b"
    r".{0,50}\b(?:request|process|procedure|how|when)\b"
    r"|(?:цалин\w*|цалинг\w*).{0,60}(?:нэмүүл|нэмэгдүүл|нэмэх|нэмэгд)"
    r".{0,60}(?:яаж|хэрхэн|хэзээ|журам|хүсэлт|процесс)?"
    r"|(?:цалин\w*|цалинг\w*).{0,60}(?:яаж|хэрхэн|хэзээ)"
    r".{0,60}(?:нэмүүл|нэмэгдүүл|нэмэх|нэмэгд)"
    r"|(?:tsalin\w*|tsalinii).{0,60}(?:nemuul|nemegde|nemeh|increase|raise)"
    r".{0,70}(?:yaj|yaaj|hezee|huselt|jural|process)?"
    r"|(?:tsalin\w*|tsalinii).{0,60}(?:yaj|yaaj|hezee)"
    r".{0,60}(?:nemuul|nemegde|nemeh)"
    r")",
    re.IGNORECASE,
)
SALARY_CHANGE_AMOUNT = re.compile(
    r"(?:\d[\d,.]*\s*(?:₮|төгрөг\w*|мянга\w*|сая\w*|%|хувиар|хувь\w*|percent\b)"
    r"|\b(?:хэдээр|хэдэн хувь|by how much|what percentage)\b)",
    re.IGNORECASE,
)

LATIN_NEW_YEAR_LEAVE_QUESTION = re.compile(
    r"(?:ene\s*shine\s*jil.{0,40}(?:amrah|amral|chuluu|ajillah)"
    r"|(?:amrah|amral|chuluu|ajillah).{0,40}ene\s*shine\s*jil)",
    re.IGNORECASE,
)

EDUCATION_BENEFIT_QUESTION = re.compile(
    r"(?:\b(?:tuition|training|certification|education)\b.{0,80}"
    r"\b(?:fees?|cost|reimburse\w*|benefit|support|company pay\w*)\b"
    r"|(?:сургалт\w*|сертификат\w*|хүүхэд\w*).{0,80}"
    r"(?:төлбөр|зардал|нөхөн төлбөр|дэмжлэг|даах|буцаан олголт)"
    r"|(?:surgalt\w*|sertifikat\w*|huuhed\w*).{0,80}"
    r"(?:tulbur|zardal|demjleg|daah|nuHun|nohon|butsa\w*))",
    re.IGNORECASE,
)

DOCUMENT_CONTEXT = re.compile(
    r"(?:\b(?:document|uploaded file|file|article|paper|report|source)\b"
    r"|баримт|оруулсан файл|өгүүлэл|тайлан|эх сурвалж)",
    re.IGNORECASE,
)
DOCUMENT_EXPLANATION = re.compile(
    r"(?:\b(?:explain|describe|summari[sz]e|what does .{0,60} mean|"
    r"what does .{0,60} do)\b|тайлбарла|утгыг нь хэл|юу гэсэн|юу хийдэг)",
    re.IGNORECASE,
)
CODE_CREATION = re.compile(
    r"(?:\b(?:write|generate|create|build|implement|fix|debug|complete)\b"
    r".{0,60}\b(?:code|program|script|function|class)\b"
    r"|\b(?:code|program|script)\b.{0,60}\b(?:write|generate|create|build|fix|debug)\b"
    r"|код\s*(?:бич|зохио|гарга|зас|дуусга)|программ\s*(?:бич|зохио|зас)|"
    r"алдааг\s*(?:зас|олж))",
    re.IGNORECASE,
)
MATH_REQUEST = re.compile(
    r"(?:\b(?:solve|calculate|evaluate|simplify|prove|derive)\b"
    r".{0,70}\b(?:equation|problem|expression|formula|integral|derivative|math)\b"
    r"|\b(?:solve|calculate|evaluate|simplify|prove|derive)\s+(?:for\s+)?[a-z0-9(]"
    r"|\b(?:what is|explain|define)\b.{0,60}\b(?:algebra|calculus|derivative|integral|matrix|theorem|equation|geometry|probability|statistics)\b"
    r"|\b(?:what is|how (?:do i|to) calculate|calculate|find)\b.{0,50}\b(?:sum|average|mean|total|product)\b"
    r"|\b(?:sum|average|mean|product)\s+of\s+(?:\w+|[a-z0-9])\b"
    r"|математикийн\s+бодлого|тэгшитгэл\s+(?:бод|шийд)|бодлогыг\s+бод|"
    r"бодож\s+өг|тооцоолж\s+өг|"
    r"(?:математик|алгебр|геометр|интеграл|дериватив|теорем|магадлал|статистик)"
    r".{0,30}(?:тайлбарла|гэж юу|бод|шийд))",
    re.IGNORECASE,
)

REFUSALS = {
    module_id: module["refusal"]
    for module_id, module in RESTRICTION_MODULES.items()
}


def _is_composite_policy_request(text: str) -> bool:
    """Let the semantic router split mixed questions before applying a restriction."""
    has_connector = bool(
        re.search(r"(?:\b(?:and|also|plus)\b|бас|мөн|тэгээд|\?)", text, re.IGNORECASE)
    )
    has_other_topic = bool(
        re.search(
            r"(?:ирц|хоцролт|хоцор\w*|амралт|чөлөө|зөвшөөрөл|сургалт|тэтгэмж|attendance|leave|"
            r"remote work|work from home|benefit|training|it request)",
            text,
            re.IGNORECASE,
        )
    )
    return has_connector and has_other_topic


def is_salary_schedule_question(query: str) -> bool:
    """Identify pay-date questions, excluding any that also ask for an amount."""
    text = unicodedata.normalize("NFC", " ".join(query.split()))
    return bool(SALARY_SCHEDULE.search(text) and not SALARY_AMOUNT.search(text))


def is_salary_process_question(query: str) -> bool:
    """Allow salary-review procedure questions that do not request an amount."""
    text = unicodedata.normalize("NFC", " ".join(query.split()))
    return bool(
        SALARY_PROCESS.search(text)
        and not SALARY_AMOUNT.search(text)
        and not SALARY_CHANGE_AMOUNT.search(text)
    )


def is_leave_policy_question(query: str) -> bool:
    """Identify explicit New Year leave questions to avoid contextual salary leakage."""
    text = unicodedata.normalize("NFC", " ".join(query.split())).casefold()
    words = text.split()
    has_cyrillic_new_year = "шинэ" in words and any(
        word.startswith("жил") for word in words
    )
    has_cyrillic_leave = any(
        word.startswith(("амр", "амра", "чөлөө", "ажиллах")) for word in words
    )
    has_latin_new_year = bool(LATIN_NEW_YEAR_LEAVE_QUESTION.search(text))
    is_leave_question = (
        has_cyrillic_new_year and has_cyrillic_leave
    ) or has_latin_new_year
    return bool(is_leave_question and not SALARY_AMOUNT.search(text))


def is_education_benefit_question(query: str) -> bool:
    """Keep education-cost benefits out of the salary amount restriction."""
    text = unicodedata.normalize("NFC", " ".join(query.split())).casefold()
    return bool(
        EDUCATION_BENEFIT_QUESTION.search(text)
        and not SALARY_AMOUNT.search(text)
    )


def precheck_query(
    query: str,
    enabled_modules=None,
) -> tuple[str, str] | None:
    """Return (status, refusal) for restricted requests, otherwise None."""
    text = unicodedata.normalize("NFC", " ".join(query.split()))
    enabled = set(active_restriction_ids(enabled_modules))
    asks_for_salary_amount = SALARY_AMOUNT.search(text) or (
        SALARY_PROCESS.search(text) and SALARY_CHANGE_AMOUNT.search(text)
    )
    if (
        "salary" in enabled
        and asks_for_salary_amount
        and not _is_composite_policy_request(text)
    ):
        return "salary", REFUSALS["salary"]

    # Permit explanation of material explicitly located in an uploaded source,
    # but never permit requests to create, complete, or debug code.
    source_explanation = (
        DOCUMENT_CONTEXT.search(text)
        and DOCUMENT_EXPLANATION.search(text)
        and not CODE_CREATION.search(text)
    )
    if "programming" in enabled and CODE_CREATION.search(text):
        return "programming", REFUSALS["programming"]
    if "math" in enabled and MATH_REQUEST.search(text) and not source_explanation:
        return "math", REFUSALS["math"]
    return None


def composite_restrictions(query: str, enabled_modules=None) -> list[str]:
    """Find restricted subrequests in a mixed question without blocking its other intents."""
    text = unicodedata.normalize("NFC", " ".join(query.split()))
    enabled = set(active_restriction_ids(enabled_modules))
    matched = []
    asks_for_salary_amount = SALARY_AMOUNT.search(text) or (
        SALARY_PROCESS.search(text) and SALARY_CHANGE_AMOUNT.search(text)
    )
    if (
        "salary" in enabled
        and asks_for_salary_amount
        and _is_composite_policy_request(text)
    ):
        matched.append("salary")
    return matched
