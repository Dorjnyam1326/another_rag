"""Fast, local precheck for request types this document assistant declines."""

import re
import unicodedata

from domain_rules import DOMAIN_RULES
from restriction_modules import RESTRICTION_MODULES, active_restriction_ids

SALARY_AMOUNT = re.compile(
    r"(?:"
    r"\b(?:salary (?:of|for|amount)|pay (?:range|band|scale)|"
    r"compensation (?:of|for|amount)|(?:how much|what (?:is|'s|does))"
    r".{0,50}(?:salary|compensation|earn|make|get paid))\b"
    r"|\b(?:salary|compensation|pay)\b.{0,35}\b(?:how much|amount|range|band)\b"
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

LATIN_NEW_YEAR_LEAVE_QUESTION = DOMAIN_RULES["new_year_leave_pattern"]
EDUCATION_BENEFIT_QUESTION = DOMAIN_RULES["education_benefit_pattern"]

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
PROGRAMMING_CONCEPT = re.compile(
    r"(?:\b(?:what is|explain|define|how does)\b.{0,70}"
    r"\b(?:programming|pointer|variable|function|class|algorithm|loop|array|"
    r"data structure|api|python|java|javascript|c\+\+|sql|code)\b"
    r"|\b(?:pointer|variable|function|class|algorithm|data structure)\b"
    r".{0,45}\b(?:in|using|with)\b.{0,25}\b(?:python|java|javascript|c\+\+|sql)\b"
    r"|\bpointer\b.{0,20}(?:what|mean|concept|юу|гэж юу)"
    r"|програмчлалын\s+(?:асуулт|ойлголт|талаар)|кодын\s+(?:ойлголт|талаар))",
    re.IGNORECASE,
)
MATH_REQUEST = re.compile(
    r"(?:\b(?:solve|calculate|evaluate|simplify|prove|derive)\b"
    r".{0,70}\b(?:equation|problem|expression|formula|integral|derivative|math)\b"
    r"|\b(?:solve|evaluate|simplify|prove|derive)\s+(?:for\s+)?[a-z0-9(]"
    r"|\b(?:what is|explain|define)\b.{0,60}\b(?:algebra|calculus|derivative|integral|matrix|theorem|equation|geometry|probability|statistics)\b"
    r"|\b(?:what is|calculate|compute)\s+\d+(?:\.\d+)?\s*(?:plus|minus|times|multiplied by|divided by)\s*\d+\b"
    r"|\b(?:what is|how (?:do i|to) calculate|calculate|find)\b.{0,50}\b(?:sum|average|mean|total|product)\b"
    r"|\b(?:sum|average|mean|product)\s+of\s+(?:\w+|[a-z0-9])\b"
    r"|(?<!\d)\d[\d,.]*(?:\s*[+*/×÷=]\s*|\s+-\s+)\s*[\da-z]+"
    r"|(?<!\d)\d[\d,.]*[a-z]\s*[+*/=]\s*[\da-z]+"
    r"|\d[\d,.]*-?ийг\s+\d[\d,.]*-?(?:т|д).{0,30}(?:хува|бод|тооцоол)"
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


_REQUEST_SEPARATOR = re.compile(
    r"(?:[?!;\n]+|(?<!\d),(?!\d)|(?<!\d)\.(?=\s|$)|\b(?:and|also|plus|bas|mun|tegeed)\b|\b(?:бас|мөн|тэгээд)\b)",
    re.IGNORECASE,
)
_OTHER_INTENT = re.compile(
    r"(?:ирц|хоцролт|хоцор\w*|амралт|чөлөө|зөвшөөрөл|сургалт|тэтгэмж|"
    r"attendance|leave|remote work|work from home|benefit|training|it request|"
    r"computer|helpdesk|document|uploaded file|report|policy|procedure|"
    r"баримт|файл|тайлан|журам|компьютер|мэдэгдэх|засуулах|бүртгүүлэх|"
    r"irts|hotsrolt|hotsor\w*|amralt\w*|chuluu|zuvshuur\w*|"
    r"surgalt\w*|tetgemj\w*|medegdeh|zasuulah|burtguuleh)",
    re.IGNORECASE,
)


def _request_clauses(text: str) -> list[str]:
    clauses = _REQUEST_SEPARATOR.split(text)
    # A plus sign is an intent separator here only when salary procedure and
    # salary amount are both present; otherwise it may be part of a math task.
    if "+" in text and SALARY_PROCESS.search(text) and SALARY_AMOUNT.search(text):
        clauses = [part for clause in clauses for part in clause.split("+")]
    return [clause.strip(" ,.") for clause in clauses if clause.strip(" ,.?")]


def _direct_restrictions(text: str, enabled: set[str]) -> list[str]:
    """Apply a restriction to one request-sized text fragment."""
    restricted = []
    asks_for_salary_amount = SALARY_AMOUNT.search(text) or (
        SALARY_PROCESS.search(text) and SALARY_CHANGE_AMOUNT.search(text)
    )
    if "salary" in enabled and asks_for_salary_amount:
        restricted.append("salary")

    source_explanation = (
        DOCUMENT_CONTEXT.search(text)
        and DOCUMENT_EXPLANATION.search(text)
        and not CODE_CREATION.search(text)
    )
    if "programming" in enabled and CODE_CREATION.search(text):
        restricted.append("programming")
    if "math" in enabled and MATH_REQUEST.search(text) and not source_explanation:
        restricted.append("math")
    return restricted


def _composite_candidate_restrictions(text: str, enabled: set[str]) -> list[str]:
    """Include router-classified programming concepts for fail-closed mixed routing."""
    candidates = _direct_restrictions(text, enabled)
    source_explanation = (
        DOCUMENT_CONTEXT.search(text)
        and DOCUMENT_EXPLANATION.search(text)
        and not CODE_CREATION.search(text)
    )
    if (
        "programming" in enabled
        and PROGRAMMING_CONCEPT.search(text)
        and not source_explanation
        and "programming" not in candidates
    ):
        candidates.append("programming")
    return candidates


def request_restriction_candidates(query: str, enabled_modules=None) -> list[str]:
    """Conservatively classify a routed request before allowing downstream use."""
    text = unicodedata.normalize("NFC", " ".join(query.split()))
    enabled = set(active_restriction_ids(enabled_modules))
    return _composite_candidate_restrictions(text, enabled)


def _is_composite_restriction(text: str, module_id: str, enabled: set[str]) -> bool:
    clauses = _request_clauses(text)
    if len(clauses) < 2:
        return False
    restricted_clauses = [
        (clause, _composite_candidate_restrictions(clause, enabled))
        for clause in clauses
    ]
    module_positions = {
        index
        for index, (_, modules) in enumerate(restricted_clauses)
        if module_id in modules
    }
    has_other_request = any(
        index not in module_positions
        and (
            _OTHER_INTENT.search(clause)
            or SALARY_SCHEDULE.search(clause)
            or (module_id == "salary" and SALARY_PROCESS.search(clause))
            or any(other_id != module_id for other_id in modules)
        )
        for index, (clause, modules) in enumerate(restricted_clauses)
    )
    has_module_request = bool(module_positions)
    return has_module_request and has_other_request


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
    has_latin_new_year = bool(
        LATIN_NEW_YEAR_LEAVE_QUESTION
        and LATIN_NEW_YEAR_LEAVE_QUESTION.search(text)
    )
    is_leave_question = (
        has_cyrillic_new_year and has_cyrillic_leave
    ) or has_latin_new_year
    return bool(is_leave_question and not SALARY_AMOUNT.search(text))


def is_education_benefit_question(query: str) -> bool:
    """Keep education-cost benefits out of the salary amount restriction."""
    text = unicodedata.normalize("NFC", " ".join(query.split())).casefold()
    return bool(
        EDUCATION_BENEFIT_QUESTION
        and EDUCATION_BENEFIT_QUESTION.search(text)
        and not SALARY_AMOUNT.search(text)
    )


def precheck_query(
    query: str,
    enabled_modules=None,
) -> tuple[str, str] | None:
    """Return (status, refusal) for restricted requests, otherwise None."""
    text = unicodedata.normalize("NFC", " ".join(query.split()))
    enabled = set(active_restriction_ids(enabled_modules))
    for module_id in _direct_restrictions(text, enabled):
        if not _is_composite_restriction(text, module_id, enabled):
            return module_id, REFUSALS[module_id]
    return None


def composite_restrictions(query: str, enabled_modules=None) -> list[str]:
    """Find restricted subrequests in a mixed question without blocking its other intents."""
    text = unicodedata.normalize("NFC", " ".join(query.split()))
    enabled = set(active_restriction_ids(enabled_modules))
    clauses = _request_clauses(text)
    if len(clauses) < 2:
        return []
    present = list(dict.fromkeys(
        module_id
        for clause in clauses
        for module_id in _composite_candidate_restrictions(clause, enabled)
    ))
    return list(dict.fromkeys(
        module_id
        for module_id in present
        if _is_composite_restriction(text, module_id, enabled)
    ))
