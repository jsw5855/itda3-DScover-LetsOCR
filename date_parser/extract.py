from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

DIGIT = r"[0-9OoUu]"

MONTH_NAMES = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
_MONTH_RE = "|".join(MONTH_NAMES)

CONFUSABLE_MAP = str.maketrans({"O": "0", "o": "0", "U": "0", "u": "0"})


def normalize_confusable(raw: str) -> str:
    """Digits that OCR commonly confuses with letters (O/o/U/u -> 0)."""
    return raw.translate(CONFUSABLE_MAP)


@dataclass(frozen=True)
class RawField:
    raw: str
    kind: str  # 'num' or 'month_name'


@dataclass(frozen=True)
class RawDateToken:
    span: Tuple[int, int]
    fields: Tuple[RawField, ...]
    role_universe: Tuple[str, ...]
    fixed_roles: Optional[Tuple[str, ...]] = None  # None => ambiguous order


_NUM = RawField
# Real OCR output sometimes has more than one separator character in a row
# (e.g. "2021. 03.20" has a period AND a space between year and month), so
# this allows one or more, not just exactly one.
_SEP = r"[.\-/\s·×]+"
_OPTIONAL_SEP = r"[.\-/\s·×]*"

# (regex, field kinds per group, role_universe, fixed_roles)
# fixed_roles is only used where the source text itself names the unit
# (년/월/일), so the order is read directly off the text rather than assumed.
_PATTERN_DEFS = [
    (
        re.compile(rf"(?<!\d)(\d{{4}})년\s*(\d{{1,2}})월\s*(\d{{1,2}})일"),
        ("num", "num", "num"),
        ("year", "month", "day"),
        ("year", "month", "day"),
    ),
    (
        re.compile(rf"(?<!\d)(\d{{4}})년\s*(\d{{1,2}})월(?!\s*\d{{1,2}}일)"),
        ("num", "num"),
        ("year", "month"),
        ("year", "month"),
    ),
    (
        re.compile(rf"(?<!\d)(\d{{1,2}})월\s*(\d{{1,2}})일"),
        ("num", "num"),
        ("month", "day"),
        ("month", "day"),
    ),
    (
        re.compile(rf"(?<!\d)(\d{{1,2}}){_SEP}({_MONTH_RE}){_SEP}(\d{{2,4}})(?!\d)", re.IGNORECASE),
        ("num", "month_name", "num"),
        ("year", "month", "day"),
        None,
    ),
    (
        # "04NOV 2021" style: day glued directly to the month name with no
        # separator at all (OCR dropped the space), but a real separator
        # before the year. Safe to allow zero separator here for the same
        # reason as the "AUG292020" pattern above - the month name is a
        # strong, small anchor that can't accidentally swallow an unrelated
        # digit run.
        re.compile(rf"(?<!\d)(\d{{1,2}}){_OPTIONAL_SEP}({_MONTH_RE}){_SEP}(\d{{2,4}})(?!\d)", re.IGNORECASE),
        ("num", "month_name", "num"),
        ("year", "month", "day"),
        None,
    ),
    (
        # "07FEB2022" / "31JUL21": day, month name and year printed with no
        # separator at all (common on US/EU cans and pouches). The month name
        # fixes which run of digits is which; order between day and year is
        # resolved by validation and the month-name prior (day first).
        re.compile(rf"(?<!\d)({DIGIT}{{1,2}})({_MONTH_RE})({DIGIT}{{2}}|{DIGIT}{{4}})(?!\d)", re.IGNORECASE),
        ("num", "month_name", "num"),
        ("year", "month", "day"),
        None,
    ),
    (
        re.compile(rf"(?<!\d)(\d{{2,4}}){_SEP}({_MONTH_RE}){_SEP}(\d{{1,2}})(?!\d)", re.IGNORECASE),
        ("num", "month_name", "num"),
        ("year", "month", "day"),
        None,
    ),
    (
        # "JUN 28 2021" style: month name first, then day and year (day/year
        # order between the two numeric fields is still resolved by
        # validation + the existing order-prior, not assumed here).
        re.compile(rf"(?<!\d)({_MONTH_RE}){_SEP}(\d{{1,2}}){_SEP}(\d{{2,4}})(?!\d)", re.IGNORECASE),
        ("month_name", "num", "num"),
        ("year", "month", "day"),
        None,
    ),
    (
        # "AUG292020" style: month name immediately butted up against the
        # digits with no separator at all (OCR dropped the space/punctuation).
        # Safe to allow zero separators here because the month name is a
        # strong, small, fixed anchor - it can't accidentally appear inside
        # an unrelated digit run the way a bare number pattern could.
        re.compile(rf"(?<!\d)({_MONTH_RE}){_OPTIONAL_SEP}(\d{{1,2}}){_OPTIONAL_SEP}(\d{{4}})(?!\d)", re.IGNORECASE),
        ("month_name", "num", "num"),
        ("year", "month", "day"),
        None,
    ),
    (
        # "JUL2023" / "JUL 2023" style: month name + year only, no day found
        # in the text at all -> partial-NONE day. The month name itself
        # (not an assumed country convention) is what fixes the field order.
        re.compile(rf"(?<!\d)({_MONTH_RE}){_OPTIONAL_SEP}(\d{{4}})(?!\d)", re.IGNORECASE),
        ("month_name", "num"),
        ("month", "year"),
        ("month", "year"),
    ),
    (
        # A comma where a period/dash was probably intended (OCR visually
        # confuses the two), e.g. "26,07.14" or "27,02,18" or "22.01,06".
        # Requires an actual comma in at least one of the two separators
        # (unlike a plain all-period triple, which the generic pattern below
        # already handles correctly) - a pure "20XX.MM.DD" match must never
        # be intercepted here, since a longer text can have a coincidental
        # extra 2-digit group later on (e.g. an hour, "2025.10.11.22시") that
        # this pattern could otherwise latch onto instead of the real date.
        # Restricted to 1-2 digit groups only (never the {1,4} the generic
        # pattern below allows) so this can't misfire on a thousands-
        # separated number like "1,350" or "20,000" - those always have a
        # 3-digit group after the comma, which this cannot match. Uses plain
        # [0-9] rather than the O/o/U/u-confusable DIGIT class: DIGIT here
        # would let this pattern start matching midway through an unrelated
        # longer confusable-digit run (e.g. wrongly split "2O26.O1.15" after
        # its first character), which the generic pattern below - tried
        # after this one - already handles correctly as one full token.
        re.compile(r"(?<![0-9])([0-9]{1,2}),([0-9]{1,2})[.,]([0-9]{1,2})(?![0-9])"),
        ("num", "num", "num"),
        ("year", "month", "day"),
        None,
    ),
    (
        re.compile(r"(?<![0-9])([0-9]{1,2})\.([0-9]{1,2}),([0-9]{1,2})(?![0-9])"),
        ("num", "num", "num"),
        ("year", "month", "day"),
        None,
    ),
    (
        # "30,12,2021" style: day,month,year all comma-separated with an
        # explicit 4-digit year at the end. Same thousands-separator safety
        # argument as above (the two middle groups are capped at 1-2 digits,
        # so "1,234,567" or "20,000,000" can never match), plus the trailing
        # group must be exactly 4 digits, which narrows it further.
        re.compile(r"(?<![0-9])([0-9]{1,2}),([0-9]{1,2}),([0-9]{4})(?![0-9])"),
        ("num", "num", "num"),
        ("year", "month", "day"),
        None,
    ),
    (
        # Two 1-2 digit groups then a 4-digit year, with the mixed or unusual
        # separators OCR produces: "01,07 2021", "30.12,2021", "23.10·2020",
        # "18×04>2021". The trailing 4-digit year keeps this from matching
        # thousands separators ("1,350") or nutrition values, and it may not
        # start inside a longer date ("2025.10.03 2025.10.12" is two dates,
        # not "10.03 2025").
        re.compile(r"(?<![0-9])(?<![0-9][.,/\-·×])([0-9]{1,2})\s*[.,/\-·×]\s*([0-9]{1,2})\s*[.,/\-·×>\s]\s*([0-9]{4})(?![0-9])"),
        ("num", "num", "num"),
        ("year", "month", "day"),
        None,
    ),
    (
        # "202006 03": year and month glued, day after a space.
        re.compile(r"(?<![0-9])([0-9]{4})([0-9]{2})\s+([0-9]{2})(?![0-9])"),
        ("num", "num", "num"),
        ("year", "month", "day"),
        ("year", "month", "day"),
    ),
    (
        # "EXP:2606-2026": day and month glued, then the year. Only right
        # after an EXP keyword - a bare 4+4 digit run is too often a code.
        re.compile(r"(?i:EXP)[:.\s]*([0-9]{2})([0-9]{2})[\s\-./]+([0-9]{4})(?![0-9])"),
        ("num", "num", "num"),
        ("day", "month", "year"),
        ("day", "month", "year"),
    ),
    (
        # "EXP 102021": month and year glued, only right after EXP/BB/BBE.
        re.compile(r"(?i:EXP|BBE|BB)[:.\s]*([0-9]{2})([0-9]{4})(?![0-9])"),
        ("num", "num"),
        ("month", "year"),
        ("month", "year"),
    ),
    (
        # "03112021": eight digits read as DDMMYYYY / MMDDYYYY when the
        # YYYYMMDD reading (tried earlier) is not a valid date.
        re.compile(r"(?<![0-9])([0-9]{2})([0-9]{2})([0-9]{4})(?![0-9])"),
        ("num", "num", "num"),
        ("day", "month", "year"),
        None,
    ),
    (
        # The last group must not be the hour of a following HH:MM time -
        # "12.18. 10:41" is Dec 18 at 10:41, not 2018-12-10.
        re.compile(rf"(?<!\d)({DIGIT}{{1,4}}){_SEP}({DIGIT}{{1,4}}){_SEP}({DIGIT}{{1,4}})(?!\d)(?!\s*:\s*\d)"),
        ("num", "num", "num"),
        ("year", "month", "day"),
        None,
    ),
    (
        re.compile(rf"(?<!\d)({DIGIT}{{4}})({DIGIT}{{2}})({DIGIT}{{2}})(?!\d)"),
        ("num", "num", "num"),
        ("year", "month", "day"),
        None,
    ),
    (
        # "2021.0326" (month+day glued together with no internal separator),
        # "2022.11:02" / "2026:07.08" (a stray punctuation mark, e.g. OCR
        # misreading "." as ":", either between year/month or month/day) and
        # "2026.09,05" (comma instead of period). Anchored by an unambiguous
        # 4-digit year up front, so allowing a loose separator throughout is
        # low-risk - this can't accidentally swallow an unrelated HH:MM:SS
        # timestamp since those never start with a 4-digit number.
        re.compile(rf"(?<!\d)({DIGIT}{{4}})[:.\-/\s,()xX×·]+({DIGIT}{{2}})[:.,()/\-\s·]?({DIGIT}{{2}})(?!\d)"),
        ("num", "num", "num"),
        ("year", "month", "day"),
        None,
    ),
    (
        re.compile(rf"(?<!\d)({DIGIT}{{4}}){_SEP}({DIGIT}{{1,2}})(?!\d)"),
        ("num", "num"),
        ("year", "month"),
        None,
    ),
    (
        # Same year+month pair as above, but with a comma as the separator
        # (e.g. "2026,01") - OCR visually confusing a period/dash for a
        # comma, same as the dedicated comma patterns for 3-field tokens
        # above. Kept as its own pattern rather than folding into _SEP so
        # the already-tested plain pattern is untouched.
        re.compile(rf"(?<!\d)({DIGIT}{{4}}),({DIGIT}{{1,2}})(?!\d)"),
        ("num", "num"),
        ("year", "month"),
        None,
    ),
    (
        # Month-first, year-last 2-field pattern (e.g. "02/2023", "01,2022")
        # - the mirror image of the year-first pattern above. Only matches
        # when the trailing field is a full 4-digit year, so this can't
        # misfire the way a bare "NN.NN" pair could; interpret.py's existing
        # order-ambiguity handling (4-digit field always read as year,
        # regardless of position) resolves the actual role assignment.
        re.compile(rf"(?<!\d)({DIGIT}{{1,2}})[.\-/\s,]+({DIGIT}{{4}})(?!\d)"),
        ("num", "num"),
        ("year", "month"),
        None,
    ),
]


_FULL_DATE = r"[0-9]{4}[.\-/][0-9]{1,2}[.\-/][0-9]{2}"
_GLUED_DATE_RE = re.compile(rf"({_FULL_DATE})(?={_FULL_DATE})")
_GLUED_TIME_RE = re.compile(rf"({_FULL_DATE})(?=[0-9]{{1,2}}:[0-9]{{2}})")


# "2020.C6.30", "17/C7/2021": a lone C between separators, followed by one
# digit, is a 0 misread (dot-matrix 0 with a broken right side).
_C_AS_ZERO_RE = re.compile(r"(?<=[.\-/])[Cc](?=[0-9][.\-/])")


# "20 21.09.07": OCR split the 4-digit year in two ("2021").
_SPLIT_YEAR_RE = re.compile(r"(?<![0-9])(20)\s+([0-9]{2})(?=[.\-/][0-9]{1,2}[.\-/][0-9]{1,2}(?![0-9]))")


def split_glued(text: str) -> str:
    """Insert a space where OCR glued a full YYYY.MM.DD date to what follows:
    another full date ("2025.10.032025.10.12까지") or a time
    ("2026.01.2512:42"). Without the space the digit run after the day makes
    every date pattern fail."""
    text = _C_AS_ZERO_RE.sub("0", text)
    text = _SPLIT_YEAR_RE.sub(r"\1\2", text)
    text = _GLUED_DATE_RE.sub(r"\1 ", text)
    return _GLUED_TIME_RE.sub(r"\1 ", text)


def _trim_trailing_noise(fields: Tuple[RawField, ...]) -> Tuple[RawField, ...]:
    """"2026.08.267" / "2021.10.028": a 4-digit year, a month, then a 3-digit
    "day". Dot-matrix OCR often glues one stray character (a letter or the
    next symbol read as a digit) right after the day; a day never has three
    digits, so keep its first two. Only applied when the token starts with a
    4-digit year, where the year-month-day order is certain."""
    if (
        len(fields) == 3
        and all(f.kind == "num" for f in fields)
        and len(normalize_confusable(fields[0].raw)) == 4
        and len(fields[2].raw) == 3
    ):
        return (fields[0], fields[1], RawField(raw=fields[2].raw[:2], kind="num"))
    return fields


def _overlaps(span: Tuple[int, int], claimed: List[Tuple[int, int]]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in claimed)


def extract_date_tokens(text: str, accept: Optional[Callable[[RawDateToken], bool]] = None) -> List[RawDateToken]:
    """Find date-shaped substrings in ``text`` without assuming field order.

    Patterns are tried most-specific-first; once a span is claimed, later
    (more generic) patterns skip anything overlapping it.

    ``accept`` (optional) is asked about every match before it claims its
    span. A match it rejects - e.g. one with no valid calendar reading, like
    "U 2026. 01" read as 0/2026/01 via the O/U-as-zero rule - is dropped and
    leaves its span free, so a later pattern can still find the real date
    inside it ("2026. 01" as year+month).
    """
    text = split_glued(text)
    claimed: List[Tuple[int, int]] = []
    tokens: List[RawDateToken] = []
    for pattern, kinds, role_universe, fixed_roles in _PATTERN_DEFS:
        for match in pattern.finditer(text):
            span = match.span()
            if _overlaps(span, claimed):
                continue
            fields = _trim_trailing_noise(tuple(RawField(raw=g, kind=k) for g, k in zip(match.groups(), kinds)))
            token = RawDateToken(span=span, fields=fields, role_universe=role_universe, fixed_roles=fixed_roles)
            if accept is not None and not accept(token):
                continue
            tokens.append(token)
            claimed.append(span)
    tokens.sort(key=lambda t: t.span[0])
    return tokens


# Month + two-digit year ("05.21" = May 2021). Never extracted by default:
# "05.21" is just as often May 21st. Only used when the image carries a
# month/year format hint (see hints.py).
_MONTH_YY_RE = re.compile(r"(?<![0-9.,:])([0-9]{1,2})\s*[./\-]\s*([0-9]{2})(?![0-9.,:])")


def extract_month_yy_tokens(text: str, taken: List[Tuple[int, int]]) -> List[RawDateToken]:
    tokens = []
    for match in _MONTH_YY_RE.finditer(text):
        if _overlaps(match.span(), taken):
            continue
        fields = (RawField(raw=match.group(1), kind="num"), RawField(raw=match.group(2), kind="num"))
        tokens.append(RawDateToken(span=match.span(), fields=fields, role_universe=("month", "year"), fixed_roles=("month", "year")))
    return tokens
