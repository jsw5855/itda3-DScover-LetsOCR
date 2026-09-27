from __future__ import annotations

import re
from typing import Iterable, Optional

# Some packages print how to read their date code, e.g. a Korean import
# sticker "(읽는법: 일,월,년순)", "Best before (dd/mm/yyyy)", or German
# "Mindestens haltbar bis Ende: Monat/Jahr". When such a hint is on the image,
# it is direct evidence of the field order and overrides the default
# year-month-day preference for ambiguous numeric dates on that image only.
# Without a hint nothing changes: on the 701 labeled images, switching the
# default for every two-digit-year date to day-month-year fixed 7 and broke 42.

DMY = "dmy"
MONTH_YEAR = "month_year"

_DMY_PATTERNS = [
    re.compile(r"일\s*[,./·]\s*월\s*[,./·]\s*년"),
    re.compile(r"\b[DJ][D0OJ]\s*[./\-]\s*MM\s*[./\-]\s*(?:YY|AA)", re.IGNORECASE),
]
_MONTH_YEAR_PATTERNS = [
    re.compile(r"Monat\s*/\s*Jahr", re.IGNORECASE),
    re.compile(r"Month\s*/\s*Year", re.IGNORECASE),
    re.compile(r"(?<![DJ]\s[./\-]\s)(?<![DJ][./\-])\bMM\s*[./]\s*(?:YY|AA)", re.IGNORECASE),
    re.compile(r"(?<!일\s)(?<!일)월\s*/\s*년"),
]


def detect_format_hint(texts: Iterable[str]) -> Optional[str]:
    """Return DMY, MONTH_YEAR or None for all OCR text of one image."""
    joined = " | ".join(texts)
    if any(p.search(joined) for p in _DMY_PATTERNS):
        return DMY
    if any(p.search(joined) for p in _MONTH_YEAR_PATTERNS):
        return MONTH_YEAR
    return None
