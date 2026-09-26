from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

from .types import Bbox

# Marks a date as the expiration date being sought.
ANCHOR_KEYWORDS = ["소비기한", "유통기한", "사용기한", "까지", "EXP", "EXD", "BBD", "BB", "BEST BEFORE", "BEST BY"]

# Official rule: if 소비기한/사용기한 is present, it wins even when 유통기한/etc
# is also present on the same image (그건 소비기한이 없을 때의 fallback이지
# 동급이 아님). Kept separate from ANCHOR_KEYWORDS - which still means "any
# expiration-style keyword" for the general anchor-vs-exclude distance logic
# - because that broader check is unaffected by this priority.
PRIMARY_ANCHOR_KEYWORDS = ["소비기한", "사용기한"]

# Marks a date as a different kind of date (manufacture/packaging), not the
# expiration date, so it should lose to an anchor-flagged candidate nearby.
EXCLUDE_KEYWORDS = ["제조일자", "제조년월일", "제조일", "포장일자", "포장일", "산란일자", "산란일", "제조", "PRO", "PROD", "PRD", "MFD", "PACK", "MFG"]


def has_keyword(text: str, keywords: Iterable[str]) -> bool:
    """Substring match for Korean keywords (unambiguous enough on their own);
    word-boundary match for ASCII ones, since short abbreviations like "PRO"
    or "BB" would otherwise false-positive inside unrelated words like
    "PROTEIN" or "ABBA"."""
    text_upper = text.upper()
    for kw in keywords:
        kw_upper = kw.upper()
        if kw_upper.isascii():
            if re.search(rf"\b{re.escape(kw_upper)}\b", text_upper):
                return True
        elif kw_upper in text_upper:
            return True
    return False


def bbox_center(bbox: Bbox) -> Tuple[float, float]:
    xs = [point[0] for point in bbox]
    ys = [point[1] for point in bbox]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def min_distance(point: Tuple[float, float], others: Sequence[Tuple[float, float]]) -> float:
    if not others:
        return float("inf")
    return min(distance(point, other) for other in others)
