from __future__ import annotations

from datetime import date
from itertools import permutations
from typing import List, Optional, Sequence, Tuple

from .extract import MONTH_NAMES, RawDateToken, RawField, normalize_confusable
from .types import DateResult

# Widened from the 150-image label sample (observed range 2018-2028), with
# margin on both sides since real printed dates weren't found to cluster
# tightly around "now" - both very old manufacture dates and long-shelf-life
# expiration dates well into the future occur.
DEFAULT_YEAR_MIN = 2015
DEFAULT_YEAR_MAX = 2040

# Weak tiebreaker only: applied after calendar validation, never used to
# discard an otherwise-valid candidate. Ranking (YMD > DMY > MDY) calibrated
# against the 150-image label sample. Note: the sample's notes column tags
# DD/MM/YYYY-style images specifically because they deviate from the
# untagged majority's format (the labeler had no reason to annotate the
# "normal" case) - counting only tagged notes (DD/MM: 21, MM/DD: 2) would
# wrongly suggest DMY should be the default. Spot-checking untagged
# genuinely-ambiguous cases (e.g. "26.09.24" -> 2026-09-24, "21.02.22" ->
# 2021-02-22) confirms the untagged majority is in fact YMD. DMY still
# clearly beats MDY as the second choice (21:2 in the explicit tags).
_ORDER_PRIOR = {
    ("year", "month", "day"): 3,
    ("day", "month", "year"): 2,
    ("month", "day", "year"): 1,
    ("year", "month"): 3,
    ("month", "year"): 0,
}

# When the token spells the month out in English (e.g. "23-Jul-21"), the
# two-digit-year reading empirically goes day-month-year, not year-month-day
# ("23-Jul-21" -> 2021-07-23, "19-Mar-22" -> 2022-03-19 in the label
# sample - both wrong under the numeric-default table above). The English
# month name is itself direct textual evidence of a different labeling
# convention than the numeric-only majority, not an assumed country rule.
_MONTH_NAME_ORDER_PRIOR = {
    ("day", "month", "year"): 3,
    ("year", "month", "day"): 2,
    ("month", "day", "year"): 1,
}


class ScoredCandidate:
    __slots__ = ("date", "score", "perm")

    def __init__(self, date_result: DateResult, score: float, perm: Tuple[str, ...]):
        self.date = date_result
        self.score = score
        self.perm = perm

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ScoredCandidate({self.date!r}, score={self.score}, perm={self.perm})"


def _assign_role(role: str, field: RawField, year_min: int, year_max: int) -> Optional[int]:
    if field.kind == "month_name":
        return MONTH_NAMES[field.raw.upper()] if role == "month" else None

    num = normalize_confusable(field.raw)
    if not num.isdigit():
        return None
    n = int(num)

    if role == "year":
        if len(num) == 4:
            return n if year_min <= n <= year_max else None
        if len(num) <= 2:
            for century in (2000, 1900):
                candidate = century + n
                if year_min <= candidate <= year_max:
                    return candidate
            return None
        return None
    if role == "month":
        return n if 1 <= n <= 12 else None
    if role == "day":
        return n if 1 <= n <= 31 else None
    return None


# Day-of-month upper bound that holds in *every* year, used only when year
# is unknown (so a full calendar check via datetime.date isn't possible).
# February gets 29 (not 28) so a leap-year "02/29" isn't wrongly degraded.
_MAX_DAY_FOR_MONTH = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}


def _build_candidate(
    fields: Sequence[RawField], perm: Tuple[str, ...], year_min: int, year_max: int
) -> Tuple[Optional[DateResult], bool]:
    """Returns (candidate, is_degraded). is_degraded means year/month were each
    individually valid but the (year, month, day) combination doesn't exist on
    the calendar (e.g. Feb 30) — day is the field actually at fault, since
    day-of-month validity is the only one of the three that depends on the
    other two. Partial-NONE keeps year/month and drops just the day, instead
    of discarding an otherwise-readable date."""
    values = {}
    for field, role in zip(fields, perm):
        value = _assign_role(role, field, year_min, year_max)
        if value is None:
            return None, False
        values[role] = value

    result = DateResult(year=values.get("year"), month=values.get("month"), day=values.get("day"))
    if result.year is not None and result.month is not None and result.day is not None:
        try:
            date(result.year, result.month, result.day)
        except ValueError:
            return DateResult(year=result.year, month=result.month, day=None), True
    elif result.month is not None and result.day is not None and result.year is None:
        # No year to run a full calendar check against (e.g. "04월 31일"),
        # but April 31st doesn't exist in any year - catch what a
        # year-independent check still can.
        if result.day > _MAX_DAY_FOR_MONTH[result.month]:
            return DateResult(year=None, month=result.month, day=None), True
    return result, False


# Used instead of _ORDER_PRIOR when the image itself states a day-month-year
# reading order (see hints.py).
_DMY_HINT_ORDER_PRIOR = {
    ("day", "month", "year"): 3,
    ("year", "month", "day"): 2,
    ("month", "day", "year"): 1,
    ("year", "month"): 3,
    ("month", "year"): 0,
}


def _score(fields: Sequence[RawField], perm: Tuple[str, ...], order_hint: Optional[str] = None) -> float:
    if any(f.kind == "month_name" for f in fields):
        table = _MONTH_NAME_ORDER_PRIOR
    elif order_hint == "dmy":
        table = _DMY_HINT_ORDER_PRIOR
    else:
        table = _ORDER_PRIOR
    score = table.get(perm, 0)
    if "year" in perm:
        year_field = fields[perm.index("year")]
        if year_field.kind == "num" and len(normalize_confusable(year_field.raw)) == 4:
            score += 10
    return score


def generate_candidates(
    token: RawDateToken,
    year_min: int = DEFAULT_YEAR_MIN,
    year_max: int = DEFAULT_YEAR_MAX,
    order_hint: Optional[str] = None,
) -> List[ScoredCandidate]:
    """Turn one raw token into every valid (year, month, day) interpretation, scored.

    Fully valid (calendar-checked) candidates are always preferred. Only when
    none exist do calendar-invalid-but-otherwise-readable candidates (see
    ``_build_candidate``) get returned, with their day dropped to NONE.
    """
    strict: List[ScoredCandidate] = []
    degraded: List[ScoredCandidate] = []
    seen_strict = set()
    seen_degraded = set()

    if token.fixed_roles is not None:
        perms = [token.fixed_roles]
    else:
        perms = list(permutations(token.role_universe))

    for perm in perms:
        month_name_positions = [i for i, f in enumerate(token.fields) if f.kind == "month_name"]
        if any(perm[i] != "month" for i in month_name_positions):
            continue
        candidate, is_degraded = _build_candidate(token.fields, perm, year_min, year_max)
        if candidate is None:
            continue
        key = (candidate.year, candidate.month, candidate.day)
        bucket, seen = (degraded, seen_degraded) if is_degraded else (strict, seen_strict)
        if key in seen:
            continue
        seen.add(key)
        bucket.append(ScoredCandidate(candidate, _score(token.fields, perm, order_hint), perm))

    results = strict if strict else degraded
    results.sort(key=lambda c: -c.score)
    return results
