from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional, Sequence, Tuple

from .crossref import apply_manufacture_constraint
from .hints import MONTH_YEAR, detect_format_hint
from .extract import RawDateToken, RawField, extract_date_tokens, extract_month_yy_tokens, extract_yearless_month_day_tokens, extract_hint_month_name_year_tokens
from .interpret import DEFAULT_YEAR_MAX, DEFAULT_YEAR_MIN, ScoredCandidate, generate_candidates
from .keywords import ANCHOR_KEYWORDS, EXCLUDE_KEYWORDS, PRIMARY_ANCHOR_KEYWORDS, bbox_center, has_keyword, min_distance, nearest_keyword_is_exclude
from .types import DateResult, TextBox


@dataclass
class PositionedCandidate:
    result: DateResult
    center: Tuple[float, float]
    source_text: str
    candidates: List[ScoredCandidate]
    span: Optional[Tuple[int, int]] = None


def _has_position(box: TextBox) -> bool:
    """A box with no bbox points has no defined center, so it can't take part
    in any position-based logic here (candidate ranking or keyword-distance).
    Real OCR output always includes a polygon, but a malformed/degenerate
    entry should be skipped rather than crash the whole batch on one bad
    image out of thousands."""
    return len(box.bbox) > 0


def find_all_candidates(
    boxes: Sequence[TextBox], year_min: int = DEFAULT_YEAR_MIN, year_max: int = DEFAULT_YEAR_MAX
) -> List[PositionedCandidate]:
    """Every date interpretation found across all OCR boxes, each keeping its position."""
    positioned: List[PositionedCandidate] = []
    hint = detect_format_hint(box.text for box in boxes)
    for box in boxes:
        if not _has_position(box):
            continue
        scored_by_token = {}

        def has_reading(token):
            scored_by_token[token] = generate_candidates(token, year_min, year_max, order_hint=hint)
            return bool(scored_by_token[token])

        tokens = extract_date_tokens(box.text, accept=has_reading)
        if hint == MONTH_YEAR:
            extra = [t for t in extract_month_yy_tokens(box.text, [t.span for t in tokens]) if has_reading(t)]
            extra += [t for t in extract_hint_month_name_year_tokens(
                box.text, [t.span for t in tokens + extra]) if has_reading(t)]
            tokens = sorted(tokens + extra, key=lambda t: t.span[0])
        for token in tokens:
            scored = scored_by_token[token]
            positioned.append(
                PositionedCandidate(
                    result=scored[0].date,
                    center=bbox_center(box.bbox),
                    source_text=box.text,
                    candidates=scored,
                    span=token.span,
                )
            )
    if not positioned:
        positioned = _yearless_candidates(boxes, year_min, year_max)
    if not positioned:
        positioned = _six_digit_candidates(boxes, year_min, year_max)
    if not positioned:
        positioned = _padded_eight_digit_candidates(boxes, year_min, year_max)
    return positioned


# A full YYYYMMDD with stray OCR digits glued after it: one digit
# ("EXP202906147", "EXP202802191지1"), or up to three right before a 까지/지
# fragment, i.e. a misread 까지 ("2028062911지"). Longer runs without that
# fragment stay unread ("EXP 2021112116" is not 2021-11-21). Last resort only,
# with the same expiry evidence as six-digit dates or an EXP-like prefix.
_PADDED_EIGHT_RE = re.compile(
    r"(?<![0-9])(20[0-9]{2})(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])"
    r"(?:[0-9](?![0-9])|[0-9]{1,3}(?=\s*[가-힣]?지))"
)
_EXP_PREFIX_RE = re.compile(r"(?:E?XP+|EXD|BBE?)[^0-9A-Z]{0,2}$")


def _padded_eight_digit_candidates(
    boxes: Sequence[TextBox], year_min: int, year_max: int
) -> List[PositionedCandidate]:
    positioned: List[PositionedCandidate] = []
    anchor_boxes = [b for b in boxes if _has_position(b) and has_keyword(b.text, ANCHOR_KEYWORDS)]
    for box in boxes:
        if not _has_position(box):
            continue
        for match in _PADDED_EIGHT_RE.finditer(box.text):
            if not (_EXP_PREFIX_RE.search(box.text[:match.start()].upper())
                    or _has_expiry_evidence(box, match.end(), anchor_boxes)):
                continue
            roles = ("year", "month", "day")
            token = RawDateToken(span=match.span(), fields=tuple(RawField(raw=g, kind="num") for g in match.groups()),
                                 role_universe=roles, fixed_roles=roles)
            scored = generate_candidates(token, year_min, year_max)
            if scored:
                positioned.append(PositionedCandidate(result=scored[0].date, center=bbox_center(box.bbox),
                                                      source_text=box.text, candidates=scored, span=match.span()))
    return positioned


# Six digits with no separator: "202812" (YYYYMM) or "271017" (YYMMDD).
# Lot codes and barcode fragments look the same, so these are read only as a
# last resort (no other candidate in the stage) and only with expiry evidence:
# an expiration keyword in the same or an adjacent box, or a "까지" fragment
# ("지", "마지") glued right after the digits. Six digits continued by a
# separator and more digits ("202504/28") are a cut-off full date, not this.
_SIX_DIGIT_RE = re.compile(r"(?<![0-9A-Za-z])([0-9]{6})(?![0-9])(?!\s*[./\-:]\s*[0-9])")
_UNTIL_FRAGMENT_RE = re.compile(r"[가-힣]?지")


def _six_digit_candidates(
    boxes: Sequence[TextBox], year_min: int, year_max: int
) -> List[PositionedCandidate]:
    positioned: List[PositionedCandidate] = []
    anchor_boxes = [b for b in boxes if _has_position(b) and has_keyword(b.text, ANCHOR_KEYWORDS)]
    for box in boxes:
        if not _has_position(box):
            continue
        for match in _SIX_DIGIT_RE.finditer(box.text):
            if not _has_expiry_evidence(box, match.end(), anchor_boxes):
                continue
            digits = match.group(1)
            for fields, roles in (
                ((digits[:4], digits[4:]), ("year", "month")),
                ((digits[:2], digits[2:4], digits[4:]), ("year", "month", "day")),
            ):
                token = RawDateToken(span=match.span(), fields=tuple(RawField(raw=f, kind="num") for f in fields),
                                     role_universe=roles, fixed_roles=roles)
                scored = generate_candidates(token, year_min, year_max)
                if scored:
                    positioned.append(PositionedCandidate(result=scored[0].date, center=bbox_center(box.bbox),
                                                          source_text=box.text, candidates=scored, span=match.span()))
                    break
    return positioned


def _has_expiry_evidence(box: TextBox, end: int, anchor_boxes: Sequence[TextBox]) -> bool:
    if has_keyword(box.text, ANCHOR_KEYWORDS) or _UNTIL_FRAGMENT_RE.match(box.text, end):
        return True
    ys = [p[1] for p in box.bbox]
    height = (max(ys) - min(ys)) or 1.0
    center = bbox_center(box.bbox)
    return any(min_distance(center, [bbox_center(a.bbox)]) <= 3 * height for a in anchor_boxes)


def _yearless_candidates(
    boxes: Sequence[TextBox], year_min: int, year_max: int
) -> List[PositionedCandidate]:
    """Fallback for images with no dated candidate at all: year-less
    month.day tokens ("01.24", "02.12까지"), reported with year NONE."""
    positioned: List[PositionedCandidate] = []
    for box in boxes:
        if not _has_position(box):
            continue
        for token in extract_yearless_month_day_tokens(box.text):
            scored = generate_candidates(token, year_min, year_max)
            if not scored:
                continue
            year = _adjacent_year(box, boxes, year_min, year_max)
            if year is not None:
                month, day = scored[0].date.month, scored[0].date.day
                try:
                    date(year, month, day)
                    scored = [ScoredCandidate(DateResult(year=year, month=month, day=day), scored[0].score, ("year", "month", "day"))]
                except ValueError:
                    pass
            positioned.append(PositionedCandidate(
                result=scored[0].date, center=bbox_center(box.bbox), source_text=box.text, candidates=scored,
            ))
    return positioned


_YEAR_ONLY_RE = re.compile(r"^\s*(20[0-9]{2})\s*[.,]?\s*$")


def _adjacent_year(box: TextBox, boxes: Sequence[TextBox], year_min: int, year_max: int) -> Optional[int]:
    """A box holding only a 4-digit year right next to the month.day box
    (OCR split "2027 04.28" into two boxes)."""
    xs = [p[1] for p in box.bbox]
    height = max(xs) - min(xs) or 1.0
    center = bbox_center(box.bbox)
    best = None
    for other in boxes:
        if other is box or not _has_position(other):
            continue
        m = _YEAR_ONLY_RE.match(other.text)
        if not m or not (year_min <= int(m.group(1)) <= year_max):
            continue
        d = min_distance(center, [bbox_center(other.bbox)])
        if d <= 3 * height and (best is None or d < best[0]):
            best = (d, int(m.group(1)))
    return best[1] if best else None


# "유통기한: 제조일로부터 5년까지" - the package states the expiry as a period
# from the manufacture date instead of printing it.
_PERIOD_FROM_MANUFACTURE_RE = re.compile(r"제조일\s*로\s*부터\s*([0-9]{1,2})\s*(년|개월|일)")


def _period_from_manufacture(
    boxes: Sequence[TextBox], positioned: List[PositionedCandidate]
) -> Optional[PositionedCandidate]:
    """Only reached when every candidate is a manufacture date. With a stated
    period, expiry = manufacture date + period; otherwise no candidate."""
    periods = {m.groups() for b in boxes for m in _PERIOD_FROM_MANUFACTURE_RE.finditer(b.text)}
    bases = {pc.result for pc in positioned if pc.result.year is not None and pc.result.month is not None}
    if len(periods) != 1 or len(bases) != 1:
        return None
    (amount, unit), base = periods.pop(), bases.pop()
    n = int(amount)
    if unit == "일":
        if not base.is_complete():
            return None
        d = date(base.year, base.month, base.day) + timedelta(days=n)
        result = DateResult(year=d.year, month=d.month, day=d.day)
    else:
        months = base.month - 1 + (n * 12 if unit == "년" else n)
        year, month = base.year + months // 12, months % 12 + 1
        day = None if base.day is None else min(base.day, calendar.monthrange(year, month)[1])
        result = DateResult(year=year, month=month, day=day)
    source = next(pc for pc in positioned if pc.result == base)
    return PositionedCandidate(result=result, center=source.center, source_text=source.source_text,
                               candidates=[ScoredCandidate(result, 0, ("year", "month", "day"))], span=source.span)


def _self_excluded(pc: PositionedCandidate) -> bool:
    """The box's own text names this date as a non-expiration date. When the
    same box also carries an expiration keyword ("[제조번호]별도표기
    [사용기한]2029년 04월"), the keyword nearest to the date decides."""
    if not has_keyword(pc.source_text, EXCLUDE_KEYWORDS):
        return False
    if pc.span is None or not has_keyword(pc.source_text, ANCHOR_KEYWORDS):
        return True
    return nearest_keyword_is_exclude(pc.source_text, pc.span)


def _latest_ordinal(result: DateResult) -> int:
    """Latest calendar day the reading can mean, for the "later date first"
    tie-break: a year-month reading ("2029년 04월") runs to its month end, so it
    is compared on the same scale as complete dates instead of losing to all
    of them. Readings without a year stay neutral (0)."""
    if result.year is None or result.month is None:
        return 0
    if result.day is not None:
        return date(result.year, result.month, result.day).toordinal()
    return date(result.year, result.month, calendar.monthrange(result.year, result.month)[1]).toordinal()


def _is_closer_to(center: Tuple[float, float], own: List[Tuple[float, float]], other: List[Tuple[float, float]]) -> bool:
    return min_distance(center, own) < min_distance(center, other)


def _pick_manufacture_reference(
    positioned: List[PositionedCandidate],
    anchor_centers: List[Tuple[float, float]],
    exclude_centers: List[Tuple[float, float]],
) -> Optional[DateResult]:
    """The nearest exclude-keyword-associated (e.g. 제조일자) date, if it's
    fully known, to use as a plausibility reference for expiration dates."""
    manufacture_side = [
        pc for pc in positioned
        if pc.result.is_complete() and _is_closer_to(pc.center, exclude_centers, anchor_centers)
    ]
    if not manufacture_side:
        return None
    manufacture_side.sort(key=lambda pc: min_distance(pc.center, exclude_centers))
    return manufacture_side[0].result


def _earliest_of_labelled_set(positioned: List[PositionedCandidate]) -> Optional[PositionedCandidate]:
    """A set of products: two or more different complete dates, each in its
    own box, each box naming its date with the same expiry keyword and no
    manufacture keyword ("본품 사용기한2029-03-26" / "토너패드사용기한2029-07-10").
    The earliest one limits the whole item."""
    if len(positioned) < 2 or len({pc.source_text for pc in positioned}) != len(positioned):
        return None
    kinds = set()
    for pc in positioned:
        if not pc.result.is_complete() or has_keyword(pc.source_text, EXCLUDE_KEYWORDS):
            return None
        found = frozenset(k for k in ANCHOR_KEYWORDS if has_keyword(pc.source_text, [k]))
        if not found:
            return None
        kinds.add(found)
    if len(kinds) != 1 or len({pc.result.final_date_string() for pc in positioned}) < 2:
        return None
    return min(positioned, key=lambda pc: date(pc.result.year, pc.result.month, pc.result.day))


def select_final_date(
    boxes: Sequence[TextBox], year_min: int = DEFAULT_YEAR_MIN, year_max: int = DEFAULT_YEAR_MAX
) -> Optional[PositionedCandidate]:
    """Pick the expiration-date candidate: nearest to an anchor keyword and
    not nearer to an exclude keyword (e.g. 제조일자)."""
    positioned = find_all_candidates(boxes, year_min, year_max)
    if not positioned:
        return None
    # Every date the stage found is named by its own box as a manufacture /
    # packaging date ("PROD 02/2021", "HFG 2024.05.07제조"): the expiration date
    # was not read, so report no candidate instead of the manufacture date.
    if all(_self_excluded(pc) for pc in positioned):
        return _period_from_manufacture(boxes, positioned)

    set_pick = _earliest_of_labelled_set(positioned)
    if set_pick is not None:
        return set_pick

    positionable_boxes = [b for b in boxes if _has_position(b)]
    anchor_centers = [bbox_center(b.bbox) for b in positionable_boxes if has_keyword(b.text, ANCHOR_KEYWORDS)]
    exclude_centers = [bbox_center(b.bbox) for b in positionable_boxes if has_keyword(b.text, EXCLUDE_KEYWORDS)]
    # Official rule: 소비기한 outranks other expiration-style keywords
    # (유통기한/사용기한/EXP/...) when both appear on the same image, not just
    # "whichever is spatially closer". Empty when 소비기한 doesn't appear
    # anywhere, which makes the priority tier below a no-op for every
    # candidate (falls through to the existing distance-based ranking).
    primary_anchor_centers = [bbox_center(b.bbox) for b in positionable_boxes if has_keyword(b.text, PRIMARY_ANCHOR_KEYWORDS)]
    secondary_anchor_centers = [
        bbox_center(b.bbox)
        for b in positionable_boxes
        if has_keyword(b.text, ANCHOR_KEYWORDS) and not has_keyword(b.text, PRIMARY_ANCHOR_KEYWORDS)
    ]

    if anchor_centers and exclude_centers:
        reference = _pick_manufacture_reference(positioned, anchor_centers, exclude_centers)
        if reference is not None:
            for pc in positioned:
                if _is_closer_to(pc.center, anchor_centers, exclude_centers):
                    pc.candidates = apply_manufacture_constraint(pc.candidates, reference)
                    pc.result = pc.candidates[0].date

    if anchor_centers:
        def rank(pc: PositionedCandidate) -> Tuple[int, int, int, int, float]:
            # A box whose own text carries an exclude keyword (e.g. "PROD",
            # 제조일자) names itself as a non-expiration date - that should
            # outrank pure bbox distance, which can otherwise pick the
            # manufacture-date box itself when it happens to sit closer to
            # the (possibly distant) anchor text than the real expiration
            # date box does.
            self_excluded = _self_excluded(pc)
            d_anchor = min_distance(pc.center, anchor_centers)
            d_exclude = min_distance(pc.center, exclude_centers)
            penalty = 0 if d_anchor <= d_exclude else 1

            # 소비기한 priority: only meaningful when 소비기한 appears
            # somewhere AND some other expiration-style keyword also appears
            # somewhere - otherwise every candidate ties on this tier and it
            # has no effect.
            if primary_anchor_centers and secondary_anchor_centers:
                not_nearest_to_primary = 0 if _is_closer_to(pc.center, primary_anchor_centers, secondary_anchor_centers) else 1
            else:
                not_nearest_to_primary = 0

            # Among candidates tied on every keyword-based tier so far, an
            # expiration date is virtually always the later of the two, so
            # prefer it over raw pixel distance - a box that merely sits
            # closer to the keyword text by coincidence (e.g. a manufacture
            # date on the line right above "소비기한") shouldn't win against
            # a plausible later date only because it's a few pixels nearer.
            recency = -_latest_ordinal(pc.result)

            return (int(self_excluded), penalty, not_nearest_to_primary, recency, d_anchor)

        positioned.sort(key=rank)
    elif len(positioned) > 1:
        # No anchor keyword (e.g. 소비기한/EXP) was found anywhere in the
        # image, so there's no positional signal to pick among multiple
        # date candidates at all. An expiration date is virtually always
        # later than any other date printed on packaging (manufacture,
        # packaging, etc.), so as a last resort - not a real selection,
        # just a weak guess - prefer the chronologically latest complete
        # date over an arbitrary "whichever OCR box came first" default.
        def latest_first(pc: PositionedCandidate) -> Tuple[int, int]:
            if not pc.result.is_complete():
                return (1, 0)
            ordinal = date(pc.result.year, pc.result.month, pc.result.day).toordinal()
            return (0, -ordinal)

        positioned.sort(key=latest_first)

    return positioned[0]
