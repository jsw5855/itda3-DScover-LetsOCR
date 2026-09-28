from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Sequence, Tuple

from .crossref import apply_manufacture_constraint
from .hints import MONTH_YEAR, detect_format_hint
from .extract import extract_date_tokens, extract_month_yy_tokens, extract_yearless_month_day_tokens, extract_hint_month_name_year_tokens
from .interpret import DEFAULT_YEAR_MAX, DEFAULT_YEAR_MIN, ScoredCandidate, generate_candidates
from .keywords import ANCHOR_KEYWORDS, EXCLUDE_KEYWORDS, PRIMARY_ANCHOR_KEYWORDS, bbox_center, has_keyword, min_distance
from .types import DateResult, TextBox


@dataclass
class PositionedCandidate:
    result: DateResult
    center: Tuple[float, float]
    source_text: str
    candidates: List[ScoredCandidate]


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
    slash_dmy = hint is None and _has_unambiguous_slash_dmy(boxes, year_min, year_max)
    for box in boxes:
        if not _has_position(box):
            continue
        scored_by_token = {}

        def has_reading(token):
            token_hint = hint
            if slash_dmy and len(token.fields) == 3 and all(f.kind == 'num' for f in token.fields):
                # Transfer only between slash triples with the exact same
                # numeric fields; dotted or compact codes keep their prior.
                fields = tuple(f.raw for f in token.fields)
                if any(m.groups() == fields for m in _SHORT_SLASH_DATE.finditer(box.text)):
                    token_hint = 'dmy'
            scored_by_token[token] = generate_candidates(token, year_min, year_max, order_hint=token_hint)
            return bool(scored_by_token[token])

        tokens = extract_date_tokens(box.text, accept=has_reading)
        if hint == MONTH_YEAR:
            extra = [t for t in extract_month_yy_tokens(box.text, [t.span for t in tokens]) if has_reading(t)]
            extra += [t for t in extract_hint_month_name_year_tokens(
                box.text, [t.span for t in tokens + extra]) if has_reading(t)]
            tokens = sorted(tokens + extra, key=lambda t: t.span[0])
        for token in tokens:
            scored = scored_by_token[token]
            if (len(token.fields) == 2 and token.fields[0].kind == 'num'
                    and len(token.fields[0].raw) == 4
                    and scored[0].date.year is not None and scored[0].date.month is not None
                    and scored[0].date.day is None
                    and re.search(r'20[0-9]{2}[./-][0-9]{2}\s*$', box.text)):
                day = _right_adjacent_day(box, boxes)
                if day is not None:
                    value = scored[0].date
                    try:
                        date(value.year, value.month, day)
                    except ValueError:
                        pass
                    else:
                        scored = [ScoredCandidate(DateResult(value.year, value.month, day),
                            scored[0].score, ('year', 'month', 'day'))]
            positioned.append(
                PositionedCandidate(
                    result=scored[0].date,
                    center=bbox_center(box.bbox),
                    source_text=box.text,
                    candidates=scored,
                )
            )
    if not positioned:
        positioned = _anchored_compact_year_month(boxes, year_min, year_max)
    if not positioned:
        positioned = _yearless_candidates(boxes, year_min, year_max)
    return positioned


def _right_adjacent_day(box, boxes):
    """Join a separated day only on the same line, with a small physical gap.

    Require its confidence to be at least that of the source date box so the
    production source confidence does not overstate the assembled reading.
    Ambiguous neighbors, vertical text and distant numbers are left alone.
    """
    left, right = min(p[0] for p in box.bbox), max(p[0] for p in box.bbox)
    height = max(p[1] for p in box.bbox) - min(p[1] for p in box.bbox)
    if height <= 0 or right - left <= height:
        return None
    cy = bbox_center(box.bbox)[1]
    matches = []
    for other in boxes:
        if other is box or not _has_position(other) or other.confidence < box.confidence:
            continue
        if not re.fullmatch(r'(?:0[1-9]|[12][0-9]|3[01])', other.text.strip()):
            continue
        gap = min(p[0] for p in other.bbox) - right
        other_height = max(p[1] for p in other.bbox) - min(p[1] for p in other.bbox)
        if (0 <= gap <= height and .5 * height <= other_height <= 1.5 * height
                and abs(bbox_center(other.bbox)[1] - cy) <= .4 * height):
            matches.append(int(other.text))
    return matches[0] if len(matches) == 1 else None


_SHORT_SLASH_DATE = re.compile(r'(?<![0-9])([0-9]{2})/([0-9]{2})/([0-9]{2})(?![0-9])')
_LONG_SLASH_DATE = re.compile(r'(?<![0-9])([0-9]{2})/([0-9]{2})/(20[0-9]{2})(?![0-9])')


def _has_unambiguous_slash_dmy(boxes, year_min, year_max):
    found = False
    for box in boxes:
        if re.search(r'(?<![0-9])20[0-9]{2}/[0-9]{1,2}/[0-9]{1,2}(?![0-9])', box.text):
            return False  # Mixed conventions are not a usable format hint.
        for match in _LONG_SLASH_DATE.finditer(box.text):
            day, month, year = map(int, match.groups())
            if 1 <= day <= 12 and 13 <= month <= 31:
                return False  # An explicit MM/DD/YYYY reference conflicts.
            if day <= 12 or not year_min <= year <= year_max:
                continue
            try:
                date(year, month, day)
            except ValueError:
                continue
            found = True
    return found


def _anchored_compact_year_month(boxes, year_min, year_max):
    """Bare YYYYMM needs a nearby standalone expiry label, not a lot guess.

    This is a fallback only: an ordinary dated candidate always takes priority.
    The label must be within three text heights and aligned on the same line.
    """
    labels = [b for b in boxes if _has_position(b)
              and b.text.strip().rstrip(':. ').upper() in ANCHOR_KEYWORDS]
    result = []
    for box in boxes:
        if not _has_position(box):
            continue
        match = re.fullmatch(r'\s*(20[0-9]{2})(0[1-9]|1[0-2])\s*', box.text)
        if not match or not year_min <= int(match[1]) <= year_max:
            continue
        center = bbox_center(box.bbox)
        height = max(p[1] for p in box.bbox) - min(p[1] for p in box.bbox)
        if not any(abs(center[1] - bbox_center(b.bbox)[1]) <= height
                   and min_distance(center, [bbox_center(b.bbox)]) <= 3 * height for b in labels):
            continue
        value = DateResult(year=int(match[1]), month=int(match[2]), day=None)
        result.append(PositionedCandidate(value, center, box.text,
            [ScoredCandidate(value, 13, ('year', 'month'))]))
    return result


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


def select_final_date(
    boxes: Sequence[TextBox], year_min: int = DEFAULT_YEAR_MIN, year_max: int = DEFAULT_YEAR_MAX
) -> Optional[PositionedCandidate]:
    """Pick the expiration-date candidate: nearest to an anchor keyword and
    not nearer to an exclude keyword (e.g. 제조일자)."""
    positioned = find_all_candidates(boxes, year_min, year_max)
    if not positioned:
        return None

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
            self_excluded = has_keyword(pc.source_text, EXCLUDE_KEYWORDS)
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
            if pc.result.is_complete():
                recency = -date(pc.result.year, pc.result.month, pc.result.day).toordinal()
            else:
                recency = 0

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
