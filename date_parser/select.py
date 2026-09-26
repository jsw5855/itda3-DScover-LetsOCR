from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Sequence, Tuple

from .crossref import apply_manufacture_constraint
from .extract import extract_date_tokens
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
    for box in boxes:
        if not _has_position(box):
            continue
        scored_by_token = {}

        def has_reading(token):
            scored_by_token[token] = generate_candidates(token, year_min, year_max)
            return bool(scored_by_token[token])

        for token in extract_date_tokens(box.text, accept=has_reading):
            scored = scored_by_token[token]
            positioned.append(
                PositionedCandidate(
                    result=scored[0].date,
                    center=bbox_center(box.bbox),
                    source_text=box.text,
                    candidates=scored,
                )
            )
    return positioned


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
