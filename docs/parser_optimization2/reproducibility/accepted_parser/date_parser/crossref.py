from __future__ import annotations

from datetime import date
from typing import List, Optional

from .interpret import ScoredCandidate
from .types import DateResult

# A large enough bonus to always outrank interpret.py's weak order-prior/
# 4-digit-year tie-break (max ~13), so real chronological evidence wins over
# a guess when they disagree.
_PLAUSIBILITY_BONUS = 100


def apply_manufacture_constraint(
    candidates: List[ScoredCandidate], reference: Optional[DateResult]
) -> List[ScoredCandidate]:
    """Re-rank expiration-date candidates using a second, already-known date
    from the same image (typically 제조일자/포장일자) as a plausibility check:
    an expiration date should fall after the reference date.

    This never discards a candidate — a reading that fails the check can
    still win if every other reading also fails it (e.g. no reference date,
    or the reference is itself unreliable). It only promotes readings that
    pass the check over the ones that don't, which is a much stronger,
    evidence-based signal than the format-order tie-break in interpret.py.
    """
    if reference is None or not reference.is_complete():
        return candidates
    reference_ordinal = date(reference.year, reference.month, reference.day)

    def passes(candidate: ScoredCandidate) -> bool:
        d = candidate.date
        return d.is_complete() and date(d.year, d.month, d.day) > reference_ordinal

    return sorted(candidates, key=lambda c: -(c.score + (_PLAUSIBILITY_BONUS if passes(c) else 0)))
