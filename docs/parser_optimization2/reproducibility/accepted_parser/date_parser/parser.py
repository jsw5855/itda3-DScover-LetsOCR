from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Union

from .interpret import DEFAULT_YEAR_MAX, DEFAULT_YEAR_MIN
from .select import select_final_date
from .types import DateResult, TextBox

OcrEntry = Union[TextBox, Mapping[str, Any]]


def _to_text_box(entry: OcrEntry) -> TextBox:
    return entry if isinstance(entry, TextBox) else TextBox.from_dict(entry)


def parse_expiration_date(
    ocr_results: Sequence[OcrEntry],
    year_min: int = DEFAULT_YEAR_MIN,
    year_max: int = DEFAULT_YEAR_MAX,
) -> Dict[str, str]:
    """Entry point: common OCR output -> {year, month, day, final_date}.

    ``ocr_results`` is the shared OCR interface, a list of
    ``{"text": ..., "confidence": ..., "bbox": [...]}`` (or ``TextBox``)
    entries. Any field that cannot be determined is reported as ``"NONE"``
    independently of the others (partial-NONE).
    """
    boxes = [_to_text_box(entry) for entry in ocr_results]
    best = select_final_date(boxes, year_min=year_min, year_max=year_max)
    result = best.result if best is not None else DateResult()
    year, month, day = result.as_strings()
    return {"year": year, "month": month, "day": day, "final_date": result.final_date_string()}
