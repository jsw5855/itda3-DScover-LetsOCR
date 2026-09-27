from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

Bbox = Sequence[Sequence[float]]


@dataclass(frozen=True)
class TextBox:
    """One entry of the common OCR output interface: {text, confidence, bbox}."""

    text: str
    confidence: float
    bbox: Bbox

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TextBox":
        return cls(text=d["text"], confidence=d["confidence"], bbox=d["bbox"])


@dataclass(frozen=True)
class DateResult:
    """Parsed date, any field may be unknown (partial-NONE)."""

    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None

    def is_empty(self) -> bool:
        return self.year is None and self.month is None and self.day is None

    def is_complete(self) -> bool:
        return self.year is not None and self.month is not None and self.day is not None

    def as_strings(self) -> tuple[str, str, str]:
        y = f"{self.year:04d}" if self.year is not None else "NONE"
        m = f"{self.month:02d}" if self.month is not None else "NONE"
        d = f"{self.day:02d}" if self.day is not None else "NONE"
        return y, m, d

    def final_date_string(self) -> str:
        if self.is_empty():
            return "NONE"
        return "-".join(self.as_strings())
