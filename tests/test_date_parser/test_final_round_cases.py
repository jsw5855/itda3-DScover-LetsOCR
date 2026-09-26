"""Regression cases from the final-round review and the 701-image run.

Each case is OCR text actually produced by the submitted v6 pipeline (or the
text printed on the package, where noted) together with the expected
parse_expiration_date() output. Image ids refer to the competition images.
"""
import pytest

from date_parser import parse_expiration_date


def _parse(*lines):
    boxes = [
        {"text": t, "confidence": 0.9, "bbox": [[0, 40 * i], [300, 40 * i], [300, 40 * i + 30], [0, 40 * i + 30]]}
        for i, t in enumerate(lines)
    ]
    return parse_expiration_date(boxes)["final_date"]


@pytest.mark.parametrize("lines, expected", [
    # T6: a match with no valid reading must not hide the real date inside it.
    (["201045 46 +U 2026. 01"], "2026-01-NONE"),            # 515 (printed text)
    (["20104546 +0 2026.01"], "2026-01-NONE"),              # 515 (v6 OCR)
    # T3: one stray digit glued after the day of a YYYY.MM.DD date.
    (["2021.10.028"], "2021-10-02"),                        # 1730
    (["2022. 11.013"], "2022-11-01"),                       # 1741
    (["2026.08.267"], "2026-08-26"),                        # 36
    (["2026.07.080"], "2026-07-08"),                        # 384
])
def test_final_round_cases(lines, expected):
    assert _parse(*lines) == expected
