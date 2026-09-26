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
    # T1: a printed format hint decides the field order on that image only.
    (["05.21 L095 2", "Mindestens haltbar bis Ende: Monat/Jahr"], "2021-05-NONE"),  # 1205
    (["30/04/21", "CH 2371", "유통기한: 제품 뒷면에 별도표기 일까지 (읽는법: 일,월,년순)"], "2021-04-30"),  # 2917
    (["Best before (dd/mm/yy)", "17.03.22"], "2022-03-17"),
    # ...and without a hint the default year-month-day reading stays.
    (["26.04.24"], "2026-04-24"),
    (["05.21 L095 2"], "NONE"),
    # T2: full date glued to a time or to a second date; hour is not a date field.
    (["2026.01.2512:427m1U1"], "2026-01-25"),              # 682
    (["2025.10.032025.10.12까지"], "2025-10-12"),           # 686
    (["2021.10.20 13:40"], "2021-10-20"),
    (["12.18. 10:41", "F5", "12.06. 10:41"], "NONE"),       # 824: no longer 2018-12-10
    # T5: day + month name + year with no separators.
    (["07FEB2022 PV  07:17  2"], "2022-02-07"),             # 892
    (["O7FEB2022 PV 07:17 2"], "2022-02-07"),               # 893
    (["09MAR2023"], "2023-03-09"),                          # 1757
    (["BEST BY 31JUL21"], "2021-07-31"),                    # 2048
])
def test_final_round_cases(lines, expected):
    assert _parse(*lines) == expected
