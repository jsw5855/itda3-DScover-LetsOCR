"""Regression cases from the cosmetics development photos (900001-900150).

Each case is OCR text actually produced by the submitted pipeline on that photo
(full-stage dump, 2026-09-27) with the expected parse_expiration_date() output.
The cosmetics validation photos (900301-900400) are not used anywhere here.
"""
import pytest

from date_parser import parse_expiration_date
from date_parser.keywords import nearest_keyword_is_exclude


def _parse(*lines):
    boxes = [
        {"text": t, "confidence": 0.9, "bbox": [[0, 40 * i], [300, 40 * i], [300, 40 * i + 30], [0, 40 * i + 30]]}
        for i, t in enumerate(lines)
    ]
    return parse_expiration_date(boxes)["final_date"]


# Rule 1: a box holding both kinds of keyword is judged by the keyword nearest
# the date, and a year-month reading competes on its month end.
@pytest.mark.parametrize("lines, expected", [
    (["크 삼각형 한 면 기준※2 롱 래시,워터프루프 효과 시험2SC안티에이징랩 2023.06.19~21만",
      "[용량]4.5g [제조번호]별도표기 [사용기한]2029년 04월"], "2029-04-NONE"),   # 900130
    (["MFD20260611E", "제조번호 및 사용기한별도표기끝xP20290610지"], "2029-06-10"),  # 900082 (subset)
    # a box naming its own date as a manufacture date keeps losing to a real expiry date
    (["HFG 2024.05.07제조", "EXP 2026.11.06"], "2026-11-06"),
    (["제조일자 2025.01.01", "소비기한 2027.01.01"], "2027-01-01"),
])
def test_rule1_in_box_keyword_and_partial_recency(lines, expected):
    assert _parse(*lines) == expected


@pytest.mark.parametrize("text, span, expected", [
    ("[제조번호]별도표기 [사용기한]2029년 04월", (19, 27), False),
    ("MFD 2025.01.01 EXP 2027.01.01", (4, 14), True),    # tie: the preceding MFD wins
    ("MFD 2025.01.01 EXP 2027.01.01", (19, 29), False),
    ("2024.05.07제조 EXP", (0, 10), True),
])
def test_nearest_keyword_is_exclude(text, span, expected):
    assert nearest_keyword_is_exclude(text, span) is expected


# Rule 2: a lot code or a clock time right before a full date is not a date
# field (a four-digit year is never the middle field).
@pytest.mark.parametrize("lines, expected", [
    (["E26L1 2027.11.25 까지"], "2027-11-25"),                  # 900089
    (["260729–01TJ32 10:12 2029.07.29 "], "2029-07-29"),       # 900095
    # unchanged readings with a real day-month-year order
    (["17/09/2027"], "2027-09-17"),
    (["30.12,2021"], "2021-12-30"),
])
def test_rule2_year_never_middle(lines, expected):
    assert _parse(*lines) == expected


# Rule 3: six digits (YYYYMM / YYMMDD) only as a last resort and only with
# expiry evidence (keyword in the same or an adjacent box, or a glued 까지 fragment).
@pytest.mark.parametrize("lines, expected", [
    (["SALE", "FEB", "EXP", "202812"], "2028-12-NONE"),      # 900132
    (["PS241018", "271017마지"], "2027-10-17"),              # 900028
    (["271017까지"], "2027-10-17"),
    # no evidence: lot codes and barcode fragments stay out
    (["PS241018"], "NONE"),
    (["8 809576 260618"], "NONE"),
    # a cut-off full date is not a six-digit date (food 000160, original 512)
    (["R0:202504/28", "EXP:2026/102"], "NONE"),
    # a spaced "가지" is not a glued 까지 fragment (food 000749, OCR misread)
    (["202602 가지스31"], "NONE"),
    # never used when another candidate exists
    (["EXP 2027.03.01", "202812"], "2027-03-01"),
])
def test_rule3_six_digit_fallback(lines, expected):
    assert _parse(*lines) == expected


# Rule 4: when every candidate is named as a manufacture / packaging date, the
# expiration date was not read: no candidate (the cascade moves on; NONE if
# nothing else is found) instead of reporting the manufacture date.
@pytest.mark.parametrize("lines, expected", [
    (["DE68S", "HFG 2024.05.07제조", "EXP 2026.106 ", "8 809576 260618"], "NONE"),   # 900026, 1st stage
    (["PROD 02/2021"], "NONE"),
    (["제조2020.11.27"], "NONE"),                                                       # food 001686, 1st stage
    # unchanged when a non-manufacture candidate exists
    (["PROD 02/2021", "EXP 02/2024"], "2024-02-NONE"),
])
def test_rule4_manufacture_only_is_no_candidate(lines, expected):
    assert _parse(*lines) == expected


# Rule 5: "제조일로부터 N년/개월/일(까지)" with only a manufacture date read:
# expiry = manufacture date + period (month end clamped).
@pytest.mark.parametrize("lines, expected", [
    (["제조2020.11.27", "직품의유성자제소금유통기한", "제조일로부터 5년까지"], "2025-11-27"),  # food 001686
    (["제조일자 2025.01.10", "제조일로부터 18개월까지"], "2026-07-10"),
    (["제조일자 2025.01.10", "제조일로부터 90일"], "2025-04-10"),
    (["제조 2024.02.29", "제조일로부터 1년"], "2025-02-28"),
    (["제조 2026년 02월", "제조일로부터 5년"], "2031-02-NONE"),
    # a printed expiry date still wins over a computed one
    (["제조2020.11.27", "유통기한 2025.11.20", "제조일로부터 5년까지"], "2025-11-20"),
    # no period stated: rule 4 applies
    (["제조2020.11.27"], "NONE"),
])
def test_rule5_period_from_manufacture(lines, expected):
    assert _parse(*lines) == expected
