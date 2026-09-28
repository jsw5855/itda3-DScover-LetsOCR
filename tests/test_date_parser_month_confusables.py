from date_parser import parse_expiration_date
from date_parser.types import TextBox


def parse(text):
    return parse_expiration_date([
        TextBox(
            text=text,
            confidence=0.9,
            bbox=[[0, 0], [100, 0], [100, 20], [0, 20]],
        )
    ])["final_date"]


def test_month_name_date_normal():
    assert parse("11 Oct 2021") == "2021-10-11"


def test_day_i_and_zero_oct():
    assert parse("1i 0ct/2021") == "2021-10-11"


def test_glued_zero_oct():
    assert parse("280ct2023") == "2023-10-28"


def test_aug_g_as_six():
    assert parse("10 AU6 2022") == "2022-08-10"


def test_month_confusable_does_not_enable_arbitrary_text():
    assert parse("2026.01.2512:427m1U1") != "2021-07-11"
