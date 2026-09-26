from date_parser.hints import DMY, MONTH_YEAR, detect_format_hint


def test_dmy_hints():
    for text in ["(읽는법: 일,월,년순)", "EXP: 일.월.년순", "Best before (dd/mm/yyyy)", "DD.MM.YY", "D0/MM/YYY"]:
        assert detect_format_hint([text]) == DMY, text


def test_month_year_hints():
    for text in ["Mindestens haltbar bis Ende: Monat/Jahr", "MM/YY", "Month/Year", "Best before:월/년의 01일까지"]:
        assert detect_format_hint([text]) == MONTH_YEAR, text


def test_no_hint():
    for text in ["소비기한 2026.01.02까지", "년 월 일", "EXP 26.06.20", "MMA"]:
        assert detect_format_hint([text]) is None, text
