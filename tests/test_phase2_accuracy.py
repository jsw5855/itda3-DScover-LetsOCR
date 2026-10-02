"""General syntax and selection boundaries; no image IDs or GT lookups."""
import pytest

from date_parser import parse_expiration_date
import ocr_pipeline as p


def boxes(text, q=.95):
    return [{'text': text, 'confidence': q, 'bbox': [[0, 0], [250, 0], [250, 25], [0, 25]]}]


@pytest.mark.parametrize('text, expected', [
    ('2024FEB29', '2024-02-29'),
    ('2027 sep 07', '2027-09-07'),
    ('2028-OCT-19', '2028-10-19'),
    ('2027,03.14', '2027-03-14'),
    ('2027.03,14', '2027-03-14'),
    ('EXP 2027,03.141지', '2027-03-14'),
    ('270319까-지', '2027-03-19'),
    ('270319까 지', '2027-03-19'),
])
def test_exact_date_syntax(text, expected):
    assert parse_expiration_date(boxes(text))['final_date'] == expected


@pytest.mark.parametrize('text', [
    '12024FEB29', 'A2024FEB29B', '2024FE829',
    '2027,300.14', '2027.300,14', '1234,567.890',
    '270319까-치', '270319가-지', '8800270319까-지',
])
def test_no_new_code_or_spelling_guesses(text):
    assert parse_expiration_date(boxes(text))['final_date'] == 'NONE'


@pytest.mark.parametrize('text, damaged', [
    ('EXP 2028.04.99', True),
    ('EXP 2028.04.0', True),
    ('EXP 2028년 04월', False),
    ('EXP 2028.04', False),
    ('EXP 2028.04.17', False),
])
def test_damaged_day_distinct_from_intentional_partial(text, damaged):
    _, ev = p.stage_result(boxes(text, .99))
    assert ev['damaged_day'] is damaged
    assert p.retry_triggered(ev) is damaged


def evidence(value, q, anchor=False):
    return dict(selected_date=value, q=q, M=False, self_anchor=anchor, self_exclude=False)


def test_retry_cannot_erase_fields_but_can_correct_conflicting_fields():
    original = evidence('2028-04-17', .85)
    assert not p.prefer_retry(original, evidence('2028-04-NONE', .99))
    assert p.prefer_retry(original, evidence('2028-05-NONE', .99))
    assert p.prefer_retry(evidence('2028-04-NONE', .85), evidence('2028-04-17', .99))
    assert not p.prefer_retry(evidence('2028-04-NONE', .99), evidence('2028-04-17', .85))


@pytest.mark.parametrize('labelled_original, labelled_retry, retry_q, expected, extra', [
    (True, True, .96, '2028-04-17', True),
    (True, False, .96, '2028-04-11', True),
    (True, True, .70, '2028-04-11', True),
    (False, True, .96, '2028-04-11', False),
])
def test_empty_highres_requires_explicit_expiry_and_higher_confidence(
        labelled_original, labelled_retry, retry_q, expected, extra):
    stages = {
        'original_512': p.stage_result(boxes(('EXP ' if labelled_original else '') + '2028.04.11', .80)),
        'highres_1024': p.stage_result([]),
        'clahe': p.stage_result(boxes(('EXP ' if labelled_retry else '') + '2028.04.17', retry_q)),
    }
    result, method, attempts = p.run_cascade(stages.__getitem__)
    assert result['final_date'] == expected
    assert attempts == ['original_512', 'highres_1024'] + (['clahe'] if extra else [])
    assert method == ('clahe_retry' if expected == '2028-04-17' else 'original_512_retry_kept')


def test_successful_or_unnecessary_highres_does_not_add_contrast_stage():
    for confidence in (.80, .99):
        original = p.stage_result(boxes('EXP 2028.04.11', confidence))
        stages = {'original_512': original, 'highres_1024': original}
        result, _, attempts = p.run_cascade(stages.__getitem__)
        assert result['final_date'] == '2028-04-11'
        assert attempts == ['original_512'] + (['highres_1024'] if confidence < .90 else [])
