from date_parser import parse_expiration_date
from date_parser.types import TextBox


def parse(text):
    return parse_expiration_date([TextBox(text=text, confidence=.99,
        bbox=[[0, 0], [200, 0], [200, 20], [0, 20]])])['final_date']


def test_split_day_in_complete_numeric_box():
    assert parse('2022 .01. 2 2') == '2022-01-22'
    assert parse('2024-02-2 9') == '2024-02-29'


def test_split_day_does_not_join_time_or_lot():
    assert parse('2022.01.2 2:30') == '2022-01-02'
    assert parse('2022.01.2 2 LOT') == '2022-01-02'
    assert parse('2022.01.2 22') == '2022-01-02'
    assert parse('2023.02.2 9') == '2023-02-NONE'


def test_clock_minutes_are_not_date_year():
    assert parse('L021937076 22:33 14.03.21') == '2021-03-14'
    assert parse('22:33:59 14.03.21') == '2021-03-14'
    assert parse('22:33') == 'NONE'


def test_clock_mask_preserves_colon_dates_and_glued_time():
    assert parse('BE0:24/11/21') == '2024-11-21'
    assert parse('2022.11:02') == '2022-11-02'
    assert parse('2026:07.08') == '2026-07-08'
    assert parse('2026.01.2512:42') == '2026-01-25'
    assert parse('12.18. 10:41') == 'NONE-12-18'


def test_split_month_requires_exact_month_and_date_bounds():
    assert parse('07SE P21') == '2021-09-07'
    assert parse('18 N O V 2023') == '2023-11-18'
    assert parse('07SX P21') == 'NONE'
    assert parse('LOT07SE P21') == 'NONE'
    assert parse('07SE P2100') == 'NONE'


def test_standalone_compact_day_month_with_explicit_year():
    assert parse('1112 2021') == '2021-12-11'
    assert parse('2606 2026') == '2026-06-26'
    assert parse('0229 2024') == '2024-02-29'
    assert parse('1992 2025') == 'NONE'
    assert parse('02)2049-2000') == 'NONE'
    assert parse('LOT 1112 2021') == 'NONE'


def test_retry_does_not_promote_repeated_earlier_candidate():
    from ocr_pipeline import prefer_retry
    old = dict(q=.85, M=True, selected_date='2025-06-30',
               candidate_dates=['2024-12-30', '2025-06-30'])
    new = dict(q=.99, M=False, selected_date='2024-12-30')
    assert not prefer_retry(old, new)
    assert prefer_retry(old, dict(new, self_anchor=True))
    assert prefer_retry(dict(old, self_exclude=True), new)
    assert prefer_retry(old, dict(new, selected_date='2025-06-29'))
    assert prefer_retry(old, dict(new, M=True))
    assert prefer_retry(dict(old, selected_date='2025-06-NONE'), new)
    assert not prefer_retry(dict(old, candidate_dates=['2024-12-NONE', '2025-06-30']), new)
    assert prefer_retry(dict(old, candidate_dates=['2025-06-NONE', '2025-06-30']),
                        dict(new, selected_date='2025-06-29'))


def test_compact_year_month_needs_nearby_standalone_expiry_label():
    def box(text, x, y=0):
        return TextBox(text=text, confidence=.99, bbox=[[x,y],[x+40,y],[x+40,y+20],[x,y+20]])
    def run(boxes):
        return parse_expiration_date(boxes)['final_date']
    assert run([box('EXP:',0),box('202603',45)]) == '2026-03-NONE'
    assert run([box('202603',45)]) == 'NONE'
    assert run([box('LOT:',0),box('202603',45)]) == 'NONE'
    assert run([box('EXP:',0),box('202603',200)]) == 'NONE'
    assert run([box('EXP:',0),box('202603',45,80)]) == 'NONE'
    assert run([box('EXP:',0),box('202613',45)]) == 'NONE'
    assert run([box('EXP:',0),box('202603',45),box('2026.03.12',100)]) == '2026-03-12'


def test_unambiguous_slash_date_teaches_only_same_format():
    from date_parser.select import find_all_candidates
    def candidates(*texts):
        return [c.result.final_date_string() for c in find_all_candidates([
            TextBox(text=t, confidence=.99, bbox=[[0,0],[100,0],[100,20],[0,20]]) for t in texts])]
    assert candidates('29/09/20', '28/09/2022') == ['2020-09-29', '2022-09-28']
    assert candidates('29.09.20', '28/09/2022')[0] == '2029-09-20'
    assert candidates('29/09/20', '08/09/2022')[0] == '2029-09-20'
    assert candidates('29/09/20', '31/02/2022')[0] == '2029-09-20'
    assert candidates('29/09/20', '28/09/2022', '2022/09/28')[0] == '2029-09-20'
    assert candidates('29/09/20', '28/09/2022', '09/28/2022')[0] == '2029-09-20'


def test_separate_day_requires_aligned_close_confident_unique_box():
    def box(text,x,y=0,q=.99):
        return TextBox(text=text,confidence=q,bbox=[[x,y],[x+20,y],[x+20,y+20],[x,y+20]])
    base=TextBox(text='2023.10',confidence=.9,bbox=[[0,0],[100,0],[100,20],[0,20]])
    def run(*others):
        return parse_expiration_date([base,*others])['final_date']
    assert run(box('15',105)) == '2023-10-15'
    assert run(box('15',140)) == '2023-10-NONE'
    assert run(box('15',105,30)) == '2023-10-NONE'
    assert run(box('15',105,q=.5)) == '2023-10-NONE'
    assert run(box('15',105),box('16',108)) == '2023-10-NONE'
    assert run(box('99',105)) == '2023-10-NONE'


def test_retry_uses_real_stage_evidence_without_more_attempts():
    import ocr_pipeline as pipeline
    def detection(text, confidence, y):
        return dict(text=text, confidence=confidence,
                    bbox=[[0,y],[100,y],[100,y+20],[0,y+20]])
    stages = {
        'original_512': pipeline.stage_result([
            detection('2024.12.30', .98, 0), detection('2025.06.30', .85, 40)]),
        'highres_1024': pipeline.stage_result([detection('2024.12.30', .99, 0)]),
    }
    prediction, method, calls = pipeline.run_cascade(stages.__getitem__)
    assert prediction['final_date'] == '2025-06-30'
    assert method == 'original_512_retry_kept'
    assert calls == ['original_512', 'highres_1024']
