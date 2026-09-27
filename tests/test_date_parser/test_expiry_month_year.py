"""Expiry-labelled month/year fields; no image IDs or reference truths."""
import pytest
from date_parser import parse_expiration_date
from date_parser.extract import extract_date_tokens


def box(text):
    return {'text':text,'confidence':.95,'bbox':[[0,0],[100,0],[100,20],[0,20]]}


@pytest.mark.parametrize('text,expected',[
    ('Exp.Date: 12/22','2022-12-NONE'),
    ('EXP 05/27','2027-05-NONE'),
    ('Expiry Date: 9 / 28','2028-09-NONE'),
    ('expiration: 03/26','2026-03-NONE'),
    ('  exp. date 11/25  ','2025-11-NONE'),
])
def test_expiry_label_slash_month_year(text,expected):
    result=parse_expiration_date([box(text)])
    assert result['final_date']==expected
    assert result['day']=='NONE'


@pytest.mark.parametrize('text',[
    '12/22','Batch: 12/22','MFD: 12/22','EXP 12/22 LOT7',
    'EXP 12/22/27','EXP 12.22','EXP 12/222','EXPERIMENT 12/22',
    'PROD 12/22 EXP','EXP 12/22mg',
])
def test_no_implicit_month_year_outside_standalone_expiry_field(text):
    assert not any(t.fixed_roles==('month','year') and len(t.fields[1].raw)==2
                   for t in extract_date_tokens(text))


@pytest.mark.parametrize('text',['EXP 00/27','EXP 13/27','EXP 05/99','EXP 05/14'])
def test_invalid_month_year_preserves_none(text):
    assert parse_expiration_date([box(text)])==dict(year='NONE',month='NONE',day='NONE',final_date='NONE')


def test_expiry_field_does_not_apply_to_other_boxes():
    assert parse_expiration_date([box('EXP'),box('12/22')])['final_date']=='NONE'


def test_ambiguous_numeric_order_not_changed():
    assert parse_expiration_date([box('17.12.20')])['final_date']=='2017-12-20'


def test_complete_date_not_truncated():
    assert parse_expiration_date([box('EXP 2027/05/21')])['final_date']=='2027-05-21'
