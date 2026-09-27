"""Textual boundary rules, with no image/product-specific fixtures."""
import pytest
from date_parser import parse_expiration_date
from date_parser.extract import extract_hint_month_name_year_tokens

def parse(*texts):
    return parse_expiration_date([{'text':t,'confidence':.95,'bbox':[[0,40*i],[200,40*i],[200,40*i+20],[0,40*i+20]]} for i,t in enumerate(texts)])['final_date']

@pytest.mark.parametrize('text,result',[
    ('2027,05,123','2027-05-12'),
    ('2028, 04, 097','2028-04-09'),
    ('EXP:2026,12,25','2026-12-25'),
    ('2027,02,301','2027-02-NONE'),
    ('2027,05,1234','2027-05-NONE'),
    ('2027,05,12','2027-05-12'),
])
def test_comma_year_uses_existing_noise_and_calendar_rules(text,result):
    assert parse(text)==result

@pytest.mark.parametrize('text',['1,234,567','20,000,000','12345,67,890','LOT12345,67,890','9999,00,111','2027,,05,,12'])
def test_comma_numeric_code_and_malformed_negatives(text):
    assert parse(text)=='NONE'

@pytest.mark.parametrize('text,result',[
    ('09.1523시A7','NONE-09-15'),
    ('02. 2800시','NONE-02-28'),
    ('12.3105시F2','NONE-12-31'),
    ('02.3005시','NONE-02-NONE'),
])
def test_yearless_date_hour_boundary(text,result):
    assert parse(text)==result

@pytest.mark.parametrize('text',['09.1524시','09.1599시','09.1505','09.1505mg','LOT 09.1505시','LOT09.1505시','13.1505시','00.1505시','09.0005시','09.3205시'])
def test_hour_requires_explicit_unit_valid_fields_and_start_boundary(text):
    assert parse(text)=='NONE'

def test_full_date_still_suppresses_yearless_fallback():
    assert parse('2027.05.12','09.1505시')=='2027-05-12'

@pytest.mark.parametrize('text,result',[
    ('JANq2028','2028-01-NONE'),
    ('-MAYx2027','2027-05-NONE'),
    ('[SEPz2029]','2029-09-NONE'),
    ('octb2030','2030-10-NONE'),
])
def test_named_month_separator_requires_printed_hint(text,result):
    assert parse(text,'Month/Year')==result
    assert parse(text)=='NONE'

@pytest.mark.parametrize('text',['BATCH JANq2028','JANqq2028','JANq202','JANq20281','JANq9999','JANq2028A','JANq20O8',''])
def test_named_month_no_digit_repair_or_lot_contamination(text):
    assert parse(text,'Month/Year')=='NONE'

def test_month_candidate_span_not_duplicated():
    text='JANq2028'
    assert extract_hint_month_name_year_tokens(text,[(0,len(text))])==[]

def test_compact_month_day_year_is_not_a_corrupted_separator():
    assert parse('JAN72028','Month/Year')=='2028-01-07'
    assert extract_hint_month_name_year_tokens('JAN72028',[])==[]

def test_empty_all_none_contract():
    assert parse_expiration_date([])==dict(year='NONE',month='NONE',day='NONE',final_date='NONE')

def test_two_digit_year_order_unchanged():
    assert parse('17.12.20')=='2017-12-20'
