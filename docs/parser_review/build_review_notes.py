"""Draft structured review_note text for the 58 rows reviewed by 이수민.

Format (one line per image):
인쇄: … / 판정: … / 근거: … / 기존라벨: … / 오류유형: … / 왜 검수대상·애매: … / Parser: …

Facts come from label_review.csv (both teams' raw labels, final decision,
reviewer's own note), printed_text_58.json and parser_cases_58.csv.
Items marked [확인 필요] need the reviewer's own reasoning.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
rows = list(csv.DictReader(open(ROOT / "labels/review/label_review.csv", encoding="utf-8-sig")))
printed = json.loads((HERE / "printed_text_58.json").read_text(encoding="utf-8"))
cases = {r["image_id"]: r for r in csv.DictReader(open(HERE / "parser_cases_58.csv", encoding="utf-8-sig"))}

REASON = {
    "partial_none": "연·월·일 중 일부가 NONE인 부분 NONE 사례",
    "ambiguous_date_order": "날짜 순서(월/일, 연/일)가 모호하다고 자동 표시",
    "label_conflict": "기존 300 라벨과 신규 라벨이 서로 다름",
    "component_final_mismatch": "신규 라벨의 year/month/day와 final_date가 서로 다름",
    "unusual_year": "연도가 비정상 범위(2001)",
    "all_none": "신규 라벨이 전부 NONE",
    "ocr_visual_review": "이전 이미지 확인 대상(판정 미확정)",
}

# Reviewer's earlier notes that were only copied prior-review text; replaced by the basis below.
STALE_NOTE = {"106", "157", "2521"}

# Image-specific basis / error type / why, where the default wording is not enough.
SPECIFIC = {
    "106": dict(basis="\"10:20 13:53\"에서 앞의 10:20은 월.일(10월 20일), 뒤는 시각. 연도 미인쇄",
                etype="신규 라벨 판독 누락(읽을 수 있는 월·일을 전부 NONE 처리)",
                why="도트 인쇄 + 월·일 구분자가 ':'라 시각처럼 보임"),
    "157": dict(basis="\"08/07/2027\"을 일/월/년 순으로 판정 [확인 필요: 일/월/년으로 본 근거]",
                etype="날짜 순서(DD/MM vs MM/DD) 해석 차이",
                why="08/07은 8월 7일(MM/DD)과 7월 8일(DD/MM) 모두 가능한 날짜"),
    "515": dict(basis="\"2026. 01\"이 소비기한(연.월). 앞의 201045·46은 로트/코드 숫자",
                etype="라벨 오류 없음",
                why="앞쪽 숫자열이 2010으로 시작해 날짜처럼 보일 수 있음"),
    "824": dict(basis="유통기한 줄 \"12.18.\"(12월 18일), 아래 제조일자 줄 \"12.06.\"은 제외. 연도 미인쇄",
                etype="신규 라벨 숫자 오독(18을 19로 기록)",
                why="유통기한·제조일자 두 날짜가 붙어 있고 도트 인쇄의 8/9가 비슷함"),
    "1070": dict(basis="\"01.24 A\"는 요거트 월.일 표기(1월 24일). 연도 미인쇄",
                 etype="신규 라벨 형식 오해(월.일을 월.연도=2024년 1월로 해석)",
                 why="01.24가 MM.DD와 MM.YY 둘 다로 읽힘"),
    "1071": dict(basis="1070과 같은 제품. \"01.24 A\" = 1월 24일",
                 etype="신규 라벨 형식 오해(월.일을 월.연도=2024년 1월로 해석)",
                 why="01.24가 MM.DD와 MM.YY 둘 다로 읽힘 + 과노출"),
    "1205": dict(basis="아래 문구 \"Mindestens haltbar bis Ende: Monat/Jahr\"(월/연도 말까지)가 형식을 명시",
                 etype="라벨 오류 없음(검수 초안 NONE-05-21은 수정함)",
                 why="05.21이 5월 21일(MM.DD)과 2021년 5월(MM.YY) 둘 다로 읽힘"),
    "2034": dict(basis="유럽 제품(Ferrero) DD.MM.YY 관행, 로트 L352(352번째 날=12월 17~18일)가 일.월.년 해석과 일치",
                 etype="라벨 오류 없음",
                 why="17.12.20이 2017-12-20(YY.MM.DD)과 2020-12-17(DD.MM.YY) 둘 다 가능"),
    "2152": dict(basis="한국 제품 연.월.일 순. \"19.06.12 부터\" / \"19.07.11 까지\" 중 '까지' 줄이 소비기한",
                 etype="신규 라벨 final_date 기록 오류(component는 2019-07-11인데 final_date를 2021-07-19로 기록)",
                 why="빛 반사로 일자 11 일부가 가려짐 + 부터/까지 두 날짜"),
    "2521": dict(basis="\"02/10/2023\"을 일/월/년(외국식) 순으로 판정 [확인 필요: 일/월/년으로 본 근거]",
                 etype="신규 라벨 일자 오류(01로 기록)",
                 why="02/10은 2월 10일(MM/DD)과 10월 2일(DD/MM) 모두 가능"),
    "2586": dict(basis="\"Exp.Date: 12/22\" = 2022년 12월(월/2자리 연도)",
                 etype="라벨 오류 없음",
                 why="12/22가 12월 22일(MM/DD)로도 읽힘"),
    "2657": dict(basis="이탈리아 제품 일/월/년. \"01/07/21\" = 2021년 7월 1일",
                 etype="신규 라벨 연도 오류(2자리 연도 21을 2001로 기록)",
                 why="2자리 연도 표기"),
    "2917": dict(basis="한글 표시사항에 \"(읽는법: 일,월,년순)\" 명시 → \"30/04/21\" = 2021년 4월 30일",
                 etype="라벨 오류 없음",
                 why="30/04/21을 연.월.일로 읽으면 2030-04-21이 되어 순서 모호"),
    "2310": dict(basis="\"03.13. 13:47 까지\"가 소비기한, 아래 \"03.01. 부터\"는 제외. 연도 미인쇄",
                 etype="라벨 오류 없음",
                 why="부터/까지 두 날짜 + 시각 숫자"),
    "3277": dict(basis="\"04.05\" = 4월 5일. 연도 미인쇄 [확인 필요: 스티커 오른쪽이 잘려 이후 글자 미확인]",
                 etype="라벨 오류 없음",
                 why="연도 없는 월.일 + 비닐 반사"),
}


def canon(v):
    return "" if v in ("", "[]") else v


def label_status(src_final, final):
    if not src_final:
        return "없음"
    return "맞음" if src_final == final else f"틀림({src_final})"


def default_basis(final):
    y, m, d = final.split("-") if final != "NONE" else ("NONE",) * 3
    if y == "NONE" and m != "NONE":
        return "연도 미인쇄, 월.일만 표기"
    if d == "NONE" and y != "NONE":
        return "일 미인쇄, 월·연도만 표기"
    return "인쇄된 날짜 그대로"


def parser_text(case):
    if case["category"] == "Parser":
        return f"Parser 문제 - {case['detail']} (인쇄 글자 입력 시 현재 Parser 결과: {case['parser_on_printed_text']})"
    if case["category"] == "OCR":
        return f"OCR 추정 - 인쇄 글자로는 Parser 정답, 예선 v6 실제 실행은 {case['v6_baseline_pred']} → 정서현 전달"
    if case["v6_baseline_pred"]:
        return "문제 없음 - 현재 Parser 정답, 예선 v6 실제 실행도 정답"
    return "문제 없음 - 현재 Parser 정답 (예선 v6 실행 기록 없음, OCR 미검증)"


notes = {}
for r in rows:
    i = r["image_id"]
    if r["reviewer"] != "이수민":
        continue
    final = r["final_date"]
    spec = SPECIFIC.get(i, {})
    own = "" if i in STALE_NOTE else r["review_note"].strip()
    if i == "1205":
        own = ""
    basis = spec.get("basis", default_basis(final))
    if own:
        basis += f" (검수 메모: {own})"
    e = label_status(canon(r["existing_300_canonical_final_date"]), final)
    n = label_status(canon(r["incoming_432_canonical_final_date"]), final)
    if i == "2152":  # canonical value was rebuilt from components; the raw final_date was wrong
        n = f"일부 틀림(year/month/day는 맞음, final_date를 {r['incoming_432_raw_final_date']}로 기록)"
    etype = spec.get("etype") or ("라벨 오류 없음" if "틀림" not in e + n else "라벨 값 오류")
    reasons = [REASON[x] for x in r["review_reason"].split(";") if x]
    why = spec.get("why")
    why_text = "; ".join(reasons) + (f" / {why}" if why else "")
    notes[i] = (f"인쇄: {' / '.join(printed[i]['lines'])} / 판정: {final} / 근거: {basis} / "
                f"기존라벨: 기존300 {e}, 신규 {n} / 오류유형: {etype} / 왜 검수대상·애매: {why_text} / "
                f"Parser: {parser_text(cases[i])}")

out = HERE / "review_notes_58_draft.csv"
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image_id", "final_date", "review_note_draft"])
    for i in sorted(notes, key=int):
        w.writerow([i, next(r["final_date"] for r in rows if r["image_id"] == i), notes[i]])
print(len(notes), "notes ->", out)
