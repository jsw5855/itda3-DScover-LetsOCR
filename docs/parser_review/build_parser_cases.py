"""Build parser_cases_58.csv: run the current date_parser on the text actually
printed on each reviewed image and compare with the approved label.

Input : labels/review/label_review.csv (approved rows reviewed by 이수민)
        docs/parser_review/printed_text_58.json (manual transcription)
        docs/baseline_300_reproduction_*/evaluation/evaluation_300.csv (real v6 run)
Output: docs/parser_review/parser_cases_58.csv
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from date_parser import parse_expiration_date  # noqa: E402

HERE = Path(__file__).resolve().parent
texts = json.loads((HERE / "printed_text_58.json").read_text(encoding="utf-8"))
labels = {r["image_id"]: r for r in csv.DictReader(open(ROOT / "labels/review/label_review.csv", encoding="utf-8-sig"))}
baseline = {}
for p in ROOT.glob("docs/baseline_300_reproduction_*/evaluation/evaluation_300.csv"):
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        baseline[str(int(r["image_id"]))] = r


def run_parser(lines):
    # One OCR box per printed line, stacked top to bottom.
    boxes = [{"text": t, "confidence": 1.0, "bbox": [[0, 40 * i], [300, 40 * i], [300, 40 * i + 30], [0, 40 * i + 30]]}
             for i, t in enumerate(lines)]
    return parse_expiration_date(boxes)["final_date"]


# Parser failure type, assigned by reading each failing case's printed text.
PARSER_TYPE = {
    **{i: "연도 없는 월.일(MM.DD) 미지원" for i in
       ["106", "763", "789", "994", "1070", "1071", "1131", "2051", "2159", "2265", "2310", "2345", "2927", "2928", "3277"]},
    "824": "연도 없는 월.일 + 시각 숫자가 날짜로 섞임(오답 생성)",
    "1205": "월.2자리연도(MM.YY) 미지원 - 형식 안내 문구(Monat/Jahr) 미활용",
    "2586": "월/2자리연도(MM/YY) 미지원",
    "2034": "2자리 연도 순서(DD.MM.YY를 YY.MM.DD로 해석)",
    "2917": "2자리 연도 순서(DD/MM/YY) - 형식 안내 문구(읽는법: 일,월,년순) 미활용",
    "515": "문자 U를 숫자로 보는 패턴이 잘못된 구간을 먼저 차지",
}


def classify(image_id, parser_ok, base):
    if not parser_ok:
        return "Parser", PARSER_TYPE[image_id]
    if base and base["final_date_correct"] != "True":
        return "OCR", "인쇄 글자로는 Parser 정답인데 v6 실제 실행은 오답"
    if base:
        return "문제 없음", "Parser 정답, v6 실제 실행도 정답"
    return "문제 없음", "Parser 정답 (v6 실행 기록 없음: OCR은 미검증)"


rows = []
for image_id, info in sorted(texts.items(), key=lambda kv: int(kv[0])):
    truth = labels[image_id]["final_date"]
    parsed = run_parser(info["lines"])
    base = baseline.get(image_id)
    category, detail = classify(image_id, parsed == truth, base)
    rows.append({
        "image_id": image_id,
        "category": category,
        "detail": detail,
        "truth": truth,
        "printed_text": " / ".join(info["lines"]),
        "parser_on_printed_text": parsed,
        "parser_ok": parsed == truth,
        "v6_baseline_pred": base["final_date"] if base else "",
        "v6_baseline_ok": (base["final_date_correct"] == "True") if base else "",
        "ocr_risk": info["ocr_risk"],
        "review_note": labels[image_id]["review_note"],
    })

with open(HERE / "parser_cases_58.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(r["image_id"], r["truth"], "|", r["parser_on_printed_text"], "OK" if r["parser_ok"] else "XX", "| v6:", r["v6_baseline_pred"], r["v6_baseline_ok"])
