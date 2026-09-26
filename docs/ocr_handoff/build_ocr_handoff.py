"""Build the OCR handoff table: misses that remain after the Parser fixes.

Input : docs/run701/ocr_dump.jsonl (real v6 OCR, cascade mode, 701 images)
        docs/parser_review/replay/after_T7.csv (new parser replayed on it)
        docs/parser_review/labels_701.csv, labels/review/label_review.csv (notes)
Output: docs/ocr_handoff/ocr_issues_v6.csv
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

# Misses that are not OCR problems (kept out of the OCR list).
PARSER_OR_LABEL = {
    # parser: 2-digit day-month-year order, 4-digit month/day order, other held-back cases
    "1862", "1975", "2034", "2211", "2604", "2726", "2917", "1940", "2552",
    "143", "844", "1035", "2829", "2586", "106", "2066", "3011",
    # OCR text is right, parser still misses it (parser follow-up candidates)
    "777", "1932", "1964", "2899", "1714",
    # label to re-check
    "1227",
}

# Type per image, assigned by reading the saved OCR text of every stage.
T_NOT_FOUND = "D1 날짜 글자를 찾지 못함(검출 실패)"
T_DIGIT = "D2 날짜는 읽었지만 숫자 오인식"
T_PARTIAL = "D3 날짜 일부만 읽음(잘림·누락)"
T_MONTH = "D4 영문 월 오인식"
T_OTHER = "D5 소비기한 줄을 놓치고 다른 날짜만 읽음"
T_EARLY = "D6 1단계가 틀린 날짜로 멈춤(뒤 단계는 정답을 읽음)"
TYPE = {}
for i in "50 90 157 232 415 438 440 468 486 648 694 722 795 859 862 877 933 978 1103 1270 1646 1689 1749 1802 1852 1925 1979 1988 2019 2209 2241 2278 2404 2492 2766 2814 2928 3014 3326 3351".split():
    TYPE[i] = T_NOT_FOUND
for i in "190 214 231 233 266 406 553 582 604 643 673 726 749 820 848 1214 1515 1539 1587 1658 1858 1860 2125 2152 2336 2436 2524 3218 3338 3350 832 2068 770".split():
    TYPE[i] = T_DIGIT
for i in "368 423 473 994 1855 2148 2227 2588 2589 2704 3016 3311 3312 763".split():
    TYPE[i] = T_PARTIAL
for i in "1000 1868 1900 1947 1948 1955".split():
    TYPE[i] = T_MONTH
for i in "386 422 466 1548 2328".split():
    TYPE[i] = T_OTHER
TYPE["965"] = T_EARLY

DATE_LIKE = re.compile(r"\d{1,4}\s*[.\-/,:·×xX]\s*\d{1,2}|\d{6,8}|(?i:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC|EXP|BEST|BB|까지|기한)")

replay = {r["image_id"]: r for r in csv.DictReader(open(HERE.parent / "parser_review/replay/after_T7.csv", encoding="utf-8-sig"))}
labels = {r["image_id"]: r for r in csv.DictReader(open(HERE.parent / "parser_review/labels_701.csv", encoding="utf-8-sig"))}
review = {r["image_id"]: r for r in csv.DictReader(open(ROOT / "labels/review/label_review.csv", encoding="utf-8-sig"))}
dump = {}
for line in open(HERE.parent / "run701/ocr_dump.jsonl", encoding="utf-8"):
    r = json.loads(line)
    dump[r["image_id"]] = r

rows = []
for i, r in sorted(replay.items(), key=lambda kv: int(kv[0])):
    if r["correct"] == "True" or i in PARSER_OR_LABEL:
        continue
    assert i in TYPE, f"unclassified OCR miss {i}"
    stages = dump[i]["stages"]
    per_stage = {}
    for s in stages:
        texts = [d["text"] for d in s["detections"] if DATE_LIKE.search(d["text"])]
        per_stage[s["stage"]] = " | ".join(texts) if texts else "(날짜 관련 글자 없음)"
    note = review[i]["existing_300_raw_notes"] if review[i]["existing_300_raw_notes"] not in ("", "[]") else ""
    rows.append({
        "image_id": i,
        "file_name": dump[i]["file"],
        "type": TYPE[i],
        "truth": r["truth"],
        "prediction": r["pred"],
        "cascade_result_stage": r["method"],
        "stages_run": len(stages),
        "ocr_original_512": per_stage.get("original_512", "(실행 안 됨)"),
        "ocr_rotation_270": per_stage.get("rotation_270", "(실행 안 됨: 앞 단계에서 멈춤)"),
        "ocr_highres_1024": per_stage.get("highres_1024", "(실행 안 됨: 앞 단계에서 멈춤)"),
        "ocr_clahe": per_stage.get("clahe", "(실행 안 됨: 앞 단계에서 멈춤)"),
        "label_source": labels[i]["truth_source"] + "/" + labels[i]["label_sources"],
        "labeler_note": note,
    })
missing = set(TYPE) - {r["image_id"] for r in rows}
assert not missing, missing
with open(HERE / "ocr_issues_v6.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
from collections import Counter
print(len(rows), Counter(r["type"] for r in rows))
print("stage1 stopped on a wrong date:", sum(r["cascade_result_stage"] == "original_512" for r in rows))
