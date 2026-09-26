"""Re-run the date parser on saved OCR output, reproducing the submission cascade.

The submission (ocr_pipeline.predict_image) tries four OCR stages in order and
stops at the first stage whose detections give a parser candidate
(select_final_date is not None); if none does, it returns the first stage's
result. OCR output for every stage that ran is stored in ocr_dump.jsonl, so a
parser change can be re-scored without running OCR again.

If a parser change would make the cascade continue past the last stage that was
actually run and saved, that image is reported as "unknown" instead of guessed.

Usage:
    python scripts/replay_parser.py docs/run701/ocr_dump.jsonl docs/parser_review/labels_701.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STAGE_ORDER = ["original_512", "rotation_270", "highres_1024", "clahe"]


def load_dump(path):
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            records[str(int(r["image_id"]))] = r
    return records


def load_truth(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {str(int(r["image_id"])): r for r in csv.DictReader(f)}


def replay_one(record, parse, has_candidate):
    """Return (prediction, method) or (None, 'unknown')."""
    stages = {s["stage"]: s["detections"] for s in record["stages"]}
    first = None
    for name in STAGE_ORDER:
        if name not in stages:
            return None, "unknown"
        detections = stages[name]
        prediction = parse(detections)
        if first is None:
            first = prediction
        if has_candidate(detections):
            return prediction, name
    return first, "original_no_candidate"


def replay(dump, truth, parse, has_candidate):
    rows = []
    for image_id, record in dump.items():
        prediction, method = replay_one(record, parse, has_candidate)
        t = truth.get(image_id)
        rows.append({
            "image_id": image_id,
            "method": method,
            "pred": prediction["final_date"] if prediction else "UNKNOWN",
            "truth": t["final_date"] if t else "",
            "correct": bool(prediction and t and prediction["final_date"] == t["final_date"]),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("labels")
    ap.add_argument("--out", help="write per-image CSV here")
    args = ap.parse_args()

    from date_parser import parse_expiration_date
    from date_parser.select import select_final_date
    from date_parser.types import TextBox

    def has_candidate(detections):
        return select_final_date([TextBox.from_dict(d) for d in detections]) is not None

    rows = replay(load_dump(args.dump), load_truth(args.labels), parse_expiration_date, has_candidate)
    n = len(rows)
    ok = sum(r["correct"] for r in rows)
    unknown = sum(r["method"] == "unknown" for r in rows)
    print(f"correct {ok}/{n} = {100 * ok / n:.2f}%  unknown(cascade needs unsaved stage) {unknown}")
    if args.out:
        with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: int(r["image_id"])))


if __name__ == "__main__":
    main()
