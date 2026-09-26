"""Bounded, oracle-only cascade diagnostic; never changes production selection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STAGES = ("original_512", "rotation_270", "highres_1024", "clahe")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def image_id(value):
    if not value.strip().isdigit():
        raise ValueError(f"Invalid image ID: {value!r}")
    return str(int(value)).zfill(6)


def summarize(stages, truth):
    stop = next((i for i, s in enumerate(stages) if s["has_candidate"]), None)
    normal = stages[0 if stop is None else stop]["prediction"]["final_date"]
    hits = [i for i, s in enumerate(stages) if s["prediction"]["final_date"] == truth]
    later = [i for i in hits if stop is not None and i > stop]
    extra = stages[stop + 1:] if stop is not None else []
    return {
        "truth_final_date": truth, "normal_final_date": normal,
        "normal_stopping_stage": STAGES[stop] if stop is not None else "original_no_candidate",
        "normal_last_attempted_stage": STAGES[stop] if stop is not None else STAGES[-1],
        "normal_correct": normal == truth,
        "any_later_matches_truth": bool(later),
        "first_matching_stage": STAGES[hits[0]] if hits else "NONE",
        "first_later_matching_stage": STAGES[later[0]] if later else "NONE",
        "later_matching_stages": "|".join(STAGES[i] for i in later),
        "oracle_outcome": "preserve" if normal == truth else "fix" if later else "fail",
        "extra_stage_attempts": len(extra),
        "extra_stage_sec": sum(s["stage_sec"] for s in extra),
        "extra_ocr_sec": sum(s["ocr_sec"] for s in extra),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, required=True, help="Original artifact CSV containing image_id")
    ap.add_argument("--filter-column", help="Optional exact-match selection on original artifact")
    ap.add_argument("--filter-value")
    ap.add_argument("--labels", type=Path, required=True, help="Current truth CSV: image_id/year/month/day/final_date")
    ap.add_argument("--images", type=Path, default=ROOT / "data")
    ap.add_argument("--output", type=Path, required=True, help="New output directory, or matching run to resume")
    ap.add_argument("--cpu-threads", type=int, choices=(1, 2, 4), default=2)
    ap.add_argument("--run-ocr", action="store_true", help="Otherwise preflight only; never initializes OCR")
    args = ap.parse_args()
    if (args.filter_column is None) != (args.filter_value is None):
        ap.error("Both filter arguments must be supplied together")
    cases = read_csv(args.cases)
    if args.filter_column:
        cases = [r for r in cases if r[args.filter_column] == args.filter_value]
    ids = [image_id(r["image_id"]) for r in cases]
    if len(ids) != 26 or len(set(ids)) != 26:
        raise ValueError(f"Require exactly 26 unique artifact IDs; got {len(ids)} rows/{len(set(ids))} IDs")
    from evaluation_common import normalize_truth
    labels = {}
    for row in read_csv(args.labels):
        key = image_id(row["image_id"])
        if key not in ids:
            continue
        if key in labels:
            raise ValueError(f"Duplicate truth for {key}")
        if "review_status" in row and (row["review_status"] != "approved" or row.get("master_eligible") != "true"):
            raise ValueError(f"Unapproved review truth: {key}; do not substitute candidate_* fields")
        labels[key] = normalize_truth(row)
    images = {}
    for path in args.images.iterdir():
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"} and path.stem.isdigit():
            key = image_id(path.stem)
            if key in ids:
                if key in images:
                    raise ValueError(f"Ambiguous image: {key}")
                images[key] = path
    if set(labels) != set(ids) or set(images) != set(ids):
        raise ValueError(f"Missing truth: {set(ids)-set(labels)}; missing images: {set(ids)-set(images)}")
    manifest = {
        "version": 1, "cases_path": str(args.cases.resolve()), "cases_sha256": sha(args.cases),
        "filter": [args.filter_column, args.filter_value], "labels_path": str(args.labels.resolve()),
        "labels_sha256": sha(args.labels), "image_sha256": {k: sha(images[k]) for k in ids},
        "image_ids": ids, "cpu_threads": args.cpu_threads,
        "models": ["PP-OCRv6_medium_det", "korean_PP-OCRv5_mobile_rec"],
        "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                        [Path(__file__), ROOT / "ocr_pipeline.py", *sorted((ROOT / "date_parser").glob("*.py"))]},
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if not args.run_ocr:
        print("Preflight passed: exactly 26 cases. Add --run-ocr to execute.")
        return
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.cpu_threads)
    import numpy as np
    from PIL import Image
    import ocr_pipeline as pipeline
    from date_parser.select import select_final_date
    from date_parser.types import TextBox
    manifest["packages"] = {p: importlib.metadata.version(p) for p in ("paddleocr", "paddlepaddle", "numpy", "Pillow")}
    manifest["weights_sha256"] = {str(p.relative_to(ROOT)): sha(p) for name in manifest["models"]
        for p in sorted((ROOT / "weights/paddleocr" / name).glob("inference.*"))}
    manifest_path = args.output / "manifest.json"
    if args.output.exists():
        if not manifest_path.exists() or json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Output exists without matching provenance; choose a new directory")
    else:
        args.output.mkdir(parents=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    engine = None
    rows = []
    stage_rows = []
    initialized_sec = 0.0
    fresh_attempts = 0
    for key in ids:
        rgb = pipeline.decode_image(images[key])
        base = pipeline.resize_image(rgb, 512)
        prepares = (lambda: base, lambda: np.asarray(Image.fromarray(base).rotate(270, expand=True)),
                    lambda: pipeline.resize_image(rgb, 1024), lambda: pipeline.apply_clahe(base))
        stages = []
        for index, (name, prepare) in enumerate(zip(STAGES, prepares)):
            cache = args.output / f"{key}_{name}.json"
            reused = cache.exists()
            if reused:
                record = json.loads(cache.read_text(encoding="utf-8"))
            else:
                if engine is None:
                    start = time.perf_counter()
                    engine = pipeline.initialize_engine(cpu_threads=args.cpu_threads)
                    initialized_sec = time.perf_counter() - start
                start = time.perf_counter()
                array = prepare()
                ocr_start = time.perf_counter()
                detections = pipeline.paddle_to_common(engine.predict(array,
                    text_det_limit_side_len=1024 if index == 2 else 512,
                    text_det_limit_type="max", text_det_box_thresh=0.7))
                end = time.perf_counter()
                record = {"stage": name, "detections": detections,
                          "ocr_sec": end - ocr_start, "stage_sec": end - start}
                with cache.open("x", encoding="utf-8") as stream:
                    json.dump(record, stream, ensure_ascii=False, indent=2)
                fresh_attempts += 1
            detections = record["detections"]
            prediction = pipeline.parse_expiration_date(detections)
            if all(prediction[f] == "NONE" for f in ("year", "month", "day")) and prediction["final_date"] != "NONE":
                raise AssertionError("Parser violated all-NONE output contract")
            candidate = select_final_date([TextBox.from_dict(d) for d in detections])
            record.update(prediction=prediction, has_candidate=pipeline.has_candidate(detections),
                          selected_candidate=repr(candidate), reused=reused)
            stages.append(record)
            stage_rows.append({"image_id": key, "truth_final_date": labels[key]["final_date"],
                "stage": name, **prediction, "has_candidate": record["has_candidate"],
                "selected_candidate": repr(candidate), "ocr_text": " | ".join(d["text"] for d in detections),
                "ocr_sec": record["ocr_sec"], "stage_sec": record["stage_sec"], "reused": reused})
        row = {"image_id": key, **summarize(stages, labels[key]["final_date"])}
        row.update({name: stages[i]["prediction"]["final_date"] for i, name in enumerate(STAGES)})
        rows.append(row)
        print(f"[{len(rows)}/26] {key}: {row['oracle_outcome']}; first later match={row['first_later_matching_stage']}", flush=True)
    for filename, records in (("cases.csv", rows), ("stages.csv", stage_rows)):
        with (args.output / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    summary = {
        "count": 26, "recoverable_later": sum(r["oracle_outcome"] == "fix" for r in rows),
        "not_recoverable_wrong": sum(r["oracle_outcome"] == "fail" for r in rows),
        "already_correct": sum(r["normal_correct"] for r in rows),
        "no_later_truth_match": sum(not r["any_later_matches_truth"] for r in rows),
        "normal_stage1_stops": sum(r["normal_stopping_stage"] == STAGES[0] for r in rows),
        "recoveries": {r["image_id"]: r["later_matching_stages"] for r in rows if r["oracle_outcome"] == "fix"},
        "extra_stage_attempts": sum(r["extra_stage_attempts"] for r in rows),
        "extra_stage_sec": sum(r["extra_stage_sec"] for r in rows),
        "extra_ocr_sec": sum(r["extra_ocr_sec"] for r in rows),
        "fresh_attempts_this_invocation": fresh_attempts, "initialization_sec_this_invocation": initialized_sec,
        "caveat": "Oracle recoverability only, not a deployable selector or unbiased accuracy estimate. Timings include cached original measurements, exclude decode/parser/init, and are sequential CPU costs, not parallel wall time.",
        "recommendation": "Do not change production yet. Inspect partial vs complete wrong candidates; if partial cases recover, test one highres retry for partial candidates on these cases plus a small independent correct partial-candidate control set. Measure regressions and cost before expanding.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
