"""Fresh 300-image evaluation of the production cascade with the frozen retry policy.

Runs real OCR through ocr_pipeline.predict_image_detailed (the code predict.ipynb
uses via predict_image), one engine in this process. Records per-image attempted
stages and per-attempt OCR time, so original/rotation/highres/CLAHE attempt counts
include policy retries. Truth is read before inference only to fix the image list
and is used after all predictions. Refuses an existing output directory.
"""
from __future__ import annotations

from time import perf_counter
PROCESS_STARTED = perf_counter()

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIELDS = ["year", "month", "day", "final_date"]
MODELS = ("PP-OCRv6_medium_det", "korean_PP-OCRv5_mobile_rec")

from scripts.evaluation_common import add_dataset_arguments, evaluation_dataset, score_prediction  # noqa: E402


class TimedEngine:
    """Times each predict call; call order equals the attempted stage order."""

    def __init__(self, engine):
        self.engine = engine
        self.times = []

    def predict(self, *args, **kwargs):
        started = perf_counter()
        try:
            return self.engine.predict(*args, **kwargs)
        finally:
            self.times.append(perf_counter() - started)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows, columns):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run fresh inference on the selected dataset")
    parser.add_argument("--labels", type=Path, default=ROOT / "labels/labels_300.csv")
    parser.add_argument("--images", type=Path, default=ROOT / "data")
    parser.add_argument("--cpu-threads", type=int, default=4, help="Accepted 267 baseline evaluation used 4")
    parser.add_argument("--baseline-rows", type=Path,
                        default=ROOT / "docs/final_policy_production_replay/replay_rows.json",
                        help="Saved-OCR replay rows; 'baseline' is the accepted 267 cascade prediction per image")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths refused")
    add_dataset_arguments(parser)
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required")
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    names, expected, dataset = evaluation_dataset(args)
    baseline_rows = json.loads(args.baseline_rows.read_text(encoding="utf8"))

    import ocr_pipeline

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "sample_used.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    sources = [ROOT / "ocr_pipeline.py", Path(__file__), *sorted((ROOT / "date_parser").rglob("*.py"))]
    weights = {f"{m}/{f}": sha(ROOT / "weights/paddleocr" / m / f)
               for m in MODELS for f in ("inference.json", "inference.pdiparams", "inference.yml")}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "git_head": git_head(),
        "python": sys.version, "labels_path": str(args.labels.resolve()), "labels_sha256": sha(args.labels),
        "images_path": str(args.images.resolve()), "images_in_order": names,
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        "models": list(MODELS), "weights_sha256": weights,
        "packages": {n: importlib.metadata.version(n) for n in ["paddleocr", "paddlepaddle", "paddlex", "numpy", "pillow", "opencv-contrib-python"]},
        "policy": {"retry_q_threshold": ocr_pipeline.RETRY_Q_THRESHOLD, "retry_stage": "highres_1024",
                   "trigger": "original candidate and (q < threshold or M)", "selector": "strictly higher q; tie keeps original"},
        "cache_reused": False, "warmup": False, "dataset": dataset,
        "engine_options": {"cpu_threads": args.cpu_threads, "enable_mkldnn": True, "recognition_batch_size": 6},
        "baseline_rows_sha256": sha(args.baseline_rows),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    started = perf_counter()
    engine = TimedEngine(ocr_pipeline.initialize_engine(cpu_threads=args.cpu_threads))
    init_sec = perf_counter() - started
    records = []
    infer_started = perf_counter()
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for index, name in enumerate(names, 1):
            engine.times = []
            started = perf_counter()
            prediction, method, attempts = ocr_pipeline.predict_image_detailed(engine, args.images / name)
            elapsed = perf_counter() - started
            if len(engine.times) != len(attempts):
                raise AssertionError(f"{name}: {len(engine.times)} OCR calls for attempts {attempts}")
            record = {"file_name": name, "image_id": Path(name).stem, **prediction, "method": method,
                      "attempts": attempts, "attempt_ocr_sec": engine.times, "elapsed_sec": elapsed}
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{index}/{len(names)}] {name}: {method} {'+'.join(attempts)} {elapsed:.3f}s", flush=True)
    infer_sec = perf_counter() - infer_started
    started = perf_counter()
    write_csv(args.output / "submission_300.csv", [{k: r[k] for k in ocr_pipeline.COLUMNS} for r in records], ocr_pipeline.COLUMNS)
    submission_csv_sec = perf_counter() - started
    through_submission_sec = perf_counter() - PROCESS_STARTED

    evaluated = []
    for record in records:
        truth = expected[record["file_name"]]
        key = f"{int(record['image_id']):06d}"
        baseline = baseline_rows[key]["baseline"]
        evaluated.append({**record, **{"true_" + k: truth[k] for k in FIELDS}, **score_prediction(record, truth),
                          "baseline_final_date": baseline, "baseline_correct": baseline == truth["final_date"]})
    columns = ["file_name", *ocr_pipeline.COLUMNS, "method", "attempts", "elapsed_sec",
               *["true_" + k for k in FIELDS], *[k + "_correct" for k in FIELDS], "baseline_final_date", "baseline_correct"]
    write_csv(args.output / "evaluation_300.csv", evaluated, columns)
    write_csv(args.output / "failures.csv", [r for r in evaluated if not r["final_date_correct"]], columns)

    attempts = Counter(a for r in records for a in r["attempts"])
    stage_time = {s: sum(t for r in records for a, t in zip(r["attempts"], r["attempt_ocr_sec"]) if a == s)
                  for s in ocr_pipeline.STAGES}
    correct = sum(r["final_date_correct"] for r in evaluated)
    summary = {
        "status": "complete", "count": len(records),
        "final_date_correct": correct, "final_date_accuracy": correct / len(records),
        "field_accuracy": {k: sum(r[k + "_correct"] for r in evaluated) / len(evaluated) for k in FIELDS},
        "baseline_correct": sum(r["baseline_correct"] for r in evaluated),
        "gains": sorted(r["image_id"] for r in evaluated if r["final_date_correct"] and not r["baseline_correct"]),
        "regressions": sorted(r["image_id"] for r in evaluated if r["baseline_correct"] and not r["final_date_correct"]),
        "methods": dict(Counter(r["method"] for r in records)),
        "attempt_counts": {s: attempts.get(s, 0) for s in ocr_pipeline.STAGES},
        "attempt_ocr_sec": stage_time,
        "none_none_none_outputs": [r["image_id"] for r in records if r["final_date"] == "NONE-NONE-NONE"],
        "init_sec": init_sec, "infer_sec": infer_sec, "avg_sec_per_image": infer_sec / len(records),
        "submission_csv_sec": submission_csv_sec, "script_entry_through_submission_csv_sec": through_submission_sec,
        "timing_note": "Single process, engine threads as in manifest. Not the 2-process submission layout and not an "
                       "end-to-end notebook timing; interpreter start-up is excluded.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
