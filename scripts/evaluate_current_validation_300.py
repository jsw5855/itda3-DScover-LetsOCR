"""Explicitly invoked, fresh 300-image evaluation of the current submission code."""
from __future__ import annotations

from time import perf_counter
PROCESS_STARTED = perf_counter()

import argparse
import csv
import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIELDS = ["year", "month", "day", "final_date"]


from scripts.evaluation_common import add_dataset_arguments, evaluation_dataset, normalize_truth, score_prediction


def metrics(rows):
    return {"count": len(rows), **{
        field + "_accuracy": sum(r[field + "_correct"] for r in rows) / len(rows) if rows else None
        for field in FIELDS}}


def write_csv(path, rows, columns):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run fresh inference on the selected dataset")
    parser.add_argument("--labels", type=Path, default=ROOT / "labels/labels_300.csv")
    parser.add_argument("--images", type=Path, default=ROOT / "data")
    parser.add_argument("--cpu-threads", type=int, default=4, help="Pipeline CPU threads; historical v6 baseline used 2")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths refused")
    add_dataset_arguments(parser)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if not args.run:
        parser.error("--run is required; no labels have been read")
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    names, expected, dataset = evaluation_dataset(args)

    import ocr_pipeline
    from scripts.benchmark_random_100 import TimedEngine, STAGES, METHODS

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "sample_used.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    sources = [ROOT / "ocr_pipeline.py", *sorted((ROOT / "date_parser").rglob("*.py"))]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "labels_path": str(args.labels.resolve()),
        "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "images_path": str(args.images.resolve()), "images_in_order": names,
        "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "packages": {n: importlib.metadata.version(n) for n in ["paddleocr", "paddlepaddle", "paddlex", "numpy", "pillow", "opencv-contrib-python"]},
        "cache_reused": False, "warmup": False, "dataset": dataset,
        "engine_options": {"cpu_threads": args.cpu_threads, "enable_mkldnn": True, "recognition_batch_size": 6},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started = perf_counter()
    engine = TimedEngine(ocr_pipeline.initialize_engine(cpu_threads=args.cpu_threads))
    init_sec = perf_counter() - started
    records = []
    # Only image paths are passed to inference; truth is used after all predictions.
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for index, name in enumerate(names, 1):
            engine.times = []
            started = perf_counter()
            prediction, method = ocr_pipeline.predict_image(engine, args.images / name)
            elapsed = perf_counter() - started
            record = {"file_name": name, "image_id": Path(name).stem,
                      **prediction, "method": method, "elapsed_sec": elapsed,
                      **{s + "_predict_sec": t for s, t in zip(STAGES, engine.times)}}
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{index}/{len(names)}] {name}: {method} {elapsed:.3f}s", flush=True)
    started = perf_counter()
    write_csv(args.output / "submission_300.csv",
              [{k: r[k] for k in ocr_pipeline.COLUMNS} for r in records], ocr_pipeline.COLUMNS)
    submission_csv_sec = perf_counter() - started
    through_submission_sec = perf_counter() - PROCESS_STARTED
    evaluated = []
    for record in records:
        truth = expected[record["file_name"]]
        none_count = sum(truth[k] == "NONE" for k in FIELDS[:3])
        evaluated.append({**record, "truth_group": "all_none" if none_count == 3 else "partial_none" if none_count else "complete",
                          **{"true_" + k: truth[k] for k in FIELDS},
                          **score_prediction(record, truth)})
    columns = ["file_name", *ocr_pipeline.COLUMNS, "method", "elapsed_sec",
               *[s + "_predict_sec" for s in STAGES], "truth_group",
               *["true_" + k for k in FIELDS], *[k + "_correct" for k in FIELDS]]
    started = perf_counter()
    write_csv(args.output / "evaluation_300.csv", evaluated, columns)
    write_csv(args.output / "failures.csv", [r for r in evaluated if not r["final_date_correct"]], columns)
    evaluation_csv_sec = perf_counter() - started
    infer_sec = sum(r["elapsed_sec"] for r in records)
    summary = {
        "status": "complete", "new_inferences": len(names), "sample_fingerprint": dataset["sample_fingerprint"], "cached_inferences": 0,
        "metrics": metrics(evaluated),
        "by_truth_group": {g: metrics([r for r in evaluated if r["truth_group"] == g]) for g in ["complete", "partial_none", "all_none"]},
        "init_sec": init_sec, "infer_sec": infer_sec, "avg_sec_per_image": infer_sec / len(names),
        "submission_csv_sec": submission_csv_sec, "evaluation_csv_sec": evaluation_csv_sec,
        "script_entry_through_submission_csv_sec": through_submission_sec,
        "estimated_3352_init_and_infer_min": (init_sec + infer_sec / len(names) * 3352) / 60,
        "timing_note": "Projection excludes 3352-row CSV and other overhead; not proof of the 2400-second limit. Script clock starts at entry, excludes interpreter startup, includes validation setup and JSONL diagnostics. Full submission requires external wall-clock timing through final CSV replacement.",
        "methods": {m: {"count": sum(r["method"] == m for r in records)} for m in METHODS},
        "stage_predict_timings": {},
    }
    for stage in STAGES:
        values = [r[stage + "_predict_sec"] for r in records if stage + "_predict_sec" in r]
        summary["stage_predict_timings"][stage] = {"attempt_count": len(values), "total_sec": sum(values), "avg_sec": sum(values) / len(values) if values else None}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
