from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENGINE = None

FIELDS = ["year", "month", "day", "final_date"]


from scripts.evaluation_common import add_dataset_arguments, evaluation_dataset, normalize_truth, score_prediction


def initialize_worker():
    global ENGINE

    threads = 2
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(threads)

    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    import ocr_performance_variant as variant

    ENGINE = variant.initialize_engine(
        threads=2,
        batch=6,
        mkldnn=True,
    )


def run_one(item):
    index, file_name, image_dir = item

    import ocr_pipeline

    path = Path(image_dir) / file_name

    started = time.perf_counter()
    prediction, method = ocr_pipeline.predict_image(ENGINE, path)
    elapsed = time.perf_counter() - started

    return {
        "index": index,
        "file_name": file_name,
        "image_id": Path(file_name).stem,
        **prediction,
        "method": method,
        "elapsed_sec": elapsed,
        "pid": os.getpid(),
    }


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Parallel 2-process x 2-thread validation of current submission pipeline."
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    add_dataset_arguments(parser)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"Output already exists; use a new directory: {args.output}"
        )

    names, expected, dataset = evaluation_dataset(args)
    args.output.mkdir(parents=True)
    (args.output / "sample_used.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")

    total_started = time.perf_counter()

    ctx = mp.get_context("spawn")

    jobs = [
        (i, name, str(args.images))
        for i, name in enumerate(names)
    ]

    infer_started = time.perf_counter()

    records = []

    with ctx.Pool(
        processes=2,
        initializer=initialize_worker,
    ) as pool:
        for result in pool.imap_unordered(run_one, jobs, chunksize=1):
            records.append(result)

            if len(records) % 20 == 0:
                print(
                    f"[{len(records)}/{len(names)}] "
                    f"{time.perf_counter() - infer_started:.1f}s",
                    flush=True,
                )

    infer_sec = time.perf_counter() - infer_started

    records.sort(key=lambda r: r["index"])

    submission_rows = [
        {
            "image_id": r["image_id"],
            "year": r["year"],
            "month": r["month"],
            "day": r["day"],
            "final_date": r["final_date"],
        }
        for r in records
    ]

    write_csv(
        args.output / "submission_300.csv",
        submission_rows,
        ["image_id", "year", "month", "day", "final_date"],
    )

    evaluated = []

    for r in records:
        truth = expected[r["file_name"]]

        none_count = sum(
            truth[field] == "NONE"
            for field in ["year", "month", "day"]
        )

        if none_count == 3:
            truth_group = "all_none"
        elif none_count > 0:
            truth_group = "partial_none"
        else:
            truth_group = "complete"

        row = {
            **r,
            "truth_group": truth_group,
        }

        for field in FIELDS:
            row[f"true_{field}"] = truth[field]
            row[f"{field}_correct"] = score_prediction(r, truth)[f"{field}_correct"]

        evaluated.append(row)

    def acc(rows, field):
        if not rows:
            return None
        return sum(r[f"{field}_correct"] for r in rows) / len(rows)

    complete = [r for r in evaluated if r["truth_group"] == "complete"]
    partial = [r for r in evaluated if r["truth_group"] == "partial_none"]
    all_none = [r for r in evaluated if r["truth_group"] == "all_none"]

    summary = {
        "count": len(evaluated),
        "sample_fingerprint": dataset["sample_fingerprint"],
        "processes": 2,
        "threads_per_process": 2,
        "infer_sec": infer_sec,
        "avg_sec_per_image": infer_sec / len(evaluated),
        "total_wall_sec": time.perf_counter() - total_started,
        "accuracy": {
            field: acc(evaluated, field)
            for field in FIELDS
        },
        "complete_count": len(complete),
        "complete_final_date_accuracy": acc(complete, "final_date"),
        "partial_none_count": len(partial),
        "partial_none_accuracy": {
            field: acc(partial, field)
            for field in FIELDS
        },
        "all_none_count": len(all_none),
        "all_none_final_date_accuracy": acc(all_none, "final_date"),
        "stage_counts": dict(Counter(r["method"] for r in records)),
        "worker_pids": sorted(set(r["pid"] for r in records)),
    }

    evaluation_fields = [
        "file_name",
        "image_id",
        "year",
        "month",
        "day",
        "final_date",
        "method",
        "elapsed_sec",
        "pid","index","truth_group",
        "true_year",
        "year_correct",
        "true_month",
        "month_correct",
        "true_day",
        "day_correct",
        "true_final_date",
        "final_date_correct",
    ]

    write_csv(
        args.output / "evaluation_300.csv",
        evaluated,
        evaluation_fields,
    )

    failures = [
        r for r in evaluated
        if not r["final_date_correct"]
    ]

    write_csv(
        args.output / "failures.csv",
        failures,
        evaluation_fields,
    )

    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== RESULT =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
