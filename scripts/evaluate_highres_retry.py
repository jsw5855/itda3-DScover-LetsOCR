"""Evaluate one conditional 1024px retry after the 270-degree control.

The control is reconstructed from the frozen 512px baseline and only the
cached 270-degree rotation results. Ground truth is used only after inference
and candidate-based selection to calculate metrics.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from evaluate_integrated_baseline import (
    BATCH_SIZE,
    BOX_THRESHOLD,
    CPU_THREADS,
    IMAGE_DIR,
    LABELS_PATH,
    ROOT,
    RUN_NAME,
    WEIGHTS_DIR,
    configure_offline_runtime,
    normalize_label,
    paddle_to_common,
)
from evaluate_rotation_retry import selected_candidate_details


BASELINE_DIR = ROOT / "data" / "validation" / "integrated_baseline"
BASELINE_IMAGES = BASELINE_DIR / f"{RUN_NAME}_300_images.csv"
ROTATION_DIR = BASELINE_DIR / "rotation_retry"
ROTATION_CACHE = ROTATION_DIR / "rotation_pass_cache.jsonl"
ROTATION_SUMMARY = ROTATION_DIR / "rotation_retry_summary.csv"
OUTPUT_DIR = BASELINE_DIR / "highres_retry"
HIGHRES_CACHE = OUTPUT_DIR / "highres_1024_cache.jsonl"
HIGHRES_LONG_SIDE = 1024
SPEED_SAMPLE_SIZE = 10
MAX_TOTAL_MINUTES = 35.0


def validate_inputs() -> None:
    required = [
        BASELINE_IMAGES,
        ROTATION_CACHE,
        ROTATION_SUMMARY,
        LABELS_PATH,
        IMAGE_DIR,
        WEIGHTS_DIR / "PP-OCRv5_mobile_det",
        WEIGHTS_DIR / "korean_PP-OCRv5_mobile_rec",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required inputs are missing: {missing}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_highres_cache() -> dict[str, dict[str, Any]]:
    if not HIGHRES_CACHE.exists():
        return {}
    return {record["file_name"]: record for record in read_jsonl(HIGHRES_CACHE)}


def append_highres_cache(record: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with HIGHRES_CACHE.open("a", encoding="utf-8", newline="") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def label_values(label: pd.Series) -> dict[str, str]:
    values = {
        "year": normalize_label(label["year"], 4),
        "month": normalize_label(label["month"], 2),
        "day": normalize_label(label["day"], 2),
    }
    values["final_date"] = "-".join(
        [values["year"], values["month"], values["day"]]
    )
    return values


def reconstruct_control(
    baseline: pd.DataFrame, labels_by_name: pd.DataFrame
) -> pd.DataFrame:
    rotations = {
        record["file_name"]: record
        for record in read_jsonl(ROTATION_CACHE)
        if int(record["angle"]) == 270
    }
    if len(rotations) != 80:
        raise ValueError(f"Expected 80 cached 270-degree passes, found {len(rotations)}")

    control = baseline.copy()
    control["control_source"] = "original_512"
    control["control_date_candidate_found"] = control["date_candidate_found"].astype(bool)
    control["control_pred_year"] = control["pred_year"]
    control["control_pred_month"] = control["pred_month"]
    control["control_pred_day"] = control["pred_day"]
    control["control_pred_final_date"] = control["pred_final_date"]
    for column in ("true_year", "true_month", "true_day", "true_final_date"):
        control[column] = control[column].astype(object)

    for index, row in control.loc[~control["date_candidate_found"].astype(bool)].iterrows():
        rotation = rotations[row["file_name"]]
        if not rotation["date_candidate_found"]:
            continue
        prediction = rotation["prediction"]
        control.at[index, "control_source"] = "rotation_270"
        control.at[index, "control_date_candidate_found"] = True
        control.at[index, "control_pred_year"] = prediction["year"]
        control.at[index, "control_pred_month"] = prediction["month"]
        control.at[index, "control_pred_day"] = prediction["day"]
        control.at[index, "control_pred_final_date"] = prediction["final_date"]

    for index, row in control.iterrows():
        expected = label_values(labels_by_name.loc[row["file_name"]])
        control.at[index, "true_year"] = expected["year"]
        control.at[index, "true_month"] = expected["month"]
        control.at[index, "true_day"] = expected["day"]
        control.at[index, "true_final_date"] = expected["final_date"]
    control["control_final_date_correct"] = (
        control["control_pred_final_date"] == control["true_final_date"]
    )
    if int(control["control_final_date_correct"].sum()) != 182:
        raise ValueError("Reconstructed control does not reproduce 182/300")
    if int((~control["control_date_candidate_found"]).sum()) != 71:
        raise ValueError("Reconstructed control does not contain 71 no-candidate images")
    return control


def initialize_highres_engine() -> tuple[Any, float]:
    from paddleocr import PaddleOCR

    started = time.perf_counter()
    engine = PaddleOCR(
        lang="korean",
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_detection_model_dir=str(WEIGHTS_DIR / "PP-OCRv5_mobile_det"),
        text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
        text_recognition_model_dir=str(WEIGHTS_DIR / "korean_PP-OCRv5_mobile_rec"),
        text_recognition_batch_size=BATCH_SIZE,
        text_det_limit_side_len=HIGHRES_LONG_SIDE,
        text_det_limit_type="max",
        device="cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=True,
        cpu_threads=CPU_THREADS,
    )
    return engine, time.perf_counter() - started


def load_highres_input(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    long_side = max(image.size)
    if long_side > HIGHRES_LONG_SIDE:
        scale = HIGHRES_LONG_SIDE / long_side
        size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image)


def run_highres(engine: Any, file_name: str) -> dict[str, Any]:
    from date_parser import parse_expiration_date

    total_started = time.perf_counter()
    started = time.perf_counter()
    image = load_highres_input(IMAGE_DIR / file_name)
    preprocess_seconds = time.perf_counter() - started

    started = time.perf_counter()
    detections = paddle_to_common(
        engine.predict(image, text_det_box_thresh=BOX_THRESHOLD)
    )
    ocr_seconds = time.perf_counter() - started

    started = time.perf_counter()
    prediction = parse_expiration_date(detections)
    parser_seconds = time.perf_counter() - started
    return {
        "file_name": file_name,
        "resolution": HIGHRES_LONG_SIDE,
        "detections": detections,
        "prediction": prediction,
        **selected_candidate_details(detections),
        "input_height": int(image.shape[0]),
        "input_width": int(image.shape[1]),
        "preprocess_seconds": preprocess_seconds,
        "ocr_seconds": ocr_seconds,
        "parser_seconds": parser_seconds,
        "processing_seconds": time.perf_counter() - total_started,
    }


def evaluate_target(
    engine: Any,
    row: Any,
    cache: dict[str, dict[str, Any]],
    labels_by_name: pd.DataFrame,
    position: int,
    total: int,
) -> dict[str, Any]:
    file_name = row.file_name
    if file_name in cache:
        record = dict(cache[file_name])
        source = "cache"
    else:
        record = run_highres(engine, file_name)
        append_highres_cache(record)
        cache[file_name] = record
        source = "new"
    expected = label_values(labels_by_name.loc[file_name])
    record["image_id"] = str(labels_by_name.loc[file_name, "image_id"])
    record["true_final_date"] = expected["final_date"]
    record["final_date_correct"] = (
        record["prediction"]["final_date"] == expected["final_date"]
    )
    record["result_source"] = source
    print(
        f"[{position:02d}/{total:02d}] {file_name} "
        f"candidate={record['date_candidate_found']} "
        f"ocr={record['ocr_seconds']:.3f}s total={record['processing_seconds']:.3f}s "
        f"source={source}",
        flush=True,
    )
    return record


def save_pass_outputs(records: list[dict[str, Any]]) -> None:
    rows = []
    boxes = []
    for record in records:
        row = {key: value for key, value in record.items() if key != "detections"}
        row["pred_year"] = record["prediction"]["year"]
        row["pred_month"] = record["prediction"]["month"]
        row["pred_day"] = record["prediction"]["day"]
        row["pred_final_date"] = record["prediction"]["final_date"]
        row["box_count"] = len(record["detections"])
        row["detected_text"] = " | ".join(item["text"] for item in record["detections"])
        row["detections_json"] = json.dumps(record["detections"], ensure_ascii=False)
        row["prediction"] = json.dumps(record["prediction"], ensure_ascii=False)
        rows.append(row)
        for index, item in enumerate(record["detections"]):
            boxes.append(
                {
                    "file_name": record["file_name"],
                    "image_id": record["image_id"],
                    "box_index": index,
                    "text": item["text"],
                    "confidence": item["confidence"],
                    "bbox": json.dumps(item["bbox"], ensure_ascii=False),
                }
            )
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "highres_passes.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(boxes).to_csv(
        OUTPUT_DIR / "highres_ocr_boxes.csv", index=False, encoding="utf-8-sig"
    )


def main() -> int:
    configure_offline_runtime()
    validate_inputs()
    sys.path.insert(0, str(ROOT))
    baseline = pd.read_csv(BASELINE_IMAGES, dtype={"image_id": str})
    labels = pd.read_csv(LABELS_PATH, dtype={"file_name": str, "image_id": str})
    labels_by_name = labels.set_index("file_name")
    control = reconstruct_control(baseline, labels_by_name)
    targets = control.loc[~control["control_date_candidate_found"]].copy()
    print(
        f"CONTROL exact={int(control.control_final_date_correct.sum())}/300 "
        f"no_candidate={len(targets)}",
        flush=True,
    )

    engine, model_init_seconds = initialize_highres_engine()
    print(f"Model initialization: {model_init_seconds:.3f}s", flush=True)
    cache = load_highres_cache()
    records: list[dict[str, Any]] = []

    sample = targets.iloc[:SPEED_SAMPLE_SIZE]
    for position, row in enumerate(sample.itertuples(index=False), start=1):
        records.append(
            evaluate_target(
                engine, row, cache, labels_by_name, position, SPEED_SAMPLE_SIZE
            )
        )

    sample_mean = float(np.mean([record["processing_seconds"] for record in records]))
    projected_retry_seconds = sample_mean * len(targets)
    control_seconds = float(
        pd.read_csv(ROTATION_SUMMARY).iloc[0]["estimated_total_processing_seconds"]
    )
    projected_total_seconds = control_seconds + projected_retry_seconds + model_init_seconds
    projected_total_minutes = projected_total_seconds / 60
    gate = {
        "sample_images": SPEED_SAMPLE_SIZE,
        "sample_mean_seconds": sample_mean,
        "projected_retry_seconds": projected_retry_seconds,
        "control_processing_seconds": control_seconds,
        "model_init_seconds": model_init_seconds,
        "projected_total_seconds_including_init": projected_total_seconds,
        "projected_total_minutes_including_init": projected_total_minutes,
        "limit_minutes": MAX_TOTAL_MINUTES,
        "gate_passed": projected_total_minutes <= MAX_TOTAL_MINUTES,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([gate]).to_csv(
        OUTPUT_DIR / "speed_gate_10_summary.csv", index=False, encoding="utf-8-sig"
    )
    print("SPEED_GATE", json.dumps(gate, ensure_ascii=False), flush=True)
    if not gate["gate_passed"]:
        save_pass_outputs(records)
        return 0

    remaining = targets.iloc[SPEED_SAMPLE_SIZE:]
    for offset, row in enumerate(remaining.itertuples(index=False), start=1):
        records.append(
            evaluate_target(
                engine,
                row,
                cache,
                labels_by_name,
                SPEED_SAMPLE_SIZE + offset,
                len(targets),
            )
        )

    by_name = {record["file_name"]: record for record in records}
    combined = control.copy()
    combined["highres_selected"] = False
    combined["highres_additional_processing_seconds"] = 0.0
    combined["final_pred_year"] = pd.NA
    combined["final_pred_month"] = pd.NA
    combined["final_pred_day"] = pd.NA
    combined["final_pred_final_date"] = pd.NA
    for index, row in combined.iterrows():
        record = by_name.get(row["file_name"])
        if record is None:
            continue
        combined.at[index, "highres_additional_processing_seconds"] = record[
            "processing_seconds"
        ]
        if not record["date_candidate_found"]:
            continue
        prediction = record["prediction"]
        combined.at[index, "highres_selected"] = True
        combined.at[index, "final_pred_year"] = prediction["year"]
        combined.at[index, "final_pred_month"] = prediction["month"]
        combined.at[index, "final_pred_day"] = prediction["day"]
        combined.at[index, "final_pred_final_date"] = prediction["final_date"]

    unset = combined["final_pred_final_date"].isna()
    combined.loc[unset, "final_pred_year"] = combined.loc[unset, "control_pred_year"]
    combined.loc[unset, "final_pred_month"] = combined.loc[unset, "control_pred_month"]
    combined.loc[unset, "final_pred_day"] = combined.loc[unset, "control_pred_day"]
    combined.loc[unset, "final_pred_final_date"] = combined.loc[
        unset, "control_pred_final_date"
    ]
    combined["final_date_correct"] = (
        combined["final_pred_final_date"] == combined["true_final_date"]
    )

    result_frame = pd.DataFrame(records)
    selected = result_frame["date_candidate_found"].astype(bool)
    newly_correct = (
        ~combined["control_final_date_correct"].astype(bool)
        & combined["final_date_correct"].astype(bool)
    )
    regressions = (
        combined["control_final_date_correct"].astype(bool)
        & ~combined["final_date_correct"].astype(bool)
    )
    new_success_ids = combined.loc[newly_correct, "image_id"].astype(str).tolist()
    wrong_candidate_ids = result_frame.loc[
        selected & ~result_frame["final_date_correct"].astype(bool), "image_id"
    ].astype(str).tolist()
    no_candidate_ids = result_frame.loc[~selected, "image_id"].astype(str).tolist()
    additional_ocr_seconds = float(result_frame["ocr_seconds"].sum())
    additional_processing_seconds = float(result_frame["processing_seconds"].sum())
    final_matches = int(combined["final_date_correct"].sum())
    summary = {
        "target_images": len(targets),
        "candidate_recoveries": int(selected.sum()),
        "new_exact_matches": int(newly_correct.sum()),
        "control_exact_matches": 182,
        "final_exact_matches": final_matches,
        "control_exact_accuracy": 182 / 300,
        "final_exact_accuracy": final_matches / 300,
        "accuracy_change_percentage_points": (final_matches - 182) / 300 * 100,
        "regressions": int(regressions.sum()),
        "additional_ocr_seconds": additional_ocr_seconds,
        "additional_processing_seconds": additional_processing_seconds,
        "control_processing_seconds": control_seconds,
        "estimated_total_processing_seconds": control_seconds
        + additional_processing_seconds,
        "estimated_total_minutes_excluding_init": (
            control_seconds + additional_processing_seconds
        )
        / 60,
        "model_init_seconds": model_init_seconds,
        "estimated_total_minutes_including_init": (
            control_seconds + additional_processing_seconds + model_init_seconds
        )
        / 60,
        "new_success_image_ids": json.dumps(new_success_ids, ensure_ascii=False),
        "candidate_but_wrong_image_ids": json.dumps(
            wrong_candidate_ids, ensure_ascii=False
        ),
        "no_candidate_image_ids": json.dumps(no_candidate_ids, ensure_ascii=False),
    }

    save_pass_outputs(records)
    control.to_csv(OUTPUT_DIR / "control_300_results.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(OUTPUT_DIR / "combined_300_results.csv", index=False, encoding="utf-8-sig")
    combined.loc[~combined["final_date_correct"]].to_csv(
        OUTPUT_DIR / "combined_300_failures.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([summary]).to_csv(
        OUTPUT_DIR / "highres_retry_summary.csv", index=False, encoding="utf-8-sig"
    )
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
