"""Evaluate the fixed PaddleOCR + date_parser baseline without tuning it.

The script first evaluates the existing seed=42 30-image subset. It expands to
all 300 labels only when the measured 300-image projection is at most 35 minutes.
Per-image OCR results are cached as JSONL so completed images are not inferred
again when a run is resumed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = ROOT / "labels" / "labels_300.csv"
IMAGE_DIR = ROOT / "data" / "validation" / "label_images"
WEIGHTS_DIR = ROOT / "weights" / "paddlex" / "official_models"
FIXED_SUBSET_PATH = (
    ROOT
    / "data"
    / "validation"
    / "ocr_comparison"
    / "bounded_seed42_validation_fixed_subset50.csv"
)
OUTPUT_DIR = ROOT / "data" / "validation" / "integrated_baseline"
CACHE_PATH = OUTPUT_DIR / "paddle_mobile_512_batch6_thresh07_cache.jsonl"
RUN_NAME = "paddle_mobile_512_batch6_thresh07_parser_3ff5347"

MAX_LONG_SIDE = 512
CPU_THREADS = 4
BATCH_SIZE = 6
BOX_THRESHOLD = 0.7
MAX_300_MINUTES = 35.0
PARSER_COMMIT = "3ff5347"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stop-after-30",
        action="store_true",
        help="Measure the fixed 30 images and do not expand to 300.",
    )
    return parser.parse_args()


def configure_offline_runtime() -> None:
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(ROOT / "weights" / "paddlex")
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"
    os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)


def validate_inputs() -> None:
    required = [
        LABELS_PATH,
        IMAGE_DIR,
        FIXED_SUBSET_PATH,
        WEIGHTS_DIR / "PP-OCRv5_mobile_det",
        WEIGHTS_DIR / "korean_PP-OCRv5_mobile_rec",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required local assets are missing: {missing}")


def load_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(LABELS_PATH, dtype={"file_name": str, "image_id": str})
    if len(labels) != 300 or not labels["file_name"].is_unique:
        raise ValueError("labels/labels_300.csv must contain 300 unique filenames")

    fixed = pd.read_csv(FIXED_SUBSET_PATH, dtype={"file_name": str, "image_id": str})
    fixed30_names = fixed.iloc[:30]["file_name"].tolist()
    if len(fixed30_names) != 30 or len(set(fixed30_names)) != 30:
        raise ValueError("The saved seed=42 subset does not contain 30 unique images")
    fixed30 = labels.set_index("file_name").loc[fixed30_names].reset_index()
    return labels, fixed30


def load_common_input(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    long_side = max(image.size)
    if long_side > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / long_side
        size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image)


def paddle_payload(item: Any) -> dict[str, Any]:
    payload = getattr(item, "json", item)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, dict) and isinstance(payload.get("res"), dict):
        payload = payload["res"]
    return payload if isinstance(payload, dict) else {}


def paddle_to_common(predictions: Any) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for item in predictions:
        payload = paddle_payload(item)
        texts = payload.get("rec_texts", []) or []
        scores = payload.get("rec_scores", []) or []
        boxes = payload.get("rec_polys", payload.get("dt_polys", [])) or []
        for index, text in enumerate(texts):
            detections.append(
                {
                    "text": str(text),
                    "confidence": float(scores[index]) if index < len(scores) else 0.0,
                    "bbox": np.asarray(boxes[index]).tolist() if index < len(boxes) else [],
                }
            )
    return detections


def load_cache() -> dict[str, dict[str, Any]]:
    if not CACHE_PATH.exists():
        return {}
    cached: dict[str, dict[str, Any]] = {}
    with CACHE_PATH.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                cached[record["file_name"]] = record
    return cached


def append_cache(record: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("a", encoding="utf-8", newline="") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_label(value: Any, width: int) -> str:
    if pd.isna(value) or str(value).strip().upper() in {"", "NONE", "NAN"}:
        return "NONE"
    return f"{int(float(value)):0{width}d}"


def add_evaluation_fields(record: dict[str, Any], label: pd.Series) -> dict[str, Any]:
    from date_parser.extract import extract_date_tokens

    expected = {
        "year": normalize_label(label["year"], 4),
        "month": normalize_label(label["month"], 2),
        "day": normalize_label(label["day"], 2),
    }
    expected["final_date"] = "-".join(
        [expected["year"], expected["month"], expected["day"]]
    )
    prediction = record["prediction"]
    candidate_found = any(
        extract_date_tokens(item["text"]) for item in record["detections"]
    )
    record.update(
        {
            "true_year": expected["year"],
            "true_month": expected["month"],
            "true_day": expected["day"],
            "true_final_date": expected["final_date"],
            "year_correct": prediction["year"] == expected["year"],
            "month_correct": prediction["month"] == expected["month"],
            "day_correct": prediction["day"] == expected["day"],
            "final_date_correct": prediction["final_date"] == expected["final_date"],
            "date_candidate_found": candidate_found,
        }
    )
    if record.get("missing_image"):
        stage = "missing_image"
    elif not record["detections"]:
        stage = "ocr_no_boxes"
    elif not candidate_found:
        stage = "ocr_boxes_no_date_candidate"
    elif not record["final_date_correct"]:
        stage = "parser_wrong_after_candidate"
    else:
        stage = "success"
    record["failure_stage"] = stage
    return record


def initialize_engine() -> tuple[Any, float]:
    from paddleocr import PaddleOCR

    started = time.perf_counter()
    engine = PaddleOCR(
        lang="korean",
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_detection_model_dir=str(WEIGHTS_DIR / "PP-OCRv5_mobile_det"),
        text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
        text_recognition_model_dir=str(WEIGHTS_DIR / "korean_PP-OCRv5_mobile_rec"),
        text_recognition_batch_size=BATCH_SIZE,
        text_det_limit_side_len=MAX_LONG_SIDE,
        text_det_limit_type="max",
        device="cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=True,
        cpu_threads=CPU_THREADS,
    )
    return engine, time.perf_counter() - started


def run_one(engine: Any, label: pd.Series) -> dict[str, Any]:
    from date_parser import parse_expiration_date

    image_path = IMAGE_DIR / label["file_name"]
    total_started = time.perf_counter()
    if not image_path.is_file():
        prediction = {
            "year": "NONE",
            "month": "NONE",
            "day": "NONE",
            "final_date": "NONE-NONE-NONE",
        }
        return {
            "file_name": label["file_name"],
            "image_id": str(label["image_id"]),
            "detections": [],
            "prediction": prediction,
            "missing_image": True,
            "input_height": None,
            "input_width": None,
            "preprocess_seconds": 0.0,
            "ocr_seconds": 0.0,
            "parser_seconds": 0.0,
            "processing_seconds": time.perf_counter() - total_started,
        }

    started = time.perf_counter()
    image = load_common_input(image_path)
    preprocess_seconds = time.perf_counter() - started

    started = time.perf_counter()
    raw = engine.predict(image, text_det_box_thresh=BOX_THRESHOLD)
    detections = paddle_to_common(raw)
    ocr_seconds = time.perf_counter() - started

    started = time.perf_counter()
    prediction = parse_expiration_date(detections)
    parser_seconds = time.perf_counter() - started
    return {
        "file_name": label["file_name"],
        "image_id": str(label["image_id"]),
        "detections": detections,
        "prediction": prediction,
        "missing_image": False,
        "input_height": int(image.shape[0]),
        "input_width": int(image.shape[1]),
        "preprocess_seconds": preprocess_seconds,
        "ocr_seconds": ocr_seconds,
        "parser_seconds": parser_seconds,
        "processing_seconds": time.perf_counter() - total_started,
    }


def evaluate_frame(
    engine: Any,
    frame: pd.DataFrame,
    cache: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for position, (_, label) in enumerate(frame.iterrows(), start=1):
        name = label["file_name"]
        if name in cache:
            record = cache[name]
            source = "cache"
        else:
            record = run_one(engine, label)
            record["settings"] = {
                "parser_commit": PARSER_COMMIT,
                "max_long_side": MAX_LONG_SIDE,
                "text_recognition_batch_size": BATCH_SIZE,
                "text_det_box_thresh": BOX_THRESHOLD,
                "text_det_limit_type": "max",
                "cpu_threads": CPU_THREADS,
                "enable_mkldnn": True,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
            }
            append_cache(record)
            cache[name] = record
            source = "new"
        evaluated = add_evaluation_fields(dict(record), label)
        evaluated["result_source"] = source
        records.append(evaluated)
        print(
            f"[{position:03d}/{len(frame):03d}] {name} "
            f"{evaluated['processing_seconds']:.3f}s "
            f"boxes={len(evaluated['detections'])} "
            f"stage={evaluated['failure_stage']} source={source}",
            flush=True,
        )
    return records


def make_summary(
    records: list[dict[str, Any]], model_init_seconds: float, wall_seconds: float
) -> dict[str, Any]:
    frame = pd.DataFrame(records)
    candidate = frame["date_candidate_found"].astype(bool)
    success_after_candidate = (
        float(frame.loc[candidate, "final_date_correct"].mean()) if candidate.any() else None
    )
    return {
        "run_name": RUN_NAME,
        "images": len(frame),
        "parser_commit": PARSER_COMMIT,
        "model_init_seconds": model_init_seconds,
        "measured_processing_seconds": float(frame["processing_seconds"].sum()),
        "run_wall_seconds": wall_seconds,
        "mean_seconds_per_image": float(frame["processing_seconds"].mean()),
        "median_seconds_per_image": float(frame["processing_seconds"].median()),
        "estimated_300_minutes": float(frame["processing_seconds"].mean() * 300 / 60),
        "year_accuracy": float(frame["year_correct"].mean()),
        "month_accuracy": float(frame["month_correct"].mean()),
        "day_accuracy": float(frame["day_correct"].mean()),
        "final_date_exact_accuracy": float(frame["final_date_correct"].mean()),
        "ocr_no_boxes": int(frame["failure_stage"].eq("ocr_no_boxes").sum()),
        "date_candidate_found_count": int(candidate.sum()),
        "date_candidate_found_rate": float(candidate.mean()),
        "parser_success_after_candidate_count": int(
            (candidate & frame["final_date_correct"]).sum()
        ),
        "parser_success_after_candidate_rate": success_after_candidate,
        "total_boxes": int(frame["detections"].map(len).sum()),
        "mean_boxes_per_image": float(frame["detections"].map(len).mean()),
        "failure_stage_counts": json.dumps(
            frame["failure_stage"].value_counts().to_dict(), ensure_ascii=False
        ),
        "new_inferences": int(frame["result_source"].eq("new").sum()),
        "cached_inferences": int(frame["result_source"].eq("cache").sum()),
    }


def save_outputs(
    records: list[dict[str, Any]], summary: dict[str, Any], suffix: str
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_rows = []
    box_rows = []
    for record in records:
        prediction = record["prediction"]
        image_rows.append(
            {
                "file_name": record["file_name"],
                "image_id": record["image_id"],
                "true_year": record["true_year"],
                "true_month": record["true_month"],
                "true_day": record["true_day"],
                "true_final_date": record["true_final_date"],
                "pred_year": prediction["year"],
                "pred_month": prediction["month"],
                "pred_day": prediction["day"],
                "pred_final_date": prediction["final_date"],
                "year_correct": record["year_correct"],
                "month_correct": record["month_correct"],
                "day_correct": record["day_correct"],
                "final_date_correct": record["final_date_correct"],
                "date_candidate_found": record["date_candidate_found"],
                "failure_stage": record["failure_stage"],
                "box_count": len(record["detections"]),
                "detected_text": " | ".join(x["text"] for x in record["detections"]),
                "preprocess_seconds": record["preprocess_seconds"],
                "ocr_seconds": record["ocr_seconds"],
                "parser_seconds": record["parser_seconds"],
                "processing_seconds": record["processing_seconds"],
                "input_height": record["input_height"],
                "input_width": record["input_width"],
                "result_source": record["result_source"],
                "detections_json": json.dumps(record["detections"], ensure_ascii=False),
            }
        )
        for box_index, detection in enumerate(record["detections"]):
            box_rows.append(
                {
                    "file_name": record["file_name"],
                    "image_id": record["image_id"],
                    "box_index": box_index,
                    "text": detection["text"],
                    "confidence": detection["confidence"],
                    "bbox": json.dumps(detection["bbox"], ensure_ascii=False),
                }
            )

    images = pd.DataFrame(image_rows)
    boxes = pd.DataFrame(
        box_rows,
        columns=["file_name", "image_id", "box_index", "text", "confidence", "bbox"],
    )
    failures = images.loc[~images["final_date_correct"]].copy()
    prefix = OUTPUT_DIR / f"{RUN_NAME}_{suffix}"
    images.to_csv(f"{prefix}_images.csv", index=False, encoding="utf-8-sig")
    boxes.to_csv(f"{prefix}_ocr_boxes.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(f"{prefix}_failures.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(
        f"{prefix}_summary.csv", index=False, encoding="utf-8-sig"
    )


def main() -> int:
    args = parse_args()
    configure_offline_runtime()
    validate_inputs()
    sys.path.insert(0, str(ROOT))
    labels, fixed30 = load_labels()
    cache = load_cache()

    engine, model_init_seconds = initialize_engine()
    print(f"Model initialization: {model_init_seconds:.3f}s", flush=True)

    started = time.perf_counter()
    records30 = evaluate_frame(engine, fixed30, cache)
    wall30 = time.perf_counter() - started
    summary30 = make_summary(records30, model_init_seconds, wall30)
    save_outputs(records30, summary30, "30")
    print("SUMMARY_30", json.dumps(summary30, ensure_ascii=False), flush=True)

    if args.stop_after_30 or summary30["estimated_300_minutes"] > MAX_300_MINUTES:
        reason = "requested stop" if args.stop_after_30 else "projection exceeds gate"
        print(f"FULL_300_SKIPPED: {reason}", flush=True)
        return 0

    remaining = labels.loc[~labels["file_name"].isin(set(fixed30["file_name"]))]
    started = time.perf_counter()
    remaining_records = evaluate_frame(engine, remaining, cache)
    wall_remaining = time.perf_counter() - started
    by_name = {record["file_name"]: record for record in records30 + remaining_records}
    records300 = [by_name[name] for name in labels["file_name"]]
    summary300 = make_summary(
        records300, model_init_seconds, wall30 + wall_remaining
    )
    save_outputs(records300, summary300, "300")
    print("SUMMARY_300", json.dumps(summary300, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
