"""Evaluate one keyword-nearby crop retry after the saved final cascade.

Targets and crop anchors come only from saved OCR output. Labels are joined
after inference and candidate-based selection solely to measure accuracy.
No existing result or cache is modified.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from evaluate_highres_retry import initialize_highres_engine
from evaluate_integrated_baseline import (
    BOX_THRESHOLD,
    IMAGE_DIR,
    LABELS_PATH,
    ROOT,
    configure_offline_runtime,
    normalize_label,
    paddle_to_common,
)
from evaluate_rotation_retry import selected_candidate_details


BASE = ROOT / "data" / "validation" / "integrated_baseline"
FINAL_DIR = BASE / "final_cascade"
FINAL_IMAGES = FINAL_DIR / "final_cascade_images.csv"
FINAL_BOXES = FINAL_DIR / "final_cascade_ocr_boxes.csv"
FINAL_SUMMARY = FINAL_DIR / "final_cascade_summary.csv"
OUTPUT_DIR = BASE / "keyword_crop"
CACHE_PATH = OUTPUT_DIR / "keyword_crop_cache.jsonl"

CROP_LONG_SIDE = 1024
CROP_MIN_WIDTH_FRACTION = 0.50
CROP_MIN_HEIGHT_FRACTION = 0.30
CROP_KEYWORD_WIDTH_MULTIPLIER = 8.0
CROP_KEYWORD_HEIGHT_MULTIPLIER = 10.0
SOURCE_PRIORITY = {"original_512": 3, "highres_1024": 2, "clahe": 1}


def validate_inputs() -> None:
    required = [FINAL_IMAGES, FINAL_BOXES, FINAL_SUMMARY, LABELS_PATH, IMAGE_DIR]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required inputs are missing: {missing}")


def load_source_frames() -> dict[str, pd.DataFrame]:
    run = "paddle_mobile_512_batch6_thresh07_parser_3ff5347"
    frames = {
        "original_512": pd.read_csv(
            BASE / f"{run}_300_images.csv", dtype={"image_id": str}
        ),
        "highres_1024": pd.read_csv(
            BASE / "highres_retry" / "highres_passes.csv", dtype={"image_id": str}
        ),
        "clahe": pd.read_csv(
            BASE / "contrast_retry" / "contrast_passes.csv", dtype={"image_id": str}
        ),
    }
    frames["clahe"] = frames["clahe"].loc[frames["clahe"]["method"].eq("clahe")]
    return frames


def load_cache() -> dict[str, dict[str, Any]]:
    if not CACHE_PATH.exists():
        return {}
    with CACHE_PATH.open(encoding="utf-8") as stream:
        return {
            record["file_name"]: record
            for record in (json.loads(line) for line in stream if line.strip())
        }


def append_cache(record: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("a", encoding="utf-8", newline="") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def bounded_window(center: float, span: float, limit: int) -> tuple[int, int]:
    span = min(float(limit), max(1.0, span))
    start = center - span / 2
    end = center + span / 2
    if start < 0:
        end -= start
        start = 0
    if end > limit:
        start -= end - limit
        end = limit
    return max(0, int(np.floor(start))), min(limit, int(np.ceil(end)))


def choose_keyword_anchor(
    file_name: str, frames: dict[str, pd.DataFrame]
) -> dict[str, Any] | None:
    from date_parser.keywords import ANCHOR_KEYWORDS, has_keyword

    candidates = []
    for source, frame in frames.items():
        matches = frame.loc[frame["file_name"].eq(file_name)]
        if matches.empty:
            continue
        row = matches.iloc[0]
        source_width = int(row["input_width"])
        source_height = int(row["input_height"])
        for item in json.loads(row["detections_json"]):
            if not item.get("bbox") or not has_keyword(str(item.get("text", "")), ANCHOR_KEYWORDS):
                continue
            candidates.append(
                {
                    "source": source,
                    "source_width": source_width,
                    "source_height": source_height,
                    "text": str(item["text"]),
                    "confidence": float(item.get("confidence") or 0.0),
                    "bbox": item["bbox"],
                }
            )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (item["confidence"], SOURCE_PRIORITY[item["source"]]),
    )


def crop_from_original(file_name: str, anchor: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    with Image.open(IMAGE_DIR / file_name) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")

    points = np.asarray(anchor["bbox"], dtype=float)
    scale_x = original.width / anchor["source_width"]
    scale_y = original.height / anchor["source_height"]
    xs = points[:, 0] * scale_x
    ys = points[:, 1] * scale_y
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    keyword_width = max(1.0, x_max - x_min)
    keyword_height = max(1.0, y_max - y_min)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    crop_width = max(
        keyword_width * CROP_KEYWORD_WIDTH_MULTIPLIER,
        original.width * CROP_MIN_WIDTH_FRACTION,
    )
    crop_height = max(
        keyword_height * CROP_KEYWORD_HEIGHT_MULTIPLIER,
        original.height * CROP_MIN_HEIGHT_FRACTION,
    )
    left, right = bounded_window(center_x, crop_width, original.width)
    top, bottom = bounded_window(center_y, crop_height, original.height)
    cropped = original.crop((left, top, right, bottom))
    scale = CROP_LONG_SIDE / max(cropped.size)
    resized_size = (
        max(1, round(cropped.width * scale)),
        max(1, round(cropped.height * scale)),
    )
    cropped = cropped.resize(resized_size, Image.Resampling.LANCZOS)
    metadata = {
        "anchor_source": anchor["source"],
        "anchor_text": anchor["text"],
        "anchor_confidence": anchor["confidence"],
        "anchor_bbox_source": anchor["bbox"],
        "original_width": original.width,
        "original_height": original.height,
        "crop_left": left,
        "crop_top": top,
        "crop_right": right,
        "crop_bottom": bottom,
        "crop_width": right - left,
        "crop_height": bottom - top,
        "resized_width": resized_size[0],
        "resized_height": resized_size[1],
        "crop_long_side": CROP_LONG_SIDE,
        "width_rule": "max(keyword_width*8, original_width*0.50)",
        "height_rule": "max(keyword_height*10, original_height*0.30)",
    }
    return np.asarray(cropped), metadata


def run_crop(engine: Any, file_name: str, anchor: dict[str, Any]) -> dict[str, Any]:
    from date_parser import parse_expiration_date

    total_started = time.perf_counter()
    started = time.perf_counter()
    image, crop_metadata = crop_from_original(file_name, anchor)
    preprocess_seconds = time.perf_counter() - started

    started = time.perf_counter()
    result = paddle_to_common(engine.predict(image, text_det_box_thresh=BOX_THRESHOLD))
    ocr_seconds = time.perf_counter() - started

    started = time.perf_counter()
    parsed = parse_expiration_date(result)
    parser_seconds = time.perf_counter() - started
    return {
        "file_name": file_name,
        "detections": result,
        "prediction": parsed,
        **selected_candidate_details(result),
        **crop_metadata,
        "preprocess_seconds": preprocess_seconds,
        "ocr_seconds": ocr_seconds,
        "parser_seconds": parser_seconds,
        "processing_seconds": time.perf_counter() - total_started,
    }


def true_final_date(label: pd.Series) -> str:
    return "-".join(
        [
            normalize_label(label["year"], 4),
            normalize_label(label["month"], 2),
            normalize_label(label["day"], 2),
        ]
    )


def save_pass_outputs(records: list[dict[str, Any]]) -> pd.DataFrame:
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
                    "coordinate_frame": "resized_keyword_crop",
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "keyword_crop_passes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(boxes).to_csv(
        OUTPUT_DIR / "keyword_crop_ocr_boxes.csv", index=False, encoding="utf-8-sig"
    )
    return frame


def main() -> int:
    configure_offline_runtime()
    validate_inputs()
    sys.path.insert(0, str(ROOT))
    final = pd.read_csv(FINAL_IMAGES, dtype={"image_id": str})
    labels = pd.read_csv(LABELS_PATH, dtype={"file_name": str, "image_id": str})
    labels_by_name = labels.set_index("file_name")
    targets = final.loc[
        final["failure_stage"].isin(
            ["ocr_no_boxes", "ocr_boxes_no_date_candidate"]
        )
    ].copy()
    if len(final) != 300 or len(targets) != 43:
        raise ValueError(f"Expected 300 final rows and 43 targets, got {len(final)} and {len(targets)}")

    frames = load_source_frames()
    anchors = {
        name: anchor
        for name in targets["file_name"]
        if (anchor := choose_keyword_anchor(name, frames)) is not None
    }
    print(f"TARGETS={len(targets)} KEYWORD_ELIGIBLE={len(anchors)}", flush=True)
    engine, model_init_seconds = initialize_highres_engine()
    print(f"Model initialization: {model_init_seconds:.3f}s", flush=True)
    cache = load_cache()
    records = []
    for position, file_name in enumerate(anchors, start=1):
        if file_name in cache:
            record = dict(cache[file_name])
            source = "cache"
        else:
            record = run_crop(engine, file_name, anchors[file_name])
            append_cache(record)
            cache[file_name] = record
            source = "new"
        label = labels_by_name.loc[file_name]
        truth = true_final_date(label)
        record["image_id"] = str(label["image_id"])
        record["true_final_date"] = truth
        record["final_date_correct"] = record["prediction"]["final_date"] == truth
        record["result_source"] = source
        records.append(record)
        print(
            f"[{position:02d}/{len(anchors):02d}] {file_name} "
            f"anchor={record['anchor_source']}:{record['anchor_text']} "
            f"candidate={record['date_candidate_found']} "
            f"ocr={record['ocr_seconds']:.3f}s source={source}",
            flush=True,
        )

    pass_frame = save_pass_outputs(records)
    by_name = {record["file_name"]: record for record in records}
    combined = final.copy()
    combined["keyword_crop_eligible"] = combined["file_name"].isin(anchors)
    combined["keyword_crop_selected"] = False
    combined["pre_crop_final_date_correct"] = combined["final_date_correct"].astype(bool)
    for index, row in combined.iterrows():
        record = by_name.get(row["file_name"])
        if record is None or not record["date_candidate_found"]:
            continue
        prediction = record["prediction"]
        combined.at[index, "keyword_crop_selected"] = True
        combined.at[index, "selected_method"] = "keyword_crop_1024"
        combined.at[index, "pred_year"] = prediction["year"]
        combined.at[index, "pred_month"] = prediction["month"]
        combined.at[index, "pred_day"] = prediction["day"]
        combined.at[index, "pred_final_date"] = prediction["final_date"]
        combined.at[index, "date_candidate_found"] = True
    combined["year_correct"] = combined["pred_year"].astype(str) == combined["true_year"].astype(str)
    combined["month_correct"] = combined["pred_month"].astype(str) == combined["true_month"].astype(str)
    combined["day_correct"] = combined["pred_day"].astype(str) == combined["true_day"].astype(str)
    combined["final_date_correct"] = (
        combined["pred_final_date"] == combined["true_final_date"]
    )
    for index, row in combined.loc[combined["keyword_crop_selected"]].iterrows():
        combined.at[index, "failure_stage"] = (
            "success" if row["pred_final_date"] == row["true_final_date"] else "candidate_but_wrong"
        )

    previous_summary = pd.read_csv(FINAL_SUMMARY).iloc[0]
    previous_success = combined["pre_crop_final_date_correct"].astype(bool)
    current_success = combined["final_date_correct"].astype(bool)
    new_success = ~previous_success & current_success
    regressions = previous_success & ~current_success
    candidate_recoveries = int(pass_frame["date_candidate_found"].sum())
    additional_ocr_seconds = float(pass_frame["ocr_seconds"].sum())
    additional_processing_seconds = float(pass_frame["processing_seconds"].sum())
    total_processing_seconds = float(previous_summary["total_expected_processing_seconds"]) + additional_processing_seconds
    final_matches = int(current_success.sum())
    summary = {
        "method": "keyword_nearby_crop_1024",
        "source_control": "original_512 -> rotation_270 -> highres_1024 -> clahe",
        "control_exact_matches": int(previous_summary["final_exact_matches"]),
        "control_exact_accuracy": float(previous_summary["final_exact_accuracy"]),
        "target_images": len(targets),
        "keyword_eligible_images": len(anchors),
        "additional_ocr_calls": len(records),
        "candidate_recoveries": candidate_recoveries,
        "new_exact_matches": int(new_success.sum()),
        "final_exact_matches": final_matches,
        "final_exact_accuracy": float(current_success.mean()),
        "change_vs_original_baseline_percentage_points": (final_matches - 175) / 300 * 100,
        "change_vs_final_cascade_percentage_points": (
            final_matches - int(previous_summary["final_exact_matches"])
        )
        / 300
        * 100,
        "regressions": int(regressions.sum()),
        "additional_ocr_seconds": additional_ocr_seconds,
        "additional_processing_seconds": additional_processing_seconds,
        "total_expected_processing_seconds": total_processing_seconds,
        "total_expected_minutes_excluding_init": total_processing_seconds / 60,
        "model_init_seconds_this_experiment": model_init_seconds,
        "total_expected_minutes_including_one_pipeline_init": (
            total_processing_seconds + float(previous_summary["one_time_model_init_seconds"])
        )
        / 60,
        "new_success_image_ids": json.dumps(
            combined.loc[new_success, "image_id"].astype(str).tolist(), ensure_ascii=False
        ),
        "candidate_but_wrong_image_ids": json.dumps(
            pass_frame.loc[
                pass_frame["date_candidate_found"].astype(bool)
                & ~pass_frame["final_date_correct"].astype(bool),
                "image_id",
            ].astype(str).tolist(),
            ensure_ascii=False,
        ),
        "still_no_candidate_image_ids": json.dumps(
            targets.loc[~targets["file_name"].isin(
                pass_frame.loc[pass_frame["date_candidate_found"].astype(bool), "file_name"]
            ), "image_id"].astype(str).tolist(),
            ensure_ascii=False,
        ),
        "crop_rule": json.dumps(
            {
                "anchor_keywords": "date_parser.keywords.ANCHOR_KEYWORDS",
                "anchor_selection": "highest confidence; tie original_512 > highres_1024 > clahe",
                "width": "max(keyword_width*8, original_width*0.50)",
                "height": "max(keyword_height*10, original_height*0.30)",
                "resize_long_side": 1024,
                "resize_filter": "LANCZOS",
            },
            ensure_ascii=False,
        ),
    }

    method_row = {
        "method": "keyword_nearby_crop_1024_after_final_cascade",
        "target_images": len(targets),
        "additional_ocr_calls": len(records),
        "candidate_recoveries": candidate_recoveries,
        "new_exact_matches": int(new_success.sum()),
        "exact_matches_300": final_matches,
        "accuracy_300": float(current_success.mean()),
        "change_vs_baseline_pp": (final_matches - 175) / 300 * 100,
        "change_vs_rotation_control_pp": (final_matches - 182) / 300 * 100,
        "regressions": int(regressions.sum()),
        "additional_processing_seconds": additional_processing_seconds,
        "total_expected_minutes": total_processing_seconds / 60,
        "status": "evaluated",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_DIR / "combined_300_results.csv", index=False, encoding="utf-8-sig")
    combined.loc[~combined["final_date_correct"]].to_csv(
        OUTPUT_DIR / "combined_300_failures.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([summary]).to_csv(
        OUTPUT_DIR / "keyword_crop_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([method_row]).to_csv(
        OUTPUT_DIR / "method_comparison_row.csv", index=False, encoding="utf-8-sig"
    )

    existing_boxes = pd.read_csv(FINAL_BOXES, dtype={"image_id": str})
    crop_boxes = pd.read_csv(OUTPUT_DIR / "keyword_crop_ocr_boxes.csv", dtype={"image_id": str})
    crop_boxes["pass_method"] = "keyword_crop_1024"
    crop_boxes["pass_selected"] = crop_boxes["file_name"].isin(
        combined.loc[combined["keyword_crop_selected"], "file_name"]
    )
    crop_boxes["pass_date_candidate_found"] = crop_boxes["file_name"].isin(
        pass_frame.loc[pass_frame["date_candidate_found"].astype(bool), "file_name"]
    )
    common_columns = list(existing_boxes.columns)
    for column in common_columns:
        if column not in crop_boxes:
            crop_boxes[column] = pd.NA
    pd.concat([existing_boxes, crop_boxes[common_columns]], ignore_index=True).to_csv(
        OUTPUT_DIR / "combined_300_ocr_boxes.csv", index=False, encoding="utf-8-sig"
    )

    print("METHOD_COMPARISON_ROW", json.dumps(method_row, ensure_ascii=False), flush=True)
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
