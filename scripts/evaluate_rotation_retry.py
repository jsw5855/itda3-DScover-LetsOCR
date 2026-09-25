"""Evaluate conditional rotation retries on the frozen integrated baseline.

Only baseline images without a date candidate are retried. The retry decision
and rotation selection never inspect labels; labels are joined afterward only
to measure accuracy.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from evaluate_integrated_baseline import (
    BOX_THRESHOLD,
    IMAGE_DIR,
    LABELS_PATH,
    ROOT,
    RUN_NAME,
    add_evaluation_fields,
    configure_offline_runtime,
    initialize_engine,
    load_common_input,
    normalize_label,
    paddle_to_common,
)


BASELINE_DIR = ROOT / "data" / "validation" / "integrated_baseline"
BASELINE_IMAGES = BASELINE_DIR / f"{RUN_NAME}_300_images.csv"
BASELINE_SUMMARY = BASELINE_DIR / f"{RUN_NAME}_300_summary.csv"
BASELINE_CACHE = BASELINE_DIR / "paddle_mobile_512_batch6_thresh07_cache.jsonl"
OUTPUT_DIR = BASELINE_DIR / "rotation_retry"
PASS_CACHE = OUTPUT_DIR / "rotation_pass_cache.jsonl"
ANGLE_PRIORITY = {180: 3, 90: 2, 270: 1}


def validate_inputs() -> None:
    required = [BASELINE_IMAGES, BASELINE_SUMMARY, BASELINE_CACHE, LABELS_PATH, IMAGE_DIR]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required baseline inputs are missing: {missing}")


def load_jsonl(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                records[(record["file_name"], int(record["angle"]))] = record
    return records


def append_jsonl(record: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with PASS_CACHE.open("a", encoding="utf-8", newline="") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def selected_candidate_details(detections: list[dict[str, Any]]) -> dict[str, Any]:
    from date_parser.keywords import (
        ANCHOR_KEYWORDS,
        EXCLUDE_KEYWORDS,
        bbox_center,
        has_keyword,
        min_distance,
    )
    from date_parser.select import select_final_date
    from date_parser.types import TextBox

    boxes = [TextBox.from_dict(item) for item in detections]
    selected = select_final_date(boxes)
    if selected is None:
        return {
            "date_candidate_found": False,
            "selected_source_text": "",
            "selected_token_confidence": 0.0,
            "anchor_keyword_found": False,
            "anchor_preference": 0,
            "normalized_anchor_distance": None,
            "valid_field_count": 0,
            "complete_date": False,
            "parser_candidate_score": None,
        }

    anchors = [bbox_center(box.bbox) for box in boxes if has_keyword(box.text, ANCHOR_KEYWORDS)]
    excludes = [bbox_center(box.bbox) for box in boxes if has_keyword(box.text, EXCLUDE_KEYWORDS)]
    anchor_distance = min_distance(selected.center, anchors)
    exclude_distance = min_distance(selected.center, excludes)
    anchor_preference = 0
    if anchors:
        anchor_preference = 2 if anchor_distance <= exclude_distance else 1

    matching_confidences = []
    for item in detections:
        if item["text"] != selected.source_text or not item["bbox"]:
            continue
        center = bbox_center(item["bbox"])
        if math.dist(center, selected.center) < 1e-6:
            matching_confidences.append(float(item["confidence"]))

    result = selected.result
    valid_field_count = sum(
        value is not None for value in (result.year, result.month, result.day)
    )
    return {
        "date_candidate_found": True,
        "selected_source_text": selected.source_text,
        "selected_token_confidence": max(matching_confidences, default=0.0),
        "anchor_keyword_found": bool(anchors),
        "anchor_preference": anchor_preference,
        "normalized_anchor_distance": anchor_distance if anchors else None,
        "valid_field_count": valid_field_count,
        "complete_date": result.is_complete(),
        "parser_candidate_score": (
            float(selected.candidates[0].score) if selected.candidates else None
        ),
    }


def run_rotation(engine: Any, file_name: str, angle: int) -> dict[str, Any]:
    from date_parser import parse_expiration_date

    total_started = time.perf_counter()
    base_image = load_common_input(IMAGE_DIR / file_name)
    started = time.perf_counter()
    rotated = np.asarray(Image.fromarray(base_image).rotate(angle, expand=True))
    rotation_seconds = time.perf_counter() - started

    started = time.perf_counter()
    detections = paddle_to_common(
        engine.predict(rotated, text_det_box_thresh=BOX_THRESHOLD)
    )
    ocr_seconds = time.perf_counter() - started

    started = time.perf_counter()
    prediction = parse_expiration_date(detections)
    parser_seconds = time.perf_counter() - started
    details = selected_candidate_details(detections)
    return {
        "file_name": file_name,
        "angle": angle,
        "detections": detections,
        "prediction": prediction,
        **details,
        "input_height": int(rotated.shape[0]),
        "input_width": int(rotated.shape[1]),
        "rotation_seconds": rotation_seconds,
        "ocr_seconds": ocr_seconds,
        "parser_seconds": parser_seconds,
        "processing_seconds": time.perf_counter() - total_started,
    }


def selection_key(record: dict[str, Any]) -> tuple[float, ...]:
    distance = record["normalized_anchor_distance"]
    proximity = -float(distance) if distance is not None else 0.0
    return (
        float(record["anchor_preference"]),
        proximity,
        float(record["complete_date"]),
        float(record["valid_field_count"]),
        float(record["selected_token_confidence"]),
        float(ANGLE_PRIORITY[int(record["angle"])]),
    )


def selection_reason(record: dict[str, Any]) -> str:
    distance = record["normalized_anchor_distance"]
    distance_text = "none" if distance is None else f"{distance:.3f}px"
    return (
        f"anchor_preference={record['anchor_preference']}; "
        f"anchor_distance={distance_text}; "
        f"complete_date={record['complete_date']}; "
        f"valid_fields={record['valid_field_count']}; "
        f"token_confidence={record['selected_token_confidence']:.6f}; "
        f"angle_tiebreak_priority={ANGLE_PRIORITY[int(record['angle'])]}"
    )


def evaluate_pass_against_label(record: dict[str, Any], label: pd.Series) -> dict[str, Any]:
    expected = {
        "year": normalize_label(label["year"], 4),
        "month": normalize_label(label["month"], 2),
        "day": normalize_label(label["day"], 2),
    }
    expected_final = "-".join(expected.values())
    prediction = record["prediction"]
    return {
        **record,
        "image_id": str(label["image_id"]),
        "true_final_date": expected_final,
        "final_date_correct": prediction["final_date"] == expected_final,
    }


def save_results(
    passes: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    combined: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pass_rows = []
    box_rows = []
    for record in passes:
        row = {key: value for key, value in record.items() if key != "detections"}
        row["pred_year"] = record["prediction"]["year"]
        row["pred_month"] = record["prediction"]["month"]
        row["pred_day"] = record["prediction"]["day"]
        row["pred_final_date"] = record["prediction"]["final_date"]
        row["box_count"] = len(record["detections"])
        row["detected_text"] = " | ".join(x["text"] for x in record["detections"])
        row["detections_json"] = json.dumps(record["detections"], ensure_ascii=False)
        row["prediction"] = json.dumps(record["prediction"], ensure_ascii=False)
        pass_rows.append(row)
        for index, item in enumerate(record["detections"]):
            box_rows.append(
                {
                    "file_name": record["file_name"],
                    "image_id": record["image_id"],
                    "angle": record["angle"],
                    "box_index": index,
                    "text": item["text"],
                    "confidence": item["confidence"],
                    "bbox": json.dumps(item["bbox"], ensure_ascii=False),
                }
            )

    pd.DataFrame(pass_rows).to_csv(
        OUTPUT_DIR / "rotation_passes.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(box_rows).to_csv(
        OUTPUT_DIR / "rotation_ocr_boxes.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(selections).to_csv(
        OUTPUT_DIR / "rotation_selections.csv", index=False, encoding="utf-8-sig"
    )
    combined.to_csv(
        OUTPUT_DIR / "combined_300_results.csv", index=False, encoding="utf-8-sig"
    )
    combined.loc[~combined["final_date_correct"]].to_csv(
        OUTPUT_DIR / "combined_300_failures.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([summary]).to_csv(
        OUTPUT_DIR / "rotation_retry_summary.csv", index=False, encoding="utf-8-sig"
    )


def main() -> int:
    configure_offline_runtime()
    validate_inputs()
    sys.path.insert(0, str(ROOT))
    baseline = pd.read_csv(BASELINE_IMAGES, dtype={"image_id": str})
    labels = pd.read_csv(LABELS_PATH, dtype={"file_name": str, "image_id": str})
    labels_by_name = labels.set_index("file_name")
    targets = baseline.loc[~baseline["date_candidate_found"].astype(bool)].copy()
    if len(baseline) != 300 or len(targets) != 80:
        raise ValueError(
            f"Expected 300 baseline images and 80 retry targets, got {len(baseline)} and {len(targets)}"
        )

    cache = load_jsonl(PASS_CACHE)
    engine, model_init_seconds = initialize_engine()
    print(f"Model initialization: {model_init_seconds:.3f}s", flush=True)
    passes: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []

    for position, target in enumerate(targets.itertuples(index=False), start=1):
        file_name = target.file_name
        image_passes = []
        angles = [180]
        while angles:
            angle = angles.pop(0)
            key = (file_name, angle)
            if key in cache:
                record = cache[key]
                source = "cache"
            else:
                record = run_rotation(engine, file_name, angle)
                append_jsonl(record)
                cache[key] = record
                source = "new"
            record = dict(record)
            record["result_source"] = source
            image_passes.append(record)
            passes.append(record)
            print(
                f"[{position:02d}/80] {file_name} angle={angle} "
                f"candidate={record['date_candidate_found']} "
                f"ocr={record['ocr_seconds']:.3f}s source={source}",
                flush=True,
            )
            if angle == 180 and not record["date_candidate_found"]:
                angles.extend([90, 270])

        candidates = [record for record in image_passes if record["date_candidate_found"]]
        selected = max(candidates, key=selection_key) if candidates else None
        selections.append(
            {
                "file_name": file_name,
                "image_id": str(labels_by_name.loc[file_name, "image_id"]),
                "selected": selected is not None,
                "selected_angle": selected["angle"] if selected else None,
                "selected_prediction": (
                    selected["prediction"]["final_date"] if selected else target.pred_final_date
                ),
                "selection_reason": selection_reason(selected) if selected else "no_rotation_candidate",
                "attempted_angles": json.dumps([x["angle"] for x in image_passes]),
                "additional_ocr_seconds": sum(x["ocr_seconds"] for x in image_passes),
                "additional_processing_seconds": sum(
                    x["processing_seconds"] for x in image_passes
                ),
            }
        )

    evaluated_passes = [
        evaluate_pass_against_label(record, labels_by_name.loc[record["file_name"]])
        for record in passes
    ]
    selection_by_name = {row["file_name"]: row for row in selections}
    combined = baseline.copy()
    combined["baseline_final_date_correct"] = combined["final_date_correct"].astype(bool)
    combined["rotation_retry_selected"] = False
    combined["selected_angle"] = pd.NA
    combined["rotation_additional_processing_seconds"] = 0.0
    for index, row in combined.iterrows():
        selected = selection_by_name.get(row["file_name"])
        if selected is None:
            continue
        combined.at[index, "rotation_additional_processing_seconds"] = selected[
            "additional_processing_seconds"
        ]
        if selected["selected"]:
            combined.at[index, "rotation_retry_selected"] = True
            combined.at[index, "selected_angle"] = selected["selected_angle"]
            combined.at[index, "pred_final_date"] = selected["selected_prediction"]
            prediction = next(
                record["prediction"]
                for record in evaluated_passes
                if record["file_name"] == row["file_name"]
                and record["angle"] == selected["selected_angle"]
            )
            combined.at[index, "pred_year"] = prediction["year"]
            combined.at[index, "pred_month"] = prediction["month"]
            combined.at[index, "pred_day"] = prediction["day"]

    combined["year_correct"] = combined["pred_year"] == combined["true_year"]
    combined["month_correct"] = combined["pred_month"] == combined["true_month"]
    combined["day_correct"] = combined["pred_day"] == combined["true_day"]
    combined["final_date_correct"] = (
        combined["pred_final_date"] == combined["true_final_date"]
    )

    passes_frame = pd.DataFrame(evaluated_passes)
    selected_names = {
        row["file_name"]: int(row["selected_angle"])
        for row in selections
        if row["selected"]
    }
    selected_passes = passes_frame.loc[
        [
            selected_names.get(row.file_name) == row.angle
            for row in passes_frame.itertuples(index=False)
        ]
    ]
    no_candidate_ids = [
        row["image_id"] for row in selections if not row["selected"]
    ]
    wrong_candidate_ids = selected_passes.loc[
        ~selected_passes["final_date_correct"], "image_id"
    ].tolist()
    baseline_summary = pd.read_csv(BASELINE_SUMMARY).iloc[0]
    additional_ocr_seconds = float(passes_frame["ocr_seconds"].sum())
    additional_processing_seconds = float(
        passes_frame["processing_seconds"].sum()
    )
    baseline_success = baseline["final_date_correct"].astype(bool)
    final_success = combined["final_date_correct"].astype(bool)
    summary: dict[str, Any] = {
        "target_images": 80,
        "model_init_seconds": model_init_seconds,
        "rotation_passes": len(passes_frame),
        "new_candidate_images": len(selected_names),
        "new_exact_match_images": int((~baseline_success & final_success).sum()),
        "baseline_exact_matches": int(baseline_success.sum()),
        "final_exact_matches": int(final_success.sum()),
        "baseline_exact_accuracy": float(baseline_success.mean()),
        "final_exact_accuracy": float(final_success.mean()),
        "regressions_from_baseline_success": int((baseline_success & ~final_success).sum()),
        "additional_ocr_seconds": additional_ocr_seconds,
        "additional_processing_seconds": additional_processing_seconds,
        "baseline_processing_seconds": float(baseline_summary["measured_processing_seconds"]),
        "estimated_total_processing_seconds": float(
            baseline_summary["measured_processing_seconds"] + additional_processing_seconds
        ),
        "estimated_total_minutes": float(
            (baseline_summary["measured_processing_seconds"] + additional_processing_seconds)
            / 60
        ),
        "retry_failed_image_ids": json.dumps(no_candidate_ids, ensure_ascii=False),
        "candidate_but_wrong_image_ids": json.dumps(wrong_candidate_ids, ensure_ascii=False),
    }
    for angle in (180, 90, 270):
        angle_frame = passes_frame.loc[passes_frame["angle"].eq(angle)]
        summary[f"angle_{angle}_attempts"] = len(angle_frame)
        summary[f"angle_{angle}_candidate_recoveries"] = int(
            angle_frame["date_candidate_found"].sum()
        )
        summary[f"angle_{angle}_exact_matches"] = int(
            angle_frame["final_date_correct"].sum()
        )

    save_results(evaluated_passes, selections, combined, summary)
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
