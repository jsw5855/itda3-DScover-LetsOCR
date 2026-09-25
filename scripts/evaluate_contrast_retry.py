"""Evaluate conditional CLAHE and dark-image gamma OCR retries.

The control uses only the frozen original 512px OCR and cached 270-degree
rotation results. High-resolution retry outputs are never read. Labels are
used only after candidate-based selection to calculate evaluation metrics.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from evaluate_highres_retry import (
    BASELINE_IMAGES,
    LABELS_PATH,
    ROTATION_SUMMARY,
    reconstruct_control,
)
from evaluate_integrated_baseline import (
    BOX_THRESHOLD,
    IMAGE_DIR,
    ROOT,
    configure_offline_runtime,
    initialize_engine,
    load_common_input,
    normalize_label,
    paddle_to_common,
)
from evaluate_rotation_retry import selected_candidate_details


OUTPUT_DIR = (
    ROOT / "data" / "validation" / "integrated_baseline" / "contrast_retry"
)
PASS_CACHE = OUTPUT_DIR / "contrast_pass_cache.jsonl"
SPEED_SAMPLE_SIZE = 10
MAX_TOTAL_MINUTES = 35.0

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
GAMMA_DARK_THRESHOLD = 110.0
GAMMA_VALUE = 0.7
METHOD_PRIORITY = {"clahe": 2, "gamma": 1}


def validate_inputs() -> None:
    required = [BASELINE_IMAGES, LABELS_PATH, IMAGE_DIR, ROTATION_SUMMARY]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required inputs are missing: {missing}")


def read_cache() -> dict[tuple[str, str], dict[str, Any]]:
    if not PASS_CACHE.exists():
        return {}
    records = {}
    with PASS_CACHE.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                records[(record["file_name"], record["method"])] = record
    return records


def append_cache(record: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with PASS_CACHE.open("a", encoding="utf-8", newline="") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def mean_brightness(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(gray.mean())


def apply_clahe(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE
    )
    corrected = cv2.merge((clahe.apply(lightness), channel_a, channel_b))
    return cv2.cvtColor(corrected, cv2.COLOR_LAB2RGB)


def apply_gamma(rgb: np.ndarray) -> np.ndarray:
    lookup = np.rint(
        255.0 * np.power(np.arange(256, dtype=np.float32) / 255.0, GAMMA_VALUE)
    ).astype(np.uint8)
    return cv2.LUT(rgb, lookup)


def preprocess(file_name: str, method: str) -> tuple[np.ndarray, float]:
    original = load_common_input(IMAGE_DIR / file_name)
    brightness = mean_brightness(original)
    if method == "clahe":
        return apply_clahe(original), brightness
    if method == "gamma":
        if brightness >= GAMMA_DARK_THRESHOLD:
            raise ValueError(f"Gamma requested for non-dark image: {file_name}")
        return apply_gamma(original), brightness
    raise ValueError(f"Unknown method: {method}")


def run_variant(engine: Any, file_name: str, method: str) -> dict[str, Any]:
    from date_parser import parse_expiration_date

    total_started = time.perf_counter()
    started = time.perf_counter()
    image, brightness = preprocess(file_name, method)
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
        "method": method,
        "mean_brightness": brightness,
        "gamma_applied": method == "gamma",
        "detections": detections,
        "prediction": prediction,
        **selected_candidate_details(detections),
        "input_height": int(image.shape[0]),
        "input_width": int(image.shape[1]),
        "preprocess_seconds": preprocess_seconds,
        "ocr_seconds": ocr_seconds,
        "parser_seconds": parser_seconds,
        "processing_seconds": time.perf_counter() - total_started,
        "parameters": {
            "max_long_side": 512,
            "clahe_color_space": "LAB_L_only",
            "clahe_clip_limit": CLAHE_CLIP_LIMIT,
            "clahe_tile_grid_size": list(CLAHE_TILE_GRID_SIZE),
            "gamma_dark_threshold_mean_gray_0_255": GAMMA_DARK_THRESHOLD,
            "gamma": GAMMA_VALUE,
            "gamma_formula": "round(255 * (pixel / 255) ** gamma)",
        },
    }


def expected_final_date(label: pd.Series) -> str:
    return "-".join(
        [
            normalize_label(label["year"], 4),
            normalize_label(label["month"], 2),
            normalize_label(label["day"], 2),
        ]
    )


def evaluate_one(
    engine: Any,
    file_name: str,
    method: str,
    cache: dict[tuple[str, str], dict[str, Any]],
    labels_by_name: pd.DataFrame,
    position: int,
    total: int,
) -> dict[str, Any]:
    key = (file_name, method)
    if key in cache:
        record = dict(cache[key])
        source = "cache"
    else:
        record = run_variant(engine, file_name, method)
        append_cache(record)
        cache[key] = record
        source = "new"
    label = labels_by_name.loc[file_name]
    truth = expected_final_date(label)
    record["image_id"] = str(label["image_id"])
    record["true_final_date"] = truth
    record["final_date_correct"] = record["prediction"]["final_date"] == truth
    record["result_source"] = source
    print(
        f"[{position:02d}/{total:02d}] {method:6s} {file_name} "
        f"brightness={record['mean_brightness']:.1f} "
        f"candidate={record['date_candidate_found']} "
        f"ocr={record['ocr_seconds']:.3f}s source={source}",
        flush=True,
    )
    return record


def selection_key(record: dict[str, Any]) -> tuple[float, ...]:
    distance = record["normalized_anchor_distance"]
    proximity = -float(distance) if distance is not None else 0.0
    return (
        float(record["anchor_preference"]),
        proximity,
        float(record["complete_date"]),
        float(record["valid_field_count"]),
        float(record["selected_token_confidence"]),
        float(METHOD_PRIORITY[record["method"]]),
    )


def selection_reason(record: dict[str, Any]) -> str:
    distance = record["normalized_anchor_distance"]
    distance_text = "none" if distance is None else f"{distance:.3f}px"
    return (
        f"method={record['method']}; anchor_preference={record['anchor_preference']}; "
        f"anchor_distance={distance_text}; complete_date={record['complete_date']}; "
        f"valid_fields={record['valid_field_count']}; "
        f"token_confidence={record['selected_token_confidence']:.6f}; "
        f"method_tiebreak_priority={METHOD_PRIORITY[record['method']]}"
    )


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
        row["parameters"] = json.dumps(record["parameters"], ensure_ascii=False)
        rows.append(row)
        for index, item in enumerate(record["detections"]):
            boxes.append(
                {
                    "file_name": record["file_name"],
                    "image_id": record["image_id"],
                    "method": record["method"],
                    "box_index": index,
                    "text": item["text"],
                    "confidence": item["confidence"],
                    "bbox": json.dumps(item["bbox"], ensure_ascii=False),
                }
            )
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "contrast_passes.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(boxes).to_csv(
        OUTPUT_DIR / "contrast_ocr_boxes.csv", index=False, encoding="utf-8-sig"
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

    brightness = {
        name: mean_brightness(load_common_input(IMAGE_DIR / name))
        for name in targets["file_name"]
    }
    gamma_names = [
        name
        for name in targets["file_name"]
        if brightness[name] < GAMMA_DARK_THRESHOLD
    ]
    if len(targets) != 71:
        raise ValueError(f"Expected 71 control targets, found {len(targets)}")
    print(
        f"CONTROL exact={int(control.control_final_date_correct.sum())}/300 "
        f"targets={len(targets)} clahe={len(targets)} gamma={len(gamma_names)}",
        flush=True,
    )

    engine, model_init_seconds = initialize_engine()
    print(f"Model initialization: {model_init_seconds:.3f}s", flush=True)
    cache = read_cache()
    records: list[dict[str, Any]] = []

    clahe_sample = targets["file_name"].iloc[:SPEED_SAMPLE_SIZE].tolist()
    gamma_sample = gamma_names[:SPEED_SAMPLE_SIZE]
    for position, file_name in enumerate(clahe_sample, start=1):
        records.append(
            evaluate_one(
                engine,
                file_name,
                "clahe",
                cache,
                labels_by_name,
                position,
                len(clahe_sample),
            )
        )
    for position, file_name in enumerate(gamma_sample, start=1):
        records.append(
            evaluate_one(
                engine,
                file_name,
                "gamma",
                cache,
                labels_by_name,
                position,
                len(gamma_sample),
            )
        )

    sample_frame = pd.DataFrame(records)
    clahe_mean = float(
        sample_frame.loc[sample_frame["method"].eq("clahe"), "processing_seconds"].mean()
    )
    gamma_mean = float(
        sample_frame.loc[sample_frame["method"].eq("gamma"), "processing_seconds"].mean()
    )
    control_seconds = float(
        pd.read_csv(ROTATION_SUMMARY).iloc[0]["estimated_total_processing_seconds"]
    )
    projected_retry_seconds = clahe_mean * len(targets) + gamma_mean * len(gamma_names)
    projected_total_seconds = control_seconds + projected_retry_seconds + model_init_seconds
    gate = {
        "clahe_sample_images": len(clahe_sample),
        "clahe_sample_mean_seconds": clahe_mean,
        "clahe_projected_images": len(targets),
        "gamma_sample_images": len(gamma_sample),
        "gamma_sample_mean_seconds": gamma_mean,
        "gamma_projected_images": len(gamma_names),
        "control_processing_seconds": control_seconds,
        "projected_retry_seconds": projected_retry_seconds,
        "model_init_seconds": model_init_seconds,
        "projected_total_minutes_including_init": projected_total_seconds / 60,
        "limit_minutes": MAX_TOTAL_MINUTES,
        "gate_passed": projected_total_seconds / 60 <= MAX_TOTAL_MINUTES,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([gate]).to_csv(
        OUTPUT_DIR / "speed_gate_summary.csv", index=False, encoding="utf-8-sig"
    )
    print("SPEED_GATE", json.dumps(gate, ensure_ascii=False), flush=True)
    if not gate["gate_passed"]:
        save_pass_outputs(records)
        return 0

    completed = {(record["file_name"], record["method"]) for record in records}
    full_plan = [(name, "clahe") for name in targets["file_name"]]
    full_plan.extend((name, "gamma") for name in gamma_names)
    remaining = [item for item in full_plan if item not in completed]
    for position, (file_name, method) in enumerate(remaining, start=1):
        records.append(
            evaluate_one(
                engine,
                file_name,
                method,
                cache,
                labels_by_name,
                position,
                len(remaining),
            )
        )

    by_image: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_image.setdefault(record["file_name"], []).append(record)

    selections = []
    for file_name in targets["file_name"]:
        candidates = [
            record
            for record in by_image[file_name]
            if record["date_candidate_found"]
        ]
        selected = max(candidates, key=selection_key) if candidates else None
        selections.append(
            {
                "file_name": file_name,
                "image_id": str(labels_by_name.loc[file_name, "image_id"]),
                "selected": selected is not None,
                "selected_method": selected["method"] if selected else None,
                "selected_prediction": (
                    selected["prediction"]["final_date"] if selected else None
                ),
                "selection_reason": (
                    selection_reason(selected) if selected else "no_contrast_candidate"
                ),
                "clahe_candidate": any(
                    x["method"] == "clahe" and x["date_candidate_found"]
                    for x in by_image[file_name]
                ),
                "gamma_eligible": file_name in gamma_names,
                "gamma_candidate": any(
                    x["method"] == "gamma" and x["date_candidate_found"]
                    for x in by_image[file_name]
                ),
            }
        )

    selection_by_name = {row["file_name"]: row for row in selections}
    combined = control.copy()
    combined["contrast_selected"] = False
    combined["selected_method"] = pd.NA
    combined["final_pred_year"] = combined["control_pred_year"]
    combined["final_pred_month"] = combined["control_pred_month"]
    combined["final_pred_day"] = combined["control_pred_day"]
    combined["final_pred_final_date"] = combined["control_pred_final_date"]
    for index, row in combined.iterrows():
        choice = selection_by_name.get(row["file_name"])
        if choice is None or not choice["selected"]:
            continue
        selected = next(
            record
            for record in by_image[row["file_name"]]
            if record["method"] == choice["selected_method"]
        )
        prediction = selected["prediction"]
        combined.at[index, "contrast_selected"] = True
        combined.at[index, "selected_method"] = selected["method"]
        combined.at[index, "final_pred_year"] = prediction["year"]
        combined.at[index, "final_pred_month"] = prediction["month"]
        combined.at[index, "final_pred_day"] = prediction["day"]
        combined.at[index, "final_pred_final_date"] = prediction["final_date"]

    combined["final_date_correct"] = (
        combined["final_pred_final_date"] == combined["true_final_date"]
    )
    frame = pd.DataFrame(records)
    baseline_success = combined["control_final_date_correct"].astype(bool)
    final_success = combined["final_date_correct"].astype(bool)
    newly_correct = ~baseline_success & final_success
    regressions = baseline_success & ~final_success
    selected_frame = pd.DataFrame(selections)
    selected_wrong_ids = combined.loc[
        combined["contrast_selected"].astype(bool) & ~final_success, "image_id"
    ].astype(str).tolist()
    no_candidate_ids = selected_frame.loc[
        ~selected_frame["selected"].astype(bool), "image_id"
    ].astype(str).tolist()

    summary: dict[str, Any] = {
        "control_exact_matches": 182,
        "control_exact_accuracy": 182 / 300,
        "clahe_target_images": int(frame["method"].eq("clahe").sum()),
        "clahe_candidate_recoveries": int(
            frame.loc[frame["method"].eq("clahe"), "date_candidate_found"].sum()
        ),
        "clahe_exact_matches": int(
            frame.loc[frame["method"].eq("clahe"), "final_date_correct"].sum()
        ),
        "clahe_additional_ocr_seconds": float(
            frame.loc[frame["method"].eq("clahe"), "ocr_seconds"].sum()
        ),
        "clahe_additional_processing_seconds": float(
            frame.loc[frame["method"].eq("clahe"), "processing_seconds"].sum()
        ),
        "clahe_no_box_images": int(
            frame.loc[frame["method"].eq("clahe"), "detections"].map(len).eq(0).sum()
        ),
        "gamma_target_images": int(frame["method"].eq("gamma").sum()),
        "gamma_candidate_recoveries": int(
            frame.loc[frame["method"].eq("gamma"), "date_candidate_found"].sum()
        ),
        "gamma_exact_matches": int(
            frame.loc[frame["method"].eq("gamma"), "final_date_correct"].sum()
        ),
        "gamma_additional_ocr_seconds": float(
            frame.loc[frame["method"].eq("gamma"), "ocr_seconds"].sum()
        ),
        "gamma_additional_processing_seconds": float(
            frame.loc[frame["method"].eq("gamma"), "processing_seconds"].sum()
        ),
        "gamma_no_box_images": int(
            frame.loc[frame["method"].eq("gamma"), "detections"].map(len).eq(0).sum()
        ),
        "selected_candidate_images": int(selected_frame["selected"].sum()),
        "selected_clahe_images": int(selected_frame["selected_method"].eq("clahe").sum()),
        "selected_gamma_images": int(selected_frame["selected_method"].eq("gamma").sum()),
        "new_exact_matches": int(newly_correct.sum()),
        "final_exact_matches": int(final_success.sum()),
        "final_exact_accuracy": float(final_success.mean()),
        "accuracy_change_percentage_points": float(
            (final_success.sum() - 182) / 300 * 100
        ),
        "regressions": int(regressions.sum()),
        "additional_ocr_seconds": float(frame["ocr_seconds"].sum()),
        "additional_processing_seconds": float(frame["processing_seconds"].sum()),
        "estimated_total_minutes_excluding_init": float(
            (control_seconds + frame["processing_seconds"].sum()) / 60
        ),
        "model_init_seconds": model_init_seconds,
        "estimated_total_minutes_including_init": float(
            (control_seconds + frame["processing_seconds"].sum() + model_init_seconds)
            / 60
        ),
        "new_success_image_ids": json.dumps(
            combined.loc[newly_correct, "image_id"].astype(str).tolist(),
            ensure_ascii=False,
        ),
        "candidate_but_wrong_image_ids": json.dumps(
            selected_wrong_ids, ensure_ascii=False
        ),
        "no_candidate_image_ids": json.dumps(no_candidate_ids, ensure_ascii=False),
        "parameters": json.dumps(
            {
                "clahe": {
                    "color_space": "LAB",
                    "channel": "L",
                    "clip_limit": CLAHE_CLIP_LIMIT,
                    "tile_grid_size": list(CLAHE_TILE_GRID_SIZE),
                },
                "gamma": {
                    "eligibility": f"mean grayscale < {GAMMA_DARK_THRESHOLD}/255",
                    "gamma": GAMMA_VALUE,
                    "formula": "round(255 * (pixel / 255) ** gamma)",
                },
            },
            ensure_ascii=False,
        ),
    }

    save_pass_outputs(records)
    pd.DataFrame(selections).to_csv(
        OUTPUT_DIR / "contrast_selections.csv", index=False, encoding="utf-8-sig"
    )
    control.to_csv(OUTPUT_DIR / "control_300_results.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(OUTPUT_DIR / "combined_300_results.csv", index=False, encoding="utf-8-sig")
    combined.loc[~combined["final_date_correct"]].to_csv(
        OUTPUT_DIR / "combined_300_failures.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([summary]).to_csv(
        OUTPUT_DIR / "contrast_retry_summary.csv", index=False, encoding="utf-8-sig"
    )
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
