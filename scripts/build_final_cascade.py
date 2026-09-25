"""Build the final fallback cascade entirely from saved OCR caches.

No OCR inference is performed here. Selection uses only date-candidate presence:
original 512px -> 270-degree rotation -> 1024px -> CLAHE. Ground truth is used
after selection solely to calculate development-set metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "validation" / "integrated_baseline"
RUN = "paddle_mobile_512_batch6_thresh07_parser_3ff5347"
OUTPUT = BASE / "final_cascade"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    date_dtypes = {
        "image_id": str,
        "true_year": str,
        "true_month": str,
        "true_day": str,
        "pred_year": str,
        "pred_month": str,
        "pred_day": str,
    }
    original = pd.read_csv(
        BASE / f"{RUN}_300_images.csv", dtype=date_dtypes
    )
    rotation = pd.read_csv(
        BASE / "rotation_retry" / "rotation_passes.csv", dtype=date_dtypes
    )
    rotation = rotation.loc[rotation["angle"].eq(270)].copy()
    highres = pd.read_csv(
        BASE / "highres_retry" / "highres_passes.csv", dtype=date_dtypes
    )
    contrast = pd.read_csv(
        BASE / "contrast_retry" / "contrast_passes.csv", dtype=date_dtypes
    )
    clahe = contrast.loc[contrast["method"].eq("clahe")].copy()
    if (len(original), len(rotation), len(highres), len(clahe)) != (300, 80, 71, 71):
        raise ValueError("Saved OCR result counts do not match the frozen experiments")
    return original, rotation, highres, clahe


def bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def detections(row: pd.Series) -> list[dict[str, Any]]:
    return json.loads(row["detections_json"])


def prediction(row: pd.Series) -> dict[str, str]:
    return {
        "year": str(row["pred_year"]),
        "month": str(row["pred_month"]),
        "day": str(row["pred_day"]),
        "final_date": str(row["pred_final_date"]),
    }


def index(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {row["file_name"]: row for _, row in frame.iterrows()}


def comparison_table(
    original: pd.DataFrame,
    rotation: pd.DataFrame,
    highres: pd.DataFrame,
    clahe: pd.DataFrame,
    final_summary: dict[str, Any],
) -> pd.DataFrame:
    baseline_seconds = float(original["processing_seconds"].sum())
    rotation_seconds = float(rotation["processing_seconds"].sum())
    highres_seconds = float(highres["processing_seconds"].sum())
    contrast_summary = pd.read_csv(
        BASE / "contrast_retry" / "contrast_retry_summary.csv"
    ).iloc[0]
    contrast_seconds = float(contrast_summary["additional_processing_seconds"])
    keyword_crop = pd.read_csv(
        BASE / "keyword_crop" / "method_comparison_row.csv"
    ).iloc[0]
    rows = [
        {
            "method": "original_512_baseline",
            "target_images": 300,
            "additional_ocr_calls": 0,
            "candidate_recoveries": 0,
            "new_exact_matches": 0,
            "exact_matches_300": 175,
            "accuracy_300": 175 / 300,
            "change_vs_baseline_pp": 0.0,
            "change_vs_rotation_control_pp": (175 - 182) / 300 * 100,
            "regressions": 0,
            "additional_processing_seconds": 0.0,
            "total_expected_minutes": baseline_seconds / 60,
        },
        {
            "method": "conditional_rotation_270",
            "target_images": 80,
            "additional_ocr_calls": 80,
            "candidate_recoveries": int(rotation["date_candidate_found"].sum()),
            "new_exact_matches": int(rotation["final_date_correct"].sum()),
            "exact_matches_300": 182,
            "accuracy_300": 182 / 300,
            "change_vs_baseline_pp": (182 - 175) / 300 * 100,
            "change_vs_rotation_control_pp": 0.0,
            "regressions": 0,
            "additional_processing_seconds": rotation_seconds,
            "total_expected_minutes": (baseline_seconds + rotation_seconds) / 60,
        },
        {
            "method": "highres_1024_after_rotation_control",
            "target_images": 71,
            "additional_ocr_calls": 71,
            "candidate_recoveries": int(highres["date_candidate_found"].sum()),
            "new_exact_matches": int(highres["final_date_correct"].sum()),
            "exact_matches_300": 202,
            "accuracy_300": 202 / 300,
            "change_vs_baseline_pp": (202 - 175) / 300 * 100,
            "change_vs_rotation_control_pp": (202 - 182) / 300 * 100,
            "regressions": 0,
            "additional_processing_seconds": highres_seconds,
            "total_expected_minutes": (
                baseline_seconds + rotation_seconds + highres_seconds
            )
            / 60,
        },
        {
            "method": "clahe_plus_gamma_after_rotation_control",
            "target_images": 71,
            "additional_ocr_calls": int(
                contrast_summary["clahe_target_images"]
                + contrast_summary["gamma_target_images"]
            ),
            "candidate_recoveries": int(contrast_summary["selected_candidate_images"]),
            "new_exact_matches": int(contrast_summary["new_exact_matches"]),
            "exact_matches_300": 189,
            "accuracy_300": 189 / 300,
            "change_vs_baseline_pp": (189 - 175) / 300 * 100,
            "change_vs_rotation_control_pp": (189 - 182) / 300 * 100,
            "regressions": int(contrast_summary["regressions"]),
            "additional_processing_seconds": contrast_seconds,
            "total_expected_minutes": (
                baseline_seconds + rotation_seconds + contrast_seconds
            )
            / 60,
        },
        {
            "method": str(keyword_crop["method"]),
            "target_images": int(keyword_crop["target_images"]),
            "additional_ocr_calls": int(keyword_crop["additional_ocr_calls"]),
            "candidate_recoveries": int(keyword_crop["candidate_recoveries"]),
            "new_exact_matches": int(keyword_crop["new_exact_matches"]),
            "exact_matches_300": int(keyword_crop["exact_matches_300"]),
            "accuracy_300": float(keyword_crop["accuracy_300"]),
            "change_vs_baseline_pp": float(keyword_crop["change_vs_baseline_pp"]),
            "change_vs_rotation_control_pp": float(
                keyword_crop["change_vs_rotation_control_pp"]
            ),
            "regressions": int(keyword_crop["regressions"]),
            "additional_processing_seconds": float(
                keyword_crop["additional_processing_seconds"]
            ),
            "total_expected_minutes": float(keyword_crop["total_expected_minutes"]),
            "status": "evaluated_no_effect_excluded",
        },
        {
            "method": "final_original_rotation270_highres_clahe",
            "target_images": "80 -> 71 -> 48",
            "additional_ocr_calls": int(final_summary["additional_ocr_calls"]),
            "candidate_recoveries": int(final_summary["fallback_candidate_recoveries"]),
            "new_exact_matches": int(final_summary["new_exact_matches"]),
            "exact_matches_300": int(final_summary["final_exact_matches"]),
            "accuracy_300": float(final_summary["final_exact_accuracy"]),
            "change_vs_baseline_pp": float(
                final_summary["change_vs_baseline_percentage_points"]
            ),
            "change_vs_rotation_control_pp": float(
                final_summary["change_vs_rotation_control_percentage_points"]
            ),
            "regressions": int(final_summary["regressions"]),
            "additional_processing_seconds": float(
                final_summary["additional_processing_seconds"]
            ),
            "total_expected_minutes": float(
                final_summary["total_expected_minutes_excluding_init"]
            ),
            "status": "selected",
        },
    ]
    return pd.DataFrame(rows)


def main() -> int:
    original, rotation, highres, clahe = load_inputs()
    original_by_name = index(original)
    rotation_by_name = index(rotation)
    highres_by_name = index(highres)
    clahe_by_name = index(clahe)

    image_rows = []
    attempted_passes: dict[str, list[tuple[str, pd.Series]]] = {}
    stage_saves = {"rotation_270": 0, "highres_1024": 0, "clahe": 0}
    stage_candidates = {"rotation_270": 0, "highres_1024": 0, "clahe": 0}
    stage_attempts = {"rotation_270": 0, "highres_1024": 0, "clahe": 0}

    for _, base_row in original.iterrows():
        name = base_row["file_name"]
        attempts: list[tuple[str, pd.Series]] = [("original_512", base_row)]
        chosen_method = "original_512"
        chosen_row = base_row
        chosen_has_candidate = bool_value(base_row["date_candidate_found"])

        if not chosen_has_candidate:
            rotation_row = rotation_by_name[name]
            attempts.append(("rotation_270", rotation_row))
            stage_attempts["rotation_270"] += 1
            if bool_value(rotation_row["date_candidate_found"]):
                chosen_method = "rotation_270"
                chosen_row = rotation_row
                chosen_has_candidate = True
                stage_candidates["rotation_270"] += 1

        if not chosen_has_candidate:
            highres_row = highres_by_name[name]
            attempts.append(("highres_1024", highres_row))
            stage_attempts["highres_1024"] += 1
            if bool_value(highres_row["date_candidate_found"]):
                chosen_method = "highres_1024"
                chosen_row = highres_row
                chosen_has_candidate = True
                stage_candidates["highres_1024"] += 1

        if not chosen_has_candidate:
            clahe_row = clahe_by_name[name]
            attempts.append(("clahe", clahe_row))
            stage_attempts["clahe"] += 1
            if bool_value(clahe_row["date_candidate_found"]):
                chosen_method = "clahe"
                chosen_row = clahe_row
                chosen_has_candidate = True
                stage_candidates["clahe"] += 1

        chosen_prediction = prediction(chosen_row)
        year_correct = chosen_prediction["year"] == str(base_row["true_year"])
        month_correct = chosen_prediction["month"] == str(base_row["true_month"])
        day_correct = chosen_prediction["day"] == str(base_row["true_day"])
        final_correct = chosen_prediction["final_date"] == str(base_row["true_final_date"])
        if chosen_method in stage_saves and final_correct:
            stage_saves[chosen_method] += 1

        if final_correct:
            failure_stage = "success"
        elif chosen_has_candidate:
            failure_stage = "candidate_but_wrong"
        else:
            all_boxes = sum(len(detections(row)) for _, row in attempts)
            failure_stage = (
                "ocr_no_boxes" if all_boxes == 0 else "ocr_boxes_no_date_candidate"
            )

        fallback_seconds = sum(
            float(row["processing_seconds"])
            for method, row in attempts
            if method != "original_512"
        )
        image_rows.append(
            {
                "file_name": name,
                "image_id": str(base_row["image_id"]),
                "true_year": str(base_row["true_year"]),
                "true_month": str(base_row["true_month"]),
                "true_day": str(base_row["true_day"]),
                "true_final_date": str(base_row["true_final_date"]),
                "pred_year": chosen_prediction["year"],
                "pred_month": chosen_prediction["month"],
                "pred_day": chosen_prediction["day"],
                "pred_final_date": chosen_prediction["final_date"],
                "year_correct": year_correct,
                "month_correct": month_correct,
                "day_correct": day_correct,
                "final_date_correct": final_correct,
                "selected_method": chosen_method if chosen_has_candidate else "original_no_candidate",
                "date_candidate_found": chosen_has_candidate,
                "failure_stage": failure_stage,
                "attempted_methods": json.dumps([method for method, _ in attempts]),
                "fallback_processing_seconds": fallback_seconds,
            }
        )
        attempted_passes[name] = attempts

    images = pd.DataFrame(image_rows)
    baseline_success = original["final_date_correct"].astype(bool).reset_index(drop=True)
    final_success = images["final_date_correct"].astype(bool)
    rotation_seconds = sum(
        float(row["processing_seconds"])
        for attempts in attempted_passes.values()
        for method, row in attempts
        if method == "rotation_270"
    )
    highres_seconds = sum(
        float(row["processing_seconds"])
        for attempts in attempted_passes.values()
        for method, row in attempts
        if method == "highres_1024"
    )
    clahe_seconds = sum(
        float(row["processing_seconds"])
        for attempts in attempted_passes.values()
        for method, row in attempts
        if method == "clahe"
    )
    fallback_ocr_seconds = sum(
        float(row["ocr_seconds"])
        for attempts in attempted_passes.values()
        for method, row in attempts
        if method != "original_512"
    )
    baseline_seconds = float(original["processing_seconds"].sum())
    baseline_init_seconds = float(
        pd.read_csv(BASE / f"{RUN}_300_summary.csv").iloc[0]["model_init_seconds"]
    )
    extra_seconds = rotation_seconds + highres_seconds + clahe_seconds
    failure_counts = images["failure_stage"].value_counts().to_dict()

    summary = {
        "images": 300,
        "pipeline": "original_512 -> rotation_270 -> highres_1024 -> clahe",
        "baseline_exact_matches": int(baseline_success.sum()),
        "rotation_control_exact_matches": 182,
        "final_exact_matches": int(final_success.sum()),
        "final_exact_accuracy": float(final_success.mean()),
        "change_vs_baseline_percentage_points": float(
            (final_success.sum() - baseline_success.sum()) / 300 * 100
        ),
        "change_vs_rotation_control_percentage_points": float(
            (final_success.sum() - 182) / 300 * 100
        ),
        "year_correct": int(images["year_correct"].sum()),
        "year_accuracy": float(images["year_correct"].mean()),
        "month_correct": int(images["month_correct"].sum()),
        "month_accuracy": float(images["month_correct"].mean()),
        "day_correct": int(images["day_correct"].sum()),
        "day_accuracy": float(images["day_correct"].mean()),
        "ocr_no_boxes": int(failure_counts.get("ocr_no_boxes", 0)),
        "ocr_boxes_no_date_candidate": int(
            failure_counts.get("ocr_boxes_no_date_candidate", 0)
        ),
        "candidate_but_wrong": int(failure_counts.get("candidate_but_wrong", 0)),
        "success": int(failure_counts.get("success", 0)),
        "rotation_270_attempts": stage_attempts["rotation_270"],
        "rotation_270_candidate_recoveries": stage_candidates["rotation_270"],
        "rotation_270_new_exact_matches": stage_saves["rotation_270"],
        "highres_1024_attempts": stage_attempts["highres_1024"],
        "highres_1024_candidate_recoveries": stage_candidates["highres_1024"],
        "highres_1024_new_exact_matches": stage_saves["highres_1024"],
        "clahe_attempts": stage_attempts["clahe"],
        "clahe_candidate_recoveries": stage_candidates["clahe"],
        "clahe_new_exact_matches": stage_saves["clahe"],
        "fallback_candidate_recoveries": sum(stage_candidates.values()),
        "new_exact_matches": sum(stage_saves.values()),
        "additional_ocr_calls": sum(stage_attempts.values()),
        "additional_ocr_seconds": fallback_ocr_seconds,
        "additional_processing_seconds": extra_seconds,
        "baseline_processing_seconds": baseline_seconds,
        "total_expected_processing_seconds": baseline_seconds + extra_seconds,
        "total_expected_minutes_excluding_init": (baseline_seconds + extra_seconds)
        / 60,
        "one_time_model_init_seconds": baseline_init_seconds,
        "total_expected_minutes_including_init": (
            baseline_seconds + extra_seconds + baseline_init_seconds
        )
        / 60,
        "regressions": int((baseline_success & ~final_success).sum()),
        "internal_development_evaluation": True,
    }
    if summary["final_exact_matches"] != 205:
        raise ValueError(f"Unexpected final result: {summary['final_exact_matches']}/300")
    if summary["total_expected_minutes_including_init"] > 40:
        raise ValueError("Final cascade exceeds the 40-minute limit")

    box_rows = []
    selected_by_name = images.set_index("file_name")["selected_method"].to_dict()
    for file_name, attempts in attempted_passes.items():
        selected_method = selected_by_name[file_name]
        if selected_method == "original_no_candidate":
            selected_method = "original_512"
        for method, row in attempts:
            for box_index, item in enumerate(detections(row)):
                box_rows.append(
                    {
                        "file_name": file_name,
                        "image_id": str(original_by_name[file_name]["image_id"]),
                        "pass_method": method,
                        "pass_selected": method == selected_method,
                        "pass_date_candidate_found": bool_value(
                            row["date_candidate_found"]
                        ),
                        "box_index": box_index,
                        "text": item["text"],
                        "confidence": item["confidence"],
                        "bbox": json.dumps(item["bbox"], ensure_ascii=False),
                    }
                )

    comparison = comparison_table(original, rotation, highres, clahe, summary)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    images.to_csv(OUTPUT / "final_cascade_images.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(box_rows).to_csv(
        OUTPUT / "final_cascade_ocr_boxes.csv", index=False, encoding="utf-8-sig"
    )
    images.loc[~images["final_date_correct"]].to_csv(
        OUTPUT / "final_cascade_failures.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([summary]).to_csv(
        OUTPUT / "final_cascade_summary.csv", index=False, encoding="utf-8-sig"
    )
    comparison.to_csv(OUTPUT / "method_comparison.csv", index=False, encoding="utf-8-sig")
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))
    print(comparison.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
