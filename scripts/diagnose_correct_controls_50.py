"""Frozen correct-control experiment: original_512 only; preflight unless --run-ocr."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from diagnose_early_stop_26 import ROOT, image_id, read_csv, sha

FROZEN_IDS = tuple(f"{i:06d}" for i in (
    18, 60, 75, 99, 280, 287, 321, 327, 384, 447,
    462, 645, 684, 690, 707, 725, 789, 912, 954, 1184,
    1185, 1295, 1409, 1426, 1471, 1742, 1823, 1898, 2033, 2037,
    2062, 2197, 2248, 2370, 2443, 2466, 2521, 2620, 2622, 2659,
    2674, 2835, 3006, 3010, 3043, 3090, 3129, 3148, 3238, 3280,
))
MODELS = ("PP-OCRv6_medium_det", "korean_PP-OCRv5_mobile_rec")
POLICIES = {"A": "q < 0.90", "B": "q < 0.90 OR M",
            "M": "at least two distinct find_all_candidates result tuples",
            "q": "minimum confidence matching selected source_text and bbox center"}


def validate_cohort(evaluation, failures):
    """Truth is used only to establish the already-frozen control membership."""
    from evaluation_common import normalize_truth
    rows = read_csv(evaluation)
    keys = [image_id(r["image_id"]) for r in rows]
    if len(rows) != 300 or len(set(keys)) != 300:
        raise ValueError("Require official evaluation with 300 unique images")
    eligible = sorted(image_id(r["image_id"]) for r in rows
                      if r["final_date"] == r["true_final_date"] and r["method"] == "original_512")
    if len(eligible) != 242:
        raise ValueError("Expected exactly 242 eligible correct original_512 stops")
    sampled = tuple(sorted(random.Random(42).sample(eligible, 50)))
    if len(FROZEN_IDS) != 50 or len(set(FROZEN_IDS)) != 50 or sampled != FROZEN_IDS:
        raise ValueError("Seed-42 sample does not equal the exact frozen 50 IDs")
    failure_ids = [image_id(r["image_id"]) for r in read_csv(failures)]
    if len(failure_ids) != 26 or len(set(failure_ids)) != 26:
        raise ValueError("Require the known 26-case artifact")
    if set(FROZEN_IDS) & set(failure_ids):
        raise ValueError("Frozen controls overlap known failure cases")
    return {image_id(r["image_id"]): normalize_truth(
        {k: r["true_" + k] for k in ("year", "month", "day", "final_date")})
        for r in rows if image_id(r["image_id"]) in FROZEN_IDS}


def evidence(detections):
    """Label-independent policy evaluation; no truth or image ID argument."""
    from date_parser import parse_expiration_date
    from date_parser.keywords import bbox_center
    from date_parser.select import find_all_candidates, select_final_date
    from date_parser.types import TextBox
    boxes = [TextBox.from_dict(d) for d in detections]
    for box in boxes:
        if not math.isfinite(box.confidence) or not 0 <= box.confidence <= 1:
            raise ValueError("Invalid recognition confidence")
    candidates = find_all_candidates(boxes)
    distinct = sorted({c.result.final_date_string() for c in candidates})
    selected = select_final_date(boxes)
    q = None
    indices = []
    if selected is not None:
        indices = [i for i, b in enumerate(boxes) if b.bbox
                   and b.text == selected.source_text and bbox_center(b.bbox) == selected.center]
        if not indices:
            raise ValueError("Cannot identify selected candidate source box")
        q = min(boxes[i].confidence for i in indices)
    # No selected candidate means q is undefined, not artificially low confidence.
    # Report this explicitly; do not silently add a third retry condition.
    policy_a = q < 0.90 if q is not None else None
    multiple = len(distinct) >= 2
    policy_b = True if multiple else policy_a
    return {
        "prediction": parse_expiration_date(detections),
        "has_candidate": selected is not None,
        "selected_candidate": repr(selected),
        "selected_source_text": selected.source_text if selected else None,
        "selected_source_box_indices": indices, "q": q,
        "detection_count": len(boxes), "candidate_count": len(candidates),
        "candidates": [{"final_date": c.result.final_date_string(),
                        "source_text": c.source_text, "center": c.center} for c in candidates],
        "distinct_parsed_dates": distinct, "distinct_candidate_count": len(distinct),
        "M": multiple, "policy_A_trigger": policy_a, "policy_B_trigger": policy_b,
    }


def summarize(rows):
    count = len(rows)
    total = sum(r["ocr_sec"] for r in rows)
    result = {"count": count,
              "stage1_still_correct_count": sum(r["stage1_still_correct"] for r in rows),
              "stage1_changed_ids": [r["image_id"] for r in rows if not r["stage1_still_correct"]],
              "total_original_512_ocr_sec": total,
              "mean_original_512_ocr_sec": total / count if count else None,
              "policies": POLICIES}
    for policy in ("A", "B"):
        field = f"policy_{policy}_trigger"
        ids = [r["image_id"] for r in rows if r[field] is True]
        result[f"policy_{policy}"] = {"trigger_count": len(ids), "trigger_ids": ids,
            "trigger_rate": len(ids) / count if count else None,
            "unevaluable_ids": [r["image_id"] for r in rows if r[field] is None]}
    result["caveat"] = ("Stage1 trigger measurement only, not retry/acceptance accuracy. "
        "Rates use all controls as denominator; undefined q is separately reported. "
        "OCR timings include output conversion, exclude decode/resize/parser/init; "
        "cached records retain original measured durations.")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evaluation", type=Path, default=ROOT / "docs/parser_integration_300_correct_0926/evaluation_300.csv")
    ap.add_argument("--failures", type=Path, default=ROOT / "docs/early_stop_26_run1/cases.csv")
    ap.add_argument("--images", type=Path, default=ROOT / "data")
    ap.add_argument("--output", type=Path, default=ROOT / "docs/correct_controls_50_run1")
    ap.add_argument("--cpu-threads", type=int, choices=(1, 2, 4), default=2)
    ap.add_argument("--run-ocr", action="store_true", help="Explicitly enable at most 50 original_512 attempts")
    args = ap.parse_args()
    truth = validate_cohort(args.evaluation, args.failures)
    images = {}
    for path in args.images.iterdir():
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"} and path.stem.isdigit():
            key = image_id(path.stem)
            if key in FROZEN_IDS:
                if key in images:
                    raise ValueError(f"Ambiguous image: {key}")
                images[key] = path
    if set(images) != set(FROZEN_IDS):
        raise ValueError(f"Missing controls: {set(FROZEN_IDS) - set(images)}")
    weights = [ROOT / "weights/paddleocr" / model / name for model in MODELS
               for name in ("inference.json", "inference.pdiparams", "inference.yml")]
    for path in weights:
        if not path.is_file():
            raise FileNotFoundError(path)
    code = [Path(__file__), ROOT / "scripts/diagnose_early_stop_26.py",
            ROOT / "scripts/evaluation_common.py", ROOT / "ocr_pipeline.py",
            *sorted((ROOT / "date_parser").glob("*.py"))]
    manifest = {
        "version": 1, "experiment": "frozen_correct_controls_50_original_512",
        "image_ids": FROZEN_IDS, "seed": 42, "eligible_count": 242,
        "sampling": "sorted(random.Random(42).sample(sorted(eligible_ids), 50))",
        "evaluation_path": str(args.evaluation.resolve()), "evaluation_sha256": sha(args.evaluation),
        "failures_path": str(args.failures.resolve()), "failures_sha256": sha(args.failures),
        "images": {k: {"path": str(images[k].resolve()), "sha256": sha(images[k])} for k in FROZEN_IDS},
        "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in code},
        "weights_sha256": {str(p.relative_to(ROOT)): sha(p) for p in weights},
        "models": MODELS, "python": sys.version, "policies": POLICIES,
        "packages": {p: importlib.metadata.version(p) for p in ("paddleocr", "paddlepaddle", "numpy", "Pillow")},
        "engine": {"cpu_threads": args.cpu_threads, "enable_mkldnn": True, "recognition_batch_size": 6},
        "predict": {"text_det_limit_side_len": 512, "text_det_limit_type": "max", "text_det_box_thresh": 0.7},
        "stage": "original_512",
    }
    manifest = json.loads(json.dumps(manifest))
    manifest_path = args.output / "manifest.json"
    if args.output.exists():
        if not manifest_path.is_file() or json.loads(manifest_path.read_text(encoding="utf8")) != manifest:
            raise ValueError("Output exists without matching provenance; use a new output directory")
    print(json.dumps(manifest, indent=2), flush=True)
    if not args.run_ocr:
        print("Preflight passed: exact frozen 50, no failure overlap. No OCR or output writes.")
        return
    if not args.output.exists():
        args.output.mkdir(parents=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf8")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.cpu_threads)
    import ocr_pipeline as pipeline
    engine = None
    rows = []
    fresh = 0
    init_sec = 0.0
    for key in FROZEN_IDS:
        cache = args.output / f"{key}_original_512.json"
        reused = cache.exists()
        if reused:
            record = json.loads(cache.read_text(encoding="utf8"))
            if record["image_id"] != key or record["stage"] != "original_512":
                raise ValueError(f"Invalid cache: {cache}")
        else:
            if engine is None:
                start = time.perf_counter()
                engine = pipeline.initialize_engine(cpu_threads=args.cpu_threads,
                    enable_mkldnn=True, recognition_batch_size=6)
                init_sec = time.perf_counter() - start
            array = pipeline.resize_image(pipeline.decode_image(images[key]), 512)
            start = time.perf_counter()
            detections = pipeline.paddle_to_common(engine.predict(array, **manifest["predict"]))
            duration = time.perf_counter() - start
            record = {"image_id": key, "stage": "original_512", "detections": detections,
                      "ocr_sec": duration, "stage_sec": duration}
            with cache.open("x", encoding="utf8") as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2)
            fresh += 1
        row = {"image_id": key, "stage": "original_512", **evidence(record["detections"]),
               "truth": truth[key], "ocr_sec": record["ocr_sec"], "reused": reused}
        row["stage1_still_correct"] = row["prediction"]["final_date"] == truth[key]["final_date"]
        rows.append(row)
        print(f"[{len(rows)}/50] {key}: correct={row['stage1_still_correct']} "
              f"A={row['policy_A_trigger']} B={row['policy_B_trigger']}", flush=True)
    with (args.output / "results.jsonl").open("w", encoding="utf8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {**summarize(rows), "fresh_attempts_this_invocation": fresh,
               "cached_attempts": len(rows) - fresh, "initialization_sec_this_invocation": init_sec}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
