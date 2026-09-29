"""Submission inference: local models, CPU, label-independent fallback selection."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from date_parser import parse_expiration_date
from date_parser.keywords import (
    ANCHOR_KEYWORDS,
    EXCLUDE_KEYWORDS,
    bbox_center,
    has_keyword,
)
from date_parser.select import find_all_candidates, select_final_date
from date_parser.types import TextBox

ROOT = Path(__file__).resolve().parent
COLUMNS = ["image_id", "year", "month", "day", "final_date"]


def initialize_engine(
    weights_dir=None,
    enable_mkldnn=True,
    cpu_threads=4,
    recognition_batch_size=6,
):
    weights = Path(weights_dir) if weights_dir is not None else ROOT / "weights/paddleocr"
    names = ("PP-OCRv6_medium_det", "korean_PP-OCRv5_mobile_rec")
    for name in names:
        for filename in ("inference.json", "inference.pdiparams", "inference.yml"):
            path = weights / name / filename
            if not path.is_file():
                raise FileNotFoundError(f"Required local model file: {path}")
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(ROOT / "weights/paddlex")
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_threads)
    import cv2
    cv2.setNumThreads(cpu_threads)
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang="korean", device="cpu",
        text_detection_model_name=names[0],
        text_detection_model_dir=str(weights / names[0]),
        text_recognition_model_name=names[1],
        text_recognition_model_dir=str(weights / names[1]),
        text_recognition_batch_size=recognition_batch_size,
        text_det_limit_side_len=512, text_det_limit_type="max",
        use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_textline_orientation=False, enable_mkldnn=enable_mkldnn,
        cpu_threads=cpu_threads,
    )


def decode_image(path):
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def resize_image(rgb, max_side):
    img = rgb
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize(tuple(max(1, round(v * scale)) for v in img.size), Image.Resampling.LANCZOS)
    return np.asarray(img)


def load_image(path, max_side=512):
    return resize_image(decode_image(path), max_side)


def paddle_to_common(predictions):
    detections = []
    for item in predictions:
        payload = getattr(item, "json", item)
        if callable(payload):
            payload = payload()
        if isinstance(payload, str):
            payload = json.loads(payload)
        payload = payload.get("res", payload)
        texts = payload["rec_texts"]
        scores = payload["rec_scores"]
        boxes = payload.get("rec_polys", payload.get("dt_polys"))
        if boxes is None or not (len(texts) == len(scores) == len(boxes)):
            raise ValueError("Inconsistent PaddleOCR text/score/polygon results")
        for text, score, box in zip(texts, scores, boxes):
            detections.append({"text": str(text), "confidence": float(score), "bbox": np.asarray(box).tolist()})
    return detections


def apply_clahe(rgb):
    import cv2
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2RGB)


def has_candidate(detections):
    boxes = [TextBox.from_dict(item) for item in detections]
    return select_final_date(boxes) is not None


STAGES = ("original_512", "rotation_270", "highres_1024", "clahe")
# Frozen retry policy: an original_512 date candidate is re-read once at 1024px
# when its source box recognition confidence is below this, or when the stage
# holds several distinct date readings.
RETRY_Q_THRESHOLD = 0.90


def stage_result(detections):
    """Parse one OCR stage into (prediction, evidence).

    evidence is None when the stage has no date candidate; otherwise q is the
    recognition confidence of the box the
    selected date came from (minimum over boxes with the same text and center),
    M is whether find_all_candidates yields at least two distinct dates.
    Selected/candidate dates and source keyword flags support retry comparison.
    """
    boxes = [TextBox.from_dict(item) for item in detections]
    prediction = parse_expiration_date(boxes)
    selected = select_final_date(boxes)
    if selected is None:
        return prediction, None
    q = min((box.confidence for box in boxes if box.bbox and box.text == selected.source_text
             and bbox_center(box.bbox) == selected.center), default=None)
    distinct = {candidate.result.final_date_string() for candidate in find_all_candidates(boxes)}
    self_anchor = has_keyword(selected.source_text, ANCHOR_KEYWORDS)
    self_exclude = has_keyword(selected.source_text, EXCLUDE_KEYWORDS)
    return prediction, {
        "q": q,
        "M": len(distinct) >= 2,
        "self_anchor": self_anchor and not self_exclude,
        "self_exclude": self_exclude and not self_anchor,
        "selected_date": selected.result.final_date_string(),
        "candidate_dates": sorted(distinct),
    }


def retry_triggered(evidence):
    """Policy B on an original_512 candidate: q < 0.90 OR M.

    A missing q (source box not found) cannot be trusted, so it also retries.
    """
    q = evidence["q"]
    return q is None or q < RETRY_Q_THRESHOLD or evidence["M"]


def prefer_retry(original, highres):
    """Take highres only when it has strictly higher q, except when doing so
    would replace a self-anchored expiration reading with a self-excluded
    manufacture/non-expiration reading.
    """
    if highres is None or highres["q"] is None or original["q"] is None:
        return False
    if original.get("self_anchor", False) and highres.get("self_exclude", False):
        return False
    # Higher recognition confidence on an already-rejected earlier date is
    # not evidence that the later expiry disappeared from the package. Keep
    # the original when highres merely drops that expiry and repeats the
    # original's other date, unless highres explicitly anchors it. Completing
    # an earlier year/month is the same case, but a same-month day correction
    # is not: it can be a legitimate refinement of the selected expiry.
    old_date = original.get("selected_date", "NONE")
    new_date = highres.get("selected_date", "NONE")
    if (original["M"] and not highres["M"]
            and not original.get("self_exclude", False)
            and not highres.get("self_anchor", False)
            and "NONE" not in old_date and "NONE" not in new_date
            and new_date < old_date
            and (new_date in original.get("candidate_dates", ())
                 or (new_date[:7] < old_date[:7]
                     and new_date[:7] + '-NONE' in original.get("candidate_dates", ())))):
        return False
    return highres["q"] > original["q"]

def run_cascade(run_stage):
    """Run the baseline, then retry only an exhausted NONE at 1024px CLAHE."""
    prediction, method, attempts = _run_baseline_cascade(run_stage)
    if prediction["final_date"] != "NONE":
        return prediction, method, attempts
    retry, evidence = run_stage("clahe_1024")
    attempts = attempts + ["clahe_1024"]
    if evidence is not None:
        return retry, "clahe_1024", attempts
    return prediction, method, attempts


def _run_baseline_cascade(run_stage):
    """Stage control flow, independent of how a stage is produced.

    run_stage(name) -> (prediction, evidence) as from stage_result. Returns
    (prediction, method, attempted stage names in order).
    """
    attempts = []

    def run(name):
        attempts.append(name)
        return run_stage(name)

    original, evidence = run("original_512")
    if evidence is not None:
        if not retry_triggered(evidence):
            return original, "original_512", attempts
        highres, highres_evidence = run("highres_1024")
        if prefer_retry(evidence, highres_evidence):
            return highres, "highres_1024_retry", attempts
        return original, "original_512_retry_kept", attempts
    # No original candidate: unchanged fallback, first stage with a candidate wins.
    for name in STAGES[1:]:
        prediction, stage_evidence = run(name)
        if stage_evidence is not None:
            return prediction, name, attempts
    return original, "original_no_candidate", attempts


def predict_image_detailed(engine, path):
    rgb = decode_image(path)
    base = resize_image(rgb, 512)
    stages = {
        "original_512": (lambda: base, 512),
        "rotation_270": (lambda: np.asarray(Image.fromarray(base).rotate(270, expand=True)), 512),
        "highres_1024": (lambda: resize_image(rgb, 1024), 1024),
        "clahe": (lambda: apply_clahe(base), 512),
        "clahe_1024": (lambda: apply_clahe(resize_image(rgb, 1024)), 1024),
    }

    def run_stage(name):
        prepare, side = stages[name]
        return stage_result(paddle_to_common(engine.predict(
            prepare(), text_det_limit_side_len=side,
            text_det_limit_type="max", text_det_box_thresh=0.7,
        )))

    return run_cascade(run_stage)


def predict_image(engine, path):
    prediction, method, _ = predict_image_detailed(engine, path)
    return prediction, method


_WORKER_ENGINE = None


def _initialize_submission_worker():
    global _WORKER_ENGINE
    _WORKER_ENGINE = initialize_engine(
        enable_mkldnn=True,
        cpu_threads=2,
        recognition_batch_size=6,
    )


def _predict_submission_item(item):
    index, path_str, image_id = item
    prediction, method = predict_image(
        _WORKER_ENGINE,
        Path(path_str),
    )
    return index, image_id, prediction, method


def run_submission(input_dir, output_path, engine=None):
    import time

    started = time.monotonic()

    directory = Path(input_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Input image directory: {directory.resolve()}")

    images = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise ValueError(f"No jpg/jpeg/png images in {directory}")

    ids = [p.stem for p in images]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate image IDs in input filenames")

    processes = int(os.environ.get("ITDA_OCR_PROCESSES", "2"))

    if processes not in (1, 2):
        raise ValueError("ITDA_OCR_PROCESSES must be 1 or 2")

    if engine is not None or processes == 1:
        if engine is None:
            engine = initialize_engine(
                enable_mkldnn=True,
                cpu_threads=4,
                recognition_batch_size=6,
            )

        rows = []
        for index, (path, image_id) in enumerate(zip(images, ids), 1):
            prediction, method = predict_image(engine, path)
            rows.append({"image_id": image_id, **prediction})
            if index % 20 == 0 or index == len(images):
                print(f"[{index}/{len(images)}] {path.name}: {method}", flush=True)
    else:
        jobs = [
            (index, str(path), image_id)
            for index, (path, image_id) in enumerate(zip(images, ids))
        ]

        from submission_runtime import run_workers

        def progress(count):
            if count % 20 == 0 or count == len(images):
                print(f"[{count}/{len(images)}] elapsed={time.monotonic()-started:.1f}s", flush=True)

        print(f'[runtime] starting {processes} workers x 2 threads; {len(images)} images', flush=True)
        results = run_workers(
            jobs, _initialize_submission_worker, _predict_submission_item,
            processes=processes, threads=2, progress=progress,
        )

        results.sort(key=lambda x: x[0])
        rows = [
            {"image_id": image_id, **prediction}
            for _, image_id, prediction, _ in results
        ]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(output)
    print(f'[runtime] CSV saved; returning {len(rows)} rows; {time.monotonic()-started:.1f}s', flush=True)
    return rows

