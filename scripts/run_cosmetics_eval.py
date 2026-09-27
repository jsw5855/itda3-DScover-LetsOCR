"""Run the submission pipeline on labelled cosmetics photos and keep the raw OCR.

Official predictions are exactly ocr_pipeline.predict_image (via
predict_image_detailed). A recording wrapper around engine.predict keeps every
stage's detections; in full-stage mode the stages the cascade did not need are
run afterwards with the same preprocessing and detection parameters, so any
later parser can be re-scored from the dump without OCR.

The dump (ocr_dump_<split>.jsonl) uses the docs/run701/ocr_dump.jsonl layout
(image_id, file, method, prediction, stages[stage, seconds, has_candidate,
prediction, detections]) plus extra fields, so scripts/review_routing_rates.py
--dump and the production-cascade re-scoring here read it directly.

Nothing in date_parser/ or ocr_pipeline.py is changed; labels are used only
after predictions, for scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import ocr_pipeline  # noqa: E402
from date_parser.select import find_all_candidates, select_final_date  # noqa: E402
from date_parser.types import TextBox  # noqa: E402
from scripts import review_routing_rates as rr  # noqa: E402
from scripts.evaluation_common import normalize_truth, score_prediction  # noqa: E402

STAGES = ocr_pipeline.STAGES
MODES = ("full-stage", "fast")
SPLITS = ("dev", "validation")
VALIDATION_CONFIRMATION = "개선 완료"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MODELS = ("PP-OCRv6_medium_det", "korean_PP-OCRv5_mobile_rec")
WEIGHT_FILES = ("inference.json", "inference.pdiparams", "inference.yml")
# Single-process submission settings (ocr_pipeline.run_submission with one process).
ENGINE_OPTIONS = {"enable_mkldnn": True, "cpu_threads": 4, "recognition_batch_size": 6}
DETECTION_KWARGS = {"text_det_limit_type": "max", "text_det_box_thresh": 0.7}
LABEL_FIELDS = ("file_name", "image_id", "year", "month", "day", "final_date")


class RunConditionError(RuntimeError):
    """Resume refused: the saved run was made under different conditions."""


# ---------------------------------------------------------------- inputs

def image_key(value):
    return f"{int(str(value).strip()):06d}"


def find_images(image_dir):
    """{image key: path}; .JPG and .jpg are the same file name."""
    images = {}
    for path in sorted(Path(image_dir).iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.stem.isdigit():
            key = image_key(path.stem)
            if key in images:
                raise ValueError(f"Two image files for {key}: {images[key].name}, {path.name}")
            images[key] = path
    return images


def label_digest(row):
    return hashlib.sha256(json.dumps({k: row.get(k, "") for k in LABEL_FIELDS}, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


def is_unlabeled(row):
    """Template row not filled in yet (check_labels EMPTY): all date fields blank."""
    return not any((row.get(f) or "").strip() for f in ("year", "month", "day", "final_date"))


def load_labels(csv_paths):
    """{image key: row} over all label files; duplicate IDs across files are an error.
    Rows not labelled yet are left out (check_labels reports them as EMPTY)."""
    labels = {}
    for csv_path in csv_paths:
        with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if is_unlabeled(row):
                    continue
                key = image_key(row["image_id"])
                if key in labels:
                    raise ValueError(f"Duplicate label for {key} ({csv_path})")
                labels[key] = {**row, "_label_file": Path(csv_path).name}
    return labels


def check_label_files(csv_paths, image_dir=None):
    """scripts/check_labels.check_files -> (errors, warnings, unlabeled image ids).
    ERROR stops the run; WARN is shown; EMPTY rows (not labelled yet) are skipped."""
    from scripts import check_labels
    issues = check_labels.check_files([Path(p) for p in csv_paths], Path(image_dir) if image_dir else None)

    def text(issue):
        return f"{Path(issue['file']).name}:{issue['line']} id={issue['image_id']} {issue['message']}"
    return ([text(i) for i in issues if i["level"] == "ERROR"],
            [text(i) for i in issues if i["level"] == "WARN"],
            [str(i["image_id"]) for i in issues if i["level"] == "EMPTY"])


def match_inputs(image_dir, labels):
    images = find_images(image_dir)
    runnable = sorted(k for k in labels if k in images)
    return {"images": images, "runnable": runnable,
            "labels_without_image": sorted(k for k in labels if k not in images),
            "images_without_label": sorted(k for k in images if k not in labels)}


def weights_status():
    missing = [str(ROOT / "weights/paddleocr" / m / f) for m in MODELS for f in WEIGHT_FILES
               if not (ROOT / "weights/paddleocr" / m / f).is_file()]
    return {"weights_dir": str(ROOT / "weights/paddleocr"), "missing": missing}


def code_version():
    version_file = ROOT / "CODE_VERSION.txt"
    if version_file.is_file():
        return version_file.read_text(encoding="utf-8").strip()
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    files = [ROOT / "ocr_pipeline.py", Path(__file__), *sorted((ROOT / "date_parser").glob("*.py"))]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def run_conditions(split, mode):
    return {"split": split, "mode": mode, "code_version": code_version(), "source_sha256": source_hashes(),
            "engine_options": ENGINE_OPTIONS, "detection": DETECTION_KWARGS, "models": list(MODELS)}


def check_environment(image_dir, label_csvs, split="dev"):
    """Everything the notebook shows before running; no OCR."""
    labels = load_labels(label_csvs)
    matched = match_inputs(image_dir, labels)
    label_errors, label_warnings, unlabeled = check_label_files(label_csvs, image_dir)
    return {"python": sys.version.split()[0], "repo_dir": str(ROOT), "code_version": code_version(),
            "weights": weights_status(), "split": split, "labels": len(labels),
            "images": len(matched["images"]), "runnable": len(matched["runnable"]),
            "labels_without_image": matched["labels_without_image"],
            "images_without_label": matched["images_without_label"],
            "label_errors": label_errors, "label_warnings": label_warnings, "unlabeled_rows": unlabeled}


# ---------------------------------------------------------------- OCR

class RecordingEngine:
    """Passes predict calls through and keeps each call's detections and time."""

    def __init__(self, engine):
        self.engine = engine
        self.calls = []

    def predict(self, image, **kwargs):
        started = time.perf_counter()
        output = list(self.engine.predict(image, **kwargs))
        self.calls.append({"kwargs": dict(kwargs), "detections": ocr_pipeline.paddle_to_common(output),
                           "seconds": time.perf_counter() - started})
        return output


def stage_inputs(rgb):
    """Same preprocessing as ocr_pipeline.predict_image_detailed: {stage: (prepare, side)}."""
    base = ocr_pipeline.resize_image(rgb, 512)
    return {
        "original_512": (lambda: base, 512),
        "rotation_270": (lambda: np.asarray(Image.fromarray(base).rotate(270, expand=True)), 512),
        "highres_1024": (lambda: ocr_pipeline.resize_image(rgb, 1024), 1024),
        "clahe": (lambda: ocr_pipeline.apply_clahe(base), 512),
    }


def process_image(engine, path, mode):
    """Official prediction plus saved stages for one image."""
    recorder = RecordingEngine(engine)
    started = time.perf_counter()
    prediction, method, attempts = ocr_pipeline.predict_image_detailed(recorder, path)
    official_seconds = time.perf_counter() - started
    if len(recorder.calls) != len(attempts):
        raise AssertionError(f"{path.name}: {len(recorder.calls)} OCR calls for attempts {attempts}")
    calls = dict(zip(attempts, recorder.calls))
    extra_seconds = 0.0
    if mode == "full-stage" and len(calls) < len(STAGES):
        started = time.perf_counter()
        inputs = stage_inputs(ocr_pipeline.decode_image(path))
        for name in STAGES:
            if name not in calls:
                prepare, side = inputs[name]
                recorder.predict(prepare(), text_det_limit_side_len=side, **DETECTION_KWARGS)
                calls[name] = recorder.calls[-1]
        extra_seconds = time.perf_counter() - started
    stages = []
    for name in STAGES:
        if name in calls:
            stage_prediction, evidence = ocr_pipeline.stage_result(calls[name]["detections"])
            stages.append({"stage": name, "seconds": calls[name]["seconds"], "official": name in attempts,
                           "has_candidate": evidence is not None, "prediction": stage_prediction,
                           "detection_kwargs": calls[name]["kwargs"], "detections": calls[name]["detections"]})
    return {"image_id": image_key(path.stem), "file": path.name, "method": method, "prediction": prediction,
            "attempts": attempts, "official_seconds": official_seconds, "extra_seconds": extra_seconds,
            "stages": stages}


# ---------------------------------------------------------------- dump / resume

def output_paths(output_dir, split):
    output_dir = Path(output_dir)
    return {"dump": output_dir / f"ocr_dump_{split}.jsonl", "manifest": output_dir / f"run_manifest_{split}.json",
            "results": output_dir / f"results_{split}.csv", "summary": output_dir / f"summary_{split}.json",
            "errors": output_dir / f"errors_for_analysis_{split}.csv"}


def read_dump(path):
    """Completed records by image key. A partial last line (interrupted write) is cut off."""
    path = Path(path)
    if not path.exists():
        return {}, False
    data = path.read_bytes()
    truncated = bool(data) and not data.endswith(b"\n")
    if truncated:
        data = data[:data.rfind(b"\n") + 1]
        with path.open("r+b") as stream:
            stream.truncate(len(data))
    records = {}
    for line in data.decode("utf-8").split("\n"):
        if line.strip():
            record = json.loads(line)
            key = image_key(record["image_id"])
            if key in records:
                raise ValueError(f"Duplicate record in dump: {key}. Run repair_dump(path) once, "
                                 "then continue (the original file is kept as a backup).")
            records[key] = record
    return records, truncated


def repair_dump(path):
    """Keep the first record of each image and back up the original file.
    Returns {image key: whether the duplicates had identical OCR detections}."""
    path = Path(path)
    lines = [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    first, kept, duplicates = {}, [], {}
    for line in lines:
        record = json.loads(line)
        key = image_key(record["image_id"])
        if key not in first:
            first[key] = record
            kept.append(line)
            continue
        same = [s["detections"] for s in record["stages"]] == [s["detections"] for s in first[key]["stages"]]
        duplicates[key] = duplicates.get(key, True) and same
    if duplicates:
        backup = path.with_name(f"{path.name}.bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        path.replace(backup)
        path.write_text("".join(line + "\n" for line in kept), encoding="utf-8", newline="\n")
    return duplicates


class RunLock:
    """Exclusive per-output lock so two notebooks/kernels cannot write the same dump."""

    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.stream = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if self.stream.seek(0, 2) == 0:          # lock needs one byte on some Windows versions
                    self.stream.write(b"0")
                    self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            raise RuntimeError(f"Another run is already writing {self.path.parent} "
                               "(another notebook tab or kernel). Stop it first.") from error
        return self

    def __exit__(self, *exc):
        try:
            if os.name == "nt":
                import msvcrt
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self.stream.close()


def append_record(path, record):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def check_resume(paths, conditions, done, labels):
    """Refuse to continue a run made under other conditions or with changed labels."""
    if paths["manifest"].exists():
        saved = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        different = sorted(k for k in conditions if saved.get(k) != conditions[k])
        if different:
            raise RunConditionError(f"Run conditions differ from the saved run: {different}. "
                                    "Use a new OUTPUT_DIR or restore the previous code/mode.")
    elif done:
        raise RunConditionError("Dump exists without run manifest; use a new OUTPUT_DIR.")
    changed = sorted(k for k, r in done.items() if k in labels and r.get("label_sha256") != label_digest(labels[k]))
    if changed:
        raise RunConditionError(f"Labels changed for already processed images: {changed}. "
                                "Restore those labels or use a new OUTPUT_DIR.")


def require_validation_confirmation(split, confirmation):
    if split == "validation" and confirmation != VALIDATION_CONFIRMATION:
        raise PermissionError(f'Validation runs only after parser work is finished: set CONFIRM_VALIDATION = '
                              f'"{VALIDATION_CONFIRMATION}"')


def run(image_dir, label_csvs, split, output_dir, mode="full-stage", engine=None, limit=None,
        progress=None, confirm_validation=None, label_check=True):
    """OCR every labelled image not yet in the dump. Returns a dict of counts."""
    if split not in SPLITS or mode not in MODES:
        raise ValueError(f"split must be {SPLITS}, mode {MODES}")
    require_validation_confirmation(split, confirm_validation)
    labels = load_labels(label_csvs)
    if label_check:
        errors, _, _ = check_label_files(label_csvs, image_dir)
        if errors:
            raise ValueError("Label check failed:\n" + "\n".join(errors))
    matched = match_inputs(image_dir, labels)
    paths = output_paths(output_dir, split)
    paths["dump"].parent.mkdir(parents=True, exist_ok=True)
    with RunLock(paths["dump"].with_name(f"run_{split}.lock")):
        return _run_locked(paths, labels, matched, split, mode, engine, limit, progress)


def _run_locked(paths, labels, matched, split, mode, engine, limit, progress):
    done, truncated = read_dump(paths["dump"])
    conditions = run_conditions(split, mode)
    check_resume(paths, conditions, done, labels)
    if not paths["manifest"].exists():
        paths["manifest"].write_text(json.dumps({**conditions, "created_utc": datetime.now(timezone.utc).isoformat()},
                                                indent=2, ensure_ascii=False), encoding="utf-8")
    todo = [k for k in matched["runnable"] if k not in done]
    if limit is not None:
        todo = todo[:limit]
    init_seconds = 0.0
    if todo and engine is None:
        weights = weights_status()
        if weights["missing"]:
            raise FileNotFoundError("Missing model files:\n" + "\n".join(weights["missing"]))
        started = time.perf_counter()
        engine = ocr_pipeline.initialize_engine(**ENGINE_OPTIONS)
        init_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for index, key in enumerate(todo, 1):
        record = process_image(engine, matched["images"][key], mode)
        record.update(split=split, label_sha256=label_digest(labels[key]), label_file=labels[key]["_label_file"])
        append_record(paths["dump"], record)
        if progress is not None:
            elapsed = time.perf_counter() - started
            progress(index, len(todo), elapsed, elapsed / index * (len(todo) - index))
    return {"processed_now": len(todo), "already_done": len(done), "truncated_line_removed": truncated,
            "init_seconds": init_seconds, "run_seconds": time.perf_counter() - started,
            "labels_without_image": matched["labels_without_image"],
            "images_without_label": matched["images_without_label"]}


# ---------------------------------------------------------------- scoring

def digits_visible(truth, stages):
    """first_guess rule of docs/parser_review/run_701_ocr_dump.ipynb (cell 7): all truth
    numbers inside one OCR box of any saved stage -> the parser missed it."""
    need = []
    if truth["year"] != "NONE":
        need.append({truth["year"], truth["year"][2:]})
    if truth["month"] != "NONE":
        need.append({truth["month"], str(int(truth["month"]))})
    if truth["day"] != "NONE":
        need.append({truth["day"], str(int(truth["day"]))})
    if not need:
        return False
    for stage in stages:
        for detection in stage["detections"]:
            numbers = set(re.findall(r"\d+", detection["text"]))
            for number in list(numbers):
                if len(number) == 8:
                    numbers |= {number[:4], number[4:6], number[6:]}
            if all(options & numbers for options in need):
                return True
    return False


def score_record(record, label, rescore=False):
    """One results row. rescore=True re-parses saved stages with the current code."""
    saved = {s["stage"]: s["detections"] for s in record["stages"]}

    def facts_of(name):
        if name not in saved:
            raise rr.MissingStage(name)
        return rr.stage_facts(saved[name])

    try:
        replayed, method, facts = rr.replay(facts_of, "production")
    except rr.MissingStage as missing:
        replayed, method, facts = None, f"unreproducible:{missing}", None
    prediction = replayed if rescore else record["prediction"]
    if rescore and prediction is None:
        prediction = {"year": "", "month": "", "day": "", "final_date": "UNREPRODUCIBLE"}
    if not rescore:
        method = record["method"]
    truth = normalize_truth(label)
    scores = score_prediction(prediction, truth)
    final_stage = rr.FINAL_STAGE.get(method)
    boxes = [TextBox.from_dict(d) for d in saved.get(final_stage, [])]
    selected = select_final_date(boxes) if boxes else None
    candidates = find_all_candidates(boxes) if boxes else []
    wrong = not scores["final_date_correct"]
    return {
        "image_id": image_key(record["image_id"]), "split": record.get("split", ""),
        "true_final_date": truth["final_date"],
        "pred_year": prediction["year"], "pred_month": prediction["month"], "pred_day": prediction["day"],
        "pred_final_date": prediction["final_date"],
        "correct_final": scores["final_date_correct"], "correct_year": scores["year_correct"],
        "correct_month": scores["month_correct"], "correct_day": scores["day_correct"],
        "stop_stage": method, "total_seconds": round(record.get("official_seconds", 0.0), 3),
        "route": rr.route(facts, prediction, rr.THRESHOLD) if facts is not None else "",
        "q": facts["q"] if facts is not None else None,
        "distinct_dates": "|".join(facts["distinct"]) if facts is not None else "",
        "candidate_dates": " | ".join(f"{c.result.final_date_string()} <- {c.source_text}" for c in candidates),
        "selected_source_text": selected.source_text if selected is not None else "",
        "first_guess": ("Parser 의심" if digits_visible(truth, record["stages"]) else "OCR 의심") if wrong else "",
        "ocr_text": " | ".join(d["text"] for d in saved.get(final_stage, [])),
        "notes": label.get("notes", ""),
        "replay_matches_official": replayed is not None and replayed == record["prediction"],
    }


RESULT_COLUMNS = ["image_id", "split", "true_final_date", "pred_year", "pred_month", "pred_day", "pred_final_date",
                  "correct_final", "correct_year", "correct_month", "correct_day", "stop_stage", "total_seconds",
                  "route", "q", "distinct_dates", "candidate_dates", "selected_source_text", "first_guess",
                  "ocr_text", "notes"]


def write_csv(path, rows, columns):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def build_reports(label_csvs, split, output_dir, rescore=False):
    """results/summary (and dev errors file) from the dump and current labels. No OCR."""
    labels = load_labels(label_csvs)
    paths = output_paths(output_dir, split)
    records, _ = read_dump(paths["dump"])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8")) if paths["manifest"].exists() else {}
    rows = [score_record(records[k], labels[k], rescore) for k in sorted(records) if k in labels]
    suffix = "_rescored" if rescore else ""
    write_csv(paths["results"].with_name(f"results_{split}{suffix}.csv"), rows, RESULT_COLUMNS)
    if split == "dev":
        write_csv(paths["errors"].with_name(f"errors_for_analysis_{split}{suffix}.csv"),
                  [r for r in rows if not r["correct_final"]], RESULT_COLUMNS)
    seconds = [records[k].get("official_seconds", 0.0) for k in records]
    extra = [records[k].get("extra_seconds", 0.0) for k in records]
    n = len(rows)
    accuracy = {f: (sum(r[f"correct_{f}"] for r in rows) / n if n else None) for f in ("final", "year", "month", "day")}
    summary = {
        "split": split, "rescored_with_current_code": rescore, "images": n,
        "correct_final": sum(r["correct_final"] for r in rows), "accuracy": accuracy,
        "stop_stage": dict(Counter(r["stop_stage"] for r in rows)),
        "route": dict(Counter(r["route"] for r in rows)),
        "first_guess_of_wrong": dict(Counter(r["first_guess"] for r in rows if not r["correct_final"])),
        "replay_mismatch_ids": [r["image_id"] for r in rows if not r["replay_matches_official"]],
        "seconds": {"official_mean": statistics.mean(seconds) if seconds else None,
                    "official_median": statistics.median(seconds) if seconds else None,
                    "official_total": sum(seconds), "extra_full_stage_total": sum(extra)},
        "attempt_counts": dict(Counter(a for r in records.values() for a in r.get("attempts", []))),
        "environment": {"run": manifest, "python": sys.version.split()[0], "platform": platform.platform(),
                        "report_code_version": code_version(), "report_source_sha256": source_hashes(),
                        "built_utc": datetime.now(timezone.utc).isoformat(),
                        "label_files_sha256": {Path(p).name: sha(p) for p in label_csvs}},
    }
    summary_path = paths["summary"].with_name(f"summary_{split}{suffix}.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary, rows


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("check", "run", "report", "rescore"))
    ap.add_argument("--images", type=Path)
    ap.add_argument("--labels", type=Path, nargs="+", required=True)
    ap.add_argument("--split", choices=SPLITS, default="dev")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--mode", choices=MODES, default="full-stage")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--confirm-validation")
    args = ap.parse_args()
    if args.command == "check":
        print(json.dumps(check_environment(args.images, args.labels, args.split), indent=2, ensure_ascii=False))
        return
    if args.command == "run":
        def progress(done, total, elapsed, remaining):
            if done % 10 == 0 or done == total:
                print(f"[{done}/{total}] elapsed {elapsed:.0f}s, remaining ~{remaining:.0f}s", flush=True)
        print(json.dumps(run(args.images, args.labels, args.split, args.output, args.mode, limit=args.limit,
                             progress=progress, confirm_validation=args.confirm_validation), indent=2,
                         ensure_ascii=False))
    summary, _ = build_reports(args.labels, args.split, args.output, rescore=args.command == "rescore")
    print(json.dumps({k: summary[k] for k in ("images", "correct_final", "accuracy", "route", "stop_stage")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
