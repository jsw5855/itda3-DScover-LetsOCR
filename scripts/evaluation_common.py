"""Shared evaluation normalization and immutable, reusable sample manifests."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from datetime import date
from pathlib import Path

from date_parser.interpret import DEFAULT_YEAR_MIN, DEFAULT_YEAR_MAX

FIELDS = ("year", "month", "day", "final_date")
VERSION = 1


def normalize_date(row):
    values = []
    for index, field in enumerate(FIELDS[:3]):
        value = str(row[field]).strip()
        if value.casefold() == "none":
            values.append("NONE")
            continue
        if not re.fullmatch(r"[0-9]+", value):
            raise ValueError(f"Invalid {field}: {value!r}")
        number = int(value)
        if index == 0:
            if len(value) <= 2:
                candidates = [century + number for century in (2000, 1900)
                              if DEFAULT_YEAR_MIN <= century + number <= DEFAULT_YEAR_MAX]
                if not candidates:
                    raise ValueError(f"Ambiguous/unresolved short year: {value}")
                number = candidates[0]
            elif len(value) != 4:
                raise ValueError(f"Invalid year width: {value}")
            if not 1 <= number <= 9999:
                raise ValueError("Invalid calendar year")
        elif not 1 <= number <= (12 if index == 1 else 31):
            raise ValueError(f"Invalid {field}: {value}")
        values.append(str(number).zfill(4 if index == 0 else 2))
    y, m, d = values
    if m != "NONE" and d != "NONE":
        date(2000 if y == "NONE" else int(y), int(m), int(d))
    return dict(zip(FIELDS, values + ["NONE" if values == ["NONE"] * 3 else "-".join(values)]))


def normalize_truth(row):
    truth = normalize_date(row)
    stored = str(row["final_date"]).strip()
    parts = ["NONE"] * 3 if stored.casefold() == "none" else stored.split("-")
    if len(parts) != 3 or normalize_date(dict(zip(FIELDS, parts))) != truth:
        raise ValueError(f"component/final_date mismatch: {row.get('image_id')}")
    return truth


def score_prediction(prediction, truth):
    # Keep the pipeline's actual output and strict exact-match metric. Normalize
    # truth only: reconstructing prediction.final_date would conceal output bugs.
    return {field + "_correct": prediction[field] == truth[field] for field in FIELDS}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def add_dataset_arguments(parser):
    parser.add_argument("--dataset-mode", choices=("baseline", "approved"), default="baseline")
    parser.add_argument("--sample-manifest", type=Path,
                        help="Existing immutable sample; create with scripts/prepare_evaluation_sample.py")


def load_population(labels, images, mode, statuses=None):
    with labels.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if not rows or any(None in r or None in r.values() for r in rows):
        raise ValueError("Empty or malformed labels")
    # Never let a review CSV bypass approval filtering via baseline mode.
    review = "review_status" in rows[0]
    if review != (mode == "approved"):
        raise ValueError("baseline requires legacy labels; approved requires review CSV")
    if mode == "baseline" and (len(rows) != 300 or statuses):
        raise ValueError("Baseline requires exactly 300 legacy rows and no status filter")
    statuses = set(statuses or ["approved"])
    if mode == "approved" and statuses != {"approved"}:
        raise ValueError("Ground-truth evaluation only permits review_status=approved")
    files = {}
    for path in images.iterdir():
        if path.is_file() and path.stem.isdigit() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            files.setdefault(str(int(path.stem)), []).append(path.name)
    result, seen = [], set()
    for row in rows:
        raw_id = row["image_id"].strip()
        if not re.fullmatch(r"[0-9]+", raw_id):
            raise ValueError("Invalid image_id")
        key = str(int(raw_id))
        if key in seen:
            raise ValueError(f"Duplicate image_id: {key}")
        seen.add(key)
        if review:
            if row["review_status"] not in statuses:
                continue
            if row.get("master_eligible") != "true" or not row.get("reviewer", "").strip() or not row.get("review_note", "").strip():
                raise ValueError(f"Incomplete approval: {key}")
            matches = files.get(key, [])
            if len(matches) != 1:
                raise ValueError(f"Image mapping is not unique: {key}")
            name = matches[0]
        else:
            name = row["file_name"]
            if Path(name).name != name or not (images / name).is_file() or not Path(name).stem.isdigit() or str(int(Path(name).stem)) != key:
                raise ValueError(f"Invalid image mapping: {key}")
        result.append({"image_id": key, "file_name": name, "truth": normalize_truth(row)})
    if not result:
        raise ValueError("No eligible labels")
    return result


def create_sample(labels, images, mode, size, seed, statuses=None):
    population = load_population(labels, images, mode, statuses)
    if not 1 <= size <= len(population):
        raise ValueError(f"Requested {size}; only {len(population)} eligible labels. No silent downsizing.")
    selected = random.Random(seed).sample(sorted(population, key=lambda r: int(r["image_id"])), size)
    result = {"version": VERSION, "mode": mode, "seed": seed,
              "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
              "population_fingerprint": digest(sorted(population, key=lambda r: int(r["image_id"]))),
              "entries": selected}
    result["sample_fingerprint"] = digest(selected)
    return result


def evaluation_dataset(args):
    population = load_population(args.labels, args.images, args.dataset_mode)
    if args.sample_manifest:
        manifest = json.loads(args.sample_manifest.read_text(encoding="utf-8"))
        if manifest["version"] != VERSION or manifest["mode"] != args.dataset_mode:
            raise ValueError("Sample version/mode mismatch")
        selected = manifest["entries"]
        if not selected or len({r['image_id'] for r in selected}) != len(selected):
            raise ValueError("Empty/duplicate sample")
        if digest(selected) != manifest["sample_fingerprint"]:
            raise ValueError("Sample fingerprint mismatch")
        current = {r["image_id"]: r for r in population}
        if any(current.get(r["image_id"]) != r for r in selected):
            raise ValueError("Sample label/approval/image mapping changed; create a new version")
    else:
        if args.dataset_mode != "baseline":
            raise ValueError("Approved evaluation requires --sample-manifest")
        selected = population
    metadata = {"dataset_mode": args.dataset_mode, "count": len(selected),
                "sample_fingerprint": digest(selected),
                "image_ids": [r["image_id"] for r in selected],
                "entries": selected,
                "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest()}
    return [r["file_name"] for r in selected], {r["file_name"]: r["truth"] for r in selected}, metadata


def sample_main():
    parser = argparse.ArgumentParser(description="Create reusable evaluation sample without OCR")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "approved"), required=True)
    parser.add_argument("--review-status", action="append", help="Approved only; unfinished labels are rejected")
    parser.add_argument("--size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = create_sample(args.labels, args.images, args.mode, args.size, args.seed, args.review_status)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(f"Saved {len(result['entries'])} images: {result['sample_fingerprint']}")
