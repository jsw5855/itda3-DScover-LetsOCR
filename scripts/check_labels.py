"""Check label CSVs before evaluation (e.g. the new 9xxxxx cosmetics labels).

ERROR   the evaluation scripts would reject or misread the row
        (normalize_truth fails, file_name/image_id mismatch, duplicate id).
WARN    accepted by evaluation but inconsistent with the team format
        (lowercase none, missing zero padding, 2-digit year, text in an
        unnamed column, id outside --id-range, image file missing).
EMPTY   template row not labeled yet (all of year..final_date blank).

Usage:
    python scripts/check_labels.py labels_a.csv [labels_b.csv ...]
        [--images DIR] [--id-range 900001-900400] [--out issues.csv]
Exit status is 1 when any ERROR is found.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation_common import normalize_truth  # noqa: E402

REQUIRED = ("file_name", "image_id", "year", "month", "day", "final_date")
DATE_FIELDS = ("year", "month", "day", "final_date")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WIDTH = {"year": 4, "month": 2, "day": 2}


def check_row(row, source, line, id_range=None):
    issues = []

    def add(level, message):
        issues.append({"level": level, "file": source, "line": line,
                       "image_id": row.get("image_id", ""), "message": message})

    name = (row.get("file_name") or "").strip()
    raw_id = (row.get("image_id") or "").strip()
    stem, suffix = Path(name).stem, Path(name).suffix.lower()
    if not re.fullmatch(r"[0-9]+", raw_id):
        add("ERROR", f"image_id is not a number: {raw_id!r}")
    elif not stem.isdigit() or int(stem) != int(raw_id):
        add("ERROR", f"file_name {name!r} does not match image_id {raw_id}")
    elif id_range and not id_range[0] <= int(raw_id) <= id_range[1]:
        add("WARN", f"image_id {raw_id} outside {id_range[0]}-{id_range[1]}")
    if name and suffix not in IMAGE_SUFFIXES:
        add("ERROR", f"unsupported image extension: {name!r}")

    values = {f: (row.get(f) or "").strip() for f in DATE_FIELDS}
    if not any(values.values()):
        add("EMPTY", "not labeled yet")
        return issues
    missing = [f for f, v in values.items() if not v]
    if missing:
        add("ERROR", f"blank field(s): {', '.join(missing)} (write NONE for unknown parts)")
        return issues
    try:
        truth = normalize_truth({**values, "image_id": raw_id})
    except ValueError as error:
        add("ERROR", f"invalid date label: {error}")
        return issues

    for field in DATE_FIELDS:
        if (row.get(field) or "") != values[field]:
            add("WARN", f"{field} {row.get(field)!r}: remove spaces")
    for field in WIDTH:
        if values[field].casefold() == "none" and values[field] != "NONE":
            add("WARN", f"{field} {values[field]!r}: write NONE in capitals")
    for field, width in WIDTH.items():
        value = values[field]
        if value.isdigit() and len(value) != width:
            add("WARN", f"{field} {value!r}: use {width} digits ({truth[field]})")
    if values["final_date"] != truth["final_date"]:
        add("WARN", f"final_date {values['final_date']!r}: canonical form is {truth['final_date']}")

    extra = {k: v for k, v in row.items() if k not in REQUIRED and k != "notes" and k and (v or "").strip()}
    for key, value in extra.items():
        add("WARN", f"text in column {key!r}: {value.strip()!r} (move it into notes)")
    return issues


def check_files(paths, images=None, id_range=None):
    issues, seen = [], {}
    labeled_names = set()
    for path in paths:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames or []
            lacking = [c for c in REQUIRED if c not in header]
            if lacking:
                issues.append({"level": "ERROR", "file": str(path), "line": 1, "image_id": "",
                               "message": f"missing column(s): {', '.join(lacking)}"})
                continue
            for line, row in enumerate(reader, start=2):
                if None in row:
                    issues.append({"level": "ERROR", "file": str(path), "line": line,
                                   "image_id": row.get("image_id", ""),
                                   "message": "more cells than header columns (unquoted comma in notes?)"})
                row_issues = check_row(row, str(path), line, id_range)
                issues.extend(row_issues)
                raw_id = (row.get("image_id") or "").strip()
                if raw_id.isdigit():
                    key = int(raw_id)
                    if key in seen:
                        issues.append({"level": "ERROR", "file": str(path), "line": line, "image_id": raw_id,
                                       "message": f"duplicate image_id (first at {seen[key]})"})
                    else:
                        seen[key] = f"{Path(path).name}:{line}"
                if images is not None and not any(i["level"] == "EMPTY" for i in row_issues):
                    name = (row.get("file_name") or "").strip()
                    labeled_names.add(name)
                    if name and not (Path(images) / name).is_file():
                        issues.append({"level": "WARN", "file": str(path), "line": line, "image_id": raw_id,
                                       "message": f"image file not found: {name}"})
    if images is not None:
        for image in sorted(Path(images).iterdir()):
            if image.suffix.lower() in IMAGE_SUFFIXES and image.name not in labeled_names:
                issues.append({"level": "WARN", "file": str(images), "line": "", "image_id": image.stem,
                               "message": f"image has no label row: {image.name}"})
    return issues


def parse_range(text):
    low, high = (int(v) for v in text.split("-"))
    return low, high


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", nargs="+", type=Path)
    parser.add_argument("--images", type=Path, help="image folder to cross-check file names")
    parser.add_argument("--id-range", type=parse_range, help="expected id range, e.g. 900001-900400")
    parser.add_argument("--out", type=Path, help="write every issue to this CSV")
    parser.add_argument("--show-empty", action="store_true", help="list unlabeled rows one by one")
    args = parser.parse_args(argv)

    issues = check_files(args.labels, args.images, args.id_range)
    counts = {level: sum(i["level"] == level for i in issues) for level in ("ERROR", "WARN", "EMPTY")}
    for issue in issues:
        if issue["level"] == "EMPTY" and not args.show_empty:
            continue
        print(f"{issue['level']:5} {Path(issue['file']).name}:{issue['line']} id={issue['image_id']} {issue['message']}")
    print(f"\nERROR {counts['ERROR']}  WARN {counts['WARN']}  EMPTY(미작성) {counts['EMPTY']}")
    if args.out:
        with args.out.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["level", "file", "line", "image_id", "message"])
            writer.writeheader()
            writer.writerows(issues)
    return 1 if counts["ERROR"] else 0


if __name__ == "__main__":
    sys.exit(main())
