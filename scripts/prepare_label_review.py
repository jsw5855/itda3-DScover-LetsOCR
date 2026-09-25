"""Stage lossless label review; default is an in-memory audit, never a master export."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from date_parser.interpret import DEFAULT_YEAR_MIN, DEFAULT_YEAR_MAX

FIELDS = ("year", "month", "day", "final_date")
VISUAL_REVIEW = {"106", "157", "1862", "1906", "2152", "2284", "2448", "2521"}
DOCUMENTED_AMBIGUOUS = {"2034", "2917"}


def canonical_normalize(year, month, day, *, year_min=DEFAULT_YEAR_MIN,
                        year_max=DEFAULT_YEAR_MAX):
    """One normalization contract for comparison and staged serialization.

    Reject malformed/unresolved values instead of inventing NONE. Preserve valid
    four-digit years outside parser bounds for review, rather than erasing them.
    Calendar validation is separate so invalid source evidence remains visible.
    """
    values = []
    for i, value in enumerate((year, month, day)):
        value = str(value).strip()
        if value.casefold() == "none":
            values.append("NONE")
            continue
        if not re.fullmatch(r"[0-9]+", value):
            raise ValueError(f"Invalid component {value!r}")
        number = int(value)
        if i == 0:
            if len(value) <= 2:
                candidates = [c + number for c in (2000, 1900)
                              if year_min <= c + number <= year_max]
                if not candidates:
                    raise ValueError(f"Unresolved short year {value!r}")
                number = candidates[0]
            elif len(value) != 4:
                raise ValueError(f"Invalid year width {value!r}")
        values.append(f"{number:0{4 if i == 0 else 2}d}")
    final = "NONE" if values == ["NONE"] * 3 else "-".join(values)
    return dict(zip(FIELDS, values + [final]))


def normalize_stored_final(value):
    value = value.strip()
    if value.casefold() == "none":
        return canonical_normalize("NONE", "NONE", "NONE")["final_date"]
    parts = value.split("-")
    if len(parts) != 3:
        raise ValueError(f"Invalid final_date {value!r}")
    return canonical_normalize(*parts)["final_date"]


def calendar_error(c):
    y, m, d = [None if c[k] == "NONE" else int(c[k]) for k in FIELDS[:3]]
    if y is not None and not 1 <= y <= 9999:
        return "year_out_of_calendar_range"
    if m is not None and not 1 <= m <= 12:
        return "month_out_of_range"
    if d is not None and not 1 <= d <= 31:
        return "day_out_of_range"
    if m is not None and d is not None:
        try:
            date(y if y is not None else 2000, m, d)
        except ValueError:
            return "calendar_invalid"
    return ""


def image_key(value):
    value = value.strip()
    if not re.fullmatch(r"[0-9]+", value):
        raise ValueError(f"Expected numeric image_id, got {value!r}")
    return str(int(value))


def load_source(path, source):
    raw_bytes = path.read_bytes()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        schema = reader.fieldnames
        if not schema or len(schema) != len(set(schema)):
            raise ValueError(f"Missing/duplicate headers: {path}")
        if not {"image_id", *FIELDS}.issubset(schema):
            raise ValueError(f"Missing required columns: {path}")
        originals = list(reader)
    records = []
    for number, raw in enumerate(originals, 2):
        if None in raw or None in raw.values():
            raise ValueError(f"Malformed CSV record {number}: {path}")
        key = image_key(raw["image_id"])
        reasons, details, formatting = set(), [], []
        canonical = None
        try:
            canonical = canonical_normalize(*(raw[k] for k in FIELDS[:3]))
        except ValueError as error:
            reasons.add("invalid_component")
            details.append(str(error))
        if canonical:
            count = sum(canonical[k] == "NONE" for k in FIELDS[:3])
            if count:
                reasons.add("all_none" if count == 3 else "partial_none")
            if canonical["year"] != "NONE" and not DEFAULT_YEAR_MIN <= int(canonical["year"]) <= DEFAULT_YEAR_MAX:
                reasons.add("unusual_year")
            error = calendar_error(canonical)
            if error:
                reasons.add("calendar_invalid")
                details.append(error)
            if canonical["year"] == "NONE" and all(canonical[k] != "NONE" for k in ("month", "day")):
                m, d = int(canonical["month"]), int(canonical["day"])
                if 1 <= m <= 12 and 1 <= d <= 12 and m != d:
                    reasons.add("ambiguous_date_order")
                    details.append("Unknown year; swapping month/day gives another valid date (candidate only)")
            try:
                stored = normalize_stored_final(raw["final_date"])
                if stored != canonical["final_date"]:
                    reasons.add("component_final_mismatch")
                final_parts = ["NONE"] * 3 if stored == "NONE" else stored.split("-")
                if calendar_error(dict(zip(FIELDS[:3], final_parts))):
                    reasons.add("calendar_invalid")
                    details.append("Stored final_date is calendar-invalid")
            except ValueError as error:
                reasons.add("component_final_mismatch")
                details.append(str(error))
            formatting = [k for k in FIELDS if raw[k] != canonical[k]]
        if key in DOCUMENTED_AMBIGUOUS:
            reasons.add("ambiguous_date_order")
            details.append("Existing date_parser README documents order ambiguity")
        if key in VISUAL_REVIEW:
            reasons.add("ocr_visual_review")
            details.append("Previously visually inspected; no confirmed decision supplied; retain pending")
        records.append({"source_record_id": f"{source}:{number}", "source": source,
                        "source_path": str(path.resolve()), "source_row": number,
                        "image_id": key, "raw": raw, "canonical": canonical,
                        "review_reason": sorted(reasons), "details": details,
                        "format_changes": formatting})
    return records, {"path": str(path.resolve()), "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                     "schema": schema, "rows": len(records),
                     "unique_images": len({r['image_id'] for r in records})}


def build_review(paths):
    records, manifest = [], {}
    for source, path in paths.items():
        source_records, manifest[source] = load_source(path, source)
        records.extend(source_records)
    groups = defaultdict(list)
    for record in records:
        groups[record["image_id"]].append(record)
    image_paths = defaultdict(list)
    for path in (ROOT / "data").iterdir():
        if path.is_file() and path.stem.isdigit() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            image_paths[image_key(path.stem)].append(str(path.resolve()))
    images = []
    for key in sorted(groups, key=int):
        group = groups[key]
        reasons = {reason for r in group for reason in r["review_reason"]}
        candidates = {tuple(r["canonical"][k] for k in FIELDS) for r in group if r["canonical"]}
        if len(candidates) > 1:
            reasons.add("label_conflict")
            cs = [r["canonical"] for r in group if r["canonical"]]
            if any(a["year"] == b["year"] and a["month"] == b["day"] and a["day"] == b["month"]
                   and a["month"] != a["day"] for a in cs for b in cs):
                reasons.add("ambiguous_date_order")
        source_counts = Counter(r["source"] for r in group)
        if any(n > 1 for n in source_counts.values()):
            reasons.add("duplicate_source_id")
        if len(image_paths[key]) != 1:
            reasons.add("image_mapping_issue")
        candidate = dict(zip(FIELDS, next(iter(candidates)))) if len(candidates) == 1 and all(r["canonical"] for r in group) else {}
        source_columns = {}
        for source in paths:
            source_group = [r for r in group if r["source"] == source]
            # JSON arrays preserve even unexpected internal duplicate records without selection.
            source_columns[f"{source}_records_json"] = json.dumps(source_group, ensure_ascii=False)
            for field in (*FIELDS, "file_name", "notes", "Unnamed: 7"):
                values = [r["raw"].get(field, "") for r in source_group]
                source_columns[f"{source}_raw_{field}"] = values[0] if len(values) == 1 else json.dumps(values, ensure_ascii=False)
            for field in FIELDS:
                values = [(r["canonical"] or {}).get(field, "") for r in source_group]
                source_columns[f"{source}_canonical_{field}"] = values[0] if len(values) == 1 else json.dumps(values, ensure_ascii=False)
        images.append({"image_id": key, "image_path": image_paths[key][0] if len(image_paths[key]) == 1 else "",
                       "image_paths_json": json.dumps(image_paths[key], ensure_ascii=False),
                       "sources": ";".join(sorted(source_counts)),
                       "source_record_ids": ";".join(r["source_record_id"] for r in group),
                       "source_record_count": len(group),
                       **source_columns,
                       **{f"candidate_{k}": candidate.get(k, "") for k in FIELDS},
                       **{k: "" for k in FIELDS},
                       "review_reason": ";".join(sorted(reasons)),
                       "review_status": "pending" if reasons else "ready_for_review",
                       "previously_visually_reviewed": "true" if key in VISUAL_REVIEW else "false",
                       "prior_review_decision": "",
                       "reviewer": "", "review_note": "기존 이미지 확인 이력 있음. 구체 판정 미전달; 기존 판정 입력 후 동일 기준 재검수 필요." if key in VISUAL_REVIEW else "",
                       "master_eligible": "false"})
    summary = {"sources": manifest, "source_rows": len(records), "unique_images": len(images),
               "cross_source_overlap": sum(len({r['source'] for r in g}) > 1 for g in groups.values()),
               "review_reason_counts": dict(Counter(reason for r in images for reason in r["review_reason"].split(";") if reason)),
               "status_counts": dict(Counter(r["review_status"] for r in images)),
               "master_exported": False,
               "ambiguity_scope": "Source conflicts, partial month/day swaps, documented cases; not exhaustive image interpretation",
               "normalization_year_bounds": [DEFAULT_YEAR_MIN, DEFAULT_YEAR_MAX]}
    return records, images, summary


def write_outputs(directory, records, images, summary):
    # A new directory is mandatory. Never overwrite a previous review or input.
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "source_records.jsonl").open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    for name, rows in [("label_review.csv", images)]:
        with (directory / name).open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(images[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (directory / "audit_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", type=Path, default=ROOT / "labels/labels_300.csv")
    parser.add_argument("--incoming", type=Path, default=Path("C:/Users/jsw58/Downloads/labelling data.csv"))
    parser.add_argument("--write", action="store_true", help="Explicitly create staging outputs; never exports master truth")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.write and args.output_dir is None:
        parser.error("--write requires --output-dir (must not exist)")
    records, images, summary = build_review({"existing_300": args.existing, "incoming_432": args.incoming})
    if args.write:
        if not images:
            parser.error("Refusing empty output")
        write_outputs(args.output_dir, records, images, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
