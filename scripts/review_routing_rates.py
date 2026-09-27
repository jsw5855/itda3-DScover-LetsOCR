"""Staff-review routing rates from saved OCR text. No OCR, no model weights.

Each image is re-parsed with the current date_parser on its saved stage detections,
the cascade is replayed, and the final stage is routed (first match wins):

  MANUAL   final_date contains "NONE" (partial NONE included)
  CHOOSE   at least two distinct find_all_candidates dates (M)
  RECHECK  selected candidate q < threshold, or q undefined
  CONFIRM  otherwise

q and M come from production ocr_pipeline.stage_result (same definitions as
scripts/diagnose_correct_controls_50.py evidence()). Labels are read only after
routing, for the reference accuracy tables. An image whose replay needs a stage
that was not saved is "unreproducible" and left out of the rates.

--cascade first_candidate  stop at the first stage with a candidate (pre-Policy-B)
--cascade production       ocr_pipeline.run_cascade of the checked-out code
                           (Policy B as of main bb83a74)

Sources (exactly one):
--full-stage-raw   docs/full_stage_701_run1/raw_ocr.jsonl: every image x 4 stages,
                   re-parsed here with the current code. Preferred for final numbers.
--dump             docs/run701/ocr_dump.jsonl: only the stages the 2026-09-23 run
                   reached; images needing an unsaved stage are unreproducible.
--stage-snapshots  per-stage parser results saved from the full-stage dump
                   (docs/parser_optimization2/new_parser_replay.json); the parse is
                   the saved one, not re-run here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ocr_pipeline  # noqa: E402
from date_parser.select import find_all_candidates  # noqa: E402
from date_parser.types import TextBox  # noqa: E402
from scripts.evaluation_common import normalize_truth  # noqa: E402

ROUTES = ("MANUAL", "CHOOSE", "RECHECK", "CONFIRM")
REVIEW = ("MANUAL", "CHOOSE", "RECHECK")
THRESHOLD = 0.90
SENSITIVITY = (0.85, 0.90, 0.95)
FINAL_STAGE = {"original_512": "original_512", "original_512_retry_kept": "original_512",
               "original_no_candidate": "original_512", "highres_1024_retry": "highres_1024",
               "rotation_270": "rotation_270", "highres_1024": "highres_1024", "clahe": "clahe"}


class MissingStage(Exception):
    pass


def stage_facts(detections):
    """prediction, q (None if no candidate/undefined), sorted distinct dates."""
    prediction, evidence = ocr_pipeline.stage_result(detections)
    boxes = [TextBox.from_dict(d) for d in detections]
    distinct = sorted({c.result.final_date_string() for c in find_all_candidates(boxes)})
    if evidence is not None and evidence["M"] != (len(distinct) >= 2):
        raise AssertionError("M definition drift")
    return {"prediction": prediction, "evidence": evidence,
            "q": evidence["q"] if evidence else None, "distinct": distinct}


def route(facts, prediction, threshold):
    if "NONE" in prediction["final_date"]:
        return "MANUAL"
    if len(facts["distinct"]) >= 2:
        return "CHOOSE"
    if facts["q"] is None or facts["q"] < threshold:
        return "RECHECK"
    return "CONFIRM"


def snapshot_facts(snapshot):
    evidence = {"q": snapshot["q"], "M": snapshot["M"]} if snapshot["has_candidate"] else None
    return {"prediction": snapshot["prediction"], "evidence": evidence,
            "q": evidence["q"] if evidence else None, "distinct": sorted(snapshot["distinct_dates"])}


def replay(stage_facts_of, cascade):
    """stage_facts_of(name) -> facts or raises MissingStage. Returns (prediction, method, final stage facts)."""
    cache = {}

    def facts(name):
        if name not in cache:
            cache[name] = stage_facts_of(name)
        return cache[name]

    if cascade not in ("first_candidate", "production"):
        raise ValueError(f"Unknown cascade: {cascade}")
    if cascade == "first_candidate":
        for name in ocr_pipeline.STAGES:
            if facts(name)["evidence"] is not None:
                return facts(name)["prediction"], name, facts(name)
        return facts("original_512")["prediction"], "original_no_candidate", facts("original_512")
    prediction, method, _ = ocr_pipeline.run_cascade(lambda n: (facts(n)["prediction"], facts(n)["evidence"]))
    return prediction, method, facts(FINAL_STAGE[method])


def load_labels(path):
    rows = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            rows[f"{int(row['image_id']):06d}"] = row
    return rows


def pct(n, d):
    return round(100 * n / d, 1) if d else None


def wilson(k, n, z=1.96):
    """95% Wilson score interval, in percent."""
    if not n:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * max(0.0, centre - half), 1), round(100 * min(1.0, centre + half), 1)]


def error_direction(pred, truth):
    """For a wrong date: 'late' (predicted after truth: an expired item could pass),
    'early' (predicted before truth: a sellable item could be discarded), or
    'not_comparable' when either side has a NONE field."""
    if pred == truth:
        return ""
    if "NONE" in pred or "NONE" in truth:
        return "not_comparable"
    return "late" if pred > truth else "early"


def rate_table(rows, key="route"):
    n = len(rows)
    counts = Counter(r[key] for r in rows)
    table = {r: {"count": counts.get(r, 0), "pct": pct(counts.get(r, 0), n), "ci95": wilson(counts.get(r, 0), n)}
             for r in ROUTES}
    review = sum(counts.get(r, 0) for r in REVIEW)
    return {"n": n, **table, "staff_review": {"count": review, "pct": pct(review, n), "ci95": wilson(review, n)},
            "per_100_items": {r: round(100 * counts.get(r, 0) / n, 1) if n else None for r in ROUTES}}


def accuracy_table(rows, key="route"):
    out = {}
    for status in ("approved", "candidate"):
        group = [r for r in rows if r["label_status"] == status]
        out[status] = {r: {"correct": sum(x["correct"] for x in group if x[key] == r),
                           "n": sum(1 for x in group if x[key] == r)} for r in ROUTES}
    return out


def risk(rows, key="route"):
    wrong = [r for r in rows if r[key] == "CONFIRM" and not r["correct"]]
    confirm = sum(1 for r in rows if r[key] == "CONFIRM")
    return {"count": len(wrong), "pct_of_all": pct(len(wrong), len(rows)), "ci95_of_all": wilson(len(wrong), len(rows)),
            "pct_of_confirm": pct(len(wrong), confirm), "ci95_of_confirm": wilson(len(wrong), confirm),
            "by_status": dict(Counter(r["label_status"] for r in wrong)),
            "direction": {d: sorted(r["image_id"] for r in wrong if r["error_direction"] == d)
                          for d in ("late", "early", "not_comparable")},
            "ids": {s: sorted(r["image_id"] for r in wrong if r["label_status"] == s) for s in ("approved", "candidate")}}


def dump_images(path):
    """(image id, stage facts provider, saved old prediction) per dump line."""
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            saved = {s["stage"]: s["detections"] for s in record["stages"]}

            def provider(name, saved=saved):
                if name not in saved:
                    raise MissingStage(name)
                return stage_facts(saved[name])
            yield f"{int(record['image_id']):06d}", provider, record["prediction"]["final_date"]


def full_stage_images(path):
    """raw_ocr.jsonl of scripts/dump_full_stage_701.py: one line per image x stage."""
    saved = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            stages = saved.setdefault(f"{int(record['image_id']):06d}", {})
            if record["stage"] in stages:
                raise ValueError(f"Duplicate stage record: {record['image_id']}/{record['stage']}")
            stages[record["stage"]] = record["detections"]
    for key in sorted(saved):
        def provider(name, stages=saved[key]):
            if name not in stages:
                raise MissingStage(name)
            return stage_facts(stages[name])
        yield key, provider, None


def snapshot_images(path):
    stages = json.loads(Path(path).read_text(encoding="utf-8"))["stages"]
    for key, records in stages.items():
        by_name = dict(zip(ocr_pipeline.STAGES, records))
        yield f"{int(key):06d}", (lambda name, by_name=by_name: snapshot_facts(by_name[name])), None


def analyze(images, labels_path, cascade):
    labels = load_labels(labels_path)
    rows, unreproducible = [], []
    for key, provider, old_pred in images:
        label = labels[key]
        base = {"image_id": key, "label_status": label["truth_source"],
                "group": "existing_300" if "existing_300" in label["label_sources"] else "new_401",
                "old_pred": old_pred}
        try:
            prediction, method, facts = replay(provider, cascade)
        except MissingStage as missing:
            unreproducible.append({**base, "reproducible": False, "missing_stage": str(missing)})
            continue
        truth = normalize_truth(label)["final_date"]
        row = {**base, "reproducible": True, "stop_stage": method, "q": facts["q"],
               "distinct_dates": "|".join(facts["distinct"]), "pred": prediction["final_date"],
               "truth": truth, "correct": prediction["final_date"] == truth,
               "error_direction": error_direction(prediction["final_date"], truth)}
        for threshold in SENSITIVITY:
            row[f"route_q{threshold:.2f}"] = route(facts, prediction, threshold)
        row["route"] = row[f"route_q{THRESHOLD:.2f}"]
        rows.append(row)
    return rows, unreproducible


def summarize(rows, unreproducible, args):
    groups = {"all": rows, **{g: [r for r in rows if r["group"] == g] for g in ("existing_300", "new_401")}}
    summary = {
        "cascade": args.cascade, "q_threshold": THRESHOLD, "ocr_invocations": 0,
        "inputs": {"source": display(source_path(args)), "source_sha256": sha(source_path(args)),
                   "labels": display(args.labels), "labels_sha256": sha(args.labels)},
        "code_commit": git_head(),
        "images": len(rows) + len(unreproducible), "reproducible": len(rows),
        "unreproducible": {"count": len(unreproducible),
                           "by_missing_stage": dict(Counter(r["missing_stage"] for r in unreproducible)),
                           "by_status": dict(Counter(r["label_status"] for r in unreproducible)),
                           "ids": [r["image_id"] for r in unreproducible]},
        "prediction_changed_vs_saved": (sum(r["pred"] != r["old_pred"] for r in rows)
                                        if all(r["old_pred"] is not None for r in rows) else None),
        "correct_reproducible": sum(r["correct"] for r in rows),
        "A_route_rates": {g: rate_table(v) for g, v in groups.items()},
        "B_route_accuracy_reference": accuracy_table(rows),
        "C_confirm_but_wrong": risk(rows),
        "C_approved_only": risk([r for r in rows if r["label_status"] == "approved"]),
        "wrong_direction_by_route": {rt: dict(Counter(r["error_direction"] for r in rows if r["route"] == rt and not r["correct"]))
                                     for rt in ROUTES},
        "D_sensitivity_reference_not_policy": {
            f"{t:.2f}": {"rates": rate_table(rows, f"route_q{t:.2f}"), "confirm_but_wrong": risk(rows, f"route_q{t:.2f}")}
            for t in SENSITIVITY},
    }
    assert sum(summary["A_route_rates"]["all"][r]["count"] for r in ROUTES) + len(unreproducible) == summary["images"]
    return summary


def source_path(args):
    return args.full_stage_raw or args.dump or args.stage_snapshots


def display(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--full-stage-raw", type=Path, help="raw_ocr.jsonl, every image x 4 stages (preferred)")
    source.add_argument("--dump", type=Path, help="ocr_dump.jsonl (one image per line, saved stages)")
    source.add_argument("--stage-snapshots", type=Path, help="per-stage parser results of a full four-stage dump")
    ap.add_argument("--labels", type=Path, default=ROOT / "tmp_labels_701.csv")
    ap.add_argument("--cascade", choices=("first_candidate", "production"), default="production")
    ap.add_argument("--output", type=Path, help="New directory for routing_rows.csv and routing_summary.json")
    args = ap.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    images = (full_stage_images(args.full_stage_raw) if args.full_stage_raw
              else dump_images(args.dump) if args.dump else snapshot_images(args.stage_snapshots))
    rows, unreproducible = analyze(images, args.labels, args.cascade)
    summary = summarize(rows, unreproducible, args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output is not None:
        args.output.mkdir(parents=True)
        columns = ["image_id", "label_status", "group", "stop_stage", "route", "q", "distinct_dates", "pred", "truth",
                   "correct", "error_direction", "reproducible", "old_pred", "missing_stage", *[f"route_q{t:.2f}" for t in SENSITIVITY]]
        with (args.output / "routing_rows.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sorted(rows + unreproducible, key=lambda r: r["image_id"]))
        (args.output / "routing_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                                          encoding="utf-8")


if __name__ == "__main__":
    main()
