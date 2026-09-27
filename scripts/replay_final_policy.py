"""Replay the production cascade (ocr_pipeline.run_cascade) on saved OCR. No OCR is run.

Two sources, both driving the exact production control flow:

--snapshots  per-stage parser results saved with the accepted parser
             (docs/parser_optimization2/new_parser_replay.json); only
             run_cascade/retry_triggered/prefer_retry are exercised.
--raw        the immutable full-stage dump (docs/full_stage_701_run1/raw_ocr.jsonl);
             each stage is parsed by production ocr_pipeline.stage_result, so the
             parser, q and M are exercised too. The dump is only read.

Cohort truths come from docs/parser_optimization1/old_parser_replay.json and are
used only after every prediction is made. Each result is compared with the
pre-policy cascade (first stage with a candidate) replayed on the same stages.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ocr_pipeline  # noqa: E402

COHORTS = ROOT / "docs/parser_optimization1/old_parser_replay.json"
SNAPSHOTS = ROOT / "docs/parser_optimization2/new_parser_replay.json"


def snapshot_stages(path):
    stages = json.loads(Path(path).read_text(encoding="utf8"))["stages"]
    result = {}
    for key, records in stages.items():
        result[key] = {name: (r["prediction"], {"q": r["q"], "M": r["M"]} if r["has_candidate"] else None)
                       for name, r in zip(ocr_pipeline.STAGES, records)}
    return result


def raw_stages(path):
    result = {}
    with Path(path).open(encoding="utf8") as stream:
        for line in stream:
            record = json.loads(line)
            key, name = record["image_id"], record["stage"]
            if name in result.setdefault(key, {}):
                raise ValueError(f"Duplicate stage record: {key}/{name}")
            result[key][name] = ocr_pipeline.stage_result(record["detections"])
    for key, stages in result.items():
        if set(stages) != set(ocr_pipeline.STAGES):
            raise ValueError(f"Incomplete stages for {key}")
    return result


def baseline_cascade(stages):
    """Pre-policy production behaviour, kept independent of the new code."""
    attempts = []
    for name in ocr_pipeline.STAGES:
        attempts.append(name)
        prediction, evidence = stages[name]
        if evidence is not None:
            return prediction, name, attempts
    return stages["original_512"][0], "original_no_candidate", attempts


def production_cascade(stages):
    calls = []

    def run_stage(name):
        calls.append(name)
        return stages[name]

    prediction, method, attempts = ocr_pipeline.run_cascade(run_stage)
    if attempts != calls or len(set(calls)) != len(calls):
        raise AssertionError(f"Attempt accounting mismatch or repeated stage: {calls}")
    return prediction, method, attempts


def replay(stages, cohorts):
    rows = {}
    for key in sorted(stages):
        base = baseline_cascade(stages[key])
        new = production_cascade(stages[key])
        rows[key] = {"baseline": base[0]["final_date"], "baseline_method": base[1], "baseline_attempts": base[2],
                     "prediction": new[0], "method": new[1], "attempts": new[2]}
    report = {}
    for cohort, truth in cohorts.items():
        missing = set(truth) - set(rows)
        if missing:
            raise ValueError(f"{cohort}: {len(missing)} images without saved stages")
        before = {k for k in truth if rows[k]["baseline"] == truth[k]}
        after = {k for k in truth if rows[k]["prediction"]["final_date"] == truth[k]}
        count = lambda field: dict(sorted(Counter(s for k in truth for s in rows[k][field]).items(),
                                          key=lambda item: ocr_pipeline.STAGES.index(item[0])))
        report[cohort] = {
            "n": len(truth), "baseline_correct": len(before), "correct": len(after),
            "gains": sorted(after - before), "regressions": sorted(before - after),
            "baseline_attempts": count("baseline_attempts"), "attempts": count("attempts"),
            "methods": dict(Counter(rows[k]["method"] for k in truth)),
            "retry_triggered": sum(rows[k]["method"] in ("original_512_retry_kept", "highres_1024_retry") for k in truth),
        }
    none_triples = [k for k, r in rows.items() if r["prediction"]["final_date"] == "NONE-NONE-NONE"]
    all_none_formatted = all(r["prediction"]["final_date"] == "NONE" for r in rows.values()
                             if [r["prediction"][f] for f in ("year", "month", "day")] == ["NONE"] * 3)
    return rows, report, {"none_none_none_outputs": none_triples, "all_none_is_NONE": all_none_formatted}


def compare_with_snapshots(raw, snaps):
    """Raw-mode check that production parsing reproduces the saved parser results."""
    diffs = []
    for key, stages in raw.items():
        for name, (prediction, evidence) in stages.items():
            s_prediction, s_evidence = snaps[key][name]
            if prediction != s_prediction or (evidence is None) != (s_evidence is None) or (
                    evidence is not None and (evidence["M"] != s_evidence["M"] or evidence["q"] != s_evidence["q"])):
                diffs.append(f"{key}/{name}")
    return diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshots", type=Path, nargs="?", const=SNAPSHOTS)
    source.add_argument("--raw", type=Path)
    ap.add_argument("--output", type=Path, help="New directory for replay_summary.json and per-image rows")
    args = ap.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    stages = snapshot_stages(args.snapshots) if args.snapshots else raw_stages(args.raw)
    extra = {}
    if args.raw:
        extra["stage_mismatches_vs_saved_snapshots"] = compare_with_snapshots(stages, snapshot_stages(SNAPSHOTS))
    cohorts = json.loads(COHORTS.read_text(encoding="utf8"))["cohorts"]
    rows, report, checks = replay(stages, cohorts)
    source = Path(args.snapshots or args.raw).resolve()
    summary = {"source": source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT) else str(source), "mode": "snapshots" if args.snapshots else "raw",
               "ocr_invocations": 0, "cohorts": report, "formatting": checks, **extra}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output is not None:
        args.output.mkdir(parents=True)
        (args.output / "replay_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
        (args.output / "replay_rows.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
