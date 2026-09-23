import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import evaluate_current_validation_300 as serial
from scripts import evaluate_current_validation_300_parallel as parallel
from scripts.evaluation_common import (
    create_sample, evaluation_dataset, load_population, normalize_date,
    normalize_truth, score_prediction,
)

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "labels/labels_300.csv"
REVIEW = ROOT / "labels/review/label_review.csv"
IMAGES = ROOT / "data"


@pytest.mark.parametrize("parts,expected", [
    (("none", "NONE", "none"), "NONE"),
    (("25", "1", "NONE"), "2025-01-NONE"),
    (("NONE", "2", "29"), "NONE-02-29"),
    (("2001", "7", "1"), "2001-07-01"),
    (("15", "1", "1"), "2015-01-01"),
    (("40", "12", "31"), "2040-12-31"),
])
def test_shared_contract(parts, expected):
    row = dict(zip(("year", "month", "day"), parts), final_date=expected)
    assert serial.normalize_truth(row) == parallel.normalize_truth(row)
    assert normalize_truth(row)["final_date"] == expected


@pytest.mark.parametrize("parts", [("99", "1", "1"), ("", "1", "1"),
                                    ("2023", "2", "29"), ("2026", "13", "1")])
def test_invalid_not_silently_none(parts):
    with pytest.raises(ValueError):
        normalize_date(dict(zip(("year", "month", "day"), parts)))


def test_mismatch_and_prediction_not_repaired():
    row = dict(year="2021", month="2", day="4", final_date="2021-02-02")
    with pytest.raises(ValueError):
        normalize_truth(row)
    truth = normalize_date(row)
    assert not score_prediction(row, truth)["final_date_correct"]


def test_legacy_300_truth_compatibility():
    with LABELS.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 300
    for row in rows:
        old = {k: "NONE" if row[k].casefold() == "none" else row[k].zfill(w)
               for k, w in [("year", 4), ("month", 2), ("day", 2)]}
        old["final_date"] = "-".join(old.values())
        assert normalize_truth(row) == old


def test_seed_reuse_and_changed_label_rejected(tmp_path):
    sample = create_sample(LABELS, IMAGES, "baseline", 250, 42)
    assert sample == create_sample(LABELS, IMAGES, "baseline", 250, 42)
    assert sample["entries"] != create_sample(LABELS, IMAGES, "baseline", 250, 43)["entries"]
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(sample), encoding="utf-8")
    args = SimpleNamespace(labels=LABELS, images=IMAGES, dataset_mode="baseline", sample_manifest=path)
    names, truths, meta = evaluation_dataset(args)
    assert len(names) == len(truths) == 250
    assert meta["sample_fingerprint"] == sample["sample_fingerprint"]
    with LABELS.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    chosen = sample["entries"][0]["image_id"]
    for row in rows:
        if row["image_id"] == chosen:
            row.update(year="2030", month="01", day="01", final_date="2030-01-01")
    changed = tmp_path / "changed.csv"
    with changed.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    args.labels = changed
    with pytest.raises(ValueError, match="changed"):
        evaluation_dataset(args)


def test_approval_gate():
    population = load_population(REVIEW, IMAGES, "approved")
    assert {r["image_id"] for r in population} == {"1862", "1906", "2284", "2448"}
    with pytest.raises(ValueError, match="only 4"):
        create_sample(REVIEW, IMAGES, "approved", 250, 42)
    with pytest.raises(ValueError, match="only permits"):
        load_population(REVIEW, IMAGES, "approved", ["pending"])
    with pytest.raises(ValueError):
        load_population(REVIEW, IMAGES, "baseline")


def test_both_entrypoints_share_dataset(tmp_path):
    sample = create_sample(LABELS, IMAGES, "baseline", 200, 7)
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(sample), encoding="utf-8")
    args = SimpleNamespace(labels=LABELS, images=IMAGES, dataset_mode="baseline", sample_manifest=path)
    assert serial.evaluation_dataset(args) == parallel.evaluation_dataset(args)
    args.sample_manifest = None
    assert len(serial.evaluation_dataset(args)[0]) == 300


def test_short_year_matches_parser():
    from date_parser.interpret import _assign_role, DEFAULT_YEAR_MIN, DEFAULT_YEAR_MAX
    for year in range(100):
        raw = str(year).zfill(2)
        expected = _assign_role("year", SimpleNamespace(raw=raw, kind="num"), DEFAULT_YEAR_MIN, DEFAULT_YEAR_MAX)
        if expected is None:
            with pytest.raises(ValueError):
                normalize_date(dict(year=raw, month="1", day="1"))
        else:
            assert normalize_date(dict(year=raw, month="1", day="1"))["year"] == str(expected)


def test_archived_scores_unchanged():
    checked = 0
    for path in (ROOT / "docs").glob("validation*/evaluation_300.csv"):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if not all("true_" + k in row and k + "_correct" in row for k in ("year", "month", "day", "final_date")):
                continue
            truth = normalize_truth({k: row['true_' + k] for k in ("year", "month", "day", "final_date")})
            scores = score_prediction(row, truth)
            for key, value in scores.items():
                assert str(value).lower() == row[key].lower()
            checked += 1
    assert checked >= 300
