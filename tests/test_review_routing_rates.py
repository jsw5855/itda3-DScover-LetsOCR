"""Staff-review routing on fake saved detections (no OCR)."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import review_routing_rates as rr  # noqa: E402

STAGES = ["original_512", "rotation_270", "highres_1024", "clahe"]


def box(text, confidence, row=0):
    return {"text": text, "confidence": confidence,
            "bbox": [[0, 40 * row], [100, 40 * row], [100, 40 * row + 20], [0, 40 * row + 20]]}


NOTHING = [box("nothing", 0.99)]
CASES = {
    # image id: (saved stages, expected route or None when unreproducible)
    "1": ({s: NOTHING for s in STAGES}, "MANUAL"),
    "2": ({"original_512": [box("2026.04", 0.99)]}, "MANUAL"),
    "3": ({"original_512": [box("2026.04.24", 0.99), box("2025.01.02", 0.99, row=5)]}, "CHOOSE"),
    "4": ({"original_512": [box("2026.04.24", 0.80)]}, "RECHECK"),
    "5": ({"original_512": [box("2026.04.24", 0.93)]}, "CONFIRM"),
    "6": ({"original_512": NOTHING}, None),
}


def write_inputs(tmp_path):
    dump = tmp_path / "ocr_dump.jsonl"
    with dump.open("w", encoding="utf-8") as stream:
        for key, (stages, _) in CASES.items():
            stream.write(json.dumps({"image_id": key, "prediction": {"final_date": "NONE"},
                                     "stages": [{"stage": s, "detections": d} for s, d in stages.items()]}) + "\n")
    labels = tmp_path / "labels.csv"
    with labels.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "year", "month", "day", "final_date", "truth_source", "label_sources"])
        for key in CASES:
            writer.writerow([key, "2026", "04", "24", "2026-04-24",
                             "approved" if key == "5" else "candidate", "existing_300" if key < "4" else "incoming_432"])
    return dump, labels


def test_four_routes_and_unreproducible(tmp_path):
    dump, labels = write_inputs(tmp_path)
    rows, unreproducible = rr.analyze(rr.dump_images(dump), labels, "first_candidate")
    assert {r["image_id"]: r["route"] for r in rows} == {
        f"{int(k):06d}": v for k, (_, v) in CASES.items() if v is not None}
    assert [(r["image_id"], r["missing_stage"]) for r in unreproducible] == [("000006", "rotation_270")]
    by_id = {r["image_id"]: r for r in rows}
    assert by_id["000002"]["pred"] == "2026-04-NONE"
    assert by_id["000001"]["pred"] == "NONE"
    assert by_id["000005"]["correct"] and by_id["000005"]["label_status"] == "approved"


def test_summary_counts_add_up_and_risk(tmp_path):
    dump, labels = write_inputs(tmp_path)
    rows, unreproducible = rr.analyze(rr.dump_images(dump), labels, "first_candidate")

    class Args:
        cascade = "first_candidate"
        stage_snapshots = None
        full_stage_raw = None
    Args.dump, Args.labels = dump, labels
    summary = rr.summarize(rows, unreproducible, Args)
    table = summary["A_route_rates"]["all"]
    assert sum(table[r]["count"] for r in rr.ROUTES) + summary["unreproducible"]["count"] == len(CASES)
    assert table["staff_review"]["count"] == 4
    assert summary["C_confirm_but_wrong"]["count"] == 0
    # Sensitivity only moves RECHECK/CONFIRM: q 0.80 is RECHECK everywhere, q 0.93 only at 0.95.
    assert all(summary["D_sensitivity_reference_not_policy"][t]["rates"]["RECHECK"]["count"] == 1 for t in ("0.85", "0.90"))
    assert summary["D_sensitivity_reference_not_policy"]["0.95"]["rates"]["RECHECK"]["count"] == 2


def test_policy_b_marks_missing_highres_unreproducible(tmp_path):
    dump, labels = write_inputs(tmp_path)
    rows, unreproducible = rr.analyze(rr.dump_images(dump), labels, "production")
    # Low q (4) and multiple dates (3) now need highres_1024, which was not saved.
    assert {r["image_id"]: r["missing_stage"] for r in unreproducible} == {
        "000003": "highres_1024", "000004": "highres_1024", "000006": "rotation_270"}


def test_error_direction():
    assert rr.error_direction("2026-05-01", "2026-04-24") == "late"
    assert rr.error_direction("2026-04-01", "2026-04-24") == "early"
    assert rr.error_direction("2026-04-NONE", "2026-04-24") == "not_comparable"
    assert rr.error_direction("NONE", "NONE") == ""


def test_wilson_interval():
    assert rr.wilson(0, 0) is None
    low, high = rr.wilson(256, 701)
    assert low < 36.5 < high and 32 < low and high < 41
    assert rr.wilson(0, 10)[0] == 0.0


def test_full_stage_raw_source(tmp_path):
    raw = tmp_path / "raw_ocr.jsonl"
    with raw.open("w", encoding="utf-8") as stream:
        for stage in STAGES:
            detections = [box("2026.04.24", 0.80)] if stage != "highres_1024" else [box("2026.04.21", 0.97)]
            stream.write(json.dumps({"image_id": "000004", "stage": stage, "detections": detections}) + "\n")
    _, labels = write_inputs(tmp_path)
    rows, unreproducible = rr.analyze(rr.full_stage_images(raw), labels, "production")
    assert not unreproducible
    # Low q triggers the production highres retry, which wins on higher q.
    assert (rows[0]["stop_stage"], rows[0]["pred"], rows[0]["route"]) == ("highres_1024_retry", "2026-04-21", "CONFIRM")
    assert rows[0]["error_direction"] == "early"


def test_days_off_and_magnitude():
    assert rr.days_off("2026-05-01", "2026-04-24") == 7
    assert rr.days_off("2025-04-24", "2026-04-24") == -365
    assert rr.days_off("2026-04-NONE", "2026-04-24") is None
    rows = [{"image_id": k, "pred": p, "truth": t, "error_direction": rr.error_direction(p, t), "days_off": rr.days_off(p, t)}
            for k, p, t in [("a", "2026-05-01", "2026-04-24"), ("b", "2028-04-24", "2026-04-24"), ("c", "2026-03-24", "2026-04-24")]]
    m = rr.magnitude(rows)
    assert m["late"]["bins"] == {"1-7 days": 1, "8-31 days": 0, "32-365 days": 0, "over 1 year": 1}
    assert m["late"]["median_days"] == (7 + 731) / 2
    assert m["early"]["n"] == 1 and m["early"]["bins"]["8-31 days"] == 1


def test_workload_breakdown(tmp_path):
    dump, labels = write_inputs(tmp_path)
    rows, _ = rr.analyze(rr.dump_images(dump), labels, "first_candidate")
    w = rr.workload(rows)
    assert w["manual_nothing_read"]["count"] == 1          # image 1: nothing read
    assert w["manual_partial_date"]["count"] == 1          # image 2: 2026-04-NONE
    assert w["manual_partial_date"]["no_day"] == 1 and w["manual_partial_date"]["correct_as_printed"] == 0
    assert w["choose"]["truth_in_candidates"] == 1         # 2026-04-24 is one of the two dates
    assert w["recheck"]["already_correct"] == 1
