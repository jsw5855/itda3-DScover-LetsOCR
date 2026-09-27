"""run_cosmetics_eval with a fake OCR engine (no models, no real OCR)."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ocr_pipeline  # noqa: E402
from scripts import review_routing_rates as rr  # noqa: E402
from scripts import run_cosmetics_eval as ev  # noqa: E402


def box(text, score, row=0):
    return text, score, [[0, 40 * row], [100, 40 * row], [100, 40 * row + 20], [0, 40 * row + 20]]


NOTHING = [box("nothing", 0.99)]
# Pixel colour of each test photo -> what the fake OCR reads at each stage.
SCENARIOS = {
    10: {"original_512": [box("2026.04.24", 0.95)]},                                        # CONFIRM
    20: {"original_512": [box("2026.04.24", 0.95), box("2025.01.02", 0.95, row=5)],
         "highres_1024": [box("2026.04.24", 0.95), box("2025.01.02", 0.95, row=5)]},       # CHOOSE
    30: {"original_512": [box("2026.04.24", 0.80)], "highres_1024": [box("2026.04.24", 0.70)]},  # RECHECK
    40: {},                                                                                   # MANUAL (nothing)
    50: {"original_512": [box("2026.04.24", 0.80)], "highres_1024": [box("2026.04.21", 0.97)]},  # retry wins
    60: {"clahe": [box("2026.04.24", 0.95)]},                                               # found at CLAHE
}
IDS = {10: "900001", 20: "900002", 30: "900003", 40: "900004", 50: "900005", 60: "900006"}


class FakeEngine:
    """Answers by photo colour and stage; logs every call."""

    def __init__(self, override=None):
        self.override = override
        self.calls = []
        self.unrotated = {}

    def stage_of(self, image, side):
        if side == 1024:
            return "highres_1024"
        if image.shape[0] > image.shape[1]:
            return "rotation_270"
        colour = int(image[0, 0, 0])
        self.unrotated[colour] = self.unrotated.get(colour, 0) + 1
        return "original_512" if self.unrotated[colour] == 1 else "clahe"

    def predict(self, image, **kwargs):
        stage = self.stage_of(image, kwargs["text_det_limit_side_len"])
        colour = int(image[0, 0, 0])
        self.calls.append({"colour": colour, "stage": stage, "array": np.array(image), "kwargs": kwargs})
        boxes = (SCENARIOS[colour] if self.override is None else self.override).get(stage, NOTHING)
        return [{"rec_texts": [b[0] for b in boxes], "rec_scores": np.array([b[1] for b in boxes]),
                 "rec_polys": np.array([b[2] for b in boxes]).reshape(-1, 4, 2)}]


@pytest.fixture(autouse=True)
def plain_clahe(monkeypatch):
    # Keep colours intact so the fake engine can tell photos apart.
    monkeypatch.setattr(ocr_pipeline, "apply_clahe", lambda rgb: rgb)


@pytest.fixture
def data(tmp_path):
    images = tmp_path / "Cosmetic"
    images.mkdir()
    for colour, key in IDS.items():
        suffix = ".JPG" if colour == 60 else ".jpg"
        Image.new("RGB", (1200, 600), (colour, colour, colour)).save(images / f"{key}{suffix}", quality=100)
    labels = tmp_path / "cosmetics_labels_900001-900150.csv"
    with labels.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file_name", "image_id", "year", "month", "day", "final_date", "notes"])
        for key in list(IDS.values()) + ["900007"]:                      # 900007 has no photo
            writer.writerow([f"{key}.jpg", key, "2026", "04", "24", "2026-04-24", ""])
    return {"images": images, "labels": [labels], "output": tmp_path / "out"}


def run(data, engine=None, **kwargs):
    kwargs.setdefault("label_check", False)
    return ev.run(data["images"], data["labels"], "dev", data["output"], engine=engine or FakeEngine(), **kwargs)


def dump(data, split="dev"):
    return ev.read_dump(ev.output_paths(data["output"], split)["dump"])[0]


def test_official_prediction_is_predict_image(data):
    run(data)
    for key, record in dump(data).items():
        path = next(data["images"].glob(f"{key}.*"))
        assert (record["prediction"], record["method"]) == ocr_pipeline.predict_image(FakeEngine(), path)


def test_full_stage_saves_all_stages_with_submission_parameters(data):
    path = data["images"] / "900001.jpg"
    reference = FakeEngine(override={})                    # nothing anywhere: predict_image runs all four
    ocr_pipeline.predict_image(reference, path)
    engine = FakeEngine()                                   # confident original: three stages added afterwards
    record = ev.process_image(engine, path, "full-stage")
    assert [s["stage"] for s in record["stages"]] == list(ev.STAGES)
    assert record["attempts"] == ["original_512"]
    assert [s["official"] for s in record["stages"]] == [True, False, False, False]
    by_stage = {c["stage"]: c for c in engine.calls}
    assert [c["stage"] for c in reference.calls] == list(ev.STAGES)
    for call in reference.calls:
        np.testing.assert_array_equal(by_stage[call["stage"]]["array"], call["array"])
        assert by_stage[call["stage"]]["kwargs"] == call["kwargs"]


def test_fast_mode_keeps_only_official_stages(data):
    record = ev.process_image(FakeEngine(), data["images"] / "900001.jpg", "fast")
    assert [s["stage"] for s in record["stages"]] == ["original_512"]


def test_resume_skips_done_and_refuses_other_conditions(data):
    first = run(data, limit=2)
    assert first["processed_now"] == 2
    engine = FakeEngine()
    second = run(data, engine=engine)
    assert (second["already_done"], second["processed_now"]) == (2, 4)
    assert {c["colour"] for c in engine.calls} == {30, 40, 50, 60}
    assert run(data)["processed_now"] == 0
    with pytest.raises(ev.RunConditionError, match="mode"):
        run(data, mode="fast")
    rows = list(csv.DictReader(data["labels"][0].open(encoding="utf-8-sig")))
    rows[0]["day"], rows[0]["final_date"] = "25", "2026-04-25"
    with data["labels"][0].open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ev.RunConditionError, match="900001"):
        run(data)


def test_interrupted_last_line_is_dropped_and_rerun(data):
    run(data, limit=2)
    path = ev.output_paths(data["output"], "dev")["dump"]
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"image_id": "900003", "stag')        # killed mid-write
    result = run(data, limit=1)
    assert result["truncated_line_removed"] and result["processed_now"] == 1
    assert sorted(dump(data)) == ["900001", "900002", "900003"]


def test_missing_photo_skipped_and_uppercase_extension(data):
    result = run(data)
    assert result["labels_without_image"] == ["900007"]
    assert "900006" in dump(data)                            # 900006.JPG


def test_routes_first_guess_and_replay(data):
    run(data)
    summary, rows = ev.build_reports(data["labels"], "dev", data["output"])
    by_id = {r["image_id"]: r for r in rows}
    assert {by_id[IDS[c]]["route"] for c in (10, 20, 30, 40)} == {"CONFIRM", "CHOOSE", "RECHECK", "MANUAL"}
    assert by_id["900005"]["stop_stage"] == "highres_1024_retry" and by_id["900005"]["pred_final_date"] == "2026-04-21"
    assert by_id["900005"]["first_guess"] == "Parser 의심"   # 2026.04.24 was read at original
    assert by_id["900004"]["first_guess"] == "OCR 의심"
    assert by_id["900006"]["stop_stage"] == "clahe"
    assert summary["replay_mismatch_ids"] == []
    assert ev.output_paths(data["output"], "dev")["errors"].exists()
    # The dump also feeds the routing-rate tool unchanged.
    labels = data["output"] / "labels_for_rr.csv"
    with labels.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "year", "month", "day", "final_date", "truth_source", "label_sources"])
        for key in IDS.values():
            writer.writerow([key, "2026", "04", "24", "2026-04-24", "candidate", "cosmetics"])
    rr_rows, unreproducible = rr.analyze(rr.dump_images(ev.output_paths(data["output"], "dev")["dump"]), labels,
                                         "production")
    assert not unreproducible
    assert {r["image_id"]: r["pred"] for r in rr_rows} == {k: by_id[k]["pred_final_date"] for k in by_id}


def test_rescore_uses_saved_stages(data):
    run(data)
    summary, rows = ev.build_reports(data["labels"], "dev", data["output"], rescore=True)
    assert summary["rescored_with_current_code"] and summary["replay_mismatch_ids"] == []
    assert (data["output"] / "results_dev_rescored.csv").exists()


def test_validation_needs_confirmation_and_writes_no_error_list(data):
    with pytest.raises(PermissionError):
        ev.run(data["images"], data["labels"], "validation", data["output"], engine=FakeEngine(), label_check=False)
    ev.run(data["images"], data["labels"], "validation", data["output"], engine=FakeEngine(), label_check=False,
           confirm_validation=ev.VALIDATION_CONFIRMATION)
    ev.build_reports(data["labels"], "validation", data["output"])
    assert not ev.output_paths(data["output"], "validation")["errors"].exists()
    assert ev.output_paths(data["output"], "validation")["summary"].exists()


def test_label_errors_stop_the_run(data, monkeypatch):
    monkeypatch.setattr(ev, "check_label_files", lambda paths, image_dir=None: (["ERROR row 3: bad date"], [], []))
    with pytest.raises(ValueError, match="bad date"):
        run(data, label_check=True)
    assert not ev.output_paths(data["output"], "dev")["dump"].exists()


def test_real_label_checker_errors_warnings_and_empty_rows(data):
    path = data["labels"][0]
    with path.open("a", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["900008.jpg", "900008", "", "", "", "", ""])                 # not labelled yet
        writer.writerow(["900009.jpg", "900009", "2026", "4", "24", "2026-04-24", ""])  # WARN: padding
    info = ev.check_environment(data["images"], data["labels"])
    assert info["label_errors"] == []
    assert info["unlabeled_rows"] == ["900008"]
    assert any("900009" in w for w in info["label_warnings"])
    assert "900008" not in ev.load_labels(data["labels"])
    result = ev.run(data["images"], data["labels"], "dev", data["output"], engine=FakeEngine())
    assert result["processed_now"] == 6
    with path.open("a", encoding="utf-8-sig", newline="") as stream:
        csv.writer(stream).writerow(["900010.jpg", "900010", "2026", "13", "24", "2026-13-24", ""])  # ERROR
    with pytest.raises(ValueError, match="900010"):
        ev.run(data["images"], data["labels"], "dev", data["output"], engine=FakeEngine())


def test_second_concurrent_run_is_refused(data):
    output = data["output"]
    output.mkdir(parents=True)
    with ev.RunLock(output / "run_dev.lock"):
        with pytest.raises(RuntimeError, match="Another run"):
            run(data)
    assert run(data)["processed_now"] == 6                  # lock released afterwards


def test_repair_dump_keeps_first_of_duplicates(data):
    run(data, limit=2)
    path = ev.output_paths(data["output"], "dev")["dump"]
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines + [lines[1]]) + "\n", encoding="utf-8")   # 900002 twice
    with pytest.raises(ValueError, match="repair_dump"):
        ev.read_dump(path)
    assert ev.repair_dump(path) == {"900002": True}
    assert sorted(dump(data)) == ["900001", "900002"]
    assert len(list(path.parent.glob("ocr_dump_dev.jsonl.bak-*"))) == 1
    assert run(data)["processed_now"] == 4
