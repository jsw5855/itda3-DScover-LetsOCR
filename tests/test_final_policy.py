"""Frozen retry policy in the production cascade (fake engine, no OCR)."""
import inspect
import itertools
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import ocr_pipeline as pipeline

SIDE_OF = {"original_512": 512, "rotation_270": 512, "highres_1024": 1024, "clahe": 512}


def box(text, score, row=0):
    return text, score, [[0, 40 * row], [100, 40 * row], [100, 40 * row + 20], [0, 40 * row + 20]]


class ScriptedEngine:
    """Returns one scripted list of boxes per predict call, in call order."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.sides = []

    def predict(self, image, **kwargs):
        self.sides.append(kwargs["text_det_limit_side_len"])
        boxes = self.responses.pop(0)
        return [{"rec_texts": [b[0] for b in boxes], "rec_scores": np.array([b[1] for b in boxes]),
                 "rec_polys": np.array([b[2] for b in boxes]).reshape(-1, 4, 2)}]


@pytest.fixture
def image(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "apply_clahe", lambda rgb: rgb)
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (1200, 600)).save(path)
    return path


def run(image, *responses):
    engine = ScriptedEngine(*responses)
    prediction, method, attempts = pipeline.predict_image_detailed(engine, image)
    assert engine.sides == [SIDE_OF[a] for a in attempts]
    assert not engine.responses, "unused scripted stage"
    return prediction["final_date"], method, attempts


def test_a_confident_single_candidate_no_retry(image):
    assert run(image, [box("2026.04.24", 0.95)]) == ("2026-04-24", "original_512", ["original_512"])


def test_threshold_is_strict_less_than(image):
    assert run(image, [box("2026.04.24", 0.90)])[1] == "original_512"


def test_b_low_q_runs_highres_once(image):
    date, method, attempts = run(image, [box("2026.04.24", 0.85)], [box("2026.04.24", 0.80)])
    assert attempts == ["original_512", "highres_1024"]
    assert (date, method) == ("2026-04-24", "original_512_retry_kept")


def test_c_multiple_distinct_dates_runs_highres_once(image):
    original = [box("2026.04.24", 0.99), box("2025.01.02", 0.99, row=5)]
    date, method, attempts = run(image, original, [box("2026.04.24", 0.95)])
    assert attempts == ["original_512", "highres_1024"]


# Two complete single-date readings that disagree ask clahe first (vote); with
# no clahe candidate, the q rule below decides as before.
def test_d_higher_highres_q_wins(image):
    date, method, attempts = run(image, [box("2026.04.24", 0.85)], [box("2026.04.21", 0.97)], [box("nothing", 0.99)])
    assert (date, method, attempts) == ("2026-04-21", "highres_1024_retry", ["original_512", "highres_1024", "clahe"])


def test_d2_self_anchor_is_not_replaced_by_self_exclude(image):
    original = [box("Exp.20.09.2021", 0.78)]
    highres = [box("Prod.20.09.2020", 0.97)]

    date, method, attempts = run(image, original, highres)

    assert attempts == ["original_512", "highres_1024"]
    assert (date, method) == ("2021-09-20", "original_512_retry_kept")


def test_e_higher_original_q_kept(image):
    assert run(image, [box("2026.04.24", 0.85)], [box("2026.04.21", 0.80)], [box("nothing", 0.99)])[:2] == ("2026-04-24", "original_512_retry_kept")


def test_f_tie_keeps_original(image):
    assert run(image, [box("2026.04.24", 0.85)], [box("2026.04.21", 0.85)], [box("nothing", 0.99)])[:2] == ("2026-04-24", "original_512_retry_kept")


def test_vote_clahe_agrees_with_original(image):
    # 900296: "EP 2029.05.18" at 512px, "EP 2025.05.184" (higher q) at 1024px, clahe "DP 2029.05.18".
    date, method, attempts = run(image, [box("EP 2029.05.18", 0.85)], [box("EP 2025.05.184", 0.90)], [box("DP 2029.05.18", 0.76)])
    assert (date, method, attempts) == ("2029-05-18", "original_512_retry_kept", ["original_512", "highres_1024", "clahe"])


def test_vote_clahe_agrees_with_highres(image):
    date, method, _ = run(image, [box("2026.04.24", 0.88)], [box("2026.04.21", 0.80)], [box("2026.04.21", 0.70)])
    assert (date, method) == ("2026-04-21", "highres_1024_retry")


def test_no_vote_when_original_saw_two_dates(image):
    # 000954: the original read manufacture and expiry dates (M); the vote would
    # let two stages that missed the expiry date outvote it. The q rule decides.
    original = [box("21.02.05", 0.99), box("21.01.07", 0.99, row=5)]
    date, method, attempts = run(image, original, [box("21.01.07", 0.90)])
    assert attempts == ["original_512", "highres_1024"]
    assert method == "original_512_retry_kept"


def test_short_fragment_retries_without_vote(image):
    # 900146: a confident bare fragment "7.08.22" is re-read at 1024px; the
    # retry was not triggered by q or M, so clahe is not asked.
    date, method, attempts = run(image, [box("7.08.22", 0.908)], [box("EXP20280826까 지", 0.969)])
    assert (date, method, attempts) == ("2028-08-26", "highres_1024_retry", ["original_512", "highres_1024"])


def test_long_confident_source_does_not_retry(image):
    assert run(image, [box("EXP 2026.04.24", 0.95)]) == ("2026-04-24", "original_512", ["original_512"])


def test_g_highres_without_candidate_keeps_original(image):
    assert run(image, [box("2026.04.24", 0.85)], [box("nothing", 0.99)])[:2] == ("2026-04-24", "original_512_retry_kept")


@pytest.mark.parametrize("stage", ["rotation_270", "highres_1024", "clahe"])
def test_h_no_original_candidate_keeps_fallback(image, stage):
    order = list(pipeline.STAGES)
    hit = order.index(stage)
    # Even a low-confidence, multi-date fallback candidate is accepted as before: no retry there.
    responses = [[box("nothing", 0.99)]] * hit + [[box("2026.04.24", 0.5), box("2025.01.02", 0.5, row=5)]]
    date, method, attempts = run(image, *responses)
    assert (method, attempts) == (stage, order[:hit + 1])


def test_h_no_candidate_anywhere(image):
    date, method, attempts = run(image, *[[box("nothing", 0.99)]] * 4)
    assert (date, method, attempts) == ("NONE", "original_no_candidate", list(pipeline.STAGES))


def test_i_highres_never_runs_twice_on_any_path():
    # Every combination of per-stage outcomes through the control flow alone.
    outcomes = [None, {"q": 0.5, "M": False}, {"q": 0.95, "M": False}, {"q": 0.95, "M": True}]
    prediction = {"year": "NONE", "month": "NONE", "day": "NONE", "final_date": "NONE"}
    for combo in itertools.product(outcomes, repeat=4):
        stages = dict(zip(pipeline.STAGES, combo))
        _, _, attempts = pipeline.run_cascade(lambda name: (prediction, stages[name]))
        assert len(attempts) == len(set(attempts))
        assert attempts[0] == "original_512"
        if combo[0] is not None:
            assert set(attempts) <= {"original_512", "highres_1024"}


def test_j_decisions_use_only_ocr_evidence(image, tmp_path):
    for function in (pipeline.run_cascade, pipeline.retry_triggered, pipeline.prefer_retry, pipeline.stage_result):
        parameters = set(inspect.signature(function).parameters)
        assert not parameters & {"image_id", "truth", "label", "labels", "path"}
    source = Path(pipeline.__file__).read_text(encoding="utf-8")
    for token in ("001515", "003311", "000266", "000422", "000726", "002066", "002917", "labels_", "truth"):
        assert token not in source
    # Same pixels under another file name give the same result.
    other = tmp_path / "999999.png"
    Image.open(image).save(other)
    responses = ([box("2026.04.24", 0.85)], [box("2026.04.21", 0.97)], [box("nothing", 0.99)])
    assert run(image, *responses) == run(other, *responses)


def test_all_none_final_date_format(image):
    date, _, _ = run(image, *[[box("nothing", 0.99)]] * 4)
    assert date == "NONE"
