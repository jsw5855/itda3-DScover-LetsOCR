"""0dd7a35 화장품 300장 평가 도구 (평가 전용, 제출 코드는 import만 한다).

- 공식 결과: ocr_pipeline.predict_image_detailed 를 그대로 호출한다.
  엔진을 감싸서 각 호출의 입력 이미지 해시, detections, 시간을 기록할 뿐 결과는 바꾸지 않는다.
- 분석용: 공식 경로가 돌리지 않은 단계를 predict_image_detailed 와 같은 방법으로 만들어 돌린다.
  그래서 5단계(original_512, rotation_270, highres_1024, clahe, clahe_1024)가 모두 저장된다.
"""
from __future__ import annotations

import csv
import ctypes
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

import ocr_pipeline as op

EXPECTED_COMMIT = "0dd7a35f43bdb52f7c091d916f8b5fd9e5d9b948"
EXPECTED_LABEL_SHA = "cd6417d04982b72d8984cf6a460fba96f9c8b9d0b17985b96a990790ab2befdd"
ALL_STAGES = ("original_512", "rotation_270", "highres_1024", "clahe", "clahe_1024")
IDS = [str(i) for i in range(900001, 900301)]
DET_KW = dict(text_det_limit_type="max", text_det_box_thresh=0.7)   # predict_image_detailed 와 같음
SIDE = {"original_512": 512, "rotation_270": 512, "highres_1024": 1024, "clahe": 512, "clahe_1024": 1024}


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha_array(arr):
    arr = np.ascontiguousarray(arr)
    return hashlib.sha256(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def on_ac_power():
    """Windows 전원 상태: True(전원 연결) / False(배터리) / None(알 수 없음)."""
    class SPS(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                    ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
    try:
        s = SPS()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
            return None
        return {0: False, 1: True}.get(s.ACLineStatus)
    except Exception:
        return None


def keep_awake(on):
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))
    except Exception:
        pass


def check_code(root):
    man = json.loads((Path(root) / "CODE_MANIFEST_0dd7a35.json").read_text(encoding="utf-8"))
    assert man["commit"] == EXPECTED_COMMIT, "코드 기록이 0dd7a35가 아닙니다"
    bad = [f for f, h in man["files"].items() if sha256_file(Path(root) / f) != h]
    if bad:
        raise SystemExit("제출 코드가 0dd7a35와 다릅니다(수정됨): " + ", ".join(bad))
    assert Path(op.__file__).resolve().parent == Path(root).resolve(), f"다른 폴더의 ocr_pipeline 입니다: {op.__file__}"
    return man


def check_inputs(image_dir, label_csv):
    assert not any(900301 <= int(i) <= 900400 for i in IDS), "검증용 사진이 들어 있습니다"
    label_sha = sha256_file(label_csv)
    if label_sha != EXPECTED_LABEL_SHA:
        raise SystemExit(f"라벨 파일이 기록과 다릅니다 (sha256 {label_sha[:8]}…, 기대 {EXPECTED_LABEL_SHA[:8]}…). "
                         "900259를 NONE으로 고친 최신 라벨 파일을 넣어 주세요.")
    with open(label_csv, encoding="utf-8-sig", newline="") as f:
        labels = {r["image_id"]: r for r in csv.DictReader(f)}
    assert sorted(labels) == IDS, f"라벨 줄 수/번호가 다릅니다: {len(labels)}"
    files = {}
    for p in Path(image_dir).iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.stem in labels:
            files[p.stem] = p
    lost = [i for i in IDS if i not in files]
    if lost:
        raise SystemExit("사진이 없습니다: " + ", ".join(lost))
    return labels, files


class RecordingEngine:
    """실제 엔진을 그대로 부르고, 호출마다 입력 해시·결과·시간만 기록한다."""

    def __init__(self, engine):
        self.engine = engine
        self.calls = []

    def predict(self, image, **kwargs):
        t = time.perf_counter()
        out = self.engine.predict(image, **kwargs)
        sec = time.perf_counter() - t
        self.calls.append({"input_sha": sha_array(image), "shape": list(np.asarray(image).shape),
                           "kwargs": {k: kwargs[k] for k in sorted(kwargs)}, "seconds": sec,
                           "detections": op.paddle_to_common(out)})
        return out


def stage_images(path):
    """predict_image_detailed 와 같은 방법으로 5단계 입력을 만든다(분석용)."""
    rgb = op.decode_image(path)
    base = op.resize_image(rgb, 512)
    return {
        "original_512": lambda: base,
        "rotation_270": lambda: np.asarray(Image.fromarray(base).rotate(270, expand=True)),
        "highres_1024": lambda: op.resize_image(rgb, 1024),
        "clahe": lambda: op.apply_clahe(base),
        "clahe_1024": lambda: op.apply_clahe(op.resize_image(rgb, 1024)),
    }


def process(engine, image_id, path):
    rec_engine = RecordingEngine(engine)
    ac = on_ac_power()
    t0 = time.perf_counter()
    prediction, method, attempts = op.predict_image_detailed(rec_engine, path)   # 공식 경로 그대로
    official_seconds = time.perf_counter() - t0
    assert len(rec_engine.calls) == len(attempts), "엔진 호출 수와 attempts 가 다릅니다"
    stages = {}
    for name, call in zip(attempts, rec_engine.calls):
        stages[name] = dict(call, official=True)
    prep = stage_images(path)
    for name in attempts:                       # 분석용 입력이 공식 입력과 같은지 확인
        assert sha_array(prep[name]()) == stages[name]["input_sha"], f"{image_id} {name}: 입력 이미지가 공식 경로와 다릅니다"
    extra_start = time.perf_counter()
    for name in ALL_STAGES:
        if name in stages:
            continue
        img = prep[name]()
        t = time.perf_counter()
        out = engine.predict(img, text_det_limit_side_len=SIDE[name], **DET_KW)
        sec = time.perf_counter() - t
        stages[name] = {"input_sha": sha_array(img), "shape": list(np.asarray(img).shape),
                        "kwargs": dict(text_det_limit_side_len=SIDE[name], **DET_KW), "seconds": sec,
                        "detections": op.paddle_to_common(out), "official": False}
    official = {"image_id": image_id, "prediction": prediction, "method": method, "attempts": attempts,
                "official_seconds": official_seconds,
                "official_ocr_seconds": sum(stages[n]["seconds"] for n in attempts), "ac_power": ac}
    allstages = {"image_id": image_id, "extra_seconds": time.perf_counter() - extra_start,
                 "stages": [dict(stage=n, **stages[n]) for n in ALL_STAGES]}
    return official, allstages


def replay(allstages_record):
    """저장된 5단계로 0dd7a35 run_cascade 를 다시 돌린다 (OCR 없음)."""
    by = {s["stage"]: s for s in allstages_record["stages"]}
    facts = {n: op.stage_result(by[n]["detections"]) for n in by}
    return op.run_cascade(lambda n: facts[n])


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):                  # 끊긴 마지막 줄 정리
        data = data[:data.rfind(b"\n") + 1]
        path.write_bytes(data)
    return [json.loads(l) for l in data.decode("utf-8").splitlines() if l.strip()]


def append(path, record):
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
