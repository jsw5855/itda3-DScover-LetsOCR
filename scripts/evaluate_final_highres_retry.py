"""Audit frozen final-cascade misses and run exactly one 1280px pass each."""
from __future__ import annotations

import hashlib
import json
import socket
import sys
import time

import pandas as pd

import evaluate_highres_retry as highres
from evaluate_integrated_baseline import ROOT, configure_offline_runtime
from build_final_cascade import bool_value, load_inputs

BASE = ROOT / "data/validation/integrated_baseline"
OUT = BASE / "final_highres_1280"


def parse(detections):
    from date_parser import parse_expiration_date
    from date_parser.select import find_all_candidates
    from date_parser.types import TextBox
    return (parse_expiration_date(detections)["final_date"],
            len(find_all_candidates([TextBox.from_dict(x) for x in detections])))


def deny_network(*args, **kwargs):
    raise RuntimeError("Network disabled for offline experiment")


def main():
    started = time.perf_counter()
    configure_offline_runtime()
    sys.path.insert(0, str(ROOT))
    socket.socket.connect = deny_network
    socket.create_connection = deny_network
    frozen_path = BASE / "final_cascade/final_cascade_images.csv"
    frozen_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    frozen = pd.read_csv(frozen_path, dtype=str, keep_default_na=False)
    targets = frozen.loc[frozen.failure_stage.eq("ocr_boxes_no_date_candidate")]
    correct = frozen.final_date_correct.map(bool_value)
    assert len(frozen) == 300 and int(correct.sum()) == 205
    assert len(targets) == 42 and targets.file_name.is_unique
    assert not targets.date_candidate_found.map(bool_value).any()
    stages = dict(zip(["original_512", "rotation_270", "highres_1024", "clahe"],
                      [x.set_index("file_name") for x in load_inputs()]))
    previous = {}
    # Reparse every attempted cached pass: fail closed if parser drift changes targets.
    for row in targets.itertuples(index=False):
        total_boxes = 0
        for method in json.loads(row.attempted_methods):
            boxes = json.loads(stages[method].loc[row.file_name, "detections_json"])
            prediction, count = parse(boxes)
            assert count == 0, "Current parser differs from frozen no-candidate cohort"
            total_boxes += len(boxes)
            if method == "original_512":
                assert prediction == row.pred_final_date
                previous[row.file_name] = (prediction, count)
        assert total_boxes > 0
    OUT.mkdir(parents=True, exist_ok=True)
    targets.to_csv(OUT / "targets.csv", index=False, encoding="utf-8-sig")
    highres.HIGHRES_LONG_SIDE = 1280
    engine, init_seconds = highres.initialize_highres_engine()
    rows = []
    with (OUT / "highres_1280_cache.jsonl").open("w", encoding="utf-8") as cache:
        for position, row in enumerate(targets.itertuples(index=False), 1):
            assert time.perf_counter() - started < 2300, "Experiment time budget reached"
            record = highres.run_highres(engine, row.file_name)
            prediction, count = parse(record["detections"])
            old_prediction, old_count = previous[row.file_name]
            cache.write(json.dumps(record, ensure_ascii=False) + "\n")
            cache.flush()
            rows.append(dict(image_id=row.image_id, gold_date=row.true_final_date,
                             previous_prediction=old_prediction, highres_prediction=prediction,
                             previous_candidate_count=old_count, highres_candidate_count=count,
                             recovered_candidate=count > 0,
                             recovered_exact_match=prediction == row.true_final_date and old_prediction != row.true_final_date,
                             elapsed_seconds=record["processing_seconds"], file_name=row.file_name,
                             ocr_seconds=record["ocr_seconds"],
                             input_width=record["input_width"], input_height=record["input_height"]))
            pd.DataFrame(rows).to_csv(OUT / "image_results.csv", index=False, encoding="utf-8-sig")
            print(f"{position}/{len(targets)} candidates={count} exact={rows[-1]['recovered_exact_match']} seconds={record['processing_seconds']:.2f}", flush=True)
    results = pd.DataFrame(rows)
    combined = frozen.copy()
    for row in results.itertuples(index=False):
        if row.recovered_candidate:
            combined.loc[combined.image_id.eq(row.image_id), "pred_final_date"] = row.highres_prediction
    final_correct = combined.pred_final_date.eq(combined.true_final_date)
    regressions = int((correct & ~final_correct).sum())
    assert combined.loc[correct].equals(frozen.loc[correct])
    assert hashlib.sha256(frozen_path.read_bytes()).hexdigest() == frozen_hash
    combined[["image_id", "true_final_date", "pred_final_date"]].to_csv(OUT / "simulated_300_predictions.csv", index=False, encoding="utf-8-sig")
    existing = pd.read_csv(BASE / "final_cascade/final_cascade_summary.csv").iloc[0]
    seconds = float(results.elapsed_seconds.sum())
    scale = 3352 / len(frozen)
    total_projection = (float(existing.total_expected_processing_seconds) + seconds) * scale + init_seconds
    summary = dict(target_images=len(targets), additional_ocr_calls=len(results),
                   candidate_recoveries=int(results.recovered_candidate.sum()),
                   new_exact_matches=int(results.recovered_exact_match.sum()), regressions=regressions,
                   previous_exact_matches=int(correct.sum()), final_exact_matches=int(final_correct.sum()),
                   accuracy_change_pp=float((final_correct.sum()-correct.sum())/len(frozen)*100),
                   additional_processing_seconds=seconds, additional_ocr_seconds=float(results.ocr_seconds.sum()),
                   model_init_seconds=init_seconds, experiment_wall_seconds=time.perf_counter()-started,
                   projected_3352_conditional_calls=len(targets)*scale,
                   projected_3352_additional_seconds=seconds*scale,
                   projected_3352_existing_seconds=float(existing.total_expected_processing_seconds)*scale,
                   projected_3352_total_seconds_including_init=total_projection,
                   projected_3352_unconditional_retry_seconds=seconds/len(targets)*3352,
                   frozen_sha256=frozen_hash,
                   parser_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/"date_parser").glob("*.py"))})
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# 최종 cascade 이후 1280px 재시도 실험

- 브랜치: integration/ocr-eval. 실행: `.venv/Scripts/python.exe scripts/evaluate_final_highres_retry.py`
- 최종 결과 CSV의 `failure_stage == ocr_boxes_no_date_candidate`로 {len(targets)}장 추출. 박스 없음 1장은 제외. 라벨은 평가에만 사용.
- 저장된 실제 cascade는 기본 512 → 270도 → 1024 → CLAHE이며 gamma/crop은 제외되어 있다. 이전 debug CSV 대신 최종 결과를 기준으로 삼았다.
- 대상의 모든 기존 시도 OCR을 현재 parser에 다시 전달하여 후보 0을 확인했고, 기본 OCR의 예측도 저장 결과와 일치했다. 후보 수는 `find_all_candidates`의 위치별 날짜 토큰 수다.
- 기존 1024 실험 함수에 해상도만 1280으로 지정. 원본 EXIF 보정/RGB, 긴 변 상한 1280, 작은 이미지는 확대하지 않음. Korean mobile PaddleOCR, 로컬 weights, batch 6, box threshold 0.7, CPU threads 4, MKLDNN 유지. 네트워크 연결 차단.
- 고해상도 단독 OCR을 동일 parser에 전달하고 후보가 생긴 경우만 가상 fallback에 채택. 정답으로 선택하지 않음. 기존 정답 205행 불변 및 원본 CSV SHA256 불변 검증.

| 지표 | 결과 |
|---|---:|
| 대상 / 추가 OCR 호출 | {len(targets)} / {len(results)} |
| 후보 없음 → 후보 생성 | {summary['candidate_recoveries']} |
| 새 완전일치 / 회귀 | {summary['new_exact_matches']} / {regressions} |
| 정확도 | 205/300 (68.33%) → {int(final_correct.sum())}/300 ({final_correct.mean()*100:.2f}%) |
| 변화 | {summary['accuracy_change_pp']:+.2f}%p |
| 추가 처리 / OCR 시간 | {seconds:.2f}초 / {summary['additional_ocr_seconds']:.2f}초 |
| 모델 초기화 / 실험 경과 | {init_seconds:.2f}초 / {summary['experiment_wall_seconds']:.2f}초 |
| 3,352장 조건부 추가 호출 예상 | {len(targets)*scale:.2f}회 |
| 3,352장 조건부 추가 시간 | {seconds*scale/60:.2f}분 |
| 3,352장 기존 cascade 예상 | {summary['projected_3352_existing_seconds']/60:.2f}분 |
| 3,352장 fallback 포함 예상 | {total_projection/60:.2f}분 |
| 참고: 3,352장 모두 1280 OCR 시 추가 시간 | {summary['projected_3352_unconditional_retry_seconds']/60:.2f}분 |

시간 환산은 동일 하드웨어와 대상 비율 42/300 및 평균 이미지 비용이 유지된다는 가정이다. 전체 데이터 실측이 아니며, 기존 cascade 시간은 저장된 300장 합산값을 사용했다. 실험 자체는 40분 이내 완료했다.

**결론: 현 40분 전체 실행 제약에서는 최종 fallback으로 채택하지 않는다.** 새 완전일치는 {summary['new_exact_matches']}건이지만 3,352장 전체 예상 {total_projection/60:.2f}분으로 제한을 초과한다. 기존 cascade 자체도 환산하면 40분을 초과하므로 종전 300장 시간과 전체 적용 시간을 구분해야 한다. 같은 검증셋을 이용한 탐색 결과이며 별도 검증이 필요하다.

산출물은 `data/validation/integrated_baseline/final_highres_1280/`의 `targets.csv`, `image_results.csv`, `highres_1280_cache.jsonl`, `simulated_300_predictions.csv`, `summary.json`이다. data 디렉터리는 기존 gitignore 대상이다. 기존 파일 및 parser, CONFIG, requirements, download_weights는 수정하지 않았고 commit/push하지 않았다.
"""
    (ROOT / "docs/final_highres_1280_experiment.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Experiment stopped: {type(error).__name__}; check inputs/runtime without changing parser.", flush=True)
        raise SystemExit(1)
