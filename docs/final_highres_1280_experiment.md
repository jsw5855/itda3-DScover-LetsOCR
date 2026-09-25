# 최종 cascade 이후 1280px 재시도 실험

> 실험 근거 보관 전용 — 최종 fallback 미채택. 이 실험 브랜치/PR은 병합 대상이 아니다.

- 브랜치: integration/ocr-eval. 실행: `.venv/Scripts/python.exe scripts/evaluate_final_highres_retry.py`
- 최종 결과 CSV의 `failure_stage == ocr_boxes_no_date_candidate`로 42장 추출. 박스 없음 1장은 제외. 라벨은 평가에만 사용.
- 저장된 실제 cascade는 기본 512 → 270도 → 1024 → CLAHE이며 gamma/crop은 제외되어 있다. 이전 debug CSV 대신 최종 결과를 기준으로 삼았다.
- 대상의 모든 기존 시도 OCR을 현재 parser에 다시 전달하여 후보 0을 확인했고, 기본 OCR의 예측도 저장 결과와 일치했다. 후보 수는 `find_all_candidates`의 위치별 날짜 토큰 수다.
- 기존 1024 실험 함수에 해상도만 1280으로 지정. 원본 EXIF 보정/RGB, 긴 변 상한 1280, 작은 이미지는 확대하지 않음. Korean mobile PaddleOCR, 로컬 weights, batch 6, box threshold 0.7, CPU threads 4, MKLDNN 유지. 네트워크 연결 차단.
- 고해상도 단독 OCR을 동일 parser에 전달하고 후보가 생긴 경우만 가상 fallback에 채택. 정답으로 선택하지 않음. 기존 정답 205행 불변 및 원본 CSV SHA256 불변 검증.

| 지표 | 결과 |
|---|---:|
| 대상 / 추가 OCR 호출 | 42 / 42 |
| 후보 없음 → 후보 생성 | 1 |
| 새 완전일치 / 회귀 | 1 / 0 |
| 정확도 | 205/300 (68.33%) → 206/300 (68.67%) |
| 변화 | +0.33%p |
| 추가 처리 / OCR 시간 | 80.32초 / 76.05초 |
| 모델 초기화 / 실험 경과 | 8.73초 / 107.88초 |
| 3,352장 조건부 추가 호출 예상 | 469.28회 |
| 3,352장 조건부 추가 시간 | 14.96분 |
| 3,352장 기존 cascade 예상 | 129.12분 |
| 3,352장 fallback 포함 예상 | 144.22분 |
| 참고: 3,352장 모두 1280 OCR 시 추가 시간 | 106.84분 |

시간 환산은 동일 하드웨어와 대상 비율 42/300 및 평균 이미지 비용이 유지된다는 가정이다. 전체 데이터 실측이 아니며, 기존 cascade 시간은 저장된 300장 합산값을 사용했다. 실험 자체는 40분 이내 완료했다.

**결론: 현 40분 전체 실행 제약에서는 최종 fallback으로 채택하지 않는다.** 새 완전일치는 1건이지만 3,352장 전체 예상 144.22분으로 제한을 초과한다. 기존 cascade 자체도 환산하면 40분을 초과하므로 종전 300장 시간과 전체 적용 시간을 구분해야 한다. 같은 검증셋을 이용한 탐색 결과이며 별도 검증이 필요하다.

산출물은 `data/validation/integrated_baseline/final_highres_1280/`의 `targets.csv`, `image_results.csv`, `highres_1280_cache.jsonl`, `simulated_300_predictions.csv`, `summary.json`이다. data 디렉터리는 기존 gitignore 대상이다. 기존 파일 및 parser, CONFIG, requirements, download_weights는 수정하지 않았다. 실험 근거는 별도 브랜치에 기록하며 최종 기능에 병합하지 않는다.


GitHub 보관용 근거: [이미지별 결과](experiments/final_highres_1280/image_results.csv), [요약 및 parser 해시](experiments/final_highres_1280/summary.json), [대상 목록](experiments/final_highres_1280/targets.csv), [300장 가상 예측](experiments/final_highres_1280/simulated_300_predictions.csv). 원본 이미지, weights, OCR 원문 캐시는 로컬에 유지한다.
