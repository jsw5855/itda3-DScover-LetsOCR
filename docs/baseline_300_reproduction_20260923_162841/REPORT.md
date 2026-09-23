# 기존 300장 baseline 재현 결과

2026-09-23 실행. 종료 코드 0, 신규 추론 300개, 캐시 재사용 없음.

| 항목 | 결과 |
|---|---:|
| 공통 정규화 기준 | 247/300 (82.33%) |
| 기존 채점 기준으로 새 예측 채점 | 244/300 (81.33%) |
| 저장된 기존 결과 | 244/300 (81.33%) |
| 과거 예측을 공통 정규화로 재채점 | 247/300 (82.33%) |
| 과거 대비 날짜 예측 변경 | 0/300 |
| 전체 외부 wall time | 659.0024초 (10분 59초) |
| 모델 초기화 | 12.8278초 |
| 이미지별 추론시간 합계 | 644.1327초 |
| 이미지당 평균 추론 | 2.1471초 |
| 전체 wall time / 300 | 2.1967초 |

## 실행 조건

수정된 scripts/evaluate_current_validation_300.py와 evaluation_common.py 사용.
기존 labels/labels_300.csv 300행 전체를 원래 순서로 평가했다.
현재 ocr_pipeline.initialize_engine 및 predict_image, 현재 date_parser 그대로 사용.
v6 medium detector + Korean v5 mobile recognizer, CPU 단일 프로세스,
2 threads, batch 6, MKLDNN=True. 512/rotation270/highres1024/CLAHE pipeline.
기존 v6 baseline의 threads=2를 맞추기 위해 평가 CLI에 --cpu-threads 옵션만 추가했다.
실제 명령은 command.json, 환경/모델 해시는 reproduction_environment.json,
패키지 버전과 코드 해시는 evaluation/manifest.json에 기록했다.

## 변화 원인

원본 기존 결과: outputs/ppocrv6_medium_300/v6_medium_300_results.csv 및 summary.json.
새 추론의 year/month/day/final_date는 과거 예측과 300장 모두 같다.
따라서 이번 정확도 차이는 OCR/Parser 예측 개선이 아니라 평가 정규화 효과다.

| image_id | 과거 저장 정답 | 예측 및 정규화 정답 | 판정 |
|---|---|---|---|
| 1173 | 2021-09-none | 2021-09-NONE | 오답 → 정답 |
| 2062 | 2023-02-none | 2023-02-NONE | 오답 → 정답 |
| 2480 | 2022-07-none | 2022-07-NONE | 오답 → 정답 |

기존 v6 평가기는 final_date 원문 대소문자를 그대로 비교했다.
현재 공통 정규화는 부분 NONE도 대문자로 통일한다. 이에 따라 +3장, +1.00%p다.
이번 실행 전후 OCR/Parser, 원본 CSV 두 개, 검수 CSV 해시는 동일하다.
과거 244 결과 디렉터리에는 당시 코드 해시 manifest가 없어 과거 코드와의
바이트 단위 동일성은 증명하지 않는다. 실제 저장 예측과의 동일성은 확인했다.

과거 618.8956초는 모델 초기화를 제외한 추론 루프 시간이다. 이번 추론 합계는
644.1327초로 약 25.24초(+4.08%) 길다. 루프 로그/타이밍 래퍼 및 당시 시스템
부하까지 완전히 같지는 않으므로 이 차이를 pipeline 성능 회귀로 단정하지 않는다.
과거 시간과 이번 전체 wall time을 직접 비교하지 않는다.

## 산출물

- execution.log: stdout/stderr 전체 로그
- execution_meta.json: 외부 wall time, 종료 코드, 원본 보존 확인
- comparison_summary.json: 재현 및 원인 분리 집계
- image_comparison.csv: 300개 이미지별 과거/현재 예측·정답·판정
- changed_images.csv: 정규화로 판정이 달라진 3장
- evaluation/: manifest, sample_used, predictions, submission, evaluation, failures, summary
- protected_before.json: 원본 및 OCR/Parser 실행 전 해시
- reproduction_environment.json: 환경과 모델/평가 코드 해시
- compare_results.py: 재채점·비교 재현 코드

원본과 검수 이력, OCR/Parser는 수정하지 않았다. Git commit 없음.

## 검증과 경고

평가 관련 회귀 테스트 17개 통과. 300개 이미지별 비교 및 보호 파일 해시 검증 통과.
실행 오류 없음. 로그에는 명시적 모델 사용 시 lang 무시, ccache 미설치,
OMP_NUM_THREADS=2 관련 Paddle 경고와 oneDNN 초기화 메시지가 있다.
모델 초기화와 300장 추론은 정상 완료했다.

이번 코드 변경은 scripts/evaluate_current_validation_300.py의 --cpu-threads
옵션 및 해당 옵션 기록/전달뿐이다. 나머지 신규 산출물은 이 실행 디렉터리에 있다.
