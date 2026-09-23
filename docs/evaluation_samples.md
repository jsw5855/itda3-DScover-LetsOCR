# 평가 정규화와 고정 표본

두 평가기는 evaluation_common.py의 normalize_truth, evaluation_dataset,
score_prediction을 공유한다. 원본/검수 CSV와 OCR/Parser는 수정하지 않는다.

정답은 NONE 대소문자 통일, 부분 NONE 유지, 전체 NONE의 final_date=NONE,
월/일 2자리로 정규화한다. 단축 연도는 parser의 2000/1900 후보 순서와
DEFAULT_YEAR_MIN/MAX를 사용한다. 범위 밖 4자리 연도는 달력상 유효하면 보존한다.
빈 값, 해석 불가능한 단축 연도, 달력 오류, component/final_date 불일치는
오류다. 기존 300장의 소문자 none과 0 채움 차이는 허용한다.

예측은 기존 결과와의 호환성을 위해 원문 exact-match로 평가한다.
예측 final_date를 component로 재생성하여 오류를 숨기지 않는다.
같은 예측을 받으면 두 평가 방식의 판정은 같다. 실제 OCR 프로세스/스레드
설정이 다르므로 추론 결과 동일성까지 보장하지 않는다.

## 표본 생성과 재사용

```powershell
.\.venv\Scripts\python.exe -B scripts/prepare_evaluation_sample.py --labels labels/labels_300.csv --images data --mode baseline --size 250 --seed 42 --output labels/samples/baseline_250_seed42.json
.\.venv\Scripts\python.exe -B scripts/evaluate_current_validation_300.py --run --labels labels/labels_300.csv --images data --dataset-mode baseline --sample-manifest labels/samples/baseline_250_seed42.json --output outputs/experiment_serial
.\.venv\Scripts\python.exe -B scripts/evaluate_current_validation_300_parallel.py --labels labels/labels_300.csv --images data --dataset-mode baseline --sample-manifest labels/samples/baseline_250_seed42.json --output outputs/experiment_parallel
```

표본 생성은 OCR을 실행하지 않는다. --size 기본값은 250, --seed는 42다.
기존 manifest는 덮어쓰지 않는다. image_id, 실제 파일명, 정규화 정답,
seed, 원본 해시와 sample_fingerprint를 저장한다.

실험 간에는 반드시 동일 manifest를 재사용하고 summary의 sample_fingerprint가
같은 결과만 비교한다. seed만 같고 모집단이 다르면 동일 표본이 아니다.
선택된 이미지의 라벨/승인/파일 매핑이 바뀌면 실행을 거부한다. 선택되지 않은
행의 검수 진행은 재사용을 막지 않는다. 이미지 파일 내용까지 fingerprint가
탐지하지는 않으므로 동일 원본 이미지 디렉터리를 유지한다.

실행 결과의 sample_used.json에 실제 표본과 정답/원본 해시를 남긴다.
기존 도구 호환을 위해 submission_300.csv, evaluation_300.csv 파일명은
유지하지만 실제 행 수와 시간 분모는 선택된 크기를 사용한다.
baseline 모드에서 manifest를 생략하면 기존 순서대로 300장 전체를 평가한다.
baseline은 과거 라벨 비교이며 새로운 master 승인으로 간주하지 않는다.

## 승인 통합 데이터

```powershell
.\.venv\Scripts\python.exe -B scripts/prepare_evaluation_sample.py --labels labels/review/label_review.csv --images data --mode approved --review-status approved --size 4 --seed 42 --output labels/samples/approved_4_seed42.json
```

평가 시 --dataset-mode approved와 위 manifest를 사용한다. 현재 승인 라벨은
4개뿐이므로 200~300개 요청은 오류이며 자동 축소하지 않는다. 승인 개수가
늘어나면 새 표본 버전을 만든다. approved, master_eligible=true, reviewer와
review_note가 채워진 행만 허용한다. pending/ready_for_review 필터는 거부한다.
검수 파일을 baseline으로 전달하여 승인 조건을 우회할 수도 없다.
정답은 승인된 최종 날짜 열에서만 읽으며 candidate 및 원문은 수정하지 않는다.
