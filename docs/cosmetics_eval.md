# 화장품 사진 OCR 평가: 사용 안내

작성: 이수민 · 2026-09-27
스크립트 `scripts/run_cosmetics_eval.py` · 노트북 `cosmetics_eval.ipynb`(저장소에 올리지 않고 zip으로 전달) · 테스트 `tests/test_run_cosmetics_eval.py`

## 1. 무엇을 하나

직접 찍고 라벨링한 화장품 사진을 **현재 제출 파이프라인 그대로** 실행해서 채점한다.

- 공식 예측값은 `ocr_pipeline.predict_image` 결과 그대로다.
- OCR 원문을 사진마다 저장한다(`ocr_dump_<split>.jsonl`). 나중에 Parser를 고쳐도 OCR을 다시 돌리지 않고 재채점할 수 있다.
- full-stage 모드에서는 제출 cascade가 쓰지 않은 단계도 같은 전처리·파라미터로 추가 실행해 저장한다. 공식 예측값은 바뀌지 않는다.
- 결과를 보고 설정을 바꾸지 않는다. 측정만 한다.

## 2. 내 PC에서 실행하는 순서 (초보자용)

1. **zip 풀기**: 받은 `itda_OCR_cosmetics.zip`을 `C:\Users\user\itda_OCR`에 풀고, 같은 이름 파일은 **덮어쓴다**.
   - 코드 파일(`ocr_pipeline.py`, `date_parser\`, `scripts\` 등)이 최신으로 바뀐다.
   - `weights\` 폴더와 사진, 라벨은 건드리지 않는다.
2. **파일 위치 확인**
   - 사진: `C:\Users\user\itda_OCR\Cosmetics\900001.jpg` … `900150.jpg` (설정 셀의 `IMAGE_DIR`와 같아야 함)
   - 라벨: `C:\Users\user\itda_OCR\cosmetics_labels_900001-900150.csv`
   - 모델: `C:\Users\user\itda_OCR\weights\paddleocr\PP-OCRv6_medium_det\` 와 `…\korean_PP-OCRv5_mobile_rec\`
3. **주피터 열기**: 예전처럼 가상환경을 켠 cmd 창에서 `C:\Users\user\itda_OCR`로 이동해 `jupyter notebook`을 실행한다.
   - 브라우저에서 `cosmetics_eval.ipynb`를 연다.
4. **셀을 위에서부터 하나씩 실행**(`Shift+Enter`)
   - ① 설정: 경로가 맞는지 본다.
   - ② 환경 확인: "환경 확인 통과"가 나와야 한다. 라벨 오류나 모델 파일 누락이 있으면 여기서 멈춘다.
   - ②-1 기록 점검: 중복 기록이 있으면 정리한다.
   - ③ 예상 시간: 3장만 먼저 돌려 이 PC의 장당 시간과 남은 시간을 계산한다.
   - ④ 본 실행: 진행률이 5장마다 나온다. 멈추면 이 셀만 다시 실행하면 이어서 돈다.
   - ⑤ 결과 요약
   - ⑥ 서현 전달 파일 경로
5. **결과 파일**: `C:\Users\user\itda_OCR\cosmetic_results\` (저장소 폴더 안의 코드는 건드리지 않음)
   - `errors_for_analysis_dev.csv`: 틀린 사진만. **서현 전달용**
   - `results_dev.csv`: 사진별 결과(엑셀로 열림)
   - `summary_dev.json`: 전체 수치와 실행 환경
   - `ocr_dump_dev.jsonl`: OCR 원문. **지우지 말 것**(재채점에 필요)
   - `run_manifest_dev.json`: 실행 조건(이어서 실행할 때 확인용)
6. **실행 중 주의**
   - 절전 모드를 "안 함"으로 두고, 노트북 PC는 뚜껑을 닫지 않는다.
   - 주피터의 검은 cmd 창을 닫지 않는다.
   - 멀티프로세싱은 쓰지 않는다. 한 프로세스에서 순서대로 실행한다(Windows 주피터에서 멈춤 방지).

**서현 라벨 150장이 추가되면**
1. 사진을 `Cosmetic` 폴더에 넣는다.
2. 설정 셀의 `LABEL_CSVS`에서 두 번째 줄 주석을 푼다.
3. ②부터 다시 실행하면 **새 사진만** 실행된다.

사진이 아직 없는 라벨은 건너뛰고 목록만 보여 준다.

## 3. 예상 시간 (추정)

근거: 이 PC에서 식품 701장을 돌렸을 때의 단계별 평균이다(`docs/ocr_handoff/OCR_issues_v6.md`, 브랜치 `claude/itda-ocr-github-integration-wpblqh`).

| 단계 | 장당 |
|---|---:|
| original_512 | 약 10초 |
| rotation_270 | 약 10초 |
| highres_1024 | 약 40초 |
| clahe | 약 10초 |

| 모드 | 장당 | 150장 |
|---|---:|---:|
| fast (제출 cascade가 실행한 단계만) | 약 15~20초 (1단계 + 재시도·후보 없음 사진의 추가 단계) | 약 40~50분 |
| full-stage (4단계 모두) | 약 70초 | **약 3시간** |

- 식품 사진 기준 추정이다. 화장품은 1단계에서 날짜를 못 찾는 비율이 다르면 fast 시간이 달라진다.
- 실제 시간은 노트북 ③ 셀이 이 PC에서 3장을 재서 다시 계산한다.
- full-stage를 권장한다. 시간은 더 들지만, 나중에 Parser를 고친 뒤 "재현 불가" 없이 재채점할 수 있다.

## 4. 결과 열 설명 (`results_<split>.csv`)

| 열 | 뜻 |
|---|---|
| `true_final_date` / `pred_*` | 정답 / 공식 예측 |
| `correct_final`, `correct_year` … | 날짜 전체와 연·월·일 정답 여부 (`evaluation_common.score_prediction`) |
| `stop_stage` | 최종 예측을 낸 단계. `original_512_retry_kept`: 재시도했지만 원본 유지, `highres_1024_retry`: 재시도 결과 채택 |
| `route` | MANUAL / CHOOSE / RECHECK / CONFIRM. 정의는 `docs/review_routing_rates.md`와 같다 |
| `q`, `distinct_dates` | 선택 박스의 인식 신뢰도, 서로 다른 날짜 후보 |
| `candidate_dates`, `selected_source_text` | 후보 날짜와 그 원문, 선택된 원문 |
| `first_guess` | 틀린 사진의 1차 분류. 저장된 어느 단계든 한 박스 안에 정답 숫자가 모두 보이면 "Parser 의심", 아니면 "OCR 의심" (701장 노트북과 같은 규칙) |
| `ocr_text` | 최종 단계에서 OCR이 읽은 글자 전부 |
| `notes` | 라벨의 메모 |

## 5. 이어서 실행(resume)과 거부 조건

- 사진 1장이 끝날 때마다 dump에 한 줄씩 추가한다. 중단 후 다시 실행하면 끝난 사진은 건너뛴다.
- 쓰다가 끊긴 마지막 줄은 지우고 그 사진만 다시 실행한다.
- 다음 경우에는 **이어서 실행하지 않고 멈춘다**. 새 `OUTPUT_DIR`을 쓰거나 원래대로 되돌린다.
  - 코드 버전이나 코드 파일(`ocr_pipeline.py`, `date_parser\`, 이 스크립트)이 바뀜
  - 모드(full-stage/fast)나 엔진 설정이 바뀜
  - 이미 끝난 사진의 라벨이 바뀜

**동시 실행 금지와 중복 기록 복구**
- 같은 결과 폴더에 두 실행이 동시에 쓰지 못하게 잠근다(`run_<split>.lock`). 주피터를 두 개 띄워 같은 셀을 돌리면 두 번째 실행은 멈춘다.
- 이전 버전에서 동시에 실행해 같은 사진이 두 번 기록됐다면, 노트북 2-1 셀(`repair_dump`)이 첫 기록만 남긴다. 원본은 `ocr_dump_<split>.jsonl.bak-날짜` 파일로 보관한다.

## 6. 검증용(900301~900400) 규칙

- **Parser 개선이 끝날 때까지 실행하지 않는다.** 사진 폴더 위치만 정해 둔다: `C:\Users\user\itda_OCR\Cosmetic_validation`
- 스크립트는 `split="validation"`일 때 `CONFIRM_VALIDATION = "개선 완료"`가 없으면 실행을 거부한다.
- 검증용은 노트북에 전체 수치만 보여 주고, 사진별 오류 목록 파일(`errors_for_analysis_*.csv`)은 만들지 않는다.
- `results_validation.csv`에는 사진별 결과가 저장되지만 **열람하지 않는다.**

## 7. 저장된 dump 하나로 두 Parser 버전 재채점하기

검증용을 실행할 때 "개선 전"과 "개선 후" Parser를 **같은 OCR 원문**으로 비교하는 절차다.

- 개선 전: 이번 dev 실행의 `summary_dev.json` → `environment.run.code_version`에 적힌 버전
- 개선 후: 그때의 최신 코드

재채점은 `rescore` 명령으로 한다. 저장된 단계들을 **그 폴더의 코드**(`date_parser`, 제출 cascade)로 다시 파싱하고, `results_<split>_rescored.csv`와 `summary_<split>_rescored.json`을 쓴다. OCR은 돌리지 않는다.

### 방법 A: git이 없는 지금 PC (폴더 두 개)

1. 이번에 받은 zip을 **따로** `C:\Users\user\itda_OCR_baseline`에 한 번 더 풀어서 보관한다. 이 폴더가 "개선 전" 코드다. 모델 파일은 필요 없다.
2. Parser 개선 후 최신 코드를 받은 `C:\Users\user\itda_OCR`이 "개선 후"다.
3. 각 폴더에서 같은 dump와 라벨로 재채점하고, 출력 폴더는 따로 둔다. 먼저 dump를 복사한다.

```powershell
# 개선 전
cd C:\Users\user\itda_OCR_baseline
mkdir C:\Users\user\itda_OCR\rescore_before
copy C:\Users\user\itda_OCR\cosmetic_results\ocr_dump_validation.jsonl C:\Users\user\itda_OCR\rescore_before\
copy C:\Users\user\itda_OCR\cosmetic_results\run_manifest_validation.json C:\Users\user\itda_OCR\rescore_before\
python scripts\run_cosmetics_eval.py rescore --labels C:\Users\user\itda_OCR\cosmetics_labels_900301-900400.csv --split validation --output C:\Users\user\itda_OCR\rescore_before

# 개선 후 (같은 방식, 폴더만 바꿈)
cd C:\Users\user\itda_OCR
mkdir C:\Users\user\itda_OCR\rescore_after
copy C:\Users\user\itda_OCR\cosmetic_results\ocr_dump_validation.jsonl C:\Users\user\itda_OCR\rescore_after\
copy C:\Users\user\itda_OCR\cosmetic_results\run_manifest_validation.json C:\Users\user\itda_OCR\rescore_after\
python scripts\run_cosmetics_eval.py rescore --labels C:\Users\user\itda_OCR\cosmetics_labels_900301-900400.csv --split validation --output C:\Users\user\itda_OCR\rescore_after
```

4. 두 폴더의 `summary_validation_rescored.json`에서 `correct_final`을 비교한다.

### 방법 B: git 저장소가 있을 때 (worktree)

```powershell
git fetch origin
git worktree add ..\itda_OCR_baseline <개선 전 commit>
cd ..\itda_OCR_baseline
python scripts\run_cosmetics_eval.py rescore --labels <라벨 CSV> --split validation --output <개선 전 출력 폴더>
cd ..\<저장소>
python scripts\run_cosmetics_eval.py rescore --labels <라벨 CSV> --split validation --output <개선 후 출력 폴더>
git worktree remove ..\itda_OCR_baseline
```

주의:
- `rescore`는 **full-stage로 저장한 dump**에서만 "재현 불가" 없이 동작한다. fast dump는 새 cascade가 저장되지 않은 단계를 요구하면 `UNREPRODUCIBLE`로 표시된다.
- dev 결과도 같은 방법으로 재채점할 수 있다(`--split dev`).
- 식품 701장 분기 비율과 같은 표를 만들려면 `scripts/review_routing_rates.py --dump <ocr_dump_*.jsonl>`도 그대로 쓸 수 있다. 라벨 CSV에 `truth_source`, `label_sources` 열이 필요하다.

## 8. 제출 코드와의 관계

- `ocr_pipeline.py`, `date_parser/`, `predict.ipynb`, `requirements.txt`는 수정하지 않았다. import만 한다.
- 엔진 설정은 단일 프로세스 제출과 같다: `initialize_engine(enable_mkldnn=True, cpu_threads=4, recognition_batch_size=6)`
  - 실제 제출은 기본 2프로세스 × 2스레드라서 **시간은 제출 환경과 다르다.**
- 분기와 재채점은 `scripts/review_routing_rates.py`의 함수(제출 코드의 `run_cascade`를 따름)를 그대로 쓴다.
