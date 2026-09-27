# 직원 검토 분기 비율 (재확인 비율 근거)

작성: 이수민 · 2026-09-27 · 코드 기준 `bb83a74` (main, Parser 최적화 1·2 + Policy B 포함)
스크립트: `scripts/review_routing_rates.py` · 테스트: `tests/test_review_routing_rates.py`

## 1. 목적

본선 보고서의 도메인·ROI 파트(H&B 스토어 PoC)에 쓸 수치를 만든다. OCR 결과 중
직원 검토로 넘어가는 비율과, 검토 없이 승인될 때 틀릴 위험을 **OCR을 다시 실행하지 않고**
저장된 OCR 원문만으로 계산한다.

## 2. 분기 정의 (위에서부터 먼저 해당하는 하나)

| 분기 | 조건 | 직원 행동 |
|---|---|---|
| MANUAL | 최종 `final_date`에 `NONE`이 있음 (부분 NONE 포함) | 수동 입력 |
| CHOOSE | 멈춘 단계에서 `find_all_candidates`의 서로 다른 날짜가 2개 이상 (M) | 후보 중 선택 |
| RECHECK | 선택 후보의 q < 0.90, 또는 q를 정할 수 없음 | 재확인 |
| CONFIRM | 위에 해당하지 않음 | 확인 후 승인 |

- q: 선택된 날짜가 나온 박스(글자·중심 좌표가 같은 박스)의 인식 신뢰도 최솟값
- q와 M은 제출 코드 `ocr_pipeline.stage_result`로 계산했다. `scripts/diagnose_correct_controls_50.py`의 `evidence()`와 같은 정의다.
- **직원 검토 비율 = MANUAL + CHOOSE + RECHECK**
- 기준값 0.90은 이 데이터를 보고 조정하지 않았다.

## 3. 입력과 방법

- OCR 원문: `docs/run701/ocr_dump.jsonl`
  - 701장, 이수민 PC에서 2026-09-23 실행한 기록이다. **이 브랜치에는 들어 있지 않다.** 아래 위치에서 꺼내 쓴다.

```text
repository : jsw5855/itda3-DScover-LetsOCR
branch     : claude/itda-ocr-github-integration-wpblqh
commit     : c4b2f47d74c7a0df91072e68f686635fe987dc04   (이 파일을 추가한 커밋)
path       : docs/run701/ocr_dump.jsonl
git blob   : ca0e2b33a1de7f30b72c54e309512276171597ae
sha256     : d231bf5656014a4d30b1aac35ef4d796767b2f6247497d84432379e12718cea3
lines      : 701 (한 줄 = 사진 1장)
```

  같은 커밋의 `docs/run701/evaluation.csv`(sha256 `348c0e7acb337317346d2d1127b0346a6b0d142b881da14430c1de632bd900a2`)와
  `docs/run701/summary.json`(sha256 `f16c389a9a01ae241695e046985d085cbffdb70e5c32d68e85f7c527ac4a5ff5`)은 참고만 했다.
  꺼내는 명령:

```powershell
git fetch origin claude/itda-ocr-github-integration-wpblqh
git restore --source=c4b2f47d74c7a0df91072e68f686635fe987dc04 --worktree -- docs/run701/ocr_dump.jsonl
Get-FileHash .\docs\run701\ocr_dump.jsonl -Algorithm SHA256   # 위 sha256과 같아야 한다
# PowerShell의 `git show ... > 파일`은 인코딩을 바꿔 파일을 깨뜨리므로 쓰지 않는다.
```

- Policy B용 단계별 파싱 결과: `docs/parser_optimization2/new_parser_replay.json`
  - main에 있다: commit `d12eb162261aafebc5254aee5641872ffebcbb91`, git blob `122c3e07c6610d4d7754838f15be225204d4c6ca`
  - sha256 `06b0ae5f703ae41b6771b4b374251a287bb68669d0381207945178b2fa0d5fe6`
- 라벨: `tmp_labels_701.csv` (main)
  - 현재 가장 최신 파일이다. approved 69장, candidate 632장.
  - `labels/review/label_review.csv`(main)는 approved가 4장뿐인 옛 상태라서 쓰지 않았다.
  - 정답은 `scripts/evaluation_common.normalize_truth`로 정규화했다.
- 각 사진을 현재 `date_parser`로 다시 파싱해서 cascade를 재현했다. 저장된 예측값은 옛 Parser(`39e1892`) 결과라서 쓰지 않았다.
- 멈춘 단계의 원문으로 분기를 정했다.
- 다음 단계가 필요한데 그 단계가 저장되어 있지 않으면 "재현 불가"로 두고, 비율 계산에서 뺐다.

### cascade 두 가지

1. **프롬프트 기준 (첫 후보에서 멈춤)**: `original_512 → rotation_270 → highres_1024 → clahe`, 후보가 처음 나온 단계에서 멈춘다.
   - 저장 기록만으로 **701장 모두 재현된다** (재현 불가 0장).
2. **현재 main 제출 코드 (Policy B)**: 1단계 후보가 있어도 q < 0.90 또는 M이면 highres를 한 번 더 읽고, q가 더 높은 쪽을 쓴다.
   - `docs/run701` 기록에는 highres가 없는 사진이 많아서 **124장이 재현 불가**다(`policy_b_run701_partial`, 참고용).
   - 빠지는 사진이 바로 재시도 대상이라 비율이 치우친다. 그래서 선우 PC의 701장 × 4단계 전체 기록에서 단계별로 파싱한 결과(`docs/parser_optimization2/new_parser_replay.json`)로 다시 계산했다(`policy_b_full_stage`, 재현 불가 0장).
   - **교차 확인**: 두 기록(수민 PC 기록, 선우 PC 전체 기록)으로 1번 cascade를 계산하면 701장 모두 멈춘 단계, 분기, 예측, 후보 목록이 같다. q만 3장에서 소수점 일곱째 자리 차이가 있고, 기준값을 넘나드는 경우는 없다.

## 4. 결과

### A. 분기별 비율 (라벨 불필요, 핵심 결과)

**Policy B (현재 제출 코드, 701장 전체 기록)**

| 범위 | 장수 | MANUAL | CHOOSE | RECHECK | CONFIRM | **직원 검토** |
|---|---:|---:|---:|---:|---:|---:|
| 전체 | 701 | 117 (16.7%) | 88 (12.6%) | 51 (7.3%) | 445 (63.5%) | **256 (36.5%)** |
| 기존 300 | 300 | 36 (12.0%) | 28 (9.3%) | 21 (7.0%) | 215 (71.7%) | 85 (28.3%) |
| 신규 401 | 401 | 81 (20.2%) | 60 (15.0%) | 30 (7.5%) | 230 (57.4%) | 171 (42.6%) |

**첫 후보에서 멈춤 (프롬프트 기준, `docs/run701`)**

| 범위 | 장수 | MANUAL | CHOOSE | RECHECK | CONFIRM | **직원 검토** |
|---|---:|---:|---:|---:|---:|---:|
| 전체 | 701 | 119 (17.0%) | 89 (12.7%) | 65 (9.3%) | 428 (61.1%) | **273 (38.9%)** |
| 기존 300 | 300 | 37 (12.3%) | 28 (9.3%) | 29 (9.7%) | 206 (68.7%) | 94 (31.3%) |
| 신규 401 | 401 | 82 (20.4%) | 61 (15.2%) | 36 (9.0%) | 222 (55.4%) | 179 (44.6%) |

"기존 300"은 `label_sources`에 existing_300이 있는 사진(두 팀 공통 31장 포함)이다. "신규 401"은 나머지다.

### B. 분기별 정답률 (참고용, 라벨 필요)

맞은 장수 / 해당 분기 장수. approved는 검수 확정, candidate는 미검수 후보값이다.

| cascade | 라벨 | MANUAL | CHOOSE | RECHECK | CONFIRM |
|---|---|---:|---:|---:|---:|
| Policy B | approved (69) | 45/53 | 3/4 | 1/1 | 7/11 |
| Policy B | candidate (632) | 0/64 | 76/84 | 40/50 | 414/434 |
| 첫 후보 | approved (69) | 45/53 | 4/5 | 1/1 | 6/10 |
| 첫 후보 | candidate (632) | 0/66 | 76/84 | 50/64 | 401/418 |

- approved 69장은 검수 과정에서 어려운 사례(연도 없는 날짜 등)를 골라 확정한 것이라 NONE이 들어간 정답이 많다. 그래서 MANUAL이 많고, 전체 분포를 대표하지 않는다.
- candidate의 MANUAL 0/66은 후보 라벨에 NONE이 거의 없기 때문이다. MANUAL은 어차피 직원이 입력하므로 운영상 위험이 아니다.
- 재현한 예측의 전체 정답 수: Policy B **586/701**, 첫 후보 **583/701**. 모두 미검수 라벨 포함이다.
  - 이전 Parser 개선 결과는 579장이었다. 그 뒤 Parser 최적화 1·2가 들어가서 583장이 되었고, 선우 쪽 replay 기준값(583)과 같다.
  - Policy B의 +3은 Policy B replay 결과(586)와 같다.

### C. 위험 건수: CONFIRM으로 갔는데 틀린 사진

| cascade | 장수 | 전체 대비 | CONFIRM 대비 | approved | candidate |
|---|---:|---:|---:|---:|---:|
| Policy B | **24** | 3.4% | 5.4% | 4 | 20 |
| 첫 후보 | 21 | 3.0% | 4.9% | 4 | 17 |

- approved: 001862, 002034, 002125, 002917 (두 cascade 공통)
- candidate (Policy B): 000143, 000231, 000233, 000422, 000466, 000844, 001515, 001587, 001658, 001768, 001852, 001932, 001964, 002524, 002552, 002588, 002726, 002829, 003218, 003245
  - 첫 후보 cascade는 이 중 001768, 002588, 003245가 빠진 17장이다.
  - 이 3장은 모두 같은 패턴이다. 1단계에서는 날짜가 2개 읽혀서 직원 검토로 갔다.
    - 001768, 003245: CHOOSE였고 정답이 맞았다.
    - 002588: 연도와 월만 읽혀 MANUAL이었다.
  - Policy B의 highres 결과에는 날짜가 하나만 남았고 q도 높아서 CONFIRM으로 넘어갔다. 그런데 남은 날짜는 더 이른 날짜(제조일로 보임)였다.
  - 즉 Policy B는 정답을 3장 늘리는 대신, **검토로 가던 사진 3장을 "조용한 오류"로 바꾼다.** 이 중 001768과 003245는 Policy B의 701장 기준 새로 틀린 2장이다.
- candidate 오류에는 라벨 자체의 오류가 섞여 있을 수 있다. 앞선 라벨 확인에서 OCR 결과와 크게 달랐던, 한 팀 라벨만 있는 사진 7장 중 6장이 라벨 오류였다.

### D. 참고 민감도 (정책 변경 아님, 참고용)

RECHECK 기준 q만 바꿨다. Policy B의 재시도 조건(0.90)은 그대로 두었다.

| cascade | q 기준 | RECHECK | CONFIRM | 직원 검토 | CONFIRM 오류 |
|---|---|---:|---:|---:|---:|
| Policy B | 0.85 | 15 | 481 | 220 (31.4%) | 28 (4.0%) |
| Policy B | **0.90** | 51 | 445 | **256 (36.5%)** | **24 (3.4%)** |
| Policy B | 0.95 | 180 | 316 | 385 (54.9%) | 12 (1.7%) |
| 첫 후보 | 0.85 | 23 | 470 | 231 (33.0%) | 25 (3.6%) |
| 첫 후보 | **0.90** | 65 | 428 | **273 (38.9%)** | **21 (3.0%)** |
| 첫 후보 | 0.95 | 186 | 307 | 394 (56.2%) | 10 (1.4%) |

### E. 재현성 정보

| 항목 | 값 |
|---|---|
| 코드 commit | `bb83a74` (+ 이 브랜치의 새 스크립트) |
| 라벨 파일 | `tmp_labels_701.csv` (sha256 `2cc22fab…e5593`) |
| 재현 불가 | 첫 후보 0장 · Policy B(`docs/run701`) 124장(highres 없음) · Policy B(전체 기록) 0장 |
| 옛 예측과 달라진 장수 | 50장 (첫 후보, `docs/run701` 저장 예측 대비) |
| OCR 실행 | 0회 |

결과 파일:
- `docs/review_routing_rates/policy_b_full_stage/`
- `docs/review_routing_rates/first_candidate_run701/`
- `docs/review_routing_rates/policy_b_run701_partial/`

각 폴더에 사진별 `routing_rows.csv`와 `routing_summary.json`이 있다.

## 5. 한계

- **개발 데이터 포함**: 701장에는 Parser와 cascade 개발에 쓴 사진이 섞여 있다. 독립 검증 수치가 아니다.
- **미검수 라벨**: 정답이 필요한 B·C는 대부분(632장) 한 팀 라벨만 있는 미검수 후보값 기준이다. 확정 정확도로 쓰면 안 된다. A(분기 비율)는 라벨이 필요 없다.
- **cascade 의존**: 현재 제출 코드(Policy B) 기준이다. cascade나 Parser가 바뀌면 다시 계산해야 한다. 스크립트를 같은 명령으로 다시 돌리면 된다.
- **Policy B 수치의 출처**: 701장 × 4단계 전체 기록을 선우 PC에서 파싱한 저장값을 썼다. 첫 후보 cascade에서는 이 값이 현재 Parser로 다시 파싱한 결과와 701장 모두 일치했다.
- **도메인 차이**: 식품 포장 사진 기준이다. 화장품(작은 인쇄, 곡면 용기, 영문 표기)이나 매장 현장 촬영 사진과는 비율이 다를 수 있다.

## 6. 다시 실행하는 법

```powershell
# 현재 제출 코드(Policy B), 701장 x 4단계 전체 기록의 단계별 파싱 결과
.\.venv\Scripts\python.exe -B .\scripts\review_routing_rates.py --stage-snapshots .\docs\parser_optimization2\new_parser_replay.json --cascade policy_b --output <새 폴더>
# 첫 후보 cascade, 저장된 OCR 원문을 현재 Parser로 다시 파싱
.\.venv\Scripts\python.exe -B .\scripts\review_routing_rates.py --dump .\docs\run701\ocr_dump.jsonl --cascade first_candidate --output <새 폴더>
```
