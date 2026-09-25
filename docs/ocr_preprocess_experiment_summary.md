# OCR 및 이미지 전처리 실험 요약

담당: 정서현

## 실험 목적

상품 이미지에서 소비기한 날짜 후보를 더 많이 읽으면서 CPU 4코어 환경의 실행시간을 40분 안으로 유지하는 OCR 구성을 찾았다. OCR 결과에서 날짜 후보를 추출하고 최종 소비기한을 선택하는 로직은 팀원의 `date_parser`로 고정하고, OCR 엔진·해상도·회전·대비 보정 같은 OCR 및 이미지 전처리만 비교했다.

이번 최종 수치는 `labels/labels_300.csv`의 동일한 300장을 방법 선택과 평가에 반복 사용한 **내부 개발 평가**다. 별도 테스트셋의 일반화 성능이나 최종 제출 성능을 의미하지 않는다.

## 공통 평가 기준

- 날짜 파서: 팀원 커밋 `3ff5347`
- OCR: PaddleOCR `PP-OCRv5_mobile_det` + `korean_PP-OCRv5_mobile_rec`
- 기본 입력: EXIF 방향 보정 → RGB 변환 → 긴 변 512px LANCZOS 축소
- `text_recognition_batch_size=6`
- `text_det_box_thresh=0.7`
- `text_det_limit_type="max"`
- CPU threads 4, MKLDNN 활성화
- 문서 방향 분류, 왜곡 보정, 텍스트 줄 방향 분류 비활성화
- 파서 입력: 이미지별 독립된 `text`, `confidence`, `bbox`

Fallback 실행 여부와 결과 선택에는 정답, `image_id`, 수작업 메모를 사용하지 않았다. 각 패스는 독립적으로 OCR과 파서를 실행했고 여러 패스의 텍스트나 박스를 하나의 파서 입력으로 섞지 않았다. 날짜 후보가 생기면 해당 패스 결과를 채택하고 뒤 fallback은 실행하지 않는다.

평가 지표는 연·월·일 개별 정확도, 최종 날짜 완전일치, 날짜 후보 유무, OCR 박스 유무, 이미지별 처리시간이다. `confidence`는 문자 인식 신뢰도이며 소비기한일 확률은 아니다.

## 엔진과 기본 설정 선택

초기 동일 10장 비교에서는 EasyOCR과 PaddleOCR을 같은 이미지와 공통 전처리로 비교했다. PaddleOCR은 날짜 문자열 회수율이 높았고, 이후 한국어 모바일 검출·인식 모델과 512px 입력을 명시하면서 CPU 처리시간을 줄였다.

| 초기 동일 10장 비교 | 날짜 토큰 재현율 | 평균 초/장 |
|---|---:|---:|
| EasyOCR, 긴 변 1600px | 80% | 21.417 |
| PaddleOCR 최초 구성, 긴 변 1600px | 100% | 42.926 |
| EasyOCR, 긴 변 1000px | 80% | 13.693 |
| PaddleOCR 1000px, 검출 한도 640px | 100% | 31.883 |

| 동일 10장 핵심 설정 | 날짜 토큰 재현율 | 평균 초/장 | 결정 |
|---|---:|---:|---|
| 512px, batch6, threshold 0.6 | 100% | 1.730 | 출발 기준 |
| 384px | 100% | 2.499 | 속도 악화로 제외 |
| batch1 | 100% | 3.287 | 속도 악화로 제외 |
| batch12 | 100% | 2.245 | 속도 악화로 제외 |
| threshold 0.7 | 100% | 1.341 | 통합 baseline에 채택 |
| threshold 0.75 | 90% | 1.240 | 재현율 저하로 제외 |
| threshold 0.8 | 60% | 0.864 | 재현율 저하로 제외 |

고정 seed=42의 30장 사전 검증에서는 threshold 0.7 단독의 숫자형 날짜 토큰 재현율이 22/29(75.86%), `04 NOV 2021` 수동 OCR 문자 확인을 포함하면 23/29(79.31%)였다. 이 사전 결과는 최신 파서와 300장 전체를 함께 평가한 아래 통합 baseline으로 대체한다.

## 300장 통합 baseline

최신 파서와 공통 OCR 설정을 연결한 원본 512px baseline은 다음과 같다.

| 지표 | 결과 |
|---|---:|
| 최종 날짜 완전일치 | 175/300 = 58.33% |
| 연도 정확도 | 190/300 = 63.33% |
| 월 정확도 | 189/300 = 63.00% |
| 일 정확도 | 189/300 = 63.00% |
| OCR 박스 미탐지 | 7건 |
| 날짜 후보 존재 | 220/300 = 73.33% |
| 평균 / 중앙 처리시간 | 1.282초 / 1.021초 |
| 모델 초기화 제외 총 처리시간 | 6.41분 |

OCR이 전체 처리시간의 약 90.3%를 차지했고 OCR 박스 수와 처리시간의 상관계수는 0.926이었다. 속도 병목은 파서가 아니라 OCR 검출·인식 단계였다.

## 조건부 fallback 실험

아래 시간은 최종 배치에 필요한 패스만 반영했다. 초기 회전 실험에서 측정한 180도와 90도는 후보 회수가 없으므로 최종 시간에서 제외했다.

| 방법 | 적용 대상과 결과 | 결정 |
|---|---|---|
| 270도 회전 | 원본 후보 없음 80장, 후보 9건, 새 완전일치 7건 | 채택 |
| 180도 회전 | 80장, 후보 회수 0건 | 제외 |
| 90도 회전 | 80장, 후보 회수 0건 | 제외 |
| 원본 1024px | 270도 후 후보 없음 71장, 후보 23건, 새 완전일치 20건 | 채택 |
| CLAHE | 고해상도 후 후보 없음 48장, 후보 5건, 새 완전일치 3건 | 채택 |
| Gamma | 회색조 평균 110/255 미만 12장, 후보 회수 0건 | 제외 |
| 키워드 주변 crop 1024px | 최종 후보 없음 43장 중 키워드가 있던 13장 실행, 후보 회수 0건 | 실행했으나 효과 없어 제외 |

CLAHE는 RGB 이미지를 LAB로 변환하고 L 채널에 `clipLimit=2.0`, `tileGridSize=(8, 8)`로 적용했다. Gamma는 어두운 이미지에만 `gamma=0.7`로 적용했다. 전역 histogram equalization은 사용하지 않았다.

키워드 crop은 최신 파서의 소비기한 키워드가 비회전 OCR 결과에 있을 때만 실행했다. 가장 confidence가 높은 키워드 bbox를 기준으로 원본 이미지에서 영역을 자르고 긴 변을 1024px로 확대했으나, 13회 모두 날짜 후보를 회수하지 못했다.

고해상도를 CLAHE보다 먼저 적용했다. 두 방법 모두 후보를 만든 7장 중 3장은 고해상도 결과가 맞고 CLAHE 결과가 틀렸다. CLAHE를 먼저 두면 잘못된 후보에서 조기 종료되어 202/300에 머물지만 고해상도를 먼저 두면 205/300을 얻었다.

## 최종 채택 pipeline

```text
original_512
  → 날짜 후보가 없을 때만 rotation_270
    → 그래도 없을 때만 highres_1024
      → 그래도 없을 때만 CLAHE
        → 그래도 없으면 원본 결과 유지
```

| 단계 | 실제 실행 수 | 후보 회수 | 새 완전일치 |
|---|---:|---:|---:|
| rotation_270 | 80 | 9 | 7 |
| highres_1024 | 71 | 23 | 20 |
| CLAHE | 48 | 5 | 3 |
| 합계 | 199 | 37 | 30 |

최종 날짜 완전일치는 **205/300 = 68.33%**로 원본 baseline보다 **30장, 10.00%p** 개선됐다. 연도는 226/300(75.33%), 월은 225/300(75.00%), 일은 219/300(73.00%)이다. 원본에서 맞았던 175장이 fallback 때문에 틀린 사례는 0건이다.

최종 채택 pipeline의 보고용 예상 처리시간은 모델 초기화와 실행 변동 여유를 포함해 약 **12.02분(약 12분)**이다. 저장된 패스별 시간을 재계산한 값은 모델 초기화 제외 약 **11.56분**, 초기화 포함 약 **11.71분**이며, 모두 40분 제한 안이다. 효과가 없었던 keyword crop은 최종 실행시간과 pipeline에서 제외한다.

90/180도 회전, gamma 단독, keyword crop은 실험 결과가 없어서 제외한 것이 아니다. 실제 실험에서 날짜 후보 추가 회수가 각각 0건이어서 효과가 없는 것으로 판단하고 최종 pipeline에서 제외했다.

## 최종 실패와 한계

최종 실패 95건은 다음과 같다.

| 실패 단계 | 건수 | 해석 |
|---|---:|---|
| 모든 실행 패스에서 OCR 박스 없음 | 1 | 검출 단계부터 실패 |
| OCR 박스는 있으나 날짜 후보 없음 | 42 | OCR 문자열이 날짜 후보 형식으로 이어지지 않음 |
| 날짜 후보는 있으나 최종 날짜 불일치 | 52 | 후속 원인 분석 필요 |

`candidate_but_wrong` 52건을 모두 파서 문제로 단정할 수 없다. 날짜 숫자를 잘못 읽거나 구분자를 누락한 OCR 인식 오류와, 여러 날짜 중 잘못 고른 파서 선택 오류, YMD/DMY 같은 순서 해석 오류가 섞여 있을 수 있다. OCR 원문·confidence·bbox와 파서가 선택한 후보를 함께 보면서 원인을 다시 분류해야 한다.

같은 300장으로 fallback 순서를 선택하고 결과를 계산했으므로 68.33%는 낙관적일 수 있다. 별도 검증 표본에서 정확도와 시간을 다시 확인해야 한다. 날짜 후보가 잘못 생기면 뒤 fallback이 조기 종료되는 구조도 남은 한계다. `NONE` 필드는 추정하지 않고 그대로 유지해야 한다.

## 실행 코드와 결과 위치

모든 명령은 프로젝트 루트와 기존 `.venv`에서 실행한다. 완료된 OCR 결과가 있으면 각 스크립트의 cache를 먼저 확인해 불필요한 재추론을 피한다.

| 목적 | 실행 파일 | 결과 폴더 |
|---|---|---|
| 300장 통합 baseline | `scripts/evaluate_integrated_baseline.py` | `data/validation/integrated_baseline/` |
| 회전 fallback | `scripts/evaluate_rotation_retry.py` | `data/validation/integrated_baseline/rotation_retry/` |
| 1024px fallback | `scripts/evaluate_highres_retry.py` | `data/validation/integrated_baseline/highres_retry/` |
| CLAHE/gamma | `scripts/evaluate_contrast_retry.py` | `data/validation/integrated_baseline/contrast_retry/` |
| 최종 cache 결합 | `scripts/build_final_cascade.py` | `data/validation/integrated_baseline/final_cascade/` |
| 키워드 crop 검증 | `scripts/evaluate_keyword_crop_retry.py` | `data/validation/integrated_baseline/keyword_crop/` |

PowerShell 실행 예:

```powershell
& .\.venv\Scripts\python.exe .\scripts\build_final_cascade.py
```

최종 이미지별 결과, 모든 실행 패스의 OCR 박스, 실패 목록과 요약은 `data/validation/integrated_baseline/final_cascade/` 아래에 있다. Keyword crop은 최종 pipeline에 포함하지 않으며 결과는 별도 `keyword_crop/` 폴더에 보존한다. `data/`, `labels/`, `weights/`, `.venv/`는 Git에 포함하지 않는다.

## 날짜 파서 담당자에게 전달할 형식

```json
{
  "text": "04 NOV 2021",
  "confidence": 0.98,
  "bbox": [[10, 20], [110, 20], [110, 40], [10, 40]]
}
```

좌표가 어떤 OCR 패스와 입력 크기를 기준으로 하는지 함께 보존한다. 여러 패스의 bbox는 좌표계가 다르므로 합치지 않는다. `002728.jpg`의 `04 NOV 2021`처럼 OCR 문자는 맞지만 해석이 필요한 사례도 OCR 실패와 구분한다.
