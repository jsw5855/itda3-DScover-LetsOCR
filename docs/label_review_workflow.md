# 라벨 통합 검수 준비

`scripts/prepare_label_review.py`는 원본 CSV와 기존 parser/evaluation 코드를
수정하지 않는다. 기본 실행은 메모리 내 감사만 수행한다.

```powershell
.\.venv\Scripts\python.exe -B scripts/prepare_label_review.py
```

## 생성 예정 파일 (현재 생성하지 않음)

명시적인 `--write --output-dir labels/review_staging_v1` 실행 시 아래 파일만
새 디렉터리에 생성한다. 이미 존재하는 출력 디렉터리는 거부한다.

- `label_review.csv`: 701개 unique image_id를 담는 단일 검수 목록.
  사유별/검수자별/기존 8개 사례별 별도 검수 CSV는 만들지 않는다.
  `review_status=pending`으로 우선 검수 62개를 필터링할 수 있다.
- `source_records.jsonl`: 원문 732행 전체, 출처/CSV 레코드 번호,
  원문 모든 열, canonical 값, 형식 변화 및 검출 근거를 담는 보존용 파일.
  별도 검수 업무 목록이 아니다.
- `audit_summary.json`: 원본 경로, schema, SHA-256, 집계 및 탐지 한계.

CSV 한 행에 이미지 절대 경로, 양쪽 출처의 raw/canonical 날짜,
file_name/notes/Unnamed: 7, 전체 source records JSON을 모은다.
JSON 배열은 예상치 못한 내부 중복도 손실 없이 보존한다.
`review_reason`은 세미콜론으로 연결한 복수 사유다.

8개 기존 확인 사례는 106,157,1862,1906,2152,2284,2448,2521이며,
같은 목록에서 `previously_visually_reviewed=true`로 표시한다.
구체적인 기존 판정은 전달받지 않았으므로 `prior_review_decision`은 비워두고,
`review_note`에는 확인 이력과 판정 미전달 사실을 기록한다.
한 명의 reviewer가 동일 기준으로 전체 우선 목록을 검수한다.

`candidate_*`는 원문 component들의 정규화 결과가 모두 같을 때만 채운다.
component/final_date 불일치가 있으면 candidate는 단지 component 기반 제안이며
정답이 아니다. 충돌 시 candidate도 비운다. 최종 year/month/day/final_date는
모두 비워두며 모든 행의 master_eligible은 false다. ready_for_review도 승인된
정답을 뜻하지 않는다. 이 스크립트는 master ground truth를 내보내지 않는다.
후속 확정 단계는 reviewer, review_note, 결정 날짜와 미해결 사유를 확인하고
별도로 승인된 행만 내보내도록 구현해야 한다.

## 공통 정규화 계약

`canonical_normalize` 하나를 비교 및 후보 직렬화에 사용한다.
NONE 대소문자 통일, 월/일 2자리, 일부 NONE 유지, 전체 NONE의 final_date는
NONE이다. 단축 연도는 현재 parser의 기본 범위와 2000/1900 세기 후보 순서를
사용한다. 해석할 수 없는 값은 NONE으로 추측하지 않고 검수 사유로 남긴다.
4자리 범위 밖 연도는 보존하고 unusual_year로 표시한다. parser 기본 범위는
공식 라벨 허용 범위가 아니므로 2001을 버리지 않는다.
달력 오류도 원문/후보를 보존한 채 별도 사유로 표시한다.

순서 모호성 자동 검출은 팀 간 월/일 교환, 연도 없는 월/일 교환 후보,
기존 README의 2034/2917에 한정된다. 이미지 기반 전수 판정을 대체하지 않는다.

## Evaluation 통일 제안 — 아직 적용하지 않음

현재 단일 프로세스 평가기는 전체 NONE을 NONE-NONE-NONE으로 조합하지만
병렬 평가기는 NONE으로 조합한다. 두 코드 모두 25를 2025가 아닌 0025로
채울 수 있다. 두 코드 모두 component로 정답 final_date를 재생성하므로
원본 final_date와의 불일치를 가릴 수도 있다.

추후 별도 공통 모듈에 현재 canonical_normalize 계약을 옮기고, 검수 스크립트와
두 평가기가 같은 함수를 import하도록 한다. 원문 CSV를 먼저 보존한 뒤
정답과 예측의 component를 각각 동일 계약으로 처리한다. 저장된 final_date도
동일 함수를 통해 해석하여 component와 비교하고, 불일치/미해결 값/달력 오류는
평가 시작 전에 명시적으로 실패시킨다. 예측 final_date를 무조건 재생성해서
잘못된 출력을 정답으로 바꾸면 안 된다. 출력 형식 엄격 검증과 날짜 의미 비교는
별도 지표로 유지한다. 빈 문자열/NaN을 자동 NONE으로 바꾸지 않는다.

기본 범위, 경계 연도, 단축 연도 해석 실패, 부분/전체 NONE, 윤년과
component/final_date 충돌 회귀 사례를 두 평가 경로에 동일하게 적용한다.
현재는 기존 평가 파일을 변경하지 않았다.
