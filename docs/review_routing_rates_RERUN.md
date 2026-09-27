# 최종 코드 확정 후 재계산 프롬프트 (직원 검토 분기 비율)

아래 블록 전체를 새 채팅에 붙여 넣는다. `<...>` 부분만 채운다.

```text
# 작업: 최종 제출 코드 기준으로 직원 검토 분기 비율을 다시 계산한다

## 배경
- 초안: 저장소 jsw5855/itda3-DScover-LetsOCR, 브랜치 claude/review-routing-rates,
  문서 docs/review_routing_rates.md, 스크립트 scripts/review_routing_rates.py
- 초안은 main bb83a74 기준이고, 대표값은 저장된 단계별 파싱 결과로 계산했다
  (직원 검토 36.5%, CONFIRM인데 틀림 3.4%, 그중 늦은 날짜 1.1%).
- 이번에는 최종 제출 코드로 701장 x 4단계 원본 OCR을 직접 다시 파싱해서 최종 수치를 만든다.

## 최종 제출 코드
- main commit: <최종 main commit hash, 예: abc1234>
- 최종 코드에서 OCR 모델/전처리/단계 정의가 bb83a74와 같은가: <같음 / 다름>

## 지켜야 할 규칙
- OCR을 새로 실행하지 않는다. 단, 위에서 "다름"이면 멈추고 보고한다(6번 참고).
- main에 커밋하거나 push하지 않는다. claude/ 로 시작하는 새 브랜치에서만 작업하고, force push는 하지 않는다.
- 제출 코드(ocr_pipeline.py, date_parser/, predict.ipynb, requirements.txt)는 수정하지 않는다. import만 한다.
- 분기 정의와 기준값(q 0.90)은 바꾸지 않는다. 결과를 보고 기준을 조정하지 않는다.
- docs/full_stage_701_run1/ 는 읽기만 한다. 삭제, 덮어쓰기, --run-ocr, --consolidate-only 모두 금지.
- 701장에는 개발에 쓴 사진이 섞여 있고 라벨 대부분은 미검수다. 확정 정확도나 독립 검증처럼 쓰지 않는다.
- 헷갈리거나 아래 확인이 하나라도 어긋나면 임의로 판단하지 말고 멈춰서 보고한다.

## 절차
1. git fetch origin 후, 최종 main commit에서 새 브랜치 claude/review-routing-rates-final 을 만든다.
   git switch -c claude/review-routing-rates-final <최종 main commit> 로 만들고 upstream은 해제한다.
2. 초안 브랜치에서 도구를 가져온다(제출 코드가 아니므로 가져와도 된다).
   git restore --source=origin/claude/review-routing-rates --worktree -- scripts/review_routing_rates.py tests/test_review_routing_rates.py docs/review_routing_rates.md docs/review_routing_rates_RERUN.md
3. 도구가 최종 코드와 맞는지 확인한다.
   - ocr_pipeline에 stage_result, run_cascade, STAGES 가 있고 의미가 초안과 같은가
     (stage_result -> (prediction, evidence{q, M} 또는 None), run_cascade(run_stage) -> (prediction, method, attempts)).
   - 이름이나 반환값이 바뀌었으면 스크립트의 연결 부분만 고치고, 무엇을 고쳤는지 보고한다. 제출 코드는 고치지 않는다.
   - 최종 run_cascade가 새 method 이름을 쓰면 스크립트의 FINAL_STAGE 표에 추가한다.
     각 method의 최종 예측이 어느 단계에서 나오는지 코드로 확인한 뒤 추가한다.
   - pytest tests/test_review_routing_rates.py 통과
4. 원본 OCR 기록의 무결성을 확인한다(파일을 쓰지 않는 방법만 쓴다).
   - docs/full_stage_701_run1/raw_ocr.jsonl 이 2804줄이고, 701장 x 4단계가 중복 없이 모두 있다.
   - raw_ocr.jsonl 의 sha256 이 docs/full_stage_701_run1/integrity.json 의 raw_ocr_sha256 과 같다.
   - manifest.json 의 모델이 PP-OCRv6_medium_det + korean_PP-OCRv5_mobile_rec 이다.
5. 라벨 파일을 정한다.
   - 가장 최신 검수 결과가 반영된 701장 라벨 파일을 찾는다(초안은 tmp_labels_701.csv, approved 69장).
   - 필요한 열은 image_id, year, month, day, final_date, truth_source(approved/candidate), label_sources 이다.
   - 어떤 파일을 썼는지, approved가 몇 장인지 적는다. 초안보다 approved가 늘었으면 그 사실을 적는다.
6. OCR 모델/전처리/단계 정의가 bb83a74와 "다름"이면 여기서 멈추고 보고한다.
   이 경우 먼저 최종 설정으로 full-stage dump를 새로 만들어야 한다(scripts/dump_full_stage_701.py, 선우 PC).
7. 계산한다(OCR 없음).
   python -B scripts/review_routing_rates.py --full-stage-raw docs/full_stage_701_run1/raw_ocr.jsonl --labels <라벨 파일> --cascade production --output docs/review_routing_rates/final_<최종 commit 앞 7자리>
   비교용으로 같은 입력에 --cascade first_candidate 도 돌린다(출력 폴더 이름 끝에 _first_candidate).
8. 확인한다. 하나라도 어긋나면 멈추고 보고한다.
   - 네 분기 합계 + 재현 불가 = 701, 재현 불가 = 0
   - production 결과의 official 300장 정답 수가 최종 코드의 공식 300장 평가 결과와 같다
     (routing_rows.csv 의 group 이 아니라 official 300 목록 기준. 목록은 docs/parser_optimization1/old_parser_replay.json 의 cohorts.official_300).
   - 초안 대표값(36.5% / 3.4% / 늦은 날짜 8장)과 비교해서 바뀐 사진을 분기별로 세고, 왜 바뀌었는지 설명한다
     (Parser/cascade 변경 때문인지, 초안이 저장된 파싱 결과를 썼기 때문인지).
9. 문서를 갱신한다.
   - docs/review_routing_rates.md 를 최종 수치로 바꾼다. 제목의 "초안"을 "최종"으로 바꾸고,
     코드 기준, 입력(원본 OCR 경로, sha256), 라벨 파일, 결과 폴더 이름을 새 값으로 적는다.
   - 표 구조(2절 한눈에 보기, A~F, ROI 틀, 한계)는 그대로 유지한다.
   - C절의 늦은 날짜 사진 목록은 이미지로 라벨을 확인했는지 여부를 함께 적는다.
10. 테스트 전체를 돌리고(pytest), 제출 코드가 바뀌지 않았는지 git diff <최종 main commit> -- ocr_pipeline.py date_parser predict.ipynb requirements.txt 로 확인한다.
11. 커밋·push 전에 나에게 묻는다. 허락하면 claude/review-routing-rates-final 에만 커밋·push한다.

## 보고 형식
1. 사용한 최종 commit, 라벨 파일(approved 장수), 원본 OCR 무결성 확인 결과
2. 도구를 최종 코드에 맞추려고 고친 부분(없으면 없음)
3. 핵심 수치: 직원 검토 %, CONFIRM인데 틀림 %, 그중 늦은 날짜 %, 모두 95% CI 포함
4. 초안 대비 변화와 이유
5. 확인 항목(8번) 결과
6. 서현에게 넘길 3~5줄 요약: 보고서에 인용할 수 있는 문장, 수치와 조건(데이터 범위, 검수 여부, 코드 기준) 포함
7. 변경 파일 목록, 테스트 결과, 커밋 여부 질문
```

## 참고: 결과를 더 단단하게 만드는 선택 작업
- **라벨 확인**: CONFIRM인데 틀린 사진, 특히 늦은 날짜 사진을 이미지로 확인해 approved로 올린다. 그다음 위 절차를 다시 돌린다.
  보고서의 "지난 상품 통과 위험" 수치가 미검수 라벨 기준에서 검수 기준으로 바뀐다.
- **본선 데이터**: 본선 500장은 정답이 없어도 분기 비율(A)은 계산할 수 있다.
  OCR 원문을 저장해 두면 같은 스크립트로 계산할 수 있다(정답이 필요 없는 A만).
