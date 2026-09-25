# 최종 제출 안정화 검증 보고

검증일: 2026-09-14. 최종 모델: **PP-OCRv6_medium_det + korean_PP-OCRv5_mobile_rec**.
로컬 최종 후보의 새 clone/새 venv/오프라인 notebook Run All 검증을 완료했다.
원본 저장소 main merge 및 원격 push는 수행하지 않았다.

## 수정 파일 / diff

| 파일 | 변경 |
|---|---|
| `ocr_pipeline.py` | detector 이름 한 줄만 v6 medium으로 변경 |
| `download_weights.sh` | 실행 위치와 무관하게 Python 준비 스크립트 호출; bash 내장 기능 사용 |
| `scripts/download_weights.py` | 공식 모델 다운로드, SHA256 검증, 임시 준비 후 배치, 기존 파일 보존 |
| `scripts/weights_manifest.json` | 두 공식 URL, archive 및 6개 inference 파일 크기/SHA256 고정 |
| `.gitattributes` | shell 파일 LF 보장 |
| `requirements.txt` | 검증된 ipykernel 7.3.0, nbconvert 7.17.1 고정 |
| `tests/test_weight_distribution.py` | checksum 실패, 안전한 파일 추출, 재실행 및 기존 파일 보존 테스트 |
| `README.md` | 실제 모델 구조와 설치 → 다운로드 → 오프라인 Run All 명령 반영 |
| `docs/submission_final_report.md` | 이 보고서 |

`predict.ipynb`, `submission_runtime.py`, `date_parser/`는 변경하지 않았다.
strict candidate / cascade / decode-once / batch6 / MKLDNN / 2p2t를 유지했다.
기존 사용자의 `notebooks/validation_recheck_300.ipynb` 변경 및 실험 파일은 수정·포함하지 않았다.

## 배포 및 환경 감사

- PaddleX 3.7.2 `official_models.py`의 공식 `_BosModelHoster` BOS URL 사용.
- 기존 v6 archive 기록과 새 공식 recognizer archive로 SHA256을 고정했다.
  공식 배포자 서명이 아닌, 직접 받은 공식 파일의 고정 checksum이다.
- 새 clone에서 `bash download_weights.sh`로 두 모델을 실제 다운로드하고 6개 inference 파일을 검증했다.
- 준비 스크립트를 외부 접속 차단 상태에서 재실행: 기존 파일 검증만 수행, 다운로드 없음.
- 기존 작업 폴더 weight 변경: **0건**.
- 새 venv `pip install -r requirements.txt`: 성공. `pip check`: No broken requirements found.
- cv2 / numpy / PIL / paddle import: 성공 (4.10.0 / 2.2.6 / 12.3.0 / 3.2.2).
- 직접 추론의 외부 import는 numpy, PIL, cv2, paddleocr이며 requirements에 모두 명시됨.
  README CSV 검사기의 pandas는 PaddleX 의존성으로 설치됨 (새 venv 2.3.3).
  pytest는 운영 추론에 불필요하고 requirements-dev.txt에 있음.
- 제출 코드와 notebook, parser의 하드코딩 로컬 절대경로: **0건**.
- GPU/CUDA 전용 코드: **0건**. PaddleOCR `device="cpu"` 고정.
- 입력/출력은 상대경로, 모델 경로는 `__file__` 기준. notebook 내 설치/다운로드 호출 없음.
- shell 파일 LF로 fresh checkout됨을 확인.

## Fresh clone 및 실제 Run All

후보 Git commit: `69bf0ea777c1d4807bf3af88966f6733c978dc51` (임시 후보 저장소).
원본 HEAD를 임시 저장소로 clone하고 최종 제출 변경만 commit한 뒤 새 폴더에 clone했다.
검증 중 bash 호환 수정과 테스트 추가는 임시 후보에서 fast-forward했다.
원본 작업 폴더의 venv/weight는 복사하지 않았다. validation 입력 이미지만 300장 복사했고,
labels는 추론 종료 후 평가기에만 사용했다.
`include-system-site-packages=false`인 새 Python 3.10.11 venv를 생성해 requirements를 설치했다.

환경: Windows, 논리 CPU affinity [0, 1, 2, 3] (4개), CPU 추론 2p2t.
README와 동일한 명령:

```bash
python -m jupyter nbconvert --to notebook --execute predict.ipynb --ExecutePreprocessor.timeout=2400 --output executed.ipynb
```

| 검사 | 결과 |
|---|---|
| fresh clone / 새 venv 설치 | 성공 |
| offline Run All exit code | 0 |
| 전체 notebook wall clock | 862.03초 |
| submission.csv | 300행 생성 |
| schema / 입력 stem 집합 | 정상 / 일치 |
| 측정된 외부 Python 네트워크 시도 | 0건 |
| 관찰된 외부 연결 | 0건 |
| 잔존 하위 프로세스 / orphan | 0건 |
| pytest (원본 작업 환경) | 105 passed |
| validation final_date | 244/300 = 81.33% |
| 기존 후보와 4개 출력 필드 차이 | 0건 |

worker 종료 로그:

```text
[runtime] starting 2 workers x 2 threads; 300 images
[runtime] worker exits=[(21548, 0), (25268, 0)]; forced=[]; survivors=[]
[runtime] workers stopped; 300 results; 854.9s
[runtime] CSV saved; returning 300 rows; 855.0s
```

네트워크 검증은 도구 sandbox의 외부 접속 제한 아래 수행했다.
추가로 새 venv `sitecustomize.py`의 Python audit hook을 notebook launcher/kernel/worker에 상속해
외부 DNS/connect/sendto를 기록하고 차단했다. Jupyter를 위한 loopback은 허용했다.
실행 전 외부 연결 probe가 차단되는 것을 확인했으며 probe 기록은 추론 attempt 수와 분리했다.
하위 프로세스의 연결을 주기적으로 관찰했다. 물리 NIC 차단이나 독립적인 native packet capture는 수행하지 않았다.
audit hook은 임시 검증 venv에만 설치되며 제출 코드에는 들어가지 않는다.

## 남은 BLOCKER / 검증 범위

- Windows에서의 로컬 제출 재현에는 발견된 BLOCKER가 없다.
- **운영진 Linux Standard 4-Core 환경 재현은 미검증**: 이 PC에는 사용 가능한 WSL/Linux 및 Docker가 없다.
  Linux wheel 설치와 OpenCV 시스템 공유 라이브러리 검증은 해당 환경에서 남아 있다.
- **비공개 500장 / 2400초 실측은 미검증**: 공개 300장 결과로 대체할 수 없다.
- 원격 제출 브랜치 반영/merge/push는 미수행. 공개 URL의 현재 기본 브랜치를 검증한 것으로 해석하면 안 된다.
  실제 제출 시 이 보고서의 변경 파일 모두가 제출 commit에 포함되어야 한다.
- 전체 네이티브 통신의 packet-level 0건 증명은 이번 Python hook/샘플링 검증 범위 밖이다.

## 근거 파일

- [오프라인 결과 JSON](../outputs/submission_final/offline_report.json)
- [생성 CSV](../outputs/submission_final/submission.csv)
- [실행 notebook](../outputs/submission_final/executed.ipynb)
- [worker/진행 로그](../outputs/submission_final/notebook_stream.log)
- [실행 stderr 로그](../outputs/submission_final/offline_run.log)
- [정적 감사](../outputs/submission_final/static_audit.json)
- [설치 로그](../outputs/submission_final/fresh_install.log)
- [새 환경 freeze](../outputs/submission_final/fresh_freeze.txt)
- [weight 재검증](../outputs/submission_final/post_download_checks.log)
- [임시 clone 위치와 commit](../outputs/submission_final/fresh_location.json)

대용량 실험 산출물은 제출 runtime 의존성이 아니며 이 근거 파일은 로컬 검증 기록이다.
