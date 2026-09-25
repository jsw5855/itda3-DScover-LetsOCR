# 소비기한 추출 OCR

최종 모델: **PP-OCRv6_medium_det + korean_PP-OCRv5_mobile_rec**.
CPU 전용이며 strict candidate, cascade, decode-once, recognition batch 6,
MKLDNN, 2 processes × 2 threads를 유지합니다.

## 운영진 재현 순서 (Python 3.10, 저장소 루트)

### 1. 인터넷 연결 상태에서 설치

```bash
git clone https://github.com/jsw5855/itda3-DScover-LetsOCR.git
cd itda3-DScover-LetsOCR
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3"
python -c "import cv2, numpy, PIL, paddle; print(cv2.__version__, numpy.__version__, PIL.__version__, paddle.__version__)"
bash download_weights.sh
python scripts/download_weights.py --verify-only
```

Windows에서는 Python 3.10의 `python -m venv .venv` 및
`.venv\Scripts\Activate.ps1`로 환경을 준비합니다. bash는 Git Bash를 사용합니다.
Git Bash에서 활성 venv Python을 찾지 못하면 `PYTHON` 환경변수로 해당 실행 파일을 지정합니다.
이후 명령은 같은 venv에서 실행합니다.

`download_weights.sh`는 표준 라이브러리만 사용하는 `scripts/download_weights.py`를 호출합니다.
PaddleX 3.7.2 공식 BOS 배포 URL에서 두 모델을 받고, archive SHA256 및 파일별 크기/SHA256을
`scripts/weights_manifest.json`과 비교한 다음 임시 디렉터리에서 최종 경로로 옮깁니다.
checksum은 공식 서버에서 받은 파일을 고정한 값이며 배포자 서명은 아닙니다.
기존 모델은 검증 후 그대로 사용하고, 불완전하거나 변조되었으면 덮어쓰지 않고 실패합니다.
기존 v5 detector와 기타 weight 파일은 수정하지 않습니다.

```text
weights/paddleocr/
  PP-OCRv6_medium_det/
    inference.json
    inference.pdiparams
    inference.yml
  korean_PP-OCRv5_mobile_rec/
    inference.json
    inference.pdiparams
    inference.yml
```

PaddleX가 요구하는 OpenCV 배포 하나만 설치합니다. Linux에서 `import cv2`가 시스템 공유
라이브러리 오류로 실패하면 해당 실행 이미지에 필요한 라이브러리를 먼저 준비해야 합니다.
Python 의존성은 `pip check`와 위 import 명령으로 확인합니다.

### 2. 인터넷 연결 해제 후 Run All

운영진 입력 이미지를 저장소 루트의 `val_images/`에 배치합니다.
기본 입출력은 상대경로 `./val_images`, `./submission.csv`이므로 아래를 그대로 실행합니다.

```bash
python -m jupyter nbconvert --to notebook --execute predict.ipynb --ExecutePreprocessor.timeout=2400 --output executed.ipynb
```

다른 경로는 실행 전에 `ITDA_INPUT_DIR`, `ITDA_OUTPUT_PATH` 환경변수로 지정할 수 있습니다.
notebook CONFIG 셀은 이 환경변수를 읽습니다. 설치·weight 다운로드를 수행하는 셀은 없습니다.
필수 로컬 모델 파일이 없으면 OCR 초기화 전에 오류로 종료합니다.
코드에서 `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`를 설정하고 보조 모델을 비활성화합니다.
이 설정 자체가 방화벽은 아니므로 운영진의 네트워크 차단 후 실행합니다.

입력 폴더 바로 아래 jpg/jpeg/png를 처리하며 확장자를 뺀 파일명(앞자리 0 포함)이
`image_id`입니다. 빈 입력 또는 중복 stem은 오류입니다. 출력은 UTF-8 BOM CSV이며
입력 이미지당 한 행, 다음 컬럼 순서입니다.

```text
image_id,year,month,day,final_date
```

```bash
python scripts/validate_submission.py submission.csv --image-dir val_images
```

worker 오류·시간 초과는 실패 처리하며 부분 결과 CSV를 성공으로 저장하지 않습니다.
개별 작업의 worker 무응답 제한은 180초이며, 내부 전체 worker timeout은 7200초입니다.
전체 2400초 제한 충족 여부는 실행 하드웨어와 입력으로 검증해야 합니다.

## 제출 파일과 검증

필수 파일: `predict.ipynb`, `ocr_pipeline.py`, `submission_runtime.py`, `date_parser/`,
`requirements.txt`, `download_weights.sh`, `scripts/download_weights.py`,
`scripts/weights_manifest.json`, `README.md`. `weights/`는 Git에서 제외되며 위 단계로 생성합니다.
제출 추론은 `data/`, `labels/`, 개발 notebook, 실험 스크립트에 의존하지 않습니다.
개발 테스트는 `python -m pip install -r requirements-dev.txt` 후 `python -m pytest -q`입니다.

기존 300장 validation 최종 후보는 244/300 (81.33%), v5 기준은 230/300 (76.67%)입니다.
비공개 test 500장의 정확도와 실행시간은 공개 validation 결과로 보장할 수 없습니다.
최종 fresh clone 및 오프라인 검증 결과·환경·한계는 `docs/submission_final_report.md`에 기록합니다.
