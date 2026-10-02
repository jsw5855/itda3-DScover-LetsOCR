# 제출 전 재현 절차 실행 로그 (2026-10-02)

본선 안내의 "제출 전 새 가상환경에서 수행" 절차를 그대로 실행한 기록입니다.

```
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash download_weights.sh
ITDA_INPUT_DIR=./sample ITDA_OUTPUT_PATH=/tmp/out.csv jupyter nbconvert --to notebook --execute predict.ipynb --output /tmp/executed.ipynb
```

| 항목 | 값 |
|---|---|
| 실행 커밋 | `4f30f76a5ce651ddcaaf9a6c88259888c4478a4d` (이 로그를 추가한 커밋의 부모, 코드 동일) |
| 환경 | 새로 설치한 Ubuntu 22.04.5 LTS (WSL 2), Python 3.10.12, RAM 7.6GiB |
| 저장소 | `git clone` 직후 새 `.venv` |
| weights | `download_weights.sh`로 공식 URL에서 다운로드, 크기·SHA256 검증 통과 |
| 입력 | `./sample` 10장 (화장품 개발용 사진 900001~900010) |
| 결과 | 노트북 오류 없음, 10행 CSV, 48.6초 |

- `run_log.txt`: 위 절차 전체의 터미널 출력
- `executed.ipynb`: `/tmp/executed.ipynb` 사본
- `out.csv`: `/tmp/out.csv` 사본

이 실행은 재현 절차 확인용이며 CPU 개수를 제한하지 않았습니다.
CPU 4개 제한·네트워크 차단 조건의 500장 실행 시간(442.7초)은
`docs/phase3_runtime_20261002/ubuntu_validation.md`에 있습니다.
