#!/usr/bin/env bash
# Ubuntu 22.04 x86_64 check of the organizer flow and the Linux FTZ/DAZ path.
#
# Usage, from the repository root with internet available for the install step:
#   bash scripts/ubuntu_validate.sh IMAGE_DIR
# IMAGE_DIR holds jpg/jpeg/png images, ideally 500, never the cosmetic holdout.
# Everything is written under a new /tmp/itda_ubuntu.* directory; /tmp/out.csv and
# /tmp/executed.ipynb are the organizer command's own outputs.
set -euo pipefail
IMAGES="$(cd -- "$1" && pwd)"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d /tmp/itda_ubuntu.XXXXXX)"
exec > >(tee "$WORK/log.txt") 2>&1
cd "$ROOT"

. /etc/os-release
echo "[env] $PRETTY_NAME | $(uname -m) | kernel $(uname -r) | nproc $(nproc)"
grep -m1 'model name' /proc/cpuinfo
echo "[env] work dir $WORK"

python3.10 -m venv "$WORK/venv"
. "$WORK/venv/bin/activate"
python -m pip install -r requirements.txt
python -m pip check
python -c "import cv2, numpy, PIL, paddle; print(cv2.__version__, numpy.__version__, PIL.__version__, paddle.__version__)"
bash download_weights.sh
python scripts/download_weights.py --verify-only

python scripts/linux_ftz_probe.py "$IMAGES" --count 40 --cpus 4 --out "$WORK/probe" || PROBE=FAIL

# The organizer command. Network removed when an unprivileged namespace is
# available; 4 CPUs when the machine has more.
RUN=()
if [ "$(nproc)" -gt 4 ]; then RUN+=(taskset -c "$(python -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:4])))')"); fi
if unshare -rn sh -c 'ip link set lo up' 2>/dev/null; then
    RUN+=(unshare -rn sh -c 'ip link set lo up && exec "$@"' sh)
    echo "[run] network: isolated (loopback only)"
else
    echo "[run] network: NOT isolated (no unprivileged network namespace); disconnect manually for a strict run"
fi
rm -f /tmp/out.csv /tmp/executed.ipynb
START=$(date +%s.%N)
ITDA_INPUT_DIR="$IMAGES" ITDA_OUTPUT_PATH=/tmp/out.csv "${RUN[@]}" \
    jupyter nbconvert --to notebook --execute predict.ipynb --output /tmp/executed.ipynb
END=$(date +%s.%N)

python - "$IMAGES" "$START" "$END" "$WORK" <<'EOF'
import csv, json, sys
from pathlib import Path
images, start, end, work = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
count = sum(p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'} for p in images.iterdir())
with open('/tmp/out.csv', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
wall = end - start
summary = dict(images=count, rows=len(rows), columns=list(rows[0]) if rows else [],
               nbconvert_wall_sec=round(wall, 2), projected_500_sec=round(wall * 500 / count, 1),
               limit_sec=2500)
print('[run]', json.dumps(summary))
(work / 'run_summary.json').write_text(json.dumps(summary, indent=2))
assert len(rows) == count and summary['columns'] == ['image_id', 'year', 'month', 'day', 'final_date']
EOF
echo "[done] probe: ${PROBE:-PASS} | log $WORK/log.txt | probe report $WORK/probe/report.json"
