# Portable FTZ/DAZ for the Ubuntu 22.04 grader — 2026-10-02

Official grading environment (organizer specification): Ubuntu 22.04 LTS x86_64,
Python 3.10, 4 CPU cores, 8 GB, no GPU, no internet during inference, fresh venv
from `requirements.txt`, `jupyter nbconvert --execute predict.ipynb`, 500 unseen
images, hard limit 2500 s.

**Status: implemented and tested on Windows, then validated end to end on a local
Ubuntu 22.04 WSL install** (see [ubuntu_validation.md](../ubuntu_validation.md);
local WSL timing, not organizer hardware). The static analysis below predates
that run.

## What the pinned Linux wheel actually uses (STATICALLY INSPECTED)

Source: `paddlepaddle-3.2.2-cp310-cp310-manylinux1_x86_64.whl` from PyPI, and
`libc6_2.35-0ubuntu3_amd64.deb` from the Ubuntu archive. Hashes, DT_NEEDED lists
and symbol versions are in [static_evidence.json](static_evidence.json).

| Library | OpenMP dependency |
|---|---|
| `libdnnl.so.3` (oneDNN, the convolutions) | NEEDED system `libgomp.so.1` (not bundled); imports `GOMP_parallel@GOMP_4.0` |
| `libpaddle.so`, `libphi_core.so` (Paddle ops) | NEEDED bundled `libiomp5.so`; bind `GOMP_parallel` to it |
| `libiomp5.so` (Intel OMP 5.0.20190109) | defines `GOMP_parallel@GOMP_4.0` (GNU-compatible) |

- On glibc 2.35 Paddle skips its `libgomp` preload (glibc < 2.23 only), so
  `libpaddle.so` is the dlopen root. `libiomp5.so` is its direct dependency,
  while `libgomp.so.1` is only a second-level dependency through `libdnnl`. In
  glibc's breadth-first local scope, oneDNN's `GOMP_parallel@GOMP_4.0` is
  therefore expected to resolve to libiomp5. **Expected, not observed.** The
  probe below records glibc's own binding trace on Ubuntu.
- libiomp5 propagates FP control. Disassembly shows the master storing
  `fnstcw` and `stmxcsr & 0xffffffc0` into the team (offsets 0x242/0x244) at every
  fork, and each worker running `ldmxcsr` when its MXCSR differs. This is gated on
  `KMP_INHERIT_FP_CONTROL`, which is on by default. Only the exception flags are
  dropped, so FTZ (bit 15) and DAZ (bit 6) carry over to the workers.
- With GNU libgomp (the fallback case), there is no per-region copy. A new
  Linux thread starts with its creator's FPU/SSE state (kernel `fpu_clone`).
  So the master must be switched before the first team is created.
- glibc 2.35 `fegetmode` stores MXCSR at `femode_t+4`. `fesetmode` keeps the current
  exception flags (`& 0x3f`) and loads every other bit from the caller
  (`& 0xffffffc0`), including FTZ/DAZ. It's a public ISO/IEC TS 18661-1 API
  (glibc 2.25+), and `femode_t` is from the public `<bits/fenv.h>`.
- No environment variable, Paddle flag or oneDNN API sets FTZ/DAZ. The wheel's
  `ScopedFlushDenormal` exists, but on Windows it didn't cover the convolution
  path (1-thread A/B: 356 s → 112 s with the master switched).

## Mechanism chosen (IMPLEMENTED)

`ocr_pipeline._flush_linux()` sets MXCSR FTZ|DAZ on the calling thread with
glibc `fegetmode`/`fesetmode` through ctypes. No compiled artifact, and no
OpenMP internals in the production path. `initialize_engine` calls it on
non-Windows **before** `cv2`/`paddleocr` are imported and before any OpenMP team
exists. That covers both libiomp5 (per-region copy, independent of order) and
libgomp (inherit-on-create).

Fail-safe guards, each returning 0 and leaving MXCSR untouched:
- the machine isn't `x86_64`,
- `/proc/cpuinfo` is unreadable or has no `avx` flag (Paddle requires AVX, and
  every AVX CPU implements DAZ, so `ldmxcsr` can't fault),
- `libm.so.6` or `fegetmode`/`fesetmode` is missing,
- `fegetmode` fails or the value has reserved high bits set (wrong layout),
- `fesetmode` fails, or the read-back doesn't show both bits.

Windows is unchanged: the same `_controlfp` + `_vcomp_fork` body, called once,
after engine construction.

## Results (TESTED ON WINDOWS)

- **Unit tests:** `tests/test_phase3_runtime.py` has 37 passing tests. The Linux
  branch is driven through a fake libm and a fake cpuinfo, and every unsupported
  or failing case returns 0. Call order is checked on both platforms, and the
  engine still starts when FTZ is unavailable. The full suite gives 422 passed,
  1 skipped, and 5 failures in `test_evaluation_common.py`. Those 5 are
  environmental: that test reads `labels/labels_300.csv` and `data/`, which are
  absent from this worktree, and it doesn't import `ocr_pipeline`.
- **Replay** (`phase3_linux_ftz_replay.json`): food 620/700 and cosmetic
  development 269/300. All 1,000 rows are identical to `phase3_ftz_replay.json`,
  with 0 correct→wrong. No fresh OCR, and no holdout IDs.
- **Windows fresh OCR, 100-image parity set, [0,1,2,3]:** 161.7 s versus 164.3 s
  before the change. 0 prediction and 0 attempt differences, and affinity was
  [0,1,2,3] on 100/100 images.
- **Windows fresh OCR, 500 images, [0,1,2,3]** (`fresh500_4cpu_ftz_portable`):
  808.7 s, 438/500. 0 prediction and 0 attempt differences versus the
  721.54 s run. Every image had affinity [0,1,2,3].
- **Why 808.7 s, not 721.54 s:** the laptop was on battery for this run
  (Win32_Battery status 1, 81 %). Every 100-image block is uniformly 8–13 %
  slower (first-call means 1.21–1.71 s versus 1.11–1.52 s). Without FTZ those
  means are 3.4–5.7 s, so FTZ was active. Re-measure on AC power if this number
  is needed.

## Ubuntu 22.04 WSL validation (2026-10-02)

Run with `bash scripts/ubuntu_validate.sh IMAGE_DIR`; full facts in
[ubuntu_validation.md](../ubuntu_validation.md). Local WSL, not organizer hardware.

- FTZ/DAZ verified active.
- FTZ on vs off: 0/40 prediction differences.
- Organizer-style `nbconvert` run: affinity `[0,1,2,3]`, network isolated,
  500 images → 500-row CSV, 442.7 s, peak RSS 3.53 GB, no notebook errors.

Not recorded in this note: which runtime oneDNN's `GOMP_parallel` binds to, and
the size of the denormal slowdown on Ubuntu with FTZ off.
