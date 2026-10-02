# Phase 3 runtime optimization

**One production change accepted: an early stop that skips the 1024px re-read
when the original stage already holds the strongest class of evidence.**
Food stays at **620/700**, cosmetic development at **269/300**, with **zero
prediction changes on all 1,000 replayed images**. It removes 36 `highres_1024`
calls and 2 `clahe` calls.

The larger finding is that no code change was the main runtime lever: the
submission already runs **2 processes x 2 threads**, and the frequently quoted
`2.0420 sec/image` belongs to a **single-process** measurement. Correcting that
framing, plus a new measured calibration run, projects **500 images at roughly
845-980 seconds** on this PC depending on corpus mix.

Phase 3 did not restart parser work and did not open the cosmetic holdout.

## Starting baseline

Reproduced before any change, with `scripts/phase2_saved_evidence.py`:

| Corpus | Correct | original_512 | highres_1024 | rotation_270 | clahe | clahe_1024 | Total | Per image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Food frozen 700 | 620 | 700 | 302 | 120 | 92 | 59 | 1273 | 1.8186 |
| Cosmetic development 300 | 269 | 300 | 150 | 94 | 46 | 34 | 624 | 2.0800 |

Zero unevaluable cases in both corpora. Evidence:
[phase3_baseline_repro](../phase2_accuracy_20261001/phase3_baseline_repro.json).
The replay harness asserts both protected SHA256 hashes before and after every
run, so each replay in this phase also re-verified them.

`highres_1024` is the stage worth attacking. At the fresh-300 measured cost it
runs at 2.07 sec/call against 1.04 for `original_512`, and food requests it on
43% of images, cosmetic on 50%.

## Accepted change: conclusive-evidence early stop

`ocr_pipeline.py`, new `conclusive()` consulted at the top of
`retry_triggered()`. The cascade skips `highres_1024` when the `original_512`
reading is all of:

- from a box that explicitly labels itself as the expiration date (`self_anchor`),
- recognized at or above the existing `RETRY_Q_THRESHOLD` of 0.90,
- a complete date with no `NONE` field,
- not a damaged-day reading.

No new threshold was introduced; it reuses the frozen 0.90. No image ID or
expected answer appears in the logic.

### Why this generalizes

This is the strongest class of evidence an `original_512` stage can produce: the
package itself names the field, the recognizer is confident, and every field
parsed. A 1024px re-read cannot produce a stronger *class* of evidence, and the
only thing it can do to such a reading is replace it through `prefer_retry`,
which needs strictly higher confidence. Above 0.90 that headroom is small and
the risk is asymmetric: the retry can only move an already-well-evidenced answer.

The change is a restriction on *when to spend a second OCR call*, not a change to
how any reading is parsed or ranked. Every retry path remains intact for every
weaker reading: low confidence, damaged day, partial date, unlabelled date, and
the no-candidate fallback chain are untouched.

Its practical effect is to stop treating "another date is also printed on the
package" (`M`) as a reason to re-read a confidently labelled complete expiry.
That case supplies most of the savings and is exactly where re-reading was
least justified.

### Result

| Corpus | Correct | original_512 | highres_1024 | rotation_270 | clahe | clahe_1024 | Total | Per image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Food frozen 700 | **620** | 700 | 273 | 120 | 91 | 59 | 1243 | 1.7757 |
| Cosmetic development 300 | **269** | 300 | 143 | 94 | 45 | 34 | 616 | 2.0533 |

Changes versus baseline: **0 prediction changes, 0 regressions, 0 gains,
0 changed-wrong, 0 unevaluable** across 700 + 300 images. 36 images stop earlier:

| Attempts before | Attempts after | Food | Cosmetic |
|---|---|---:|---:|
| `original_512 highres_1024` | `original_512` | 28 | 6 |
| `original_512 highres_1024 clahe` | `original_512` | 1 | 1 |

Expensive calls removed: **36 `highres_1024`** (food 29, cosmetic 7) and
**2 `clahe`**. Evidence:
[phase3_conclusive_stop](../phase2_accuracy_20261001/phase3_conclusive_stop.json).

## Rejected experiments

All replayed against both full corpora with the real cascade code.

| Experiment | Food | Cosmetic | Outcome changes | Runtime gain | Decision |
|---|---:|---:|---|---:|---|
| Fallback order `rotation, clahe, highres` | 619 | 267 | food 1 regression; cosmetic 3 regressions, 1 gain | food 1.4%, cosmetic -0.4% | **Reject** |
| Fallback order `clahe, rotation, highres` | 619 | 267 | food 1 regression; cosmetic 3 regressions, 1 gain | food 2.1%, cosmetic 0.4% | **Reject** |
| Fallback order `clahe, highres, rotation` | 618 | 266 | food 2 regressions; cosmetic 4 regressions, 1 gain | food 2.2%, cosmetic 2.0% | **Reject** |
| Drop the short-source retry | 618 | 267 | 4 regressions | ~4% | **Reject** (prior phase-3 profile) |
| Skip `rotation_270` entirely | 616 | 266 | 4 regressions, 4 unevaluable | ~5% | **Reject** |
| Skip `clahe` entirely | 603 | 266 | 6 regressions, 18 unevaluable | ~4% | **Reject** |
| Memoize the duplicated 1024px resize | n/a | n/a | none expected | <0.005 sec/image | **Reject**, see below |

The reorderings all break the no-candidate fallback's "first stage with a
complete date wins" rule, so they trade accuracy for 0.4-2.2% runtime. Rejected
rather than tuned: the gain is inside measurement noise and each reordering
loses frozen-GT answers. Fallback-order trials:
[trials.json](trials.json). The last three rows reproduce the earlier
`docs/phase3_runtime_20261001/narrowed_profile.json` trials.

### The duplicated 1024px resize

`predict_image_detailed` computes `resize_image(rgb, 1024)` separately for
`highres_1024` and `clahe_1024`, so an image running both pays the LANCZOS
resize twice. This is real duplicated work, but it is worth about 0.025 sec on
the 8% of food images that reach `clahe_1024` — under 0.005 sec/image, below
1 second per 500 images. Caching the array would also let a mutation by one
stage reach the other, which saved-evidence replay cannot detect. Rejected: no
behavioural risk is worth a sub-second gain while the accepted change already
removed 36 OCR calls.

## Computation before and after

| Stage | Food before | Food after | Cosmetic before | Cosmetic after |
|---|---:|---:|---:|---:|
| original_512 | 700 | 700 | 300 | 300 |
| highres_1024 | 302 | **273** | 150 | **143** |
| rotation_270 | 120 | 120 | 94 | 94 |
| clahe | 92 | **91** | 46 | **45** |
| clahe_1024 | 59 | 59 | 34 | 34 |
| **Total** | 1273 | **1243** | 624 | **616** |
| **Calls/image** | 1.8186 | **1.7757** | 2.0800 | **2.0533** |

Against accepted main `0dd7a35`, whose food total was 1207 calls: the Phase-2
accuracy work cost +66 calls, and this change returns 30 of them, leaving
**+36 calls (+3.0%)** for +5 correct answers. Main's per-stage breakdown was not
re-derived here, so no runtime figure is claimed for main's own cascade.

## Runtime

### The framing correction

`predict.ipynb` calls `run_submission(INPUT_DIR, OUTPUT_PATH)` with no engine.
`run_submission` then reads `ITDA_OCR_PROCESSES`, **default `"2"`**, and takes
the `submission_runtime.run_workers` path with `processes=2, threads=2`. The
submission has always shipped the 2-process layout.

The widely quoted `2.0420 sec/image` / `612.6124 sec` for 300 images comes from
a single-process run whose own note says "Single process... Not the 2-process
submission layout". It is a real measurement of **old main `0dd7a35` at
1 process x 4 threads** and must not be read as submission throughput.

Repository evidence for the layout difference, 100 images, identical code and
sample, zero prediction changes: `1p4t` 0.8490 sec/image versus `2p2t`
0.5736 sec/image, a measured **1.48x**
(`docs/performance_optimization/{A_original,C_2p2t}/summary.json`). The same
set also shows MKLDNN is essential (3.7x slower without it) and that
`batch 12` and `2p1t` are both worse than `2p2t`, so the engine configuration
is already at its best measured setting. No engine or layout change is needed.

### Measured calibration of the final candidate

| Quantity | Value | Kind |
|---|---:|---|
| Images | 60 | MEASURED |
| Layout | 2 processes x 2 threads | MEASURED |
| Wall inference | 111.5709 sec | MEASURED |
| Throughput | **1.8595 sec/image** | MEASURED |
| Sum of per-image latency | 214.5024 sec | MEASURED |
| Parallel efficiency | **1.9226x** | MEASURED |
| original_512 / highres_1024 per call | 1.2642 / 2.9736 sec | MEASURED |
| rotation_270 / clahe / clahe_1024 per call | 1.0538 / 1.3568 / 2.8780 sec | MEASURED |
| Non-OCR per image | 0.2255 sec | MEASURED |
| Accuracy on those 60 | 54/60 | MEASURED, development data |

Per-call latencies are higher than the single-process run because two workers
share the cores; throughput is what improves. Evidence:
[calib60/summary.json](calib60/summary.json),
[calib60/manifest.json](calib60/manifest.json). A 6-image
[smoke6](smoke6/summary.json) run exists only as a harness check and is
warm-up dominated; it is not used for any projection.

### Projected 500-image runtime

PROJECTED, from the measured calibration above applied to the full replayed call
profiles. Not a measured 500-image runtime.

| Corpus profile | Calls/image | Projected sec/image | Projected 500 | vs 1770 | vs 1800 | vs 2500 |
|---|---:|---:|---:|---:|---:|---:|
| Food 700, before change | 1.8186 | 1.7550 | 877 sec | 2.02x | 2.05x | 2.85x |
| **Food 700, after** | 1.7757 | **1.6899** | **845 sec** | **2.09x** | **2.13x** | **2.96x** |
| Cosmetic 300, before change | 2.0800 | 1.9978 | 999 sec | 1.77x | 1.80x | 2.50x |
| **Cosmetic 300, after** | 2.0533 | **1.9593** | **980 sec** | **1.81x** | **1.84x** | **2.55x** |

The model reproduces its own calibration run to 0.000000 sec/image
([projection.json](projection.json), `self_check_error`).

Interpretation of the margins, under the assumption that a slower machine scales
runtime roughly proportionally: taking the **cosmetic-heavy 980 sec** as the
conservative case, the organizer PC could be about **1.84x slower** before
reaching the 1800-second internal target and about **2.55x slower** before the
2500-second formal limit. On the food profile those become 2.13x and 2.96x. The
historical 1770-second preliminary-round reference sits essentially on the
1800-second line, so it is covered by the same margin.

For reference only, the old-main single-process measurement of 2.0420 sec/image
extrapolates to about 1021 sec for 500 images, and the user-reported 3,352-image
figure of 1706.36 sec was a **split/chunked** execution, not one continuous run
(`docs/runtime_audit/REPORT.md`). Neither describes this candidate.

### Runtime risks

- The calibration is 60 food images. Per-call cost varies strongly with how much
  text an image carries, so a different mix moves the projection. The
  cosmetic row partly covers this by using the heavier call profile, but it
  still applies food-measured per-call costs.
- The projection assumes the measured 1.9226x parallel efficiency holds across
  500 images. Sustained runs could drift through thermal limits or memory
  growth; peak tree RSS near 1.8 GiB was observed historically in the
  2-worker layout.
- This PC has 8 logical processors. The historical layout experiments pinned to
  4 of them; the calibration run did not pin. A 4-vCPU organizer environment is
  not equivalent to 4 pinned cores here.
- The 500-image evaluation mix is unknown. Cosmetic-like images request more
  stages than food.
- Everything above is CPU-only, Windows, single interpreter; notebook start-up is
  excluded from all figures.

## Accuracy gate

| Gate | Required | Actual | Pass |
|---|---|---|---|
| Food frozen 700 | >= 620 | 620 | yes |
| Cosmetic development 300 | >= 269 | 269 | yes |
| New correct -> wrong regression | none | none | yes |

The inherited `001686` regression against main is unchanged and was not worked
around. `000749` and `000473` remain changed-wrong. No ID-specific logic exists.

### No additional OCR accuracy improvement is claimed

Phase 2 left zero errors classified as "parser missed usable evidence" or
"wrong candidate selection". Nothing in this phase's runtime analysis surfaced a
generalizable OCR or input-level gain: the five remaining stage-selection cases
would need stages the cascade does not request, and the broader retry policies
that would request them were already measured to regress. **620/700 is kept
deliberately** over a fragile or more expensive 621.

## Validation

- Relevant suite: **383 passed, 1 skipped** in 52.42 sec. The skip is the known
  local-data preflight unavailable in this worktree.
- 13 new tests in `tests/test_phase3_runtime.py` lock the early stop: the
  conclusive case, the `M` override, inclusive threshold behaviour, each
  non-conclusive reason, missing confidence, self-excluded readings, and
  cascade-level assertions that no further stage is requested while every
  weaker reading still retries.
- `git diff --check` clean.
- Both protected SHA256 hashes re-verified unchanged after all work:
  `9edfeb14...4cbfdc7c9d` and `3c00358a...313df502a2`.
- Cosmetic holdout `900301`-`900400` never opened. Those images are not present
  on this machine at all, and `scripts/phase3_fresh_benchmark.py` additionally
  refuses any ID in that range.
- `002133` remains excluded from food scoring.

Test command, from the candidate worktree:

```powershell
../.venv/Scripts/python.exe -m pytest tests/test_date_parser tests/test_date_parser_month_confusables.py tests/test_date_parser_frozen700_optimization.py tests/test_phase2_accuracy.py tests/test_phase3_runtime.py tests/test_ocr_pipeline.py tests/test_final_policy.py tests/test_submission_runtime.py tests/test_full_stage_701.py -q
```

## Files

Production code changed in this phase:

- `ocr_pipeline.py`: added `conclusive()` and one guard clause in
  `retry_triggered()`. Nothing else.

Carried from Phase 2, unchanged here: `date_parser/extract.py`,
`date_parser/select.py`, and the rest of `ocr_pipeline.py`.

Added in this phase:

- `tests/test_phase3_runtime.py` — 13 early-stop tests.
- `scripts/phase3_runtime_trials.py` — fallback-order trials, saved-evidence only.
- `scripts/phase3_fresh_benchmark.py` — fresh-OCR benchmark in the real 2-worker
  layout, with a holdout guard.
- `scripts/phase3_project_runtime.py` — measured-calibration projection.
- `docs/phase3_runtime_20261002/` — this report, `trials.json`,
  `projection.json`, `calib60/`, `smoke6/`.

A `weights` directory junction was created in the worktree pointing at the main
checkout's `weights/`, because `initialize_engine` resolves both the models and
the PaddleX cache relative to its own directory. `weights/` is gitignored, so it
does not appear in git status, and no production code was changed to accommodate
the worktree.

Candidate git status: `date_parser/extract.py`, `date_parser/select.py`,
`ocr_pipeline.py`, `tests/test_final_policy.py` modified; Phase 1/2/3 docs,
scripts and tests untracked. HEAD remains `3cd68fd` on
`candidate/final-optimization-20261001`; **everything is uncommitted**. Main and
origin/main remain `0dd7a35` with no tracked modifications. No staging, commit,
merge or push occurred.

## Required fresh benchmark

Everything above is saved-evidence replay plus a 60-image measurement. The
remaining step is a real 500-image fresh-OCR run of this candidate in the
submission layout.

A 500-image **cosmetic** benchmark is impossible: cosmetic images do not exist on
this machine, only their saved detections. The strongest valid alternative is 500
of the frozen-700 food images, which is what the command below runs. Because
cosmetic images request more stages, treat the food result as optimistic by
roughly the food-to-cosmetic ratio in the projection table (about 16%).

```powershell
cd C:\Users\jsw58\Desktop\itda-ocr\.candidate-final-20261001; ..\.venv\Scripts\python.exe scripts\phase3_fresh_benchmark.py --tag fresh500 --count 500 --processes 2 --threads 2
```

- Expected duration: roughly **14-18 minutes** (projected 845 sec plus engine
  initialization; the 60-image run took 112 sec).
- Output directory:
  `docs/phase3_runtime_20261002/fresh500/` containing `summary.json`,
  `manifest.json`, `predictions.jsonl`, `submission_benchmark.csv`.
- Return from `summary.json`: `infer_wall_sec`, `avg_sec_per_image`,
  `projected_500_sec`, `margin_to_1800`, `margin_to_2500`, `stage_calls`,
  `calls_per_image`, `stage_ocr_sec_mean`, `non_ocr_per_image`, `correct`,
  `accuracy`.

Two results need interpreting when it returns. `correct` is fresh-OCR accuracy on
500 development images and will not equal the saved-evidence 620/700 subset
exactly, because fresh OCR is not bit-identical to the frozen dump; a gap there
is an OCR-nondeterminism question, not an accuracy regression. And if measured
throughput lands far from 1.69 sec/image, the projection's calibration
assumptions above are what need revisiting.

This phase stops here. No commit, merge, push, or holdout use.
