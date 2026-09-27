# Final production policy: highres-only retry (Policy B)

Implemented in `ocr_pipeline.py` (`stage_result`, `retry_triggered`, `prefer_retry`,
`run_cascade`, `predict_image_detailed`; `predict_image` keeps its signature, so
`run_submission` / `predict.ipynb` use it unchanged). Parser, models, thresholds
and output format are unchanged.

- Original 512 has a date candidate: B = q < 0.90 OR M. B false -> original, no
  extra OCR. B true -> `highres_1024` exactly once; take highres only if it has a
  candidate and a strictly higher q (ties keep original).
- Original 512 has no candidate: unchanged fallback rotation 270 -> highres 1024 ->
  CLAHE, first candidate wins, else original (`original_no_candidate`).
- q: recognition confidence of the box the selected date came from (minimum over
  boxes with the same text and center). M: at least two distinct
  `final_date_string()` values over `find_all_candidates`. Same definitions as
  `scripts/diagnose_correct_controls_50.py::evidence`. A candidate whose source box
  cannot be found (not observed in 2,804 saved stages) gets q=None: it retries and
  can never win the comparison.
- New `method` values: `original_512_retry_kept`, `highres_1024_retry`.
  `original_512` now means no retry was triggered.

## Saved-OCR replay of the production code

`scripts/replay_final_policy.py` drives `ocr_pipeline.run_cascade` itself.

- `--snapshots` (run here): stage parser results saved with the accepted parser
  (`docs/parser_optimization2/new_parser_replay.json`). Output:
  `docs/final_policy_production_replay/`.
- `--raw` (run on the PC holding `docs/full_stage_701_run1/`): parses each saved
  stage with production `stage_result` and reports any stage whose prediction/q/M
  differs from the snapshots (`stage_mismatches_vs_saved_snapshots`, expected empty).

| cohort | before | after | gains | regressions | O/R/H/C before -> after |
|---|---|---|---|---|---|
| official 300 | 267 | 270 | 000266, 002066, 003311 | none | 300/46/39/27 -> 300/46/89/27 |
| approved 69 | 56 | 56 | none | none | 69/16/14/12 -> 69/16/25/12 |
| independent 195 | 192 | 193 | 002066 | none | 195/0/0/0 -> 195/0/37/0 |
| provisional 701 | 583 | 586 | 000214, 000266, 000386, 002066, 003311 | 001768, 003245 | 701/123/112/82 -> 701/123/252/82 |

No `NONE-NONE-NONE` output; every all-NONE prediction is `NONE`.
Image IDs above are replay outcomes only; production code contains no IDs.

## Real 300-image evaluation (not run here)

```powershell
.\.venv\Scripts\python.exe -B .\scripts\evaluate_final_policy_300.py --run --labels .\labels\labels_300.csv --images .\data --output .\docs\final_policy_production_experiment1
```

Single process, 4 engine threads (as the accepted 267 evaluation). Records exact-date
count, gains/regressions against the replayed 267 baseline, methods, per-stage attempt
counts and OCR time, init/inference/wall time, model names and weight hashes, source
hashes and git HEAD, and `NONE-NONE-NONE` outputs. Not the 2-process submission
layout; end-to-end notebook timing still needs a separate run.
