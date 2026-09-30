# Selective NONE-only 1024px CLAHE validation

Source: `docs/ocr_clahe1024_full700_20260929_resumed1` (completed manual run).
All 700 image IDs and GT values match the accepted baseline. Fresh baseline
predictions match the accepted predictions on every image. Protected GT and
frozen raw OCR SHA-256 hashes match the run summary.

| image_id | GT | Baseline | Candidate | Outcome |
|---|---|---|---|---|
| 000050 | 2025-09-30 | NONE | 2025-09-30 | gain |
| 000994 | NONE-02-18 | NONE | NONE-02-18 | gain |
| 001979 | 2020-11-04 | NONE | 2020-11-04 | gain |
| 002766 | 2021-03-16 | NONE | 2021-03-16 | gain |

These are all changes: 4 gains, 0 regressions, 0 changed-wrong.
Score: 611/700 (87.29%) to 615/700 (87.86%), +0.57 percentage points.

The retry runs once only when the completed baseline returns exactly `NONE`.
It preserves full and partial dates, and retains the baseline result/method
if the retry has no candidate. Images are decoded once; the retry applies
CLAHE to the original RGB image resized to a maximum side of 1024px.

Calls: 1,147 to 1,207 (+60, +5.23%); 60/700 images retry (8.57%).
Recorded baseline stage time: 1,757.243 seconds. Retry stage time: 122.699
seconds (+6.98%), or 0.175 seconds per dataset image and 2.045 seconds per retry.
Total recorded stage time: 1,879.942 seconds. Resumed wall time: 1,483.086
seconds for 553 remaining images, with 147 reused images. This is not a
full-run wall-time comparison or a measured production parallel speed impact.

Verification uses recorded detections only: production parsing and cascade
match all 700 saved candidate results, methods, and attempted-stage sequences.
No additional OCR was executed and no GT/raw OCR was modified.
Relevant fake-engine, cascade, parser, notebook and runtime tests pass.
