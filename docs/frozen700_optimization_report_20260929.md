# Frozen 700 offline optimization - 2026-09-29

Final: **611/700 = 87.2857%**, from **602/700 = 86.00%**: **9 gains, 0 regressions, 0 changed-wrong**. All 700 baseline predictions and methods exactly reproduced the accepted monthfix replay. Remote main was verified at `71529a3`. No fresh OCR was run. This is in-sample frozen-GT optimization, not an independent generalization estimate.

Only `labels/review/labels_700_confirmed_gt.csv` was used for scoring. Image `002133` remains excluded. Historical provisional labels were not used. All three protected inputs have identical before/after SHA-256 hashes (recorded in the audit JSON).

## Accepted rules and exact prediction changes

Each row is a separate full-700 replay against the preceding accepted rule. Every outcome below is a gain; GT equals the new prediction. Methods are literal production method names.

| Rule | Score old -> new | ID | Old prediction | New prediction = GT | Method old -> new |
|---|---|---|---|---|---|
| Whole-box split day | 602 -> 603 (+1) | 001932 | 2022-01-02 | 2022-01-22 | original_512 -> original_512 |
| Prevent clock-minute date contamination | 603 -> 604 (+1) | 001940 | 2033-03-14 | 2021-03-14 | original_512_retry_kept -> original_512_retry_kept |
| Whitespace inside exact month name | 604 -> 605 (+1) | 001000 | NONE | 2021-09-07 | original_no_candidate -> highres_1024 |
| Standalone DDMM YYYY | 605 -> 606 (+1) | 003011 | NONE | 2021-12-11 | original_no_candidate -> original_512 |
| Preserve expiry when retry repeats earlier discarded date | 606 -> 607 (+1) | 001768 | 2020-12-30 | 2021-06-29 | highres_1024_retry -> original_512_retry_kept |
| Nearby expiry-labelled YYYYMM fallback | 607 -> 608 (+1) | 000777 | NONE | 2026-03-NONE | original_no_candidate -> highres_1024 |
| Transfer unambiguous slash DMY evidence | 608 -> 609 (+1) | 001548 | 2029-09-20 | 2022-09-28 | highres_1024_retry -> highres_1024_retry |
| Join aligned, nearby separate day | 609 -> 610 (+1) | 002589 | 2020-10-NONE | 2023-10-15 | original_512_retry_kept -> original_512_retry_kept |
| Preserve expiry when retry completes earlier discarded year/month | 610 -> 611 (+1) | 002588 | 2020-10-15 | 2023-10-15 | highres_1024_retry -> original_512_retry_kept |

## Scope and risk checks

- Split day: four-digit year, numeric separators, whole OCR box only; do not merge a following time or lot. Pattern found in 3 stage detections / 1 image across the full dump.
- Clock: valid HH:MM boundaries only; exclude alphanumeric prefixes and date separators. Two clock-followed-date detections / 1 image; only one candidate stage changes. The initial mask also consumed `BE0:24/11/21`, increasing calls by 2 with no prediction regression. That version was rejected and narrowed; the negative test preserves this case.
- Split month: accept only exact JAN-DEC spelling with internal whitespace, bounded by day and year. A broad split-letter search found 11 detections / 10 images, mostly nutrition/code text; exact month/date constraints change only `07SE P21` (1 stage / 1 image). No new character substitutions.
- Standalone compact day/month plus year: broad 4+4 search found 17 detections / 6 images, including copyright ranges and telephone numbers. Whole-box whitespace-only separation plus calendar validation changes only 3 stages / 1 image; existing EXP compact handling is retained.
- Compact YYYYMM: full-dump standalone search found 2 detections / 2 images (`202603`, `201002`). Year validation, no existing date candidate, and a standalone expiry label within 3 text heights on the same line admit only 1. A distant keyword is insufficient.
- Slash order: require calendar-valid DD/MM/YYYY with day > 12 in the same stage, no conflicting explicit YMD/MDY, and transfer only to matching slash triples. Co-occurrence found in 1 stage / 1 image. A dotted date or ambiguous reference does not qualify.
- Separate day: only explicit YYYY.MM, YYYY/MM or YYYY-MM at the end of a box; right-side gap <= one height, vertical offset <= 0.4 height, comparable height, unique neighbor, valid calendar and day-box confidence >= date-box confidence. Four candidate stages / 2 images change; source confidence therefore remains conservative.
- Retry: no additional stage requests. Earlier complete-date repetition occurs in 4 relevant original/highres pairs; one newly changes a prediction, one was already protected by Policy C, and two have lower highres confidence. Partial-year/month extension affects one additional pair and never treats a same-month day correction as a repeated earlier date. Original manufacture sources and highres expiry anchors are exceptions.

## Rejected experiments

The broader clock mask scored 603 -> 604 with the same `001940` gain shown above, but introduced 2 unnecessary stage calls for `001290`; only the narrowed mask is retained.

English expiry wording alone is not a date-order hint. The broad English-DMY experiment scored 611 -> 613 (+4 gains, -2 regressions), so it was rejected. Narrowing by selected vocabulary/country without explicit order evidence would merely fit this sample. Unconditional highres preference (retaining Policy C) scored 611 -> 600 (+1 gain, -12 regressions, 1 changed-wrong); rejected. Exact changes follow.

| Experiment | ID | GT | Old | New | Outcome | Method old -> new |
|---|---|---|---|---|---|---|
| rejected_english_dmy | 001406 | 2021-07-31 | 2021-07-31 | 2031-07-21 | regression | original_512 -> original_512 |
| rejected_english_dmy | 001862 | 2023-06-18 | 2018-06-23 | 2023-06-18 | gain | original_512 -> original_512 |
| rejected_english_dmy | 002037 | 2020-08-22 | 2020-08-22 | 2022-08-20 | regression | original_512 -> original_512 |
| rejected_english_dmy | 002552 | 2022-01-21 | 2021-01-22 | 2022-01-21 | gain | original_512 -> original_512 |
| rejected_english_dmy | 002604 | 2023-11-17 | 2022-03-22 | 2023-11-17 | gain | original_512_retry_kept -> original_512_retry_kept |
| rejected_english_dmy | 002917 | 2021-04-30 | 2030-04-21 | 2021-04-30 | gain | original_512 -> original_512 |
| rejected_highres | 000118 | 2026-03-23 | 2026-03-23 | 2026-01-23 | regression | original_512_retry_kept -> highres_1024_retry |
| rejected_highres | 000423 | 2026-06-04 | 2025-06-05 | 2026-06-NONE | changed-wrong | original_512_retry_kept -> highres_1024_retry |
| rejected_highres | 000580 | 2026-02-28 | 2026-02-28 | 2028-02-28 | regression | original_512_retry_kept -> highres_1024_retry |
| rejected_highres | 000853 | 2026-09-24 | 2026-09-24 | 2025-09-24 | regression | original_512_retry_kept -> highres_1024_retry |
| rejected_highres | 000954 | 2021-02-05 | 2021-02-05 | 2021-01-07 | regression | original_512_retry_kept -> highres_1024_retry |
| rejected_highres | 001185 | 2021-09-28 | 2021-09-28 | 2023-12-02 | regression | original_512_retry_kept -> highres_1024_retry |
| rejected_highres | 001556 | 2022-03-16 | 2022-03-16 | NONE-03-16 | regression | original_512_retry_kept -> highres_1024_retry |
| rejected_highres | 001768 | 2021-06-29 | 2021-06-29 | 2020-12-30 | regression | original_512_retry_kept -> highres_1024_retry |
| rejected_highres | 001806 | 2020-11-09 | 2020-11-09 | 2029-02-10 | regression | original_512_retry_kept -> highres_1024_retry |
| rejected_highres | 001947 | 2022-08-10 | 2022-08-10 | 2022-06-NONE | regression | original_512_retry_kept -> highres_1024_retry |
| rejected_highres | 002284 | 2022-01-06 | 2022-01-06 | 2021-01-07 | regression | original_512_retry_kept -> highres_1024_retry |
| rejected_highres | 002336 | 2021-11-19 | 2021-11-18 | 2021-11-19 | gain | original_512_retry_kept -> highres_1024_retry |
| rejected_highres | 002588 | 2023-10-15 | 2023-10-15 | 2020-10-15 | regression | original_512_retry_kept -> highres_1024_retry |
| rejected_highres | 002883 | 2023-12-08 | 2023-12-08 | 2023-12-NONE | regression | original_512_retry_kept -> highres_1024_retry |

## Remaining 89 errors

Baseline decomposition: 63 NONE + 35 wrong dates; 12 errors had a correct stage-selected result somewhere, and 13 had a correct top candidate somewhere. Final: **60 NONE + 29 wrong dates** (25 full, 4 partial); 10 have a correct stage-selected result. These oracle counts are diagnostics, not an executable policy.

| Disjoint final category | Count |
|---|---:|
| NONE: no coherent target-date evidence found | 32 |
| Wrong date: required GT digits missing/corrupted | 13 |
| NONE: corrupted/incomplete/ambiguous date-like text | 28 |
| GT selected in another saved stage; cascade/confidence limitation | 10 |
| Date-order ambiguity; GT only an alternative reading | 6 |

All four saved stages were inspected for the NONE cases. The absent/coherent-evidence split is a conservative manual text audit, not proof of what is visible in the image. Missing dates cannot be reconstructed from GT. Date-like corruptions include `2026.19.15`, `22.91-18`, `2105 202`, `BEST BY ML 2022`, `BBE MOU 29 2021`, and `202N06/01`: repairing these needs unsupported digit insertion/substitution or ambiguous order. Bare six-digit codes and `Exp: 09 1122` remain ambiguous; do not invent a day/year or assume locale.
The 10 stage-selection cases are `000231, 000422, 000423, 000726, 001455, 001515, 001964, 002148, 002336, 002917`. Most correct stages are not requested by production; consulting them would add OCR calls. The remaining confidence disagreement cannot safely be resolved by favoring highres, as the rejected experiment demonstrates.
The 6 alternative-order cases (`001862, 001975, 002034, 002552, 002604, 002726`) lack sufficiently explicit same-stage format evidence. `002604` additionally has a date-shaped lot number; selecting its EXP token alone still leaves the year/day ambiguity. The other 13 wrong-date cases require missing or misrecognized GT digits, not merely re-ranking an existing correct candidate.
No further clearly safe rule was found in this pass. New OCR/model/input work may recover absent or corrupted evidence, but would require separate approval. None was launched.

## Validation, runtime and reproduction

` .venv/Scripts/python.exe -m pytest tests/test_date_parser tests/test_date_parser_month_confusables.py tests/test_date_parser_frozen700_optimization.py tests/test_final_policy.py -q `: **231 passed**. The new file adds 11 focused tests covering positives, boundaries, invalid dates, mixed format references, geometry, confidence, ambiguous neighbors and real production stage evidence. Historical failing tests/artifacts were not changed.
Final replay: `.venv/Scripts/python.exe scripts/optimize_frozen700_offline.py docs/<new-directory> docs/frozen700_optimization_20260929_baseline`. The replay pins frozen GT/raw hashes, rejects duplicate/incomplete stages, excludes 002133, blocks OCR/preprocessing imports and verifies protected hashes afterward. Production `stage_result` and `run_cascade` are executed on saved actual detections.
Simulated production stage calls: original 700 -> 700; rotation 121 -> 120; highres 250 -> 249; CLAHE 81 -> 78. Total **1152 -> 1147 (-5)**. No image requires an additional stage in the accepted result. Parser-only text/geometry checks add small CPU work; no end-to-end inference timing claim is made.
Final local replay: `docs/frozen700_optimization_20260929_final/`. Each experiment has its own new directory. Large per-image diagnostics remain untracked. The committed [audit JSON](frozen700_optimization_audit_20260929.json) preserves every candidate score/change, protected hashes and all 89 remaining IDs/GT/predictions/categories.

Exact intended files: `date_parser/extract.py`, `date_parser/select.py`, `ocr_pipeline.py`, `tests/test_date_parser_frozen700_optimization.py`, `scripts/optimize_frozen700_offline.py`, this report and `docs/frozen700_optimization_audit_20260929.json`. No protected input or historical artifact was modified.
