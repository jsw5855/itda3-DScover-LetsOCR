# Phase 2 accuracy decision

**Freeze the current uncommitted accuracy candidate for runtime assessment.**
Saved-evidence replay scores **620/700 food (88.57%)** and **269/300 cosmetic
development (89.67%)**. No new regression against the Phase 1 candidate.
These are development-set results, not independent generalization estimates.

| Comparison | Food gains | Food regressions | Food changed-wrong | Net correct |
|---|---:|---:|---:|---:|
| Phase 1 `3cd68fd`, 618/700 | 2 | 0 | 1 | +2 |
| Accepted main `0dd7a35`, 615/700 | 6 | 1 | 2 | +5 |

Cosmetic development improves 265 -> 269: four gains, zero regressions,
zero changed-wrong. Against main's saved official 244, it has 25 gains,
zero regressions and the inherited changed-wrong 900206 (NONE-10-14 -> NONE;
GT 2029-08-13). The original food fresh-300 score of 270/300 still describes
accepted main; no fresh-300 score is claimed for this uncommitted candidate.

## Every additional prediction change

| Dataset / ID | GT | Phase 1 | Final | Outcome |
|---|---|---|---|---|
| Food 000473 | 2026-05-21 | 2026-03-01 | 2026-05-02 | changed-wrong |
| Food 000726 | 2026-09-16 | 2026-08-16 | 2026-09-16 | gain |
| Food 002241 | 2021-05-26 | NONE | 2021-05-26 | gain |
| Cosmetic 900071 | 2029-05-17 | 2029-05-NONE | 2029-05-17 | gain |
| Cosmetic 900181 | 2028-02-09 | 2028-02-NONE | 2028-02-09 | gain |
| Cosmetic 900225 | 2029-08-02 | 2029-08-NONE | 2029-08-02 | gain |
| Cosmetic 900285 | 2028-09-04 | NONE | 2028-09-04 | gain |

All original Phase 1 gains (000231, 002148, 002336, 002917) remain correct.
001686 remains a regression against main; 000749 remains changed-wrong.
Full changes against both baselines: [decision_summary.json](decision_summary.json).

## Accepted general rules

Each isolated trial below uses unchanged Phase 1 as its reference, except the
last row, which was tested after combining the first five. Calls are simulated
production OCR-stage requests, not fresh OCR executed in this task.

| Rule | Gains | Regressions | Changed-wrong | Food / cosmetic score | Call delta food / cosmetic |
|---|---|---|---|---|---|
| Four-digit year + exact English month + day, with optional separators | 002241 | none | none | 619 / 265 | 0 / 0 |
| Explicit four-digit year with mixed comma/period separators | 900071 | none | 000473 | 618 / 266 | 0 / 0 |
| Preserve a complete date when higher-confidence retry merely loses compatible fields | 900225 | none | none | 618 / 266 | 0 / 0 |
| Retry a YMD reading with an unparseable day; distinguish from intentional year/month labels | 900181 | none | none | 618 / 266 | +1 / +2 |
| Recognize exact Korean 까 and 지 separated by whitespace or hyphen as an expiry suffix | 900285 | none | none | 618 / 266 | 0 / -2 |
| Uncertain self-labelled expiry + empty highres: try CLAHE, accept only a higher-confidence self-labelled expiry | 000726 | none | none | 620 / 269 | +5 / +1 versus combined first five |

Why these rules generalize:

- `2021MAY26` contains every required field with unambiguous order. The rule
  handles all twelve exact month names and calendar-valid years/days without
  repairing letters or inserting digits. Its decisive detection is the completed
  supplemental raw run, line 867, `clahe_1024`; that stage was already requested.
- Mixed comma/period handling is consistent with existing all-comma and
  all-period dates, retains field length/calendar validation, and does not
  parse thousands-separated values as dates. Food 000473 now reads the literal
  truncated `2026,05.2`; no missing digit is invented to match its GT.
- A compatible partial retry has less information, not evidence contradicting
  the complete date. Conflicting year/month readings retain existing confidence
  selection; this is not a blanket preference for complete dates.
- A three-field numeric token that degraded to year/month is evidence of a
  damaged day. Ordinary month-only expiry labels do not trigger this new retry.
  Confidence and manufacture/expiry protections still govern replacement.
- Splitting the exact two-syllable expiry word is a punctuation/whitespace
  normalization, not a new fuzzy spelling rule or a barcode interpretation.
- The final cascade rule extends an existing retry only when highres produced
  no candidate. It requires expiry evidence on both original and replacement
  and strictly higher replacement confidence. No thresholds were fitted.
  Food 000726 uses immutable raw lines 621, 623 and 624; no supplemental reading
  replaces these detections. Five food images and one cosmetic image request
  the additional CLAHE stage; only 000726 changes prediction.

## Rejected alternatives

These were actually replayed against Phase 1 unless otherwise stated.

| Trial | Food / cosmetic score | Gains | Regressions | Changed-wrong | Calls food / cosmetic | Decision |
|---|---|---|---|---|---|---|
| Disable manufacture-period calculation | 619 / 265 | 001686 | none | none | +3 / 0 | Reject: gain depends on later OCR losing manufacture label |
| Return printed manufacture base when a period is stated | 619 / 265 | 001686 | none | none | 0 / 0 | Reject: relabels manufacture as expiry to fit GT |
| Disable six-digit fallback | 618 / 263 | none | 900028, 900198 | 000749 -> NONE | +1 / +3 | Reject: loses valid anchored compact dates, does not fix food GT |
| Retry every partial date | 618 / 266 | 900181 | none | none | +17 / +6 | Replace with damaged-day rule: same gain with fewer requests |
| Retry every bare numeric date regardless of source length | 617 / 266 | 001515, 900181 | 000912, 002581 | none | +177 / +20 | Reject: unjustified regressions and many extra requests |
| CLAHE after every uncertain original's empty highres retry | 619 / 265 | 000726 | none | 000553 -> 2028-01-24 | +17 / +5 | Reject broad version; require explicit expiry evidence in retained version |

Every trial's scores, exact old/new predictions, outcomes, and stage-call
counts are in [variant_audit.json](variant_audit.json).

### 001686 and 000749

001686 contains `제조2020.11.27` and `제조일로부터 5년까지`. The candidate's
2025-11-27 follows its manufacture-plus-period semantics. Frozen GT remains
2020-11-27. No general expiry-semantic rule was found that legitimately turns
this into the manufacture date. The apparent gain from disabling arithmetic
occurs because CLAHE reads `제2020.11.27`, dropping the manufacture keyword;
it is not a reliable semantic correction. Both workarounds were rejected.

000749 contains `202602 까지스31` and no saved stage contains a defensible
2026-01-21 reading. The current partial February answer is wrong. Blocking
six-digit dates does not recover January 21 and loses two correct cosmetic
cases. No digit substitutions or special suffix blacklist were introduced.

## All remaining food errors

All 82 Phase 1 errors were reviewed across the four immutable stages and every
available saved supplemental stage. Two are now fixed, leaving 80.

| Primary category | Phase 1 | Final |
|---|---:|---:|
| Parser missed usable evidence | 1 | 0 |
| Wrong candidate selection | 0 | 0 |
| Partial date | 12 | 12 |
| Corrupted OCR | 34 | 34 |
| No usable OCR evidence | 20 | 20 |
| Ambiguous order/context | 8 | 8 |
| Retry/stage-selection issue | 6 | 5 |
| Other: semantics / frozen-target mismatch | 1 | 1 |
| Total | 82 | 80 |

These are conservative, mutually exclusive primary diagnoses, not proof that
the image itself lacks a date. Partial/corrupted/missing evidence can overlap
mechanistically. Classification starts with the existing manual food audit,
then updates it for all currently saved stages; adjustments and explanations
are recorded in [classification_review.csv](classification_review.csv).
[food_error_audit.csv](food_error_audit.csv) lists all 82 IDs, GT, initial/final
predictions, primary category, exact-stage and candidate-alternative oracle
availability, and whether the final cascade already requested that stage.
These annotations are analysis outputs, never inputs to production decisions.

The five remaining stage-selection cases are 000422, 000423, 001455, 001515,
001964. Their exact readings exist in saved stages the final cascade does not
request, but no further low-risk trigger was established. In particular, the
broader bare-date retry already demonstrated regressions. 001686 separately has
a matching stage, but it loses its manufacture label and remains classified
as a semantic mismatch.

The eight ambiguous cases are 001714, 001862, 001975, 002034, 002068, 002552,
002604, 002726. Some have all target digits or the correct interpretation as
an alternative, but lack decisive order/context evidence. 002604 also selects
a date-like lot code; selecting EXP alone still does not establish the correct
year/day order, so its primary category is ambiguity rather than selection.
Country/brand guesses, arbitrary digit repairs, and threshold sweeps were not
used. No further clearly generalizable improvement was established in this pass.

## OCR-call impact; no runtime work

| Corpus | Original | Highres | Rotation | CLAHE | CLAHE-1024 | Total | Per image |
|---|---:|---:|---:|---:|---:|---:|---:|
| Food Phase 1 | 700 | 301 | 120 | 87 | 59 | 1267 | 1.8100 |
| Food final | 700 | 302 | 120 | 92 | 59 | 1273 | 1.8186 |
| Cosmetic Phase 1 | 300 | 148 | 94 | 46 | 35 | 623 | 2.0767 |
| Cosmetic final | 300 | 150 | 94 | 46 | 34 | 624 | 2.0800 |

Net versus Phase 1: food +6 calls, cosmetic +1; total +7 across 1,000 images.
Food versus accepted main: 1207 -> 1273 (+66). Cosmetic versus main: 610 -> 624
(+14). No fresh OCR calls occurred. Runtime impact is not measured or
extrapolated in Phase 2; additional request costs and parser overhead belong
to the next phase. No runtime optimization was performed.

## Validation and reproducibility

- Original Phase 1 replay is reproduced exactly for all food predictions,
  methods and requested stages, and cosmetic score 265.
- Final replay: 620 food and 269 cosmetic; **zero unevaluable cases**.
- Immutable detections always take precedence. Only missing CLAHE-1024 data
  comes from the completed full-700 saved raw run. Both protected hashes match
  before and after; the supplemental hash also remains unchanged.
- Cosmetic membership is asserted to be exactly 900001-900300. Holdout
  900301-900400 was not opened or used. 002133 remains excluded from food scoring.
- Relevant tests: **370 passed, 1 skipped** in 53.11 seconds. The skip is the
  known local-data preflight unavailable in the isolated worktree. Tests use
  fake/injected engines. `git diff --check` passes.
- One existing fake-engine test needed a third scripted response to exercise
  the new contrast retry. Its assertion still ensures manufacture-only text
  cannot replace the original expiry. The initial fixture exhaustion was
  corrected and the full relevant scope rerun successfully.

Test command from candidate worktree:

```powershell
../.venv/Scripts/python.exe -m pytest tests/test_date_parser tests/test_date_parser_month_confusables.py tests/test_date_parser_frozen700_optimization.py tests/test_phase2_accuracy.py tests/test_ocr_pipeline.py tests/test_final_policy.py tests/test_submission_runtime.py tests/test_full_stage_701.py -q
```

Replay with unused output names:

```powershell
../.venv/Scripts/python.exe scripts/phase2_saved_evidence.py final_new
../.venv/Scripts/python.exe scripts/phase2_saved_evidence.py baseline_new baseline --phase1-code
../.venv/Scripts/python.exe scripts/phase2_saved_evidence.py trial_new year_month_word --phase1-code
```

`--phase1-code` creates a read-only-in-purpose snapshot of just parser/pipeline
Python files using `git show 3cd68fd`; it neither checks out a branch nor copies
datasets. Experimental monkeypatches require this flag to avoid applying rules
twice to the final code. `combined_reproduced` applies all six accepted variants
to that pinned snapshot as a cross-check of the production implementation.

[final_verified.json](final_verified.json) and
[baseline_verified.json](baseline_verified.json) contain full diagnostics;
their `.manifest.json` companions pin every parser/pipeline source hash and
all replay input hashes. No old-parser predictions or expected answers are
substituted for detections. The replay blocks OCR-engine imports.

## Exact changes and stop point

Production changes relative to 3cd68fd:

- `date_parser/extract.py`: explicit-year month-name dates and mixed separators.
- `date_parser/select.py`: whitespace/hyphen within the exact Korean expiry suffix.
- `ocr_pipeline.py`: damaged-day evidence/retry, compatible complete-date
  preservation, and anchored CLAHE fallback after empty highres.

Validation/analysis changes:

- `tests/test_phase2_accuracy.py`: 28 syntax, boundary and cascade tests.
- `tests/test_final_policy.py`: adapt the manufacture-rejection fixture to the
  additional legitimate retry while preserving the semantic assertion.
- `scripts/phase2_saved_evidence.py`: guarded food/development replay and trials.
- `docs/phase2_accuracy_20261001/`: report, variant evidence, code/input manifests,
  classification audit, and pinned Phase 1 parser/pipeline snapshot.

Phase 1 script/artifacts were preserved. Unrelated files were preserved.
Candidate HEAD remains `3cd68fd` on `candidate/final-optimization-20261001`;
all Phase 2 changes are uncommitted. Main and origin/main remain unchanged at
`0dd7a35`. No staging, commit, merge or push occurred.

Recommendation: freeze this accuracy candidate and proceed to runtime
assessment when requested, retaining explicit visibility of the inherited
001686 regression. Do not claim production acceptance or untouched-holdout
validation yet. This task stops after Phase 2.
