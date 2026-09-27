# Parser optimization 1

The parser-only production cascade improves **263 -> 264/300 (88.00%)**, with one gain (`002586`) and **zero regressions**. The official four-stage oracle improves **270 -> 271**. One of the three clear opportunities is fixed; the other two lack defensible date-order evidence in the saved text. Parser-only accuracy does not reach 266 or 90%.

## Root causes established before editing

All four saved OCR stages, boxes, source texts, candidates, hints and keyword evidence for these cases are preserved in `root_cause_evidence.json`. Approved means the immutable dump's approved metadata, not a new visual review. Official truth is never replaced by candidate/approved metadata.

### 002726 — provisional; truth 2022-03-17

Original/highres/CLAHE contain `L0076 17.03.22`; rotation separates `L0076` and `17.03.22`. No stage has a recognized expiry/manufacture keyword or date-order hint. Original/highres/CLAHE return `NONE`. The generic numeric matcher initially consumes `0076 17.03` from the lot-prefixed box. It has no valid interpretation, but non-overlapping regex iteration misses the overlapping date start. Rotation produces a candidate from `17.03.22` with YMD `2017-03-22` ranked ahead of DMY `2022-03-17`; the cascade stops there.

This has both a token-boundary problem and an unresolved order ambiguity. Extracting the date after the lot code alone would not produce the correct final answer. A lot code is not evidence of DMY, and changing the default order to fix the label would be unjustified. No change made for this case.

### 002034 — approved; truth 2020-12-17

All four stages contain `17.12.20`, with source candidate alternatives `2017-12-20` and `2020-12-17`. All select the former under the existing YMD prior. Nearby `L352MW` and `12:14` are lot/time evidence, not a date-order instruction. There is no recognized expiry/manufacture keyword and no format hint. Both dates are valid; neither the text nor a manufacture constraint resolves the ambiguity. No change made. In particular, neither brand recognition nor a hand-written DMY exception is introduced.

### 002586 — approved; truth 2022-12-NONE

Every stage reads the same complete box: `Exp.Date: 12/22`. Other numeric text is supplement nutrition or `Batch No: 652332`; there is no manufacture date. EXP supplies local expiry context but no explicit MM/YY hint. The old parser extracts no candidate and returns exactly `NONE`; its short month/year helper runs only when a printed month/year hint exists.

The new extraction rule represents this standalone expiry-labelled slash pair as fixed month/year. Existing candidate validation resolves 22 to 2022 and validates month 12; the normal selection path returns `2022-12-NONE`. All four stages now have the correct candidate; the unchanged production cascade stops on original rather than attempting all four stages.

## Rule and risk

Only `date_parser/extract.py` changes: an 11-line pattern entry in the existing extraction table. It accepts a complete OCR box consisting of EXP/EXPIRY/EXPIRATION, optional DATE and punctuation, followed by a one/two-digit month, slash, and two-digit year. It uses the existing fixed-role month/year representation and calendar/year-range validation. No new postprocessing layer, image ID, product exception, truth lookup, current-date heuristic, OCR operation, or public-interface change is involved.

The rule applies to textual variants such as `EXP 05/27`, `Expiry Date: 9 / 28`, and `expiration: 03/26`, not just the diagnostic text. Bare pairs, dot-separated month/day, manufacture/batch fields, longer dates, code suffixes, invalid months and out-of-range years are excluded by the rule/tests. The expiry label and value must occupy the same box; no global context is propagated into unrelated boxes.

Expected pre-evaluation gain was one case, not three. The principal remaining semantic risk is that an otherwise identical expiry-labelled slash pair could mean month/day with omitted year; MM/YY here is a constrained format convention, not a mathematical disambiguation. Partial-date candidates may also interact with competing dates. Whole-box anchoring limits exposure, but one observed approved success does not establish universal safety. In this dump, only four stage observations of one image change; all other 2,800 observations are identical, including diagnostic signals.

The all-NONE rule is preserved: year/month/day all NONE yields exactly `final_date="NONE"`; a known month/year yields `YYYY-MM-NONE`.

## Parser-only comparison

| Cohort | Old current cascade | New current cascade | Gains | Regressions |
|---|---:|---:|---:|---:|
| Official 300 | 263 | 264 | 1 | 0 |
| Approved 69 | 53 | 54 | 1 | 0 |
| Historical independent 195, reused | 192 | 192 | 0 | 0 |
| All 701 provisional/noisy | 579 | 580 | 1 | 0 |

The sole gain is `002586` in each cohort containing it. There are no parser-induced regressions or other prediction changes. The historical independent 195 contain no affected example, so their unchanged score is not positive validation of the new pattern. Cohorts overlap; the three appearances of the gain are not three independent successes. Among 701 labels, 632 are not fully verified and 606 are single-team labels per handoff.

| Four-stage oracle | Old | New | Added | Lost |
|---|---:|---:|---|---|
| Official 300 | 270 | 271 | 002586 | none |
| Approved 69 | 54 | 55 | 002586 | none |
| Historical 195 | 194 | 194 | none | none |
| Provisional 701 | 594 | 595 | 002586 | none |

## Policy interactions, not policy adoption

The trigger/selector implementations are unchanged and use the new parser's signals. Current cascade is the actual first-candidate policy. Frozen B retains its original q < .90 OR M definition and no-candidate cascade fallback. Previous best is B-or-ambiguity / rotation then conditional highres / highest-q. Agreement variants retain their prior definitions, including conservative higher-q, complete, singleton, unambiguous retry agreement. No new trigger or threshold was tuned.

| Official policy | Old parser | New parser | Gains / regressions vs old 263 | Gains / regressions vs new 264 |
|---|---:|---:|---|---|
| Current cascade | 263 | 264 | 1 / 0 | 0 / 0 |
| Frozen B + highest-q | 267 | 268 | 5 / 0 | 4 / 0 |
| Previous best 269 policy | 269 | 270 | 7 / 0 | 6 / 0 |
| No-anchor agreement extension | 270 | 271 | 8 / 0 | 7 / 0 |
| Unrestricted agreement extension | 270 | 271 | 8 / 0 | 7 / 0 |

Thus an existing policy plus the new parser reaches exactly 90% offline; agreement reaches 90.33%. Neither result is a production approval. `policy_comparison.csv` includes all cohort counts, gains/regressions relative to both parser baselines and the old same policy, changed attempt counts, and projected costs. `changed_cases.csv` lists every old/new parser change under every policy/cohort.

On approved 69, frozen B gives 52, and previous-best/agreement give 53, versus the new current-cascade 54. Their regressions are `000824` and `002310`; these existed before this parser change. Historical 195 scores are 193 for B and 194 for previous-best/agreement. Provisional 701 scores are 581 for B/previous-best and 582 for agreement. The parser adds one correct answer to each applicable old policy without adding a regression.

For completeness, provisional-policy regressions versus the old/new current cascade are:

- Frozen B: `000071`, `000824`, `001768`, `002310`, `003245`.
- Previous-best and both agreement variants: `000071`, `000697`, `000824`, `001768`, `002310`, `002734`, `003245`.

All official policy regression lists are empty. These comparisons are development evidence on reused data, not an independent validation of new policy selection.

## Runtime

No OCR ran. The new parser-only cascade adds **zero OCR attempts** and eliminates one rotation, one highres and one CLAHE attempt on `002586`. Official stage counts change from 300/47/40/28 to 300/46/39/27. On the full 701, counts change from 701/124/113/83 to 701/123/112/82; no image gains an attempt.

Saved-call accounting decreases by 9.743 seconds in cohorts containing the gain. Official-mix projected 500-image OCR service cost becomes 1,052.8 seconds, and 701-mix becomes 1,106.6 seconds. These are selected saved elapsed-call durations, not target-machine end-to-end benchmarks; outliers are retained. Diagnostic whole-parser replay with additional signal extraction took roughly 3.5 seconds both before and after on this machine; that single observation is not a precise production CPU-overhead measurement.

Combined policies remain runtime-sensitive. Previous-best projects 2,478.9 OCR seconds from the 701 mix, leaving effectively no overhead margin. No-anchor agreement projects 2,608.2 seconds and unrestricted agreement 2,747.4, both exceeding the limit before overhead. Diagnostic dump wall time is not used. No combined policy is certified for Ubuntu 22.04, 4 CPU cores, 8 GB RAM and the 2,500-second end-to-end limit.

## Decision and validation

Consider the **parser-only change a production candidate**: small textual rule, one approved gain, no observed regressions, preserved interface/NONE behavior, and fewer OCR attempts. This is not a claim of 266/300, parser-only 90%, or independently validated universal format inference. Leave the two unresolved order cases unchanged until actual textual/contextual evidence supports them. Do not adopt a trigger policy automatically.

Before editing, all 134 existing parser tests passed. After editing, **156 parser tests passed**, including 22 new focused tests; no existing expectation changed. The previous 11 offline diagnostic tests also pass. No OCR, label modifications, immutable-dump writes, staging or commits occurred. Prior uncommitted diagnostics are preserved.

## Reproducibility

`scripts/replay_parser_optimization.py --capture --output NEW_DIRECTORY` captures the unchanged parser after the original read-only preflight and exact 300 prediction/method gate. `--compare --output SAME_DIRECTORY` compares the parser candidate against that frozen snapshot after tests pass. The capture must run with the old parser (0aefa14); comparison must run with the candidate. Do not recapture a candidate and call it the production baseline.

The existing historical analyzer deliberately rejects parser hash drift; it was not weakened or overwritten. The new comparison tool validates pinned raw data, source metadata, old replay, and unchanged OCR/cascade/runtime/notebook hashes, then permits only the explicitly intended parser comparison. Input/code hashes and old/new replay snapshots are stored in this directory. The source dump was rehashed after replay and remains canonical.
