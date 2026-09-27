# Parser Optimization 2 — verified final review

Accepted comparison: `9b14803cd42459d4bcf91104bcb26e4fb161a372` (Optimization 1), not the older 263-case diagnostic baseline.

**Result: 267/300 (89.00%), +3 gains / 0 regressions; four-stage oracle 274/300.** Approved 54→56/69, reused historical independent 192→192/195, provisional 580→583/701. Three of the six cases are reasonably recoverable with constrained general rules; all three are fixed. The other three remain unresolved. “Recoverable” means a defensible textual heuristic, not certainty about the image.

The working tree already contained the three parser edits, 43 focused test instances, probe artifacts and summary when this continuation began. Those bytes were preserved. This continuation independently reconstructed the accepted parser from git objects, reran all four probes and an additional rejected probe, verified the implementation against all 2,804 saved stages, completed tests, and added this review. It does not claim the pre-existing edits were newly written in this turn. Existing root summary/CSV files remain byte-identical; refreshed verification and the additional probe are under `reproducibility/`.

No OCR, image recognition, label edits, OCR/cascade edits, staging, commit, push, or policy adoption occurred. The immutable OCR tree, prior analysis tree, Optimization 1 artifacts, labels, and existing Optimization 2 files were hashed and verified unchanged.

## Architecture and rule scope

`parser.py` converts saved detections to positioned text boxes. `extract.py` runs narrow normalizations, then ordered regexes with claimed spans; rejected non-date readings do not consume a span. Four-digit-year triples already trim a third-field trailing digit. `interpret.py` performs field permutations, year bounds, calendar checks, and existing order scoring. `select.py` detects printed format hints, adds hint-specific candidates, uses yearless MM.DD only when no dated candidates exist, and ranks by expiry/manufacture context and geometry. OCR confidence and retry policy are separate offline signals.

The retained implementation changes only `date_parser/extract.py` and the hint-specific extraction call in `date_parser/select.py`:

1. Four-digit-year comma triple extraction (day length ≤3) before partial extraction. Reuses existing trailing noise/calendar behavior; requires an alphanumeric left boundary and excludes longer numeric runs. This does not globally replace commas or change numeric date order.
2. At the start of a box, split MM.DDHH시 only for a two-digit valid clock hour (00–23), inside the yearless fallback. No unit, invalid hour, lot prefix, or competing dated candidate means no new yearless interpretation.
3. With a printed month/year hint in the same stage, admit one alphabetic character in the separator position of a standalone English month abbreviation and intact four-digit year. Whole-box bounds allow adjacent punctuation/brackets, exclude lot prefixes/extra digits, and do not repair date digits.

Existing O/o/U/u→0, narrow C→0, split-year, compact-date and glued-date/time handling were inspected and left intact. No new I/1 or S/5 conversion, arbitrary numeric slicing, expiry-label fuzzy matching, cross-stage text merging, or broad DMY/YMD preference was added. Standard separated dates and month names remain handled by existing rules. The suffix rule is heuristic; calendar validity alone does not establish that a numeric code is a date.

## Six cases

Stages: O=original_512, R=rotation_270, H=highres_1024, C=clahe. Outputs below show accepted baseline → current implementation. Text is copied verbatim from the saved stage; “none” means no reliable date field in that stage. `six_cases.csv` additionally includes **all** OCR boxes for all four stages, avoiding omissions from the relevant-text excerpt. Label status is frozen metadata, not a new visual approval.

### 000763 — approved; truth `NONE-11-12`

**G (B+A); recoverable and fixed.**

| Stage | Relevant saved OCR / raw date-like substrings | Baseline → current |
|---|---|---|
| original_512 | `11. 120트시F로` | `NONE` → `NONE` |
| rotation_270 | (none) | `NONE` → `NONE` |
| highres_1024 | (none) | `NONE` → `NONE` |
| clahe | `11.1205시F2` | `NONE` → `NONE-11-12` |

The numeric right boundary rejects 1205 as a day. CLAHE preserves MM.DDHH시: 11.12 plus valid hour 05 and explicit Korean hour unit 시. Split only this anchored field in the yearless fallback. Original 11. 120트시F로 has a corrupted hour and is deliberately not repaired; rotation/highres omit the date. No year is invented.

### 000820 — candidate; truth `2026-05-18`

**G (A+B+C); not safely recoverable by the tested general rules.**

| Stage | Relevant saved OCR / raw date-like substrings | Baseline → current |
|---|---|---|
| original_512 | `SSS SI-18` | `NONE` → `NONE` |
| rotation_270 | `ESTeBEFERE18405-2026` | `NONE` → `NONE` |
| highres_1024 | `BEST`<br>`EEFORE`<br>`VARIOEE` | `NONE` → `NONE` |
| clahe | `2026` | `NONE` → `NONE` |

Rotation preserves 18405-2026, not a clean day/month/year token. Getting 18/05/2026 requires deleting the internal digit 4 or asserting that 4 was a separator. There is no text-supported reason to choose that repair over other corrupted-digit readings. The other stages do not preserve reliable day/month fields. The damaged BEST BEFORE label does not resolve the missing boundary.

### 002328 — candidate; truth `2021-10-20`

**G (B+E+F); not safely recoverable by the tested general rules.**

| Stage | Relevant saved OCR / raw date-like substrings | Baseline → current |
|---|---|---|
| original_512 | `2020 10.21부터` | `2020-10-21` → `2020-10-21` |
| rotation_270 | `2부` | `NONE` → `NONE` |
| highres_1024 | `유통기한`<br>`후면 표기일까지`<br>`200.10. 21부통`<br>`21 10 201` | `NONE` → `NONE` |
| clahe | `유통기한연기일까지`<br>`2020.10.21부터` | `2020-10-21` → `2020-10-21` |

Highres has 21 10 201. Dropping the final 1 gives 21 10 20, but both 2021-10-20 (YMD) and 2020-10-21 (DMY) are calendar-valid. A two-digit first field does not establish which field is a year or whether the third field has a suffix. Current cascade stops on the earlier manufacturing-like 2020.10.21부터 observation. Highres manufacturing text is itself corrupt. This is a plausible hypothesis, not defensible recovery without a boundary/order assumption; it is not simply missing all digits.

### 002899 — candidate; truth `2021-11-09`

**G (C+B); recoverable and fixed.**

| Stage | Relevant saved OCR / raw date-like substrings | Baseline → current |
|---|---|---|
| original_512 | `2021 11098` | `NONE` → `NONE` |
| rotation_270 | `2021 11098` | `NONE` → `NONE` |
| highres_1024 | `온021 11 098` | `NONE` → `NONE` |
| clahe | `2021,11,098` | `2021-11-NONE` → `2021-11-09` |

CLAHE retains a four-digit year, separate valid month, and three-digit day field. Commas are not accepted by the existing generic triple pattern, so a year/month partial wins. Admit the comma triple and reuse the already accepted four-digit-year trailing-noise rule (098 -> 09), with unchanged calendar/order interpretation. This is a bounded suffix-noise heuristic, not proof of the printed date. No internal digit repair, unrestricted numeric-token slicing, or inferred year is added. Earlier fused/corrupted stages remain unchanged.

### 001955 — approved; truth `2021-10-NONE`

**G (A+C+E); recoverable and fixed.**

| Stage | Relevant saved OCR / raw date-like substrings | Baseline → current |
|---|---|---|
| original_512 | `제품 법도표시원 Best belore월/년의 O1일끼지`<br>`-OCT.202` | `NONE` → `NONE` |
| rotation_270 | `OCTb2021` | `NONE` → `NONE` |
| highres_1024 | `제품 별도표시된 Best before:월/년의 01일까지`<br>`--OCTbOore:` | `NONE` → `NONE` |
| clahe | `제품 범도n시 Best belore월/년의 01일끼지`<br>`-OCTb2021` | `NONE` → `2021-10-NONE` |

Month-name extraction does not admit b in the separator slot. CLAHE preserves OCT and 2021 and independently reads 월/년. A standalone month-name plus one alphabetic separator plus four-digit year, gated on the same stage having a printed month/year hint, is sufficient. Rotation has intact fields but no format hint and stays NONE. Original truncates the year; highres corrupts it: neither is guessed. Generic 01일 instructions are not substituted for a missing stamped day.

### 003350 — candidate; truth `2025-12-10`

**G (A+B); not safely recoverable by the tested general rules.**

| Stage | Relevant saved OCR / raw date-like substrings | Baseline → current |
|---|---|---|
| original_512 | `소비김한년 월 읽 시간까지`<br>`25.2.18137`<br>`소비기한`<br>`한2025:1` | `NONE` → `NONE` |
| rotation_270 | (none) | `NONE` → `NONE` |
| highres_1024 | `소비김한(년 월 일 시간)까지`<br>`25.412.1015시` | `NONE` → `NONE` |
| clahe | `소비김한년 월 읽 시간까지`<br>`25.42.1515`<br>`소비기한`<br>`한2025:1` | `NONE` → `NONE` |

Highres retains a Y/M/D/hour instruction and a recoverable 10+15시 boundary, but the month is 412. Removing leading 4 to obtain 12 is arbitrary digit deletion; it is not a separator normalization. Original/CLAHE have different damaged month/day digits. The explicit format instruction settles order but cannot establish which month digits were corrupted. Hour splitting alone cannot recover the labeled full date.

## Offline probes

Each probe runs in an isolated process with the accepted git parser, never in the production parser. Truths are used only to score outputs. Original probe snapshots and current production outputs reproduce exactly. Counts below are correct/gains/regressions/net versus the accepted current cascade; cohort order is official 300, approved 69, reused independent 195, provisional 701.

| Probe | Official | Approved | Independent | Provisional |
|---|---|---|---|---|
| comma_year | 265/1/0/1 | 54/0/0/0 | 192/0/0/0 | 581/1/0/1 |
| yearless_hour | 265/1/0/1 | 55/1/0/1 | 192/0/0/0 | 581/1/0/1 |
| hint_month_separator | 265/1/0/1 | 55/1/0/1 | 192/0/0/0 | 581/1/0/1 |
| combined | 267/3/0/3 | 56/2/0/2 | 192/0/0/0 | 583/3/0/3 |
| rejected_two_digit_trailing_noise | 264/0/0/0 | 54/0/0/0 | 192/0/0/0 | 580/0/0/0 |

Accepted probes change final predictions only for 002899 (comma), 000763 (hour boundary), and 001955 (hinted separator); combined changes their union. None breaks a currently correct official, approved, independent, or provisional case. All three prediction changes are at CLAHE. There is also a signal-only change for 000015 at original_512: its final date stays 2026-03-01, but candidate diagnostics change (see changed_stages.csv). Exact per-cohort IDs and predictions are in the root `probe_changed_cases.csv`; full rerun including the rejected probe is in `reproducibility/probe_changed_cases.csv` and `probe_changed_stages.csv`.

The additional **two-digit trailing-noise** probe extends trimming a three-digit third field to triples with a two-digit first field, without changing order scoring. It is rejected: the first field no longer establishes the year, a trailing digit can belong to a code or damaged year, and 21 10 20 remains order-ambiguous. It changes no current-cascade output in any cohort. Stage changes are 000245/highres (wrong to wrong), 000503/CLAHE (already oracle-recoverable elsewhere), 002328/highres (new correct interpretation), plus signal-only changes for 002134/original and 002459/highres. It raises the official oracle 271 to 272 and provisional oracle 595 to 596 solely through 002328, with approved and independent oracles unchanged. A useful oracle interpretation is insufficient to justify choosing it. No such change was made to production. Exact changed outputs, including wrong→wrong changes, are in the rerun CSVs.

Rejected without implementation: delete internal 4 in 18405-2026; delete 4 from month 412; infer truncated 202/202N years; loosen whole-box bounds into batch/lot words; slice dates out of arbitrary longer digit runs. These lack a reliable textual boundary. No gain is counted for such hypotheses.

## Regression and policy interaction

The full parser suite plus offline replay/policy unit tests were run before the final implementation replay. See `reproducibility/tests.txt`. Focused tests cover comma dates, calendar-invalid fields, long numeric runs, lot prefixes, malformed separators, valid/invalid hour units, NONE, competing full dates, printed hints, brackets, truncated years, no digit repair, and unchanged ambiguous two-digit-year order.

| Policy (offline only) | Official /300 | Approved /69 | Independent /195 | Provisional /701 |
|---|---:|---:|---:|---:|
| current_cascade | 267 | 56 | 192 | 583 |
| frozen_B_highest_q | 271 | 54 | 193 | 584 |
| previous_best269 | 273 | 55 | 194 | 584 |
| agreement_no_anchor | 274 | 55 | 194 | 585 |
| agreement_all | 274 | 55 | 194 | 585 |

Frozen B + highest_q alone reaches **271/300**; the previous best policy reaches **273/300**, and agreement reaches the **274/300 oracle**. Relative to Optimization 1 with the same policies, each gains the same three official cases with no parser-induced official regression. Relative to the new current cascade, frozen B adds 4 official successes, previous best adds 6, agreement adds 7. Thus ≥270 no longer needs the aggressive agreement extension, but the parser itself remains at 267.

Policies are not adopted. Frozen B regresses two approved cases relative to current cascade (56→54); previous best/agreement have one gain and two approved regressions (56→55). Provisional policy results also include regressions (5 for frozen B, 7 for previous best/agreement versus accepted cascade). These are inherited policy tradeoffs, not hidden parser regressions. Exact IDs are in `reproducibility/policy_changes_vs_accepted_cascade.csv`.

No OCR ran and current-cascade stage attempts are **identical for every one of 701 images**. The gains occur in the already-attempted last stage, so there is no additional OCR runtime and no retry reduction under the unchanged cascade. Frozen B simulations add one rotation and one highres attempt on the 701 mix relative to its old-parser version; this is an offline projection, not work executed.

Official O/R/H/C attempts: current 300/46/39/27; frozen B 300/96/89/27; previous best 300/174/97/27; no-anchor agreement 300/240/103/27. Recorded OCR service-time projections per 500 images (official mix) are about 1053, 1415, 1628, 1780 seconds respectively. On the 701 mix: 1107, 1529, 2479, 2608 seconds. They include retained measured outliers and omit startup/decode/I/O; they are not an end-to-end runtime qualification. Parser diagnostic replay time is in verification.json and includes signal extraction.

## Remaining errors and decision

Of the 33 current-cascade official errors, seven are recoverable through saved-stage selection. Of the 26 still outside the oracle:

- 18 lack reliable date fields or contain unresolved digit corruption; improved OCR/preprocessing is needed for reliable automatic recovery from those observations: 000438, 001214, 002019, 002278, 002492, 002814, 003014, 003312, 003326, 000582, 000694, 000749, 000978, 001539, 001646, 002404, 002436, 003338.
- 3 mixed cases remain: 000820 and 003350 require unsupported internal digit repair; 002328 has an unresolved boundary/order/context hypothesis and needs better evidence, rather than being declared completely digit-missing: 000820, 002328, 003350.
- 2 retain date-order alternatives (002726, 002034): parser interpretation ambiguity, not proven OCR deficiency; do not change DMY/YMD defaults to fit labels: 002726, 002034.
- 3 require label/image review before blaming the parser or OCR; 000157 is a documented official/approved conflict, 001587 and 003218 are suspected conflicts or repeated OCR errors: 000157, 001587, 003218.

Decision: 3/6 reasonably recoverable and 3/6 safely fixed on this replay; official 264→267 (89.00%), +3/0/net +3; approved 54→56; independent 192→192; provisional 580→583; oracle 271→274. Parser-only exceeds 266. A simple existing frozen-B policy exceeds 270 (271), while the best combined offline result is 274. Additional OCR runtime is zero. The reused historical cohort and provisional labels are not fresh independent validation; no labels were changed.

## Reproduction and artifact inventory

Run from the repository root using the existing environment:

```powershell
.\.venv\Scripts\python.exe -B scripts/audit_parser_optimization2.py --output docs/parser_optimization2/reproducibility_new
```

The destination must be new. It reconstructs the accepted parser via read-only git show, verifies input hashes, reruns isolated probes, executes all parser and offline diagnostic tests, replays 2,804 immutable saved stages, checks current-cascade attempts, and asserts exact equivalence with preserved results. No OCR imports or calls are needed.

Root deliverables: `findings.md`, preserved `summary.json`, `six_cases.csv`, preserved `probe_comparison.csv`, `probe_changed_cases.csv`, `changed_cases.csv`, `policy_comparison.csv`, `probe_snapshots.json`, `new_parser_replay.json`, and `manifest.json`. New `decision.json` records the final answers. `reproducibility/` holds reconstructed accepted parser, verified probe/policy CSVs, complete stage changes, outside-oracle IDs, tests, protected-input hashes, initial git status/diff, and `verification.json`. Final status and hashes are recorded separately after report generation.

Parser diff versus accepted commit remains the pre-existing edits to `extract.py` and `select.py`; focused tests remain `tests/test_date_parser/test_boundary_normalization.py`. This continuation adds `scripts/audit_parser_optimization2.py` and `scripts/report_parser_optimization2.py`. All pre-existing diagnostic files are preserved, including the existing Optimization 2 directory. Nothing is staged or committed.
