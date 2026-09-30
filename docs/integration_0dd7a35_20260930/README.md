# Work branch + main `0dd7a35` integration check (2026-09-30)

Merge of GitHub main `0dd7a35` (NONE-only 1024px CLAHE retry) into the work branch
(`305efb3`: Parser rules 1-11, retry T3 · C2s, fallback C6; already contains main `5d77cf8`).
No merge conflict. One main test was adapted: a 7-character partial date (`2026.04`) is a
short fragment, so the work branch's T3 re-reads it at 1024px; the test now uses a labelled
partial date and lets C6 read the later fallback stages. Its purpose (a partial date never
triggers `clahe_1024`) is unchanged. `pytest --ignore=tests/test_evaluation_common.py`:
345 passed, 1 skipped (`test_evaluation_common` needs local label files, as before).

## Replay on saved OCR (no new OCR)

Both code trees run their own `stage_result` + `run_cascade` on the same saved detections.

| Data | main `0dd7a35` | merged | new correct / new wrong | OCR stage calls |
|---|---:|---:|---|---:|
| Cosmetics dev 300 (all 5 stages saved, main run 09-30) | 244 | **265** | 21 / 0 | 610 → 623 |
| Food frozen GT 700 (hybrid, see below) | 614 | **617** | 4 / 1 | 1,207 → 1,267 |
| ├ rule-development 401 | 344 | 346 | 3 / 1 | 729 → 762 |
| └ check 299 | 270 | 271 | 1 / 0 | 478 → 505 |

- Cosmetics main replay equals the official 0dd7a35 run on all 300 images.
- Food: run701 dump with missing stages filled from the old-parser four-stage snapshot
  (same hybrid as round 6). `clahe_1024` is not saved for food; every image that reached it in
  either tree was one main left NONE, whose outcome is known from
  `docs/clahe1024_production_validation_20260929.md` (4 recovered, the rest no candidate).
  0 images had an unknown outcome. Main scores 614 here, 615 on Seonwoo's full-stage dump.
- Food gains: 000231, 002148, 002336, 002917. Loss: 001686 (GT 2020-11-27 is the manufacture
  date under "유통기한: 제조일로부터 5년까지"; Parser rule 5 computes 2025-11-27). This conflicts
  with the team label rule "manufacture date only → NONE"; the frozen GT is not changed here.
- Cosmetics results are in-sample: the Parser rules were written from these 300 images.
  Cosmetics validation 900301-900400 was not opened.
