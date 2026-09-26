# Full-stage 701 raw OCR dump

This is an isolated data-generation tool. Production cascade, Parser, Policy B,
thresholds and acceptance rules are unchanged. No real OCR was run while preparing it.

## Frozen dataset

Use the 701 IDs in `tmp_labels_701.csv`, independently cross-checked against
`labels/review/label_review.csv`. Both sorted, normalized six-digit ID lists have
SHA-256 `4a1c04e97c285caa12a00841616ff56d3eaf8f374238c45105a5ae139a8f0e4b`
(IDs joined by LF, no terminal LF). Images come exclusively from `data/`.
Missing images, duplicate normalized image filenames, duplicate CSV IDs and
any substituted/missing/extra membership fail. `data/` is the larger image pool;
images outside the frozen membership are deliberately not selected.

The review audit records the union of existing 300 and incoming 432 label rows,
with 31 overlapping IDs. The 26-case diagnostic names this exact temporary label
CSV as its label source. This establishes membership, not approved ground truth:
the review source has unfinished decisions and the temporary CSV includes candidate
labels. Label rows are copied verbatim as metadata; no truth normalization,
correctness test, candidate selection or policy evaluation occurs in generation.

## Commands (PowerShell, repository root)

Read-only preflight (no OCR imports, initialization, inference, or output writes):

```powershell
.\.venv\Scripts\python.exe -B .\scripts\dump_full_stage_701.py --output .\docs\full_stage_701_run1
```

Real run, or resume using exactly the same command:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\dump_full_stage_701.py --output .\docs\full_stage_701_run1 --run-ocr
```

Rebuild a complete consolidated dump without OCR:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\dump_full_stage_701.py --output .\docs\full_stage_701_run1 --consolidate-only
```

No-OCR tests (one engine is mocked; the 701-image test uses synthetic detections;
a synthetic repository with 701 placeholder image files exercises the full preflight;
the real-environment preflight test is skipped unless `data/`, `weights/` and the
shadow run artifacts exist locally):

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_full_stage_701.py -v
```

## Stage and runtime contract

One sequential CPU engine, two threads, MKLDNN enabled, recognition batch 6
(two threads matches the submission worker and the retry diagnostics; the official
300 baseline evaluation used four threads, and `initialize_engine` defaults to four);
local `PP-OCRv6_medium_det` and `korean_PP-OCRv5_mobile_rec`. No warmup inference.
EXIF transpose and RGB decode; shrink-only LANCZOS using current rounded dimensions.
Every image receives original 512, PIL rotation 270 of the 512 input (expand=True),
decoded original resized to 1024, and LAB-lightness CLAHE of the 512 input
(clipLimit=2.0, tileGridSize=(8,8)). Detection side limits are 512/512/1024/512,
limit type max, box threshold 0.7, exactly as current diagnostics/production.
Orientation classification, unwarping and textline orientation stay disabled by
the existing initializer. No Parser/has_candidate calls; no early stop.

## Cache audit and planning

The audit pins each inspected source manifest, checks current producer code and
model weight hashes, checks runtime versions, and validates every raw record.
Only `frozen_b_shadow_independent_run1` qualifies for import: 195 original,
37 rotation and 37 high-resolution records (269 total). It includes fingerprints
for PaddleOCR, PaddlePaddle, NumPy, Pillow, PaddleX and OpenCV, per-image hashes,
and per-record identity/checksums. These all match the current environment.

The 104 early-stop, 50 correct-control, and 18 triggered-control records match
code/weights but are rejected for two independent reasons: their manifests omit
historical PaddleX/OpenCV versions, and their schema has no per-record checksum
bound to a manifest run id. Conservatively reject all 172 for reuse; never infer
historical package identity from today's install.
Their valid raw timings remain useful as approximate timing samples.
Every historical manifest must also match the current detection options
(`text_det_limit_type`, `text_det_box_thresh`) and use the checksummed shadow schema;
anything else is reported as not reusable instead of being imported.

`docs/run701/ocr_dump.jsonl` (Sumin's PC, `claude/itda-ocr-github-integration-wpblqh`
branch only; stage-1 raw OCR for all 701 plus the later stages that early stop reached)
is **not reusable** and is deliberately not audited as a source: it records no package
versions, model weight hashes, engine/thread settings, image hashes, run id or record
checksums, and it came from a different machine. Older audit/benchmark dumps are excluded:
they are outside the inspected producer allowlist and lack an established
equivalence chain to this four-stage, two-thread schema. The official 300
prediction JSONL does not contain raw detections and cannot supply these records.

`docs/full_stage_701_preflight.json` is the report produced on Seonwoo's PC at
`c13fb32`, before the audit hardening above. Re-run the read-only preflight before
`--run-ocr`; the reuse count and reasons printed then are authoritative.

Initial plan: maximum 2,804 stage records; 269 imported; 2,535 fresh attempts:
506 original, 664 rotation, 664 highres, 701 CLAHE.
Pooled historical mean OCR/conversion times are 1.481, 1.308, 3.427 and 1.190
seconds (sample counts 271, 72, 72 and 26 respectively). Remaining count times
each stage mean gives 4,727.7 seconds, about 78.8 minutes of OCR/conversion.
Allow roughly 90–120 minutes for planning, including initialization, image work,
fingerprinting and durable writes. This is not measured full-run wall time or a
bound: selected cohorts, host load and disk performance can shift it considerably.

## Artifacts and offline replay

Under `docs/full_stage_701_run1/`:

- `manifest.json`: exact IDs, image bytes/paths, producer and pipeline hashes,
  all six local model file hashes, runtime versions, engine/stage definitions,
  source metadata hashes, and historical import paths/checksums/audit decisions.
- `labels_metadata.json`: verbatim label rows, separate from OCR input/selection.
- `cache/IMAGE_STAGE.json`: one authoritative committed raw record per stage.
- `progress.json`: committed count and fresh/imported/cached invocation counters.
- `raw_ocr.jsonl`: exactly 2,804 records ordered by image ID, then the four stages.
- `summary.json`: invocation counters, wall time, initialization time and dump hash.
- `timing_summary.json`: stage totals retaining original cache measurements.
- `integrity.json`: run fingerprint, final JSONL hash and all record checksums.
- `writer.lock`: persistent file with an OS-held exclusive lock only while writing.

Each raw record has `image_id`, `stage`, ordered `detections` (text, recognition
confidence, polygon), `ocr_sec`, `stage_sec`, `stage_metadata`, `source`, `run_id`,
and `record_sha256`. Fresh records also include input dimensions, decoded size,
decode and preparation timings. Historical records retain the original timing
fields; dimensions absent from source are not invented. Polygons are in stage
pixel coordinates. `manifest.json` plus the input bytes defines their transform.
`detections` is the current Parser's complete raw input, before parsing or acceptance;
it is not a full PaddleOCR result serialization (intermediate tensors/unrecognized
detector proposals are not retained).

Read `raw_ocr.jsonl` by iterating the file or splitting on `\n` only. Do not use
`str.splitlines()`: OCR text may contain U+0085/U+2028, which are left unescaped
and which `splitlines()` treats as line breaks.

Offline replay example:

```python
import json
from date_parser import parse_expiration_date

with open('docs/full_stage_701_run1/raw_ocr.jsonl', encoding='utf8') as stream:
    for line in stream:
        record = json.loads(line)
        result = parse_expiration_date(record['detections'])
        # Group by image_id and stage for future trigger/acceptance experiments.
```

## Integrity and interruption behavior

Record writes flush and fsync a same-directory temporary file, then atomically
replace the destination. Only final validated records count as completed. A crash
between inference and the durable commit may require that uncommitted attempt to
run again; every committed image/stage is reused. Recognized `.json.tmp` remnants
are ignored and replaced only when their stage needs writing. Corrupt committed
records, unexpected cache files, identity/checksum mismatches and provenance
changes fail before inference; no silent repair or rerun of corrupt records.
OS locks prevent simultaneous writers and release when the process dies.

Progress is informational; the validated record set is authoritative after a
crash. Consolidation requires exactly all four stages for every frozen image and
reads back the JSONL before emitting integrity information. A partial run has
per-stage caches and progress but no complete summary/dump. Rerunning consolidation
updates the invocation summary to zero fresh work; retained per-stage timings
are the original measurements, not that invocation's wall time.

Run the preflight and the run in the working tree that produced the shadow artifacts;
do not re-clone. The audit pins code files by byte hash, so a fresh Windows clone that
converts line endings to CRLF changes those hashes and the 269 records are reported as
code drift (reuse falls to 0 - safe, but every stage is then run fresh).

Do not hold `progress.json` (or other output files) open while the run is writing,
e.g. with `Get-Content -Wait`, an editor, or file sync. On Windows the atomic replace
can then fail with a permission error and stop the run; rerun the same command to resume.

Any change to this script changes its hash in `manifest.json`, so an output directory
must be resumed and consolidated with the same script version that started it.

Do not alter inputs, producer code, packages or model files during a run. Strict
resume also fingerprints metadata files; label edits require restoring that
snapshot or choosing a new output directory, even though truth never influences
OCR. Imported source files must remain available and unchanged for preflight.
Filesystem atomic replacement/durability assumes normal local filesystem behavior;
it is not protection from hardware failure or malicious edits with recomputed hashes.
The resulting 701 cohort contains development cases, so future tuning on it must
not be reported as independent shadow validation.

## Policy B `M` / `q` definition check

No trigger, threshold or acceptance logic is used by this tool; this only records
that the two analyses agree. Seonwoo's `evidence()` in
`scripts/diagnose_correct_controls_50.py` (reused by `validate_frozen_b_shadow.py`):
M = at least two distinct `c.result.final_date_string()` over `find_all_candidates(boxes)`;
q = minimum confidence of the boxes whose text and bbox center equal the selected
candidate's. Sumin's 701 estimate: M = at least two distinct `pc.result.as_strings()`
over the same candidates; q = confidence of the first such box.

- M is identical: both take each candidate's top-scoring reading (`result`), and
  `as_strings()` tuples map one-to-one onto `final_date_string()` (an empty result is
  one `NONE` value in both), so the distinct counts are equal.
- q can differ only when several boxes share the selected text and exact center
  (min vs first), and Seonwoo's version raises when no source box is found. No saved
  record had more than one such box, so "q differs 0" does not test min vs first.
- Checked on all 1,170 saved stage records of `docs/run701/ocr_dump.jsonl` with the
  current `date_parser`: M differs 0, q differs 0; stage-1 Policy B triggers 139 in both.
