"""Render the six-case review from verified saved OCR; never runs OCR."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import probe_parser_boundaries as p

OUT = ROOT / 'docs/parser_optimization2'
CASES = {
    '000763': dict(category='G (B+A)', recoverable=True,
        relevant=[['11. 120트시F로'], [], [], ['11.1205시F2']],
        reason='The numeric right boundary rejects 1205 as a day. CLAHE preserves MM.DDHH시: 11.12 plus valid hour 05 and explicit Korean hour unit 시. Split only this anchored field in the yearless fallback. Original 11. 120트시F로 has a corrupted hour and is deliberately not repaired; rotation/highres omit the date. No year is invented.'),
    '000820': dict(category='G (A+B+C)', recoverable=False,
        relevant=[['SSS SI-18'], ['ESTeBEFERE18405-2026'], ['BEST', 'EEFORE', 'VARIOEE'], ['2026']],
        reason='Rotation preserves 18405-2026, not a clean day/month/year token. Getting 18/05/2026 requires deleting the internal digit 4 or asserting that 4 was a separator. There is no text-supported reason to choose that repair over other corrupted-digit readings. The other stages do not preserve reliable day/month fields. The damaged BEST BEFORE label does not resolve the missing boundary.'),
    '002328': dict(category='G (B+E+F)', recoverable=False,
        relevant=[['2020 10.21부터'], ['2부'], ['유통기한', '후면 표기일까지', '200.10. 21부통', '21 10 201'], ['유통기한연기일까지', '2020.10.21부터']],
        reason='Highres has 21 10 201. Dropping the final 1 gives 21 10 20, but both 2021-10-20 (YMD) and 2020-10-21 (DMY) are calendar-valid. A two-digit first field does not establish which field is a year or whether the third field has a suffix. Current cascade stops on the earlier manufacturing-like 2020.10.21부터 observation. Highres manufacturing text is itself corrupt. This is a plausible hypothesis, not defensible recovery without a boundary/order assumption; it is not simply missing all digits.'),
    '002899': dict(category='G (C+B)', recoverable=True,
        relevant=[['2021 11098'], ['2021 11098'], ['온021 11 098'], ['2021,11,098']],
        reason='CLAHE retains a four-digit year, separate valid month, and three-digit day field. Commas are not accepted by the existing generic triple pattern, so a year/month partial wins. Admit the comma triple and reuse the already accepted four-digit-year trailing-noise rule (098 -> 09), with unchanged calendar/order interpretation. This is a bounded suffix-noise heuristic, not proof of the printed date. No internal digit repair, unrestricted numeric-token slicing, or inferred year is added. Earlier fused/corrupted stages remain unchanged.'),
    '001955': dict(category='G (A+C+E)', recoverable=True,
        relevant=[['제품 법도표시원 Best belore월/년의 O1일끼지', '-OCT.202'], ['OCTb2021'], ['제품 별도표시된 Best before:월/년의 01일까지', '--OCTbOore:'], ['제품 범도n시 Best belore월/년의 01일끼지', '-OCTb2021']],
        reason='Month-name extraction does not admit b in the separator slot. CLAHE preserves OCT and 2021 and independently reads 월/년. A standalone month-name plus one alphabetic separator plus four-digit year, gated on the same stage having a printed month/year hint, is sufficient. Rotation has intact fields but no format hint and stays NONE. Original truncates the year; highres corrupts it: neither is guessed. Generic 01일 instructions are not substituted for a missing stamped day.'),
    '003350': dict(category='G (A+B)', recoverable=False,
        relevant=[['소비김한년 월 읽 시간까지', '25.2.18137', '소비기한', '한2025:1'], [], ['소비김한(년 월 일 시간)까지', '25.412.1015시'], ['소비김한년 월 읽 시간까지', '25.42.1515', '소비기한', '한2025:1']],
        reason='Highres retains a Y/M/D/hour instruction and a recoverable 10+15시 boundary, but the month is 412. Removing leading 4 to obtain 12 is arbitrary digit deletion; it is not a separator normalization. Original/CLAHE have different damaged month/day digits. The explicit format instruction settles order but cannot establish which month digits were corrupted. Hour splitting alone cannot recover the labeled full date.'),
}


def main():
    assert not (OUT / 'findings.md').exists(), 'Preserve existing report'
    assert p.a.read(OUT / 'reproducibility/verification.json')['existing_files_unchanged']
    raw, old, cohorts, _ = p.load_inputs()
    new = p.a.read(OUT / 'new_parser_replay.json')['stages']
    context = p.a.read(ROOT / 'docs/parser_optimization1/old_parser_replay.json')
    rows = []
    lines = ['# Parser Optimization 2 — verified final review', '',
        'Accepted comparison: `9b14803cd42459d4bcf91104bcb26e4fb161a372` (Optimization 1), not the older 263-case diagnostic baseline.', '',
        '**Result: 267/300 (89.00%), +3 gains / 0 regressions; four-stage oracle 274/300.** Approved 54→56/69, reused historical independent 192→192/195, provisional 580→583/701. Three of the six cases are reasonably recoverable with constrained general rules; all three are fixed. The other three remain unresolved. “Recoverable” means a defensible textual heuristic, not certainty about the image.', '',
        'The working tree already contained the three parser edits, 43 focused test instances, probe artifacts and summary when this continuation began. Those bytes were preserved. This continuation independently reconstructed the accepted parser from git objects, reran all four probes and an additional rejected probe, verified the implementation against all 2,804 saved stages, completed tests, and added this review. It does not claim the pre-existing edits were newly written in this turn. Existing root summary/CSV files remain byte-identical; refreshed verification and the additional probe are under `reproducibility/`.', '',
        'No OCR, image recognition, label edits, OCR/cascade edits, staging, commit, push, or policy adoption occurred. The immutable OCR tree, prior analysis tree, Optimization 1 artifacts, labels, and existing Optimization 2 files were hashed and verified unchanged.', '',
        '## Architecture and rule scope', '',
        '`parser.py` converts saved detections to positioned text boxes. `extract.py` runs narrow normalizations, then ordered regexes with claimed spans; rejected non-date readings do not consume a span. Four-digit-year triples already trim a third-field trailing digit. `interpret.py` performs field permutations, year bounds, calendar checks, and existing order scoring. `select.py` detects printed format hints, adds hint-specific candidates, uses yearless MM.DD only when no dated candidates exist, and ranks by expiry/manufacture context and geometry. OCR confidence and retry policy are separate offline signals.', '',
        'The retained implementation changes only `date_parser/extract.py` and the hint-specific extraction call in `date_parser/select.py`:', '',
        '1. Four-digit-year comma triple extraction (day length ≤3) before partial extraction. Reuses existing trailing noise/calendar behavior; requires an alphanumeric left boundary and excludes longer numeric runs. This does not globally replace commas or change numeric date order.',
        '2. At the start of a box, split MM.DDHH시 only for a two-digit valid clock hour (00–23), inside the yearless fallback. No unit, invalid hour, lot prefix, or competing dated candidate means no new yearless interpretation.',
        '3. With a printed month/year hint in the same stage, admit one alphabetic character in the separator position of a standalone English month abbreviation and intact four-digit year. Whole-box bounds allow adjacent punctuation/brackets, exclude lot prefixes/extra digits, and do not repair date digits.', '',
        'Existing O/o/U/u→0, narrow C→0, split-year, compact-date and glued-date/time handling were inspected and left intact. No new I/1 or S/5 conversion, arbitrary numeric slicing, expiry-label fuzzy matching, cross-stage text merging, or broad DMY/YMD preference was added. Standard separated dates and month names remain handled by existing rules. The suffix rule is heuristic; calendar validity alone does not establish that a numeric code is a date.', '',
        '## Six cases', '',
        'Stages: O=original_512, R=rotation_270, H=highres_1024, C=clahe. Outputs below show accepted baseline → current implementation. Text is copied verbatim from the saved stage; “none” means no reliable date field in that stage. `six_cases.csv` additionally includes **all** OCR boxes for all four stages, avoiding omissions from the relevant-text excerpt. Label status is frozen metadata, not a new visual approval.', '']
    for key, info in CASES.items():
        label = context['labels'][key]['truth_source']
        truth = cohorts['official_300'][key]
        record = dict(image_id=key, label_status=label, truth=truth, classification=info['category'],
                      recoverable_by_reasonable_rule=info['recoverable'], safely_fixed=info['recoverable'], explanation=info['reason'])
        lines += [f'### {key} — {label}; truth `{truth}`', '',
                  f'**{info["category"]}; ' + ('recoverable and fixed.**' if info['recoverable'] else 'not safely recoverable by the tested general rules.**'), '',
                  '| Stage | Relevant saved OCR / raw date-like substrings | Baseline → current |', '|---|---|---|']
        for i, stage in enumerate(p.a.STAGES):
            all_text = [d['text'] for d in raw[key][stage]['detections']]
            assert all(t in all_text for t in info['relevant'][i])
            b, n = p.replay.pred(old[key], i), p.replay.pred(new[key], i)
            record[stage + '_all_texts_json'] = json.dumps(all_text, ensure_ascii=False)
            record[stage + '_raw_date_like_and_context_json'] = json.dumps(info['relevant'][i], ensure_ascii=False)
            record[stage + '_baseline_output'] = b
            record[stage + '_current_output'] = n
            texts = '<br>'.join('`' + t.replace('|', '\\|') + '`' for t in info['relevant'][i]) or '(none)'
            lines.append(f'| {stage} | {texts} | `{b}` → `{n}` |')
        lines += ['', info['reason'], '']
        rows.append(record)
    p.a.write_csv(OUT / 'six_cases.csv', rows)
    probes = p.a.csvread(OUT / 'reproducibility/probe_comparison.csv')
    lines += ['## Offline probes', '',
        'Each probe runs in an isolated process with the accepted git parser, never in the production parser. Truths are used only to score outputs. Original probe snapshots and current production outputs reproduce exactly. Counts below are correct/gains/regressions/net versus the accepted current cascade; cohort order is official 300, approved 69, reused independent 195, provisional 701.', '',
        '| Probe | Official | Approved | Independent | Provisional |', '|---|---|---|---|---|']
    for variant in dict.fromkeys(r['variant'] for r in probes):
        group = [r for r in probes if r['variant'] == variant]
        lines.append('| ' + variant + ' | ' + ' | '.join('/'.join(r[k] for k in ('correct','gains','regressions','net_gain')) for r in group) + ' |')
    lines += ['',
        'Accepted probes change final predictions only for 002899 (comma), 000763 (hour boundary), and 001955 (hinted separator); combined changes their union. None breaks a currently correct official, approved, independent, or provisional case. All three prediction changes are at CLAHE. There is also a signal-only change for 000015 at original_512: its final date stays 2026-03-01, but candidate diagnostics change (see changed_stages.csv). Exact per-cohort IDs and predictions are in the root `probe_changed_cases.csv`; full rerun including the rejected probe is in `reproducibility/probe_changed_cases.csv` and `probe_changed_stages.csv`.', '',
        'The additional **two-digit trailing-noise** probe extends trimming a three-digit third field to triples with a two-digit first field, without changing order scoring. It is rejected: the first field no longer establishes the year, a trailing digit can belong to a code or damaged year, and 21 10 20 remains order-ambiguous. It changes no current-cascade output in any cohort. Stage changes are 000245/highres (wrong to wrong), 000503/CLAHE (already oracle-recoverable elsewhere), 002328/highres (new correct interpretation), plus signal-only changes for 002134/original and 002459/highres. It raises the official oracle 271 to 272 and provisional oracle 595 to 596 solely through 002328, with approved and independent oracles unchanged. A useful oracle interpretation is insufficient to justify choosing it. No such change was made to production. Exact changed outputs, including wrong→wrong changes, are in the rerun CSVs.', '',
        'Rejected without implementation: delete internal 4 in 18405-2026; delete 4 from month 412; infer truncated 202/202N years; loosen whole-box bounds into batch/lot words; slice dates out of arbitrary longer digit runs. These lack a reliable textual boundary. No gain is counted for such hypotheses.', '',
        '## Regression and policy interaction', '',
        'The full parser suite plus offline replay/policy unit tests were run before the final implementation replay. See `reproducibility/tests.txt`. Focused tests cover comma dates, calendar-invalid fields, long numeric runs, lot prefixes, malformed separators, valid/invalid hour units, NONE, competing full dates, printed hints, brackets, truncated years, no digit repair, and unchanged ambiguous two-digit-year order.', '',
        '| Policy (offline only) | Official /300 | Approved /69 | Independent /195 | Provisional /701 |', '|---|---:|---:|---:|---:|']
    summary = p.a.read(OUT / 'summary.json')
    for policy in p.replay.POLICIES:
        group = [r for r in summary['policies'] if r['policy'] == policy]
        lines.append('| ' + policy + ' | ' + ' | '.join(str(r['correct']) for r in group) + ' |')
    lines += ['',
        'Frozen B + highest_q alone reaches **271/300**; the previous best policy reaches **273/300**, and agreement reaches the **274/300 oracle**. Relative to Optimization 1 with the same policies, each gains the same three official cases with no parser-induced official regression. Relative to the new current cascade, frozen B adds 4 official successes, previous best adds 6, agreement adds 7. Thus ≥270 no longer needs the aggressive agreement extension, but the parser itself remains at 267.', '',
        'Policies are not adopted. Frozen B regresses two approved cases relative to current cascade (56→54); previous best/agreement have one gain and two approved regressions (56→55). Provisional policy results also include regressions (5 for frozen B, 7 for previous best/agreement versus accepted cascade). These are inherited policy tradeoffs, not hidden parser regressions. Exact IDs are in `reproducibility/policy_changes_vs_accepted_cascade.csv`.', '',
        'No OCR ran and current-cascade stage attempts are **identical for every one of 701 images**. The gains occur in the already-attempted last stage, so there is no additional OCR runtime and no retry reduction under the unchanged cascade. Frozen B simulations add one rotation and one highres attempt on the 701 mix relative to its old-parser version; this is an offline projection, not work executed.', '',
        'Official O/R/H/C attempts: current 300/46/39/27; frozen B 300/96/89/27; previous best 300/174/97/27; no-anchor agreement 300/240/103/27. Recorded OCR service-time projections per 500 images (official mix) are about 1053, 1415, 1628, 1780 seconds respectively. On the 701 mix: 1107, 1529, 2479, 2608 seconds. They include retained measured outliers and omit startup/decode/I/O; they are not an end-to-end runtime qualification. Parser diagnostic replay time is in verification.json and includes signal extraction.', '',
        '## Remaining errors and decision', '',
        'Of the 33 current-cascade official errors, seven are recoverable through saved-stage selection. Of the 26 still outside the oracle:', '',
    ]
    remaining = p.a.csvread(OUT / 'reproducibility/remaining_outside_oracle.csv')
    for category, text in [('insufficient_ocr', '18 lack reliable date fields or contain unresolved digit corruption; improved OCR/preprocessing is needed for reliable automatic recovery from those observations'),
                           ('mixed_ocr_parser', '3 mixed cases remain: 000820 and 003350 require unsupported internal digit repair; 002328 has an unresolved boundary/order/context hypothesis and needs better evidence, rather than being declared completely digit-missing'),
                           ('parser_evidence', '2 retain date-order alternatives (002726, 002034): parser interpretation ambiguity, not proven OCR deficiency; do not change DMY/YMD defaults to fit labels'),
                           ('label_uncertainty', '3 require label/image review before blaming the parser or OCR; 000157 is a documented official/approved conflict, 001587 and 003218 are suspected conflicts or repeated OCR errors')]:
        ids = ', '.join(r['image_id'] for r in remaining if r['previous_category'] == category)
        lines += [f'- {text}: {ids}.']
    lines += ['',
        'Decision: 3/6 reasonably recoverable and 3/6 safely fixed on this replay; official 264→267 (89.00%), +3/0/net +3; approved 54→56; independent 192→192; provisional 580→583; oracle 271→274. Parser-only exceeds 266. A simple existing frozen-B policy exceeds 270 (271), while the best combined offline result is 274. Additional OCR runtime is zero. The reused historical cohort and provisional labels are not fresh independent validation; no labels were changed.', '',
        '## Reproduction and artifact inventory', '',
        'Run from the repository root using the existing environment:', '',
        '```powershell',
        '.\\.venv\\Scripts\\python.exe -B scripts/audit_parser_optimization2.py --output docs/parser_optimization2/reproducibility_new',
        '```', '',
        'The destination must be new. It reconstructs the accepted parser via read-only git show, verifies input hashes, reruns isolated probes, executes all parser and offline diagnostic tests, replays 2,804 immutable saved stages, checks current-cascade attempts, and asserts exact equivalence with preserved results. No OCR imports or calls are needed.', '',
        'Root deliverables: `findings.md`, preserved `summary.json`, `six_cases.csv`, preserved `probe_comparison.csv`, `probe_changed_cases.csv`, `changed_cases.csv`, `policy_comparison.csv`, `probe_snapshots.json`, `new_parser_replay.json`, and `manifest.json`. New `decision.json` records the final answers. `reproducibility/` holds reconstructed accepted parser, verified probe/policy CSVs, complete stage changes, outside-oracle IDs, tests, protected-input hashes, initial git status/diff, and `verification.json`. Final status and hashes are recorded separately after report generation.', '',
        'Parser diff versus accepted commit remains the pre-existing edits to `extract.py` and `select.py`; focused tests remain `tests/test_date_parser/test_boundary_normalization.py`. This continuation adds `scripts/audit_parser_optimization2.py` and `scripts/report_parser_optimization2.py`. All pre-existing diagnostic files are preserved, including the existing Optimization 2 directory. Nothing is staged or committed.', '']
    (OUT / 'findings.md').write_text('\n'.join(lines), encoding='utf8')
    p.a.write_json(OUT / 'decision.json', dict(recoverable=3, safely_fixed=3,
        official=dict(before=264, after=267, total=300, gains=['000763','001955','002899'], regressions=[]),
        approved=dict(before=54, after=56, total=69), independent_reused=dict(before=192, after=192, total=195),
        provisional=dict(before=580, after=583, total=701), oracle=dict(before=271, after=274),
        parser_reaches_266=True, frozen_B_reaches_270=True, best_combined_offline=274,
        ocr_invocations=0, additional_current_cascade_ocr_attempts=0, policies_adopted=False,
        existing_uncommitted_work_preserved=True))


if __name__ == '__main__':
    main()
