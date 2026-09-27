"""Reverify Optimization 2 without overwriting existing diagnostic work or running OCR.

Reconstructs the accepted parser from git objects in a NEW artifact directory.
Runs isolated baseline/probe workers, then tests and immutable-text replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '9b14803cd42459d4bcf91104bcb26e4fb161a372'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def worker(out, parser_root):
    sys.path.insert(0, str(parser_root))
    import date_parser
    assert Path(date_parser.__file__).resolve().parent == parser_root / 'date_parser'
    sys.path.insert(0, str(ROOT))
    from scripts import probe_parser_boundaries as p
    raw, old, cohorts, _ = p.load_inputs()
    baseline, _ = p.replay.snapshot(raw)
    assert baseline == old, 'Git baseline differs from accepted saved replay'
    expected = p.a.read(ROOT / 'docs/parser_optimization2/probe_snapshots.json')
    rows, changes, stages = [], [], []
    for name in p.PROBES:
        with p.probe(name):
            new, _ = p.replay.snapshot(raw)
        assert new == expected[name], name + ' differs from existing probe'
        r, c = p.metrics(old, new, cohorts, ['current_cascade'], name)
        rows += r
        changes += c
        for key, ss in new.items():
            for i, s in enumerate(ss):
                if s != old[key][i]:
                    stages.append(dict(variant=name, image_id=key, stage=p.a.STAGES[i],
                                       old_prediction=p.replay.pred(old[key], i),
                                       new_prediction=p.replay.pred(ss, i)))
    # Diagnostic-only extension: the existing four-digit-year suffix trim
    # applied to two-digit first fields. No ID, product, or truth is consulted.
    trim = p.ex._trim_trailing_noise
    def two_digit_tail(fields):
        if (len(fields) == 3 and all(f.kind == 'num' for f in fields)
                and len(fields[0].raw) == 2 and len(fields[2].raw) == 3):
            return fields[:2] + (p.ex.RawField(fields[2].raw[:2], 'num'),)
        return trim(fields)
    p.ex._trim_trailing_noise = two_digit_tail
    try:
        new, _ = p.replay.snapshot(raw)
    finally:
        p.ex._trim_trailing_noise = trim
    name = 'rejected_two_digit_trailing_noise'
    r, c = p.metrics(old, new, cohorts, ['current_cascade'], name)
    rows += r
    changes += c
    for key, ss in new.items():
        for i, s in enumerate(ss):
            if s != old[key][i]:
                stages.append(dict(variant=name, image_id=key, stage=p.a.STAGES[i],
                                   old_prediction=p.replay.pred(old[key], i),
                                   new_prediction=p.replay.pred(ss, i)))
    p.a.write_csv(out / 'probe_comparison.csv', rows)
    p.a.write_csv(out / 'probe_changed_cases.csv', changes)
    p.a.write_csv(out / 'probe_changed_stages.csv', stages)
    write(out / 'probe_verification.json', dict(baseline_exact=True,
          original_four_probe_snapshots_exact=True,
          rejected_probe_oracles=p.oracle(new, cohorts), ocr_invocations=0))


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    protected_dirs = ['docs/full_stage_701_run1', 'docs/full_stage_701_analysis1',
                      'docs/parser_optimization1', 'labels']
    paths = [f for d in protected_dirs for f in (ROOT / d).rglob('*') if f.is_file()]
    paths += list((ROOT / 'docs/parser_optimization2').glob('*.*'))
    paths += list((ROOT / 'date_parser').glob('*.py'))
    paths += [ROOT / p for p in ['ocr_pipeline.py', 'submission_runtime.py', 'predict.ipynb',
              'scripts/probe_parser_boundaries.py', 'tests/test_date_parser/test_boundary_normalization.py']]
    before = {str(f.relative_to(ROOT)): sha(f) for f in paths}
    write(out / 'preserved_inputs_before.json', before)
    for name, args in [('git_status_before.txt', ['status', '--short']),
                       ('git_diff_before.patch', ['diff', '--binary'])]:
        result = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, check=True)
        (out / name).write_bytes(result.stdout)
    parser_root = out / 'accepted_parser'
    (parser_root / 'date_parser').mkdir(parents=True)
    for file in (ROOT / 'date_parser').glob('*.py'):
        content = subprocess.run(['git', 'show', f'{COMMIT}:date_parser/{file.name}'],
                                 cwd=ROOT, capture_output=True, check=True).stdout
        (parser_root / 'date_parser' / file.name).write_bytes(content)
    subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--worker',
                    '--output', str(out), '--parser-root', str(parser_root)], cwd=ROOT, check=True)
    test = subprocess.run([sys.executable, '-B', '-m', 'pytest', 'tests/test_date_parser',
                           'tests/test_full_stage_offline.py', 'tests/test_oracle_gap_offline.py',
                           '-q'], cwd=ROOT, capture_output=True)
    (out / 'tests.txt').write_bytes(test.stdout + test.stderr)
    assert test.returncode == 0, 'See tests.txt'
    sys.path.insert(0, str(ROOT))
    from scripts import probe_parser_boundaries as p
    raw, old, cohorts, _ = p.load_inputs()
    new, elapsed = p.replay.snapshot(raw)
    assert new == p.a.read(ROOT / 'docs/parser_optimization2/new_parser_replay.json')['stages']
    rows, changes = p.metrics(old, new, cohorts, list(p.replay.POLICIES), 'implemented')
    assert rows == p.a.read(ROOT / 'docs/parser_optimization2/summary.json')['policies']
    p.a.write_csv(out / 'policy_comparison.csv', rows)
    p.a.write_csv(out / 'changed_cases.csv', changes)
    stage_changes = []
    for key, ss in new.items():
        for i, s in enumerate(ss):
            if s != old[key][i]:
                stage_changes.append(dict(image_id=key, stage=p.a.STAGES[i],
                    old_prediction=p.replay.pred(old[key], i), new_prediction=p.replay.pred(ss, i),
                    old_signals=json.dumps(old[key][i]), new_signals=json.dumps(s)))
    p.a.write_csv(out / 'changed_stages.csv', stage_changes)
    # Capture every policy change relative to the accepted current cascade,
    # including inherited policy regressions, not just parser-induced changes.
    policy_changes = []
    for cohort, truths in cohorts.items():
        for policy in p.replay.POLICIES:
            for key, truth in truths.items():
                before_i = p.a.cascade(old[key])[0]
                after_i, attempts = p.replay.run_policy(new[key], policy)
                b, n = p.replay.pred(old[key], before_i), p.replay.pred(new[key], after_i)
                if b != n:
                    policy_changes.append(dict(cohort=cohort, policy=policy, image_id=key,
                        truth=truth, old_prediction=b, new_prediction=n,
                        old_correct=b == truth, new_correct=n == truth,
                        attempts='|'.join(p.a.STAGES[i] for i in attempts)))
    p.a.write_csv(out / 'policy_changes_vs_accepted_cascade.csv', policy_changes)
    extra = [k for k in new if p.a.cascade(new[k])[1] != p.a.cascade(old[k])[1]]
    assert not extra
    remaining = []
    categories = {r['image_id']: r for r in p.a.csvread(
        ROOT / 'docs/full_stage_701_analysis1/oracle_gap/nonrecoverable_30.csv')}
    for key, truth in cohorts['official_300'].items():
        if any(p.replay.pred(new[key], i) == truth for i in range(4)):
            continue
        remaining.append(dict(image_id=key, truth=truth, **{s: p.replay.pred(new[key], i)
            for i, s in enumerate(p.a.STAGES)},
            previous_category=categories[key]['category'], review_note=categories[key]['review_note']))
    p.a.write_csv(out / 'remaining_outside_oracle.csv', remaining)
    after = {str(f.relative_to(ROOT)): sha(f) for f in paths}
    assert before == after, 'Existing work/input changed'
    write(out / 'verification.json', dict(accepted_commit=COMMIT, python=sys.version,
        parser_snapshot_exact=True, policy_metrics_exact=True, existing_files_unchanged=True,
        protected_tree_file_count=len(before), current_cascade_attempts_changed=extra,
        parser_replay_seconds=elapsed, changed_stage_count=len(stage_changes),
        tests_exit_code=test.returncode, ocr_invocations=0,
        artifact_sha256={str(f.relative_to(out)): sha(f) for f in out.rglob('*') if f.is_file()},
        audit_script_sha256=sha(Path(__file__))))
    print(json.dumps(dict(verified=True, tests=test.stdout.decode(errors='replace').strip(),
                         parser_replay_seconds=elapsed, changed_stage_count=len(stage_changes)), indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'docs/parser_optimization2/reproducibility')
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--parser-root', type=Path)
    args = ap.parse_args()
    if args.worker:
        worker(args.output.resolve(), args.parser_root.resolve())
    else:
        main(args.output.resolve())
