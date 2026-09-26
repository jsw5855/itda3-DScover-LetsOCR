"""Independent frozen B + highest_q shadow validation; NO OCR unless --run-ocr.

Primary estimand: official original_512 early stops, after all development IDs
are excluded. Later official stops are excluded, never replaced with stage1.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

from diagnose_early_stop_26 import ROOT, image_id, read_csv, sha
from diagnose_correct_controls_50 import FROZEN_IDS, POLICIES, evidence
from diagnose_triggered_controls_9 import STAGES, choose
from evaluation_common import normalize_truth

OFFICIAL = ROOT / 'docs/parser_integration_300_correct_0926'
CONTROLS = ROOT / 'docs/correct_controls_50_run1'
FAILURES = ROOT / 'docs/early_stop_26_run1'
PINS = {
    'docs/parser_integration_300_correct_0926/evaluation_300.csv': 'df0425cc6c09cd9c300c646e51e391cdf72c86a000d204b370f7585223889684',
    'docs/parser_integration_300_correct_0926/manifest.json': 'c311f308708f0aa4510f50b41d5dc37567b85e86bbe44cc5ff09c5780fd78e49',
    'docs/parser_integration_300_correct_0926/predictions.jsonl': '7cd6b30db33c2bdcb9eeb476e2b337414e06f2142dbbe7d91a3e11002eec4389',
    'docs/early_stop_26_run1/cases.csv': '5fb2c3f5297fb2044ae238d81900f49e24c725fa567e213854eab5eb5e46021b',
    'docs/early_stop_26_run1/manifest.json': 'a1615ef92e052553138071f0853753a87a59d371d8b4310dc4bf407a99528e50',
    'docs/correct_controls_50_run1/manifest.json': 'e24f83ff0f99dcb711c5053fd1aa6ec088a657f4051d1e98b7a97ef3eabc37cb',
    'docs/triggered_correct_controls_9_run1/manifest.json': 'bd822d4c42b16ea974e052ada26120336730ff69cdde923ed0016554092578df',
    'scripts/diagnose_triggered_controls_9.py': '1b8ce890c33b55a14d320b4dfcffec42f626de0490932f4969f31556561d7fbc',
    'scripts/diagnose_correct_controls_50.py': '60702813a433f444d046006e07add5d6c71b03b08d0e9c9694809e8621a5a130',
    'scripts/diagnose_early_stop_26.py': 'eaee2ed6a4b5282cc1f0afb83c6d0f7f6508e6fb039e1afd3d7c04a63bc9cd9c',
    'scripts/evaluation_common.py': '1117a155ce9b6306ebc2dfd3a79b35f19b83aa375a3473577fb665b7509883a3',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    allow_nan=False).encode()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    temporary.replace(path)


def unique(rows):
    result = {image_id(r['image_id']): r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate normalized image IDs')
    return result


def cohort_definition(evaluation, failures, controls):
    official = unique(evaluation)
    failure_ids = set(unique(failures))
    control_ids = set(controls)
    if len(official) != 300 or len(failure_ids) != 26 or len(control_ids) != 50:
        raise ValueError('Unexpected source cohort sizes')
    if failure_ids & control_ids:
        raise ValueError('Development cohorts overlap')
    excluded = failure_ids | control_ids
    independent = {k: v for k, v in official.items() if k not in excluded}
    primary = {k: v for k, v in independent.items() if v['method'] == STAGES[0]}
    later = {k: v for k, v in independent.items() if v['method'] != STAGES[0]}
    if (len(independent), len(primary), len(later)) != (242, 195, 47):
        raise ValueError('Frozen cohort membership/counts changed')
    info = {
        'official_count': 300, 'development_union_count': len(excluded),
        'excluded_development_ids': sorted(excluded),
        'excluded_development_ids_in_official': sorted(excluded & official.keys()),
        'excluded_failure_ids_in_official': sorted(failure_ids & official.keys()),
        'independent_count_before_method_exclusion': len(independent),
        'independent_ids_before_method_exclusion': sorted(independent),
        'official_method_counts': dict(Counter(v['method'] for v in official.values())),
        'methodological_exclusions': [{'image_id': k, 'method': v['method'],
            'reason': 'Official baseline continued beyond stage1; outside early-stop estimand'}
            for k, v in sorted(later.items())],
        'cohort_size': len(primary), 'cohort_ids': sorted(primary),
        'baseline_correct': sum(v['final_date'] == v['true_final_date'] for v in primary.values()),
        'baseline_wrong_ids': sorted(k for k, v in primary.items() if v['final_date'] != v['true_final_date']),
        'independent_242_baseline_correct': sum(v['final_date'] == v['true_final_date'] for v in independent.values()),
    }
    info['baseline_accuracy'] = info['baseline_correct'] / len(primary)
    return primary, info


def load_stage(path, key, name, run_id):
    raw = read(path)
    checksum = raw.pop('record_sha256')
    if digest(raw) != checksum or (raw['image_id'], raw['stage'], raw['run_id']) != (key, name, run_id):
        raise ValueError(f'Cache integrity/provenance mismatch: {path}')
    for field in ('ocr_sec', 'stage_sec', 'prepare_sec', 'decode_sec', 'evidence_sec'):
        if not math.isfinite(raw[field]) or raw[field] < 0:
            raise ValueError(f'Invalid timing: {path}/{field}')
    return {**raw, **evidence(raw['detections']), 'source_sha256': sha(path)}


def baseline_check(rows, stages):
    issues = []
    for key, row in rows.items():
        s = stages[key]
        expected = {f: row[f] for f in ('year', 'month', 'day', 'final_date')}
        if s['prediction'] != expected or not s['has_candidate'] or s['policy_B_trigger'] is None:
            issues.append({'image_id': key, 'official': expected, 'reconstructed': s['prediction'],
                           'has_candidate': s['has_candidate'], 'q': s['q']})
    return {'status': 'passed' if not issues else 'blocked', 'checked_count': len(rows),
            'issues': issues, 'action_on_mismatch': 'Stop entire experiment before retries; no exclusions/rebaselining'}


def preflight(output):
    for name, expected in PINS.items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Frozen input/helper changed: {name}')
    cm, fm, om = read(CONTROLS / 'manifest.json'), read(FAILURES / 'manifest.json'), read(OFFICIAL / 'manifest.json')
    if tuple(cm['image_ids']) != FROZEN_IDS:
        raise ValueError('Frozen 50 control IDs changed')
    failures = read_csv(FAILURES / 'cases.csv')
    if set(unique(failures)) != set(fm['image_ids']):
        raise ValueError('Failure manifest/case membership mismatch')
    rows, definition = cohort_definition(read_csv(OFFICIAL / 'evaluation_300.csv'), failures, FROZEN_IDS)
    code = {}
    # Normalize Windows paths before testing prefixes; check every historical hash.
    for historical in (cm['code_sha256'], fm['code_sha256'], om['source_sha256']):
        for name, expected in historical.items():
            normalized = name.replace('\\', '/')
            if sha(ROOT / normalized) != expected:
                raise ValueError(f'Historical code drift: {normalized}')
            code[normalized] = expected
    weights = cm['weights_sha256']
    if weights != fm['weights_sha256']:
        raise ValueError('Development model fingerprints differ')
    for name, expected in weights.items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Model weight drift: {name}')
    packages = {}
    for historical in (cm['packages'], fm['packages'], om['packages']):
        for name, version in historical.items():
            if importlib.metadata.version(name) != version:
                raise ValueError(f'Package drift: {name}')
            packages[name.lower()] = version
    if cm['engine'] != {'cpu_threads': 2, 'enable_mkldnn': True, 'recognition_batch_size': 6}:
        raise ValueError('Frozen engine settings changed')
    predictions = unique([json.loads(line) for line in (OFFICIAL / 'predictions.jsonl').read_text(encoding='utf8').splitlines()])
    entries = unique(om['dataset']['entries'])
    if len(predictions) != 300 or len(entries) != 300:
        raise ValueError('Official baseline/manifest count mismatch')
    images = {}
    for key, row in sorted(rows.items()):
        for field in ('file_name', 'year', 'month', 'day', 'final_date', 'method'):
            if predictions[key][field] != row[field]:
                raise ValueError(f'Official CSV/JSONL mismatch: {key}/{field}')
        truth = normalize_truth({f: row['true_' + f] for f in ('year', 'month', 'day', 'final_date')})
        if truth != entries[key]['truth'] or entries[key]['file_name'] != row['file_name']:
            raise ValueError(f'Official manifest/CSV mismatch: {key}')
        path = ROOT / 'data' / row['file_name']
        images[key] = {'path': str(path.resolve()), 'sha256': sha(path)}
    code[Path(__file__).relative_to(ROOT).as_posix()] = sha(Path(__file__))
    manifest = {
        'experiment': 'independent_original512_frozen_B_highest_q_v1',
        'cohort': definition, 'source_sha256': PINS, 'code_sha256': code,
        'images': images, 'weights_sha256': weights, 'packages': packages, 'python': sys.version,
        'engine': cm['engine'], 'predict': cm['predict'], 'policies': POLICIES,
        'acceptance': 'existing choose(stages, highest_q); ties stage1 > rotation > highres',
        'official_engine_options': om['engine_options'],
        'stages': list(STAGES), 'max_original_attempts': 195, 'max_retry_attempts': 390,
        'max_total_attempts': 585,
        'baseline_gate': 'All 195 original predictions must reproduce official baseline before any retry',
        'cache_policy': 'Reuse only this run caches. Development caches excluded; older audits have different provenance.',
        'image_provenance': 'Current image bytes fingerprinted; official manifest lacks historical image hashes. Baseline reproduction required.',
    }
    manifest = json.loads(json.dumps(manifest))
    run_id = digest(manifest)
    cached = Counter()
    originals = {}
    if output.exists():
        if not (output / 'manifest.json').is_file() or read(output / 'manifest.json') != manifest:
            raise ValueError('Output provenance mismatch; use a new output directory')
        allowed = {'manifest.json', 'cohort.csv', 'baseline_check.json', 'results.jsonl', 'comparisons.csv', 'summary.json'}
        allowed |= {f'{key}_{name}.json' for key in rows for name in STAGES}
        unexpected = [p.name for p in output.iterdir() if p.name not in allowed]
        if unexpected:
            raise ValueError(f'Unexpected output files: {unexpected}')
        for key in rows:
            for name in STAGES:
                path = output / f'{key}_{name}.json'
                if path.exists():
                    s = load_stage(path, key, name, run_id)
                    cached[name] += 1
                    if name == STAGES[0]:
                        originals[key] = s
        if originals and baseline_check({k: rows[k] for k in originals}, originals)['issues']:
            raise ValueError('Cached stage1 does not reproduce official baseline; experiment blocked')
        if cached[STAGES[1]] or cached[STAGES[2]]:
            if len(originals) != len(rows):
                raise ValueError('Retry cache exists without full baseline verification')
            for key in rows:
                if originals[key]['policy_B_trigger'] is not True:
                    if any((output / f'{key}_{name}.json').exists() for name in STAGES[1:]):
                        raise ValueError(f'Retry cache for nontriggered image: {key}')
    report = {**definition, 'preflight': 'passed', 'ocr_initialized': False,
              'new_ocr_attempts': 0, 'cached_stage_counts': dict(cached),
              'max_total_attempts': 585, 'max_remaining_attempts_upper_bound': 585 - sum(cached.values()),
              'candidate_engine': cm['engine'], 'official_engine': om['engine_options'],
              'output': str(output.resolve()), 'shadow_metrics': 'PENDING real OCR'}
    return manifest, rows, report


class StageRunner:
    """Lazy OCR import/initialization. No call path from no-OCR preflight."""
    def __init__(self, output, manifest):
        self.output, self.manifest = output, manifest
        self.run_id = digest(manifest)
        self.engine = None
        self.fresh = Counter()
        self.initialization_sec = 0.0
        self.rgb_key, self.rgb = None, None

    def get(self, key, name):
        if name not in STAGES or key not in self.manifest['images']:
            raise ValueError('Out-of-cohort/stage OCR prohibited')
        path = self.output / f'{key}_{name}.json'
        if path.exists():
            return load_stage(path, key, name, self.run_id)
        for env in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[env] = str(self.manifest['engine']['cpu_threads'])
        import numpy as np
        from PIL import Image
        import ocr_pipeline as pipeline
        if self.engine is None:
            start = time.perf_counter()
            self.engine = pipeline.initialize_engine(**self.manifest['engine'])
            self.initialization_sec = time.perf_counter() - start
        decode_sec = 0.0
        if self.rgb_key != key:
            image = Path(self.manifest['images'][key]['path'])
            if sha(image) != self.manifest['images'][key]['sha256']:
                raise ValueError(f'Image changed after preflight: {key}')
            start = time.perf_counter()
            self.rgb = pipeline.decode_image(image)
            decode_sec = time.perf_counter() - start
            self.rgb_key = key
        start = time.perf_counter()
        side = 1024 if name == STAGES[2] else 512
        array = pipeline.resize_image(self.rgb, side)
        if name == STAGES[1]:
            array = np.asarray(Image.fromarray(array).rotate(270, expand=True))
        ocr_start = time.perf_counter()
        detections = pipeline.paddle_to_common(self.engine.predict(
            array, **dict(self.manifest['predict'], text_det_limit_side_len=side)))
        ocr_end = time.perf_counter()
        parsed = evidence(detections)
        end = time.perf_counter()
        raw = {'image_id': key, 'stage': name, 'run_id': self.run_id,
               'detections': detections, 'decode_sec': decode_sec,
               'prepare_sec': ocr_start - start, 'ocr_sec': ocr_end - ocr_start,
               'stage_sec': ocr_end - start, 'evidence_sec': end - ocr_end}
        raw['record_sha256'] = digest(raw)
        with path.open('x', encoding='utf8') as stream:
            json.dump(raw, stream, ensure_ascii=False, allow_nan=False)
        self.fresh[name] += 1
        return {**raw, **parsed, 'source_sha256': sha(path)}


def outcome(key, row, stages):
    triggered = stages[0]['policy_B_trigger'] is True
    selected = stages[choose(stages, 'highest_q')] if triggered else stages[0]
    baseline, truth = row['final_date'], row['true_final_date']
    shadow = selected['prediction']['final_date']
    bc, sc = baseline == truth, shadow == truth
    category = ('unchanged_correct' if sc else 'regression') if bc else (
        'recovery' if sc else 'unchanged_wrong' if baseline == shadow else 'wrong_to_different_wrong')
    return {'image_id': key, 'truth': truth, 'baseline': baseline, 'shadow': shadow,
            'baseline_correct': bc, 'shadow_correct': sc, 'triggered': triggered,
            'selected_stage': selected['stage'], 'changed': baseline != shadow,
            'category': category, 'stages': stages}


def summarize(results, runner):
    n = len(results)
    categories = Counter(r['category'] for r in results)
    baseline = sum(r['baseline_correct'] for r in results)
    shadow = sum(r['shadow_correct'] for r in results)
    triggers = sum(r['triggered'] for r in results)
    timing = {}
    for name in STAGES:
        stages = [s for r in results for s in r['stages'] if s['stage'] == name]
        timing[name] = {'attempt_count': len(stages), **{
            field: sum(s[field] for s in stages)
            for field in ('ocr_sec', 'stage_sec', 'decode_sec', 'prepare_sec', 'evidence_sec')}}
    retry_ocr = sum(timing[s]['ocr_sec'] for s in STAGES[1:])
    retry_stage = sum(timing[s]['stage_sec'] for s in STAGES[1:])
    evidence_sec = sum(timing[s]['evidence_sec'] for s in STAGES)
    assert baseline + categories['recovery'] - categories['regression'] == shadow
    assert sum(categories.values()) == n
    assert sum(timing[s]['attempt_count'] for s in STAGES[1:]) == 2 * triggers
    return {
        'status': 'complete', 'cohort_size': n, 'baseline_correct': baseline,
        'baseline_accuracy': baseline / n, 'shadow_correct': shadow, 'shadow_accuracy': shadow / n,
        'net_correct_delta': shadow - baseline, 'accuracy_delta_percentage_points': 100 * (shadow - baseline) / n,
        **{c + '_count': categories[c] for c in ('recovery', 'regression', 'wrong_to_different_wrong', 'unchanged_correct', 'unchanged_wrong')},
        'recovery_ids': [r['image_id'] for r in results if r['category'] == 'recovery'],
        'regression_ids': [r['image_id'] for r in results if r['category'] == 'regression'],
        'wrong_to_different_wrong_ids': [r['image_id'] for r in results if r['category'] == 'wrong_to_different_wrong'],
        'trigger_count': triggers, 'trigger_rate': triggers / n,
        'triggered_baseline_correct_count': sum(r['triggered'] and r['baseline_correct'] for r in results),
        'triggered_baseline_wrong_count': sum(r['triggered'] and not r['baseline_correct'] for r in results),
        'retry_ocr_attempt_count': 2 * triggers, 'stage_timings': timing,
        'original_512_ocr_sec_for_trigger_measurement': timing[STAGES[0]]['ocr_sec'],
        'rotation_highres_added_ocr_sec': retry_ocr, 'rotation_highres_added_stage_sec': retry_stage,
        'fresh_attempts_this_invocation': dict(runner.fresh),
        'cached_attempts_this_invocation': sum(t['attempt_count'] for t in timing.values()) - sum(runner.fresh.values()),
        'initialization_sec_this_invocation': runner.initialization_sec,
        'runtime_projection_per_500_eligible_early_stops': {
            'expected_triggers': 500 * triggers / n,
            'expected_retry_attempts': 1000 * triggers / n,
            'added_retry_ocr_sec': 500 * retry_ocr / n,
            'added_retry_stage_sec': 500 * retry_stage / n,
            'evidence_analysis_sec': 500 * evidence_sec / n,
            'offline_original_ocr_measurement_sec': 500 * timing[STAGES[0]]['ocr_sec'] / n,
            'formula': '500 * observed trigger rate * mean summed rotation+highres cost per triggered image',
            'scope': 'Sequential two-thread CPU service time; not parallel wall time or a general 500-image population forecast'},
        'caveats': [
            'Independent of the specified 76 development IDs; original_512-only estimand, 47 later stops excluded.',
            'Only three baseline errors; recovery-rate precision is limited. Not a representative population accuracy estimate.',
            'q is selected-source-box recognition confidence, not date-correctness probability.',
            'Stage timings include preparation and OCR/conversion, exclude decode/evidence/init. Cached timings retain original measurements.',
            'Offline original OCR reconstruction is measurement cost, not incremental retry cost in a pipeline already running stage1.',
            'Two-phase validation decodes triggered images again; that diagnostic decode/I/O cost is not added production retry work.',
            'Official baseline used four CPU threads; candidate retains two. All original predictions must match before retries.',
        ],
    }


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def execute(output, manifest, rows):
    output.mkdir(parents=True, exist_ok=True)
    if not (output / 'manifest.json').exists():
        write(output / 'manifest.json', manifest)
    write_csv(output / 'cohort.csv', [{'image_id': k, 'file_name': r['file_name'],
        'official_method': r['method'], 'baseline': r['final_date'], 'truth': r['true_final_date']}
        for k, r in sorted(rows.items())])
    runner = StageRunner(output, manifest)
    originals = {}
    # Finish and verify ALL stage1 evidence before spending even one retry call.
    for i, key in enumerate(sorted(rows), 1):
        originals[key] = runner.get(key, STAGES[0])
        print(f'original {i}/{len(rows)} {key}', flush=True)
    check = baseline_check(rows, originals)
    write(output / 'baseline_check.json', check)
    if check['issues']:
        raise ValueError('Baseline reconstruction mismatch; stopped before retries. See baseline_check.json.')
    results = []
    for i, (key, row) in enumerate(sorted(rows.items()), 1):
        stages = [originals[key]]
        if stages[0]['policy_B_trigger'] is True:
            stages.extend(runner.get(key, name) for name in STAGES[1:])
        results.append(outcome(key, row, stages))
        print(f'shadow {i}/{len(rows)} {key} triggered={results[-1]["triggered"]}', flush=True)
    (output / 'results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in results), encoding='utf8')
    write_csv(output / 'comparisons.csv', [{k: v for k, v in r.items() if k != 'stages'} for r in results])
    summary = summarize(results, runner)
    write(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/frozen_b_shadow_independent_run1')
    parser.add_argument('--run-ocr', action='store_true', help='Explicitly enable bounded original measurement and triggered rotation/highres retries')
    args = parser.parse_args()
    manifest, rows, report = preflight(args.output)
    if not args.run_ocr:
        print(json.dumps(report, indent=2))
        print('No OCR imported/initialized; no files written. Shadow outcomes and timings pending.')
        return
    execute(args.output, manifest, rows)


if __name__ == '__main__':
    main()
