"""Frozen nine-control retry diagnostic. Default is read-only, no-OCR preflight."""
from __future__ import annotations
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from diagnose_early_stop_26 import ROOT, read_csv, sha
from diagnose_correct_controls_50 import evidence, FROZEN_IDS, POLICIES

IDS = ('000060', '000384', '000447', '000690', '000954', '001185', '002370', '002835', '003238')
A_IDS = ('000060', '000384', '000447', '002370', '002835')
STAGES = ('original_512', 'rotation_270', 'highres_1024')
RULES = ('keep_stage1', 'retry_agreement', 'highest_q', 'unique_evidence_higher_q', 'latest_highres_reference')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def enriched(path, key, stage):
    raw = read(path)
    if raw['stage'] != stage or raw.get('image_id', key) != key:
        raise ValueError(f'Wrong cache identity: {path}')
    return {**raw, **evidence(raw['detections']), 'source_path': str(path.resolve()),
            'source_sha256': sha(path)}


def choose(stages, rule):
    """No truth or ID input. Ties/missing confidence preserve the earlier stage."""
    s, r, h = stages
    date = lambda x: x['prediction']['final_date']
    if rule == 'keep_stage1':
        return 0
    if rule == 'latest_highres_reference':
        return 2
    if rule == 'retry_agreement':
        return 1 if r['has_candidate'] and h['has_candidate'] and date(r) == date(h) else 0
    eligible = [0]
    for i in (1, 2):
        x = stages[i]
        if not x['has_candidate'] or x['q'] is None:
            continue
        if rule == 'unique_evidence_higher_q' and (
                x['distinct_candidate_count'] != 1 or s['q'] is None):
            continue
        eligible.append(i)
    if rule not in ('highest_q', 'unique_evidence_higher_q'):
        raise ValueError(rule)
    return max(eligible, key=lambda i: stages[i]['q'] if stages[i]['q'] is not None else -1)


def case(key, stages, truth, cohort):
    dates = [s['prediction']['final_date'] for s in stages]
    return {'image_id': key, 'cohort': cohort, 'truth': truth, 'stages': stages,
            'stage1': dates[0], 'rotation': dates[1], 'highres': dates[2],
            'rotation_matches_truth': dates[1] == truth['final_date'],
            'highres_matches_truth': dates[2] == truth['final_date'],
            'stage1_eq_rotation': dates[0] == dates[1], 'stage1_eq_highres': dates[0] == dates[2],
            'rotation_eq_highres': dates[1] == dates[2]}


def analysis(rows):
    result = {}
    for cohort in ('recoverable_failures_10', 'other_failures_16', 'triggered_correct_controls_9'):
        group = [r for r in rows if r['cohort'] == cohort]
        result[cohort] = {'count': len(group), 'rules': {}}
        for gate in ('forced', 'A', 'B'):
            for rule in RULES:
                outcomes = []
                for row in group:
                    stages = row['stages']
                    triggered = gate == 'forced' or stages[0][f'policy_{gate}_trigger'] is True
                    index = choose(stages, rule) if triggered else 0
                    original = row['stage1']
                    selected = stages[index]['prediction']['final_date']
                    truth = row['truth']['final_date']
                    outcomes.append({'image_id': row['image_id'], 'triggered': triggered,
                        'selected_stage': STAGES[index], 'selected': selected,
                        'correct': selected == truth, 'changed': selected != original,
                        'recovery': original != truth and selected == truth,
                        'regression': original == truth and selected != truth})
                result[cohort]['rules'][gate + '/' + rule] = {
                    k + '_count': sum(o[k] for o in outcomes)
                    for k in ('triggered', 'correct', 'changed', 'recovery', 'regression')}
                result[cohort]['rules'][gate + '/' + rule]['outcomes'] = outcomes
    return result


def preflight(args):
    cm = read(args.controls / 'manifest.json')
    fm = read(args.failures / 'manifest.json')
    saved = [json.loads(line) for line in (args.controls / 'results.jsonl').read_text(encoding='utf8').splitlines()]
    if len(saved) != 50 or {r['image_id'] for r in saved} != set(FROZEN_IDS):
        raise ValueError('Expected frozen 50 cached results')
    for policy, expected in (('A', A_IDS), ('B', IDS)):
        if tuple(sorted(r['image_id'] for r in saved if r[f'policy_{policy}_trigger'] is True)) != expected:
            raise ValueError('Frozen trigger membership changed')
    sources = [args.controls / n for n in ('manifest.json', 'results.jsonl', 'summary.json')]
    sources += [args.failures / n for n in ('manifest.json', 'cases.csv', 'summary.json')]
    # Reject parser/model drift before replaying or extending historical evidence.
    for manifest in (cm, fm):
        for name, digest in manifest['code_sha256'].items():
            if name.startswith('date_parser/') or name == 'ocr_pipeline.py':
                if sha(ROOT / name) != digest:
                    raise ValueError(f'Historical code drift: {name}')
        for name, digest in manifest['weights_sha256'].items():
            if sha(ROOT / name) != digest:
                raise ValueError(f'Historical weight drift: {name}')
        for name, version in manifest['packages'].items():
            if importlib.metadata.version(name) != version:
                raise ValueError(f'Historical package drift: {name}')
    controls = {}
    images = {}
    for row in saved:
        key = row['image_id']
        path = args.controls / f'{key}_original_512.json'
        stage = enriched(path, key, STAGES[0])
        sources.append(path)
        for field in ('prediction', 'q', 'M', 'policy_A_trigger', 'policy_B_trigger'):
            if stage[field] != row[field]:
                raise ValueError(f'Cached evidence drift: {key}/{field}')
        if stage['prediction'] != row['truth']:
            raise ValueError(f'Control no longer correct: {key}')
        if key in IDS:
            controls[key] = (stage, row['truth'])
            image = Path(cm['images'][key]['path'])
            if sha(image) != cm['images'][key]['sha256']:
                raise ValueError(f'Image drift: {key}')
            images[key] = image
    failures = read_csv(args.failures / 'cases.csv')
    if len(failures) != 26 or len({r['image_id'] for r in failures}) != 26:
        raise ValueError('Expected 26 unique cached failures')
    if {r['image_id'] for r in failures} & set(FROZEN_IDS):
        raise ValueError('Failure/control overlap')
    rows = []
    for row in failures:
        key = row['image_id']
        stages = []
        for name in STAGES:
            path = args.failures / f'{key}_{name}.json'
            stage = enriched(path, key, name)
            sources.append(path)
            if stage['prediction']['final_date'] != row[name]:
                raise ValueError(f'Failure prediction drift: {key}/{name}')
            stages.append(stage)
        recoverable = row['oracle_outcome'] == 'fix'
        cohort = 'recoverable_failures_10' if recoverable else 'other_failures_16'
        truth_date = row['truth_final_date']
        if truth_date == stages[0]['prediction']['final_date'] or row['normal_stopping_stage'] != STAGES[0]:
            raise ValueError('Expected incorrect stage1 stops')
        parts = truth_date.split('-') if truth_date != 'NONE' else ['NONE'] * 3
        truth = dict(zip(('year', 'month', 'day'), parts), final_date=truth_date)
        rows.append(case(key, stages, truth, cohort))
    if sum(r['cohort'] == 'recoverable_failures_10' for r in rows) != 10:
        raise ValueError('Expected ten recoverable failures')
    code = [Path(__file__), ROOT / 'scripts/diagnose_correct_controls_50.py',
            ROOT / 'scripts/diagnose_early_stop_26.py']
    manifest = {'experiment': 'frozen_triggered_correct_controls_9', 'ids': IDS,
        'new_ocr_stages': STAGES[1:], 'max_new_attempts': 18, 'policies': POLICIES,
        'rules': RULES, 'source_sha256': {str(p.resolve()): sha(p) for p in sources},
        'code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in code},
        'engine': cm['engine'], 'packages': cm['packages'], 'python': sys.version,
        'images': {k: cm['images'][k] for k in IDS},
        'predict': cm['predict'], 'weights_sha256': cm['weights_sha256']}
    manifest = json.loads(json.dumps(manifest))
    if args.output.exists() and (not (args.output / 'manifest.json').is_file()
            or read(args.output / 'manifest.json') != manifest):
        raise ValueError('Output provenance mismatch; choose a new output directory')
    # Validate any resumed records before engine initialization.
    for key in IDS:
        for stage in STAGES[1:]:
            path = args.output / f'{key}_{stage}.json'
            if path.exists():
                enriched(path, key, stage)
    return manifest, controls, images, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--controls', type=Path, default=ROOT / 'docs/correct_controls_50_run1')
    ap.add_argument('--failures', type=Path, default=ROOT / 'docs/early_stop_26_run1')
    ap.add_argument('--output', type=Path, default=ROOT / 'docs/triggered_correct_controls_9_run1')
    ap.add_argument('--run-ocr', action='store_true', help='Enable only 18 bounded retry attempts; resume caches')
    args = ap.parse_args()
    manifest, controls, images, rows = preflight(args)
    if not args.run_ocr:
        print(json.dumps({'preflight': 'passed', 'new_ocr_attempts': 0, 'planned_ids': IDS,
            'planned_stages': STAGES[1:], 'max_attempts': 18, 'cached_failure_analysis': analysis(rows)}, indent=2))
        print('No OCR initialized; no output files written. Control retry outcomes pending.')
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if not (args.output / 'manifest.json').exists():
        write(args.output / 'manifest.json', manifest)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(manifest['engine']['cpu_threads'])
    import numpy as np
    from PIL import Image
    import ocr_pipeline as pipeline
    engine = None
    fresh = 0
    init_sec = 0.0
    for key in IDS:
        original, truth = controls[key]
        stages = [original]
        rgb = None
        for name in STAGES[1:]:
            path = args.output / f'{key}_{name}.json'
            reused = path.exists()
            if not reused:
                if engine is None:
                    start = time.perf_counter()
                    engine = pipeline.initialize_engine(**manifest['engine'])
                    init_sec = time.perf_counter() - start
                decode_sec = 0.0
                if rgb is None:
                    start = time.perf_counter()
                    rgb = pipeline.decode_image(images[key])
                    decode_sec = time.perf_counter() - start
                start = time.perf_counter()
                if name == 'rotation_270':
                    array = np.asarray(Image.fromarray(pipeline.resize_image(rgb, 512)).rotate(270, expand=True))
                else:
                    array = pipeline.resize_image(rgb, 1024)
                ocr_start = time.perf_counter()
                predict = dict(manifest['predict'], text_det_limit_side_len=512 if name == 'rotation_270' else 1024)
                detections = pipeline.paddle_to_common(engine.predict(array, **predict))
                end = time.perf_counter()
                record = {'image_id': key, 'stage': name, 'detections': detections,
                    'decode_sec': decode_sec, 'prepare_sec': ocr_start - start,
                    'ocr_sec': end - ocr_start, 'stage_sec': end - start}
                with path.open('x', encoding='utf8') as stream:
                    json.dump(record, stream, ensure_ascii=False, indent=2)
                fresh += 1
            stages.append({**enriched(path, key, name), 'reused': reused})
        rows.append(case(key, stages, truth, 'triggered_correct_controls_9'))
        print(f'[{len(rows)-26}/9] {key}: retries preserved', flush=True)
    with (args.output / 'results.jsonl').open('w', encoding='utf8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    comparisons = [{k: v for k, v in r.items() if k != 'stages'} for r in rows[-9:]]
    for row in comparisons:
        row['truth'] = row['truth']['final_date']
    with (args.output / 'comparisons.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    write(args.output / 'acceptance_analysis.json', analysis(rows))
    write(args.output / 'summary.json', {'control_count': 9, 'cached_failure_count': 26,
        'fresh_attempts_this_invocation': fresh, 'cached_retry_attempts': 18 - fresh,
        'initialization_sec_this_invocation': init_sec,
        'retry_ocr_sec': sum(s['ocr_sec'] for r in rows[-9:] for s in r['stages'][1:]),
        'retry_stage_sec': sum(s['stage_sec'] for r in rows[-9:] for s in r['stages'][1:]),
        'caveat': 'Selected diagnostic cohorts, not population accuracy. Truth evaluates only; never selects. '
                  'q is source-box recognition confidence, not date correctness probability. '
                  'Timings retain cached measurements; stage excludes decode, parser, init. '
                  'Latest-highres is a reference, not endorsed production behavior.'})


if __name__ == '__main__':
    main()
