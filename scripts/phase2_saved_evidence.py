"""Food-700 and cosmetic-development-300 replay. Never opens images or holdout.

Usage: python scripts/phase2_saved_evidence.py OUTPUT_NAME [VARIANT] [--phase1-code]
Outputs are exclusive-created under docs/phase2_accuracy_20261001.
"""
import csv
import hashlib
import importlib.abc
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent
sys.path.insert(0, str(ROOT))
from validate_candidate_phase1 import NoOCR, sha
sys.meta_path.insert(0, NoOCR())
CODE = ROOT
if '--phase1-code' in sys.argv:
    sys.argv.remove('--phase1-code')
    CODE = ROOT / 'docs/phase2_accuracy_20261001/phase1_code'
    names = ['ocr_pipeline.py'] + subprocess.check_output(
        ['git', '-C', str(ROOT), 'ls-tree', '-r', '--name-only', '3cd68fd', '--', 'date_parser'],
        text=True).splitlines()
    for name in names:
        if not name.endswith('.py'):
            continue
        content = subprocess.check_output(['git', '-C', str(ROOT), 'show', '3cd68fd:' + name])
        target = CODE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == content
        else:
            target.write_bytes(content)
    sys.path.insert(0, str(CODE))
import ocr_pipeline as op
import date_parser.select as sel
import date_parser.extract as ext
from date_parser.types import TextBox

PROTECTED = {
    DATA / 'labels/review/labels_700_confirmed_gt.csv': '9edfeb145acea9981e4d59447b6df0c0d827f8eb7ad0c7499bc281cb4fdc7c9d',
    DATA / 'docs/full_stage_701_run1/raw_ocr.jsonl': '3c00358afc9281413e47520ddcb7784b0e16ced070d1d1b74a2734313df502a2',
    DATA / 'docs/ocr_clahe1024_full700_20260929_resumed1/raw.jsonl': '2bb77aed031910773cfdd8969a0b21ea8a8f8c50fa677d41a378be3572664a55',
}


def lines(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load():
    for p, h in PROTECTED.items():
        assert sha(p) == h
    food = {r['image_id'].zfill(6): {'gt': r['final_date'], 'stages': {}}
            for r in csv.DictReader(next(iter(PROTECTED)).open(encoding='utf-8-sig'))}
    assert len(food) == 700 and '002133' not in food
    for p in list(PROTECTED)[1:]:
        for r in lines(p):
            if r['image_id'] in food:
                food[r['image_id']]['stages'].setdefault(r['stage'], r['detections'])
    cosdir = ROOT / 'docs/cosmetics300_main_0dd7a35_20260930'
    cos = {r['image_id']: {'gt': r['label'], 'stages': {}} for r in csv.DictReader(
        (cosdir / 'cos300_0dd7a35_results.csv').open(encoding='utf-8-sig'))}
    assert set(cos) == {str(i) for i in range(900001, 900301)}
    for r in lines(cosdir / 'cos300_0dd7a35_allstages.jsonl'):
        assert r['image_id'] in cos
        cos[r['image_id']]['stages'] = {s['stage']: s['detections'] for s in r['stages']}
    assert all(len(r['stages']) == 5 for r in cos.values())
    return {'food': food, 'cosmetic': cos}


def replay(corpus):
    rows = {}
    for iid, r in corpus.items():
        stages = {s: op.stage_result(ds) for s, ds in r['stages'].items()}
        try:
            prediction, method, attempts = op.run_cascade(stages.__getitem__)
            rows[iid] = dict(gt=r['gt'], prediction=prediction['final_date'], method=method, attempts=attempts)
        except KeyError as e:
            rows[iid] = dict(gt=r['gt'], prediction=None, missing=str(e), attempts=[])
        detail = {}
        for s, ds in r['stages'].items():
            boxes = [TextBox.from_dict(d) for d in ds]
            candidates = sel.find_all_candidates(boxes)
            detail[s] = dict(prediction=stages[s][0]['final_date'], evidence=stages[s][1],
                candidates=[dict(prediction=c.result.final_date_string(), text=c.source_text,
                    alternatives=[a.date.final_date_string() for a in c.candidates]) for c in candidates],
                texts=[d['text'] for d in ds])
        rows[iid]['stages'] = detail
    return dict(correct=sum(r['gt'] == r['prediction'] for r in rows.values()),
        calls=dict(Counter(s for r in rows.values() for s in r['attempts'])),
        unknown=[k for k,r in rows.items() if r['prediction'] is None], rows=rows)


def variants(name):
    if '+' in name:
        for part in name.split('+'):
            variants(part)
    elif name == 'year_month_word':
        ext._PATTERN_DEFS.insert(0, (re.compile(
            rf'(?<![0-9A-Za-z])(20[0-9]{{2}}){ext._OPTIONAL_SEP}({ext._MONTH_RE}){ext._OPTIONAL_SEP}([0-9]{{1,2}})(?![0-9A-Za-z])', re.I),
            ('num', 'month_name', 'num'), ('year', 'month', 'day'), ('year', 'month', 'day')))
    elif name == 'mixed_comma':
        for separators in [(',', r'\.'), (r'\.', ',')]:
            ext._PATTERN_DEFS.insert(0, (re.compile(
                r'(?<![0-9A-Za-z])([0-9]{4})\s*' + separators[0] + r'\s*([0-9]{1,2})\s*' + separators[1] + r'\s*([0-9]{1,3})(?![0-9])'),
                ('num', 'num', 'num'), ('year', 'month', 'day'), None))
    elif name == 'split_until':
        sel._UNTIL_FRAGMENT_RE = re.compile(r'(?:까[\s-]+지|[가-힣]?지)')
    elif name == 'preserve_complete':
        previous = op.prefer_retry
        def prefer(original, retry):
            old = original.get('selected_date', 'NONE').split('-')
            new = (retry or {}).get('selected_date', 'NONE').split('-')
            if len(old) == len(new) == 3 and 'NONE' not in old and 'NONE' in new:
                if all(b == 'NONE' or a == b for a, b in zip(old, new)):
                    return False
            return previous(original, retry)
        op.prefer_retry = prefer
    elif name == 'retry_partial':
        previous = op.retry_triggered
        op.retry_triggered = lambda e: previous(e) or 'NONE' in e.get('selected_date', '').split('-')
    elif name == 'retry_damaged_day':
        previous_stage = op.stage_result
        def stage(ds):
            pred, ev = previous_stage(ds)
            if ev is not None and pred['day'] == 'NONE' and pred['year'] != 'NONE' and pred['month'] != 'NONE':
                selected = sel.select_final_date([TextBox.from_dict(d) for d in ds])
                for token in ext.extract_date_tokens(selected.source_text):
                    fs = [ext.normalize_confusable(f.raw) for f in token.fields]
                    if len(fs) == 3 and all(f.isdigit() for f in fs) and len(fs[0]) == 4:
                        if int(fs[0]) == int(pred['year']) and int(fs[1]) == int(pred['month']):
                            ev['damaged_day'] = True
            return pred, ev
        op.stage_result = stage
        previous = op.retry_triggered
        op.retry_triggered = lambda e: previous(e) or e.get('damaged_day', False)
    elif name == 'retry_bare_date':
        previous_stage = op.stage_result
        def stage(ds):
            pred, ev = previous_stage(ds)
            if ev is not None:
                selected = sel.select_final_date([TextBox.from_dict(d) for d in ds])
                ev['bare_date'] = bool(re.fullmatch(r'[0-9\s./-]+', selected.source_text))
            return pred, ev
        op.stage_result = stage
        previous = op.retry_triggered
        op.retry_triggered = lambda e: previous(e) or e.get('bare_date', False)
    elif name in {'clahe_after_empty_retry', 'clahe_after_empty_anchored_retry'}:
        anchored = name == 'clahe_after_empty_anchored_retry'
        previous = op._run_baseline_cascade
        def cascade(run):
            seen = {}
            def wrapped(stage):
                seen[stage] = run(stage)
                return seen[stage]
            pred, method, attempts = previous(wrapped)
            original = seen['original_512'][1]
            if (original is not None and op.uncertain(original)
                    and (not anchored or original.get('self_anchor', False))
                    and 'highres_1024' in seen and seen['highres_1024'][1] is None):
                retry, ev = run('clahe')
                attempts = attempts + ['clahe']
                if op.prefer_retry(original, ev) and (not anchored or ev.get('self_anchor', False)):
                    return retry, 'clahe_retry', attempts
            return pred, method, attempts
        op._run_baseline_cascade = cascade
    elif name == 'no_period':
        sel._period_from_manufacture = lambda *args: None
    elif name == 'printed_period_base':
        original = sel._period_from_manufacture
        def base(boxes, positioned):
            return positioned[0] if original(boxes, positioned) is not None else None
        sel._period_from_manufacture = base
    elif name == 'no_six':
        sel._six_digit_candidates = lambda *args: []
    elif name != 'baseline':
        raise ValueError(name)


def main():
    name = sys.argv[1]
    variant = sys.argv[2] if len(sys.argv) > 2 else 'baseline'
    if variant != 'baseline' and CODE == ROOT:
        raise ValueError('Use --phase1-code for experiments; current code already includes accepted changes')
    variants(variant)
    result = {name: replay(corpus) for name, corpus in load().items()}
    for p, h in PROTECTED.items():
        assert sha(p) == h
    out = ROOT / 'docs/phase2_accuracy_20261001'
    out.mkdir(exist_ok=True)
    with (out / (name + '.json')).open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    inputs = dict(PROTECTED)
    for filename in ('cos300_0dd7a35_results.csv', 'cos300_0dd7a35_allstages.jsonl'):
        path = ROOT / 'docs/cosmetics300_main_0dd7a35_20260930' / filename
        inputs[path] = sha(path)
    manifest = dict(code_root=str(CODE), variant=sys.argv[2] if len(sys.argv)>2 else 'baseline',
        code_sha256={str(p.relative_to(CODE)):sha(p) for p in [CODE/'ocr_pipeline.py', *sorted((CODE/'date_parser').glob('*.py'))]},
        input_sha256={str(p):h for p,h in inputs.items()}, fresh_ocr_calls=0)
    with (out / (name + '.manifest.json')).open('x', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    for corpus, data in result.items():
        print(corpus, json.dumps({k: v for k, v in data.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
