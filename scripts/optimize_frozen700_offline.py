"""Frozen-GT-only diagnostic replay; never initializes OCR or edits inputs."""
import csv
import hashlib
import importlib.abc
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class NoFreshOCR(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'paddle', 'paddleocr', 'paddlex', 'cv2'}:
            raise AssertionError('Fresh OCR/preprocessing is forbidden in this replay')


sys.meta_path.insert(0, NoFreshOCR())
import ocr_pipeline as p
from date_parser.select import find_all_candidates
from date_parser.types import TextBox


def main():
    out = ROOT / sys.argv[1]
    out.mkdir(exist_ok=False, parents=True)
    protected = ['labels/review/labels_700_confirmed_gt.csv',
                 'labels/review/label_review_final.csv', 'docs/full_stage_701_run1/raw_ocr.jsonl']
    hashes = {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in protected}
    assert hashes[protected[0]] == '9edfeb145acea9981e4d59447b6df0c0d827f8eb7ad0c7499bc281cb4fdc7c9d'
    assert hashes[protected[2]] == '3c00358afc9281413e47520ddcb7784b0e16ced070d1d1b74a2734313df502a2'
    truth = {r['image_id'].zfill(6): r['final_date'] for r in csv.DictReader(
        (ROOT/protected[0]).open(encoding='utf-8-sig'))}
    assert len(truth) == 700 and '002133' not in truth
    raw = {}
    for line in (ROOT/protected[2]).read_text(encoding='utf8').splitlines():
        r = json.loads(line)
        assert r['stage'] not in raw.get(r['image_id'], {})
        raw.setdefault(r['image_id'], {})[r['stage']] = r['detections']
    rows = {}
    for key in sorted(truth):
        assert set(raw[key]) == set(p.STAGES)
        stages = {s: p.stage_result(raw[key][s]) for s in p.STAGES}
        pred, method, attempts = p.run_cascade(stages.__getitem__)
        detail = {}
        for s in p.STAGES:
            cs = find_all_candidates([TextBox.from_dict(d) for d in raw[key][s]])
            detail[s] = dict(prediction=stages[s][0]['final_date'], evidence=stages[s][1],
                candidates=[dict(date=c.result.final_date_string(), text=c.source_text,
                    alternatives=[x.date.final_date_string() for x in c.candidates]) for c in cs],
                texts=[d['text'] for d in raw[key][s]])
        rows[key] = dict(gt=truth[key], prediction=pred['final_date'], method=method,
                         attempts=attempts, stages=detail)
    errors = {k:r for k,r in rows.items() if r['gt'] != r['prediction']}
    summary = dict(n=700, correct=700-len(errors), none=sum(r['prediction']=='NONE' for r in errors.values()),
        stage_oracle_errors=sum(any(s['prediction']==r['gt'] for s in r['stages'].values()) for r in errors.values()),
        candidate_oracle_errors=sum(any(c['date']==r['gt'] for s in r['stages'].values() for c in s['candidates']) for r in errors.values()),
        attempts=dict(Counter(s for r in rows.values() for s in r['attempts'])), hashes=hashes, ocr_invocations=0)
    if len(sys.argv)>2:
        old=json.loads((ROOT/sys.argv[2]/'rows.json').read_text(encoding='utf8'))
        summary['old_correct']=sum(r['gt']==r['prediction'] for r in old.values())
        summary['changes']=[dict(image_id=k, gt=r['gt'], old=old[k]['prediction'], new=r['prediction'],
            old_method=old[k]['method'], new_method=r['method'],
            outcome='gain' if r['prediction']==r['gt'] else 'regression' if old[k]['prediction']==r['gt'] else 'changed-wrong')
            for k,r in rows.items() if r['prediction']!=old[k]['prediction']]
    for name,data in [('rows',rows),('errors',errors),('summary',summary)]:
        (out/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    assert hashes == {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in protected}
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
