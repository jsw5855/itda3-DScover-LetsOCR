"""Capture old parser, then compare a parser-only edit using immutable saved OCR.

Run --capture before editing. Run --compare only after parser tests pass.
No OCR imports or inference; historical analysis artifacts remain untouched.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import analyze_full_stage_offline as a
from scripts import diagnose_oracle_gap_offline as gap

POLICIES={'current_cascade':'current_cascade', 'frozen_B_highest_q':gap.FROZEN,
          'previous_best269':gap.BEST, 'agreement_no_anchor':'probe_no_anchor_agreement',
          'agreement_all':'probe_all_agreement'}
DUMP=ROOT/'docs/full_stage_701_run1'
PROTECTED=['ocr_pipeline.py','predict.ipynb','submission_runtime.py']

def hashes(paths):
    return {str(p.relative_to(ROOT)):a.sha(p) for p in paths}

def snapshot(raw):
    start=time.perf_counter()
    stages={k:[{**a.replay(g[s]['detections']),'ocr_sec':g[s]['ocr_sec']} for s in a.STAGES] for k,g in raw.items()}
    return stages,time.perf_counter()-start

def run_policy(ss,name):
    rule=POLICIES[name]
    return gap.candidate_policy(ss,rule) if name.startswith('agreement') else a.policy(ss,rule)

def pred(ss,i):
    return ss[i]['prediction']['final_date']

def capture(out):
    raw,labels,official,check=a.preflight(DUMP)
    stages,seconds=snapshot(raw)
    for k,r in official.items():
        i,_,method=a.cascade(stages[k])
        assert stages[k][i]['prediction']=={f:r[f] for f in ('year','month','day','final_date')}
        assert method==r['method']
    shadow=a.read(ROOT/'docs/frozen_b_shadow_independent_run1/manifest.json')['cohort']['cohort_ids']
    cohorts={'official_300':{k:r['true_final_date'] for k,r in official.items()},
             'approved_69':{k:a.normalize_truth(r)['final_date'] for k,r in labels.items() if r['truth_source']=='approved'},
             'historical_independent_195_reused':{k:official[k]['true_final_date'] for k in shadow},
             'all_701_provisional':{k:a.normalize_truth(r)['final_date'] for k,r in labels.items()}}
    out.mkdir(parents=True,exist_ok=False)
    a.write_json(out/'old_parser_replay.json',dict(stages=stages,cohorts=cohorts,labels=labels,parser_replay_seconds=seconds))
    sources=[DUMP/n for n in ('raw_ocr.jsonl','manifest.json','integrity.json','labels_metadata.json','timing_summary.json','summary.json')]
    sources += [a.OFFICIAL/'evaluation_300.csv',a.OFFICIAL/'manifest.json',ROOT/'docs/frozen_b_shadow_independent_run1/manifest.json',ROOT/'scripts/analyze_full_stage_offline.py',ROOT/'scripts/diagnose_oracle_gap_offline.py',ROOT/'scripts/evaluation_common.py']
    a.write_json(out/'manifest.json',dict(source_sha256=hashes(sources),old_parser_sha256=hashes(sorted((ROOT/'date_parser').glob('*.py'))),protected_sha256=hashes([ROOT/p for p in PROTECTED]),old_replay_sha256=a.sha(out/'old_parser_replay.json'),preflight=check,python=sys.version))
    print('Captured unchanged parser; official predictions/methods exactly match 263/300.')

def compare(out):
    manifest=a.read(out/'manifest.json')
    for name,h in {**manifest['source_sha256'],**manifest['protected_sha256']}.items():
        if a.sha(ROOT/name)!=h: raise ValueError('Protected input drift: '+name)
    if a.sha(out/'old_parser_replay.json')!=manifest['old_replay_sha256']:
        raise ValueError('Old replay changed')
    if (out/'summary.json').exists(): raise ValueError('Comparison already exists; do not overwrite')
    old=a.read(out/'old_parser_replay.json')
    m=a.read(DUMP/'manifest.json')
    raw=a.group_records([a.loads(line) for line in (DUMP/'raw_ocr.jsonl').read_text(encoding='utf8').splitlines() if line.strip()],m['image_ids'],a.digest(m),m['stages'])
    new,seconds=snapshot(raw)
    metrics,changed,stage_changed=[],[],[]
    for k,ss in new.items():
        for i,s in enumerate(ss):
            prev=old['stages'][k][i]
            if s!=prev:
                stage_changed.append(dict(image_id=k,stage=a.STAGES[i],old_prediction=pred(old['stages'][k],i),new_prediction=pred(ss,i),old_signals=json.dumps(prev),new_signals=json.dumps(s),label_status=old['labels'][k]['truth_source']))
    for cohort,truths in old['cohorts'].items():
        for name in POLICIES:
            vs_old,vs_new,vs_same=[],[],[]
            counts,old_counts=Counter(),Counter()
            cost=old_cost=0.
            retry=0
            gained,lost=[],[]
            for k,truth in truths.items():
                ss,prev=new[k],old['stages'][k]
                i,attempts=run_policy(ss,name)
                old_i,old_attempts=run_policy(prev,name)
                old_base=a.cascade(prev)[0]
                new_base=a.cascade(ss)[0]
                correct=pred(ss,i)==truth
                vs_old.append(dict(correct=correct,baseline_correct=pred(prev,old_base)==truth))
                vs_new.append(dict(correct=correct,baseline_correct=pred(ss,new_base)==truth))
                vs_same.append(dict(correct=correct,baseline_correct=pred(prev,old_i)==truth))
                if correct and not vs_old[-1]['baseline_correct']: gained.append(k)
                if not correct and vs_old[-1]['baseline_correct']: lost.append(k)
                counts.update(a.STAGES[j] for j in attempts)
                old_counts.update(a.STAGES[j] for j in old_attempts)
                cost+=sum(ss[j]['ocr_sec'] for j in attempts)
                old_cost+=sum(prev[j]['ocr_sec'] for j in old_attempts)
                retry+=len(attempts)>1
                if pred(prev,old_i)!=pred(ss,i) or old_attempts!=attempts or old_i!=i:
                    changed.append(dict(cohort=cohort,policy=name,image_id=k,label_status=old['labels'][k]['truth_source'],truth=truth,old_prediction=pred(prev,old_i),new_prediction=pred(ss,i),old_correct=pred(prev,old_i)==truth,new_correct=correct,old_selected_stage=a.STAGES[old_i],new_selected_stage=a.STAGES[i],old_attempts='|'.join(a.STAGES[j] for j in old_attempts),new_attempts='|'.join(a.STAGES[j] for j in attempts)))
            row=dict(cohort=cohort,policy=name,n=len(truths),old_same_policy_correct=sum(x['baseline_correct'] for x in vs_same),**a.accounting(vs_old),
                     **{'vs_new_baseline_'+f:v for f,v in a.accounting(vs_new).items() if f!='correct'},
                     **{'vs_old_same_policy_'+f:v for f,v in a.accounting(vs_same).items() if f!='correct'},
                     gains_vs_old_baseline_ids='|'.join(gained),regressions_vs_old_baseline_ids='|'.join(lost),retry_images=retry,
                     **{s:counts[s] for s in a.STAGES},**{'attempt_delta_'+s:counts[s]-old_counts[s] for s in a.STAGES},
                     ocr_service_seconds=cost,ocr_service_delta_vs_old_same_policy=cost-old_cost,projected_500_ocr_seconds=cost*500/len(truths))
            metrics.append(row)
    oracles={}
    for cohort,truths in old['cohorts'].items():
        before={k for k,t in truths.items() if any(pred(old['stages'][k],i)==t for i in range(4))}
        after={k for k,t in truths.items() if any(pred(new[k],i)==t for i in range(4))}
        oracles[cohort]=dict(old=len(before),new=len(after),gained_ids=sorted(after-before),lost_ids=sorted(before-after))
    a.write_csv(out/'policy_comparison.csv',metrics)
    a.write_csv(out/'changed_cases.csv',changed)
    a.write_csv(out/'changed_stages.csv',stage_changed)
    a.write_json(out/'new_parser_replay.json',dict(stages=new,parser_replay_seconds=seconds))
    summary=dict(baselines=[r for r in metrics if r['policy']=='current_cascade'],policy_metrics=metrics,oracles=oracles,changed_stage_count=len(stage_changed),ocr_invocations=0,
                 runtime_caveat='Selected measured OCR-call elapsed times only, including outliers; not target-machine end-to-end runtime. Parser replay timing includes diagnostic signal extraction.',
                 label_caveat='Approved status follows immutable metadata; official labels unchanged. All 701 provisional/noisy, 632 not fully verified. Historical 195 reused, not new validation.')
    a.write_json(out/'summary.json',summary)
    manifest.update(new_parser_sha256=hashes(sorted((ROOT/'date_parser').glob('*.py'))),replay_tool_sha256=a.sha(Path(__file__)),raw_sha256_after=a.sha(DUMP/'raw_ocr.jsonl'))
    a.write_json(out/'manifest.json',manifest)
    print(json.dumps({'baselines':summary['baselines'],'oracles':oracles,'changed_stages':len(stage_changed)},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'docs/parser_optimization1')
    action=ap.add_mutually_exclusive_group(required=True)
    action.add_argument('--capture',action='store_true')
    action.add_argument('--compare',action='store_true')
    args=ap.parse_args()
    (capture if args.capture else compare)(args.output)
