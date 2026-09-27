"""Offline, label-blind boundary probes against accepted Parser Optimization 1.

--probe runs before production edits; --compare verifies the chosen parser after tests.
All inputs are read-only. Runtime monkeypatches are confined to probe processes.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from collections import Counter
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import analyze_full_stage_offline as a
from scripts import replay_parser_optimization as replay
from date_parser import extract as ex
from date_parser import select as sel

COMMA=(re.compile(r'(?<![0-9A-Za-z])([0-9]{4})\s*,\s*([0-9]{1,2})\s*,\s*([0-9]{1,3})(?![0-9])'),('num','num','num'),('year','month','day'),None)
HOUR=re.compile(r'\A\s*(0[1-9]|1[0-2])\.\s*(0[1-9]|[12][0-9]|3[01])([01][0-9]|2[0-3])시')
MONTH=re.compile(r'\A\s*[-.(\[]*('+ '|'.join(ex.MONTH_NAMES) +r')[A-Za-z]([0-9]{4})[.)\]]*\s*\Z',re.IGNORECASE)
PROBES=('comma_year','yearless_hour','hint_month_separator','combined')

@contextmanager
def probe(name):
    patterns=ex._PATTERN_DEFS
    old_md=sel.extract_yearless_month_day_tokens
    old_my=sel.extract_month_yy_tokens
    if name in ('comma_year','combined'):
        ex._PATTERN_DEFS=[COMMA]+patterns
    if name in ('yearless_hour','combined'):
        sel.extract_yearless_month_day_tokens=lambda text:old_md(HOUR.sub(r'\1.\2 \3시',text))
    if name in ('hint_month_separator','combined'):
        def month(text,taken):
            tokens=old_my(text,taken)
            match=MONTH.fullmatch(text)
            if match and not ex._overlaps(match.span(),taken):
                tokens.append(ex.RawDateToken(match.span(),(ex.RawField(match[1],'month_name'),ex.RawField(match[2],'num')),('month','year'),('month','year')))
            return tokens
        sel.extract_month_yy_tokens=month
    try: yield
    finally:
        ex._PATTERN_DEFS=patterns
        sel.extract_yearless_month_day_tokens=old_md
        sel.extract_month_yy_tokens=old_my

def load_inputs():
    root=ROOT/'docs/parser_optimization1'
    m=a.read(root/'manifest.json')
    for name,h in {**m['source_sha256'],**m['protected_sha256']}.items():
        if a.sha(ROOT/name)!=h: raise ValueError('Input drift: '+name)
    if a.sha(root/'new_parser_replay.json')!=m['output_sha256']['new_parser_replay.json']:
        raise ValueError('Accepted replay snapshot changed')
    saved=a.read(root/'new_parser_replay.json')
    context=a.read(root/'old_parser_replay.json')
    dm=a.read(replay.DUMP/'manifest.json')
    raw=a.group_records([a.loads(line) for line in (replay.DUMP/'raw_ocr.jsonl').read_text(encoding='utf8').splitlines() if line.strip()],dm['image_ids'],a.digest(dm),dm['stages'])
    return raw,saved['stages'],context['cohorts'],m

def metrics(old,new,cohorts,policies,variant):
    rows,changes=[],[]
    for cohort,truths in cohorts.items():
        for policy in policies:
            outcomes,same,newbase=[],[],[]
            counts,oldcounts=Counter(),Counter()
            seconds=0.
            for k,t in truths.items():
                ss,prev=new[k],old[k]
                i,att=replay.run_policy(ss,policy)
                j,oldatt=replay.run_policy(prev,policy)
                baseline=a.cascade(prev)[0]
                nb=a.cascade(ss)[0]
                p,q=replay.pred(ss,i),replay.pred(prev,j)
                outcomes.append(dict(correct=p==t,baseline_correct=replay.pred(prev,baseline)==t))
                same.append(dict(correct=p==t,baseline_correct=q==t))
                newbase.append(dict(correct=p==t,baseline_correct=replay.pred(ss,nb)==t))
                counts.update(a.STAGES[x] for x in att)
                oldcounts.update(a.STAGES[x] for x in oldatt)
                seconds+=sum(ss[x]['ocr_sec'] for x in att)
                if p!=q or att!=oldatt or i!=j:
                    changes.append(dict(variant=variant,cohort=cohort,policy=policy,image_id=k,truth=t,old_prediction=q,new_prediction=p,old_correct=q==t,new_correct=p==t,old_attempts='|'.join(a.STAGES[x] for x in oldatt),new_attempts='|'.join(a.STAGES[x] for x in att)))
            rows.append(dict(variant=variant,cohort=cohort,policy=policy,n=len(truths),old_correct=sum(x['baseline_correct'] for x in outcomes),**a.accounting(outcomes),
                             **{'vs_same_policy_'+k:v for k,v in a.accounting(same).items() if k!='correct'},
                             **{'vs_new_baseline_'+k:v for k,v in a.accounting(newbase).items() if k!='correct'},
                             **{s:counts[s] for s in a.STAGES},
                             **{'attempt_delta_'+s:counts[s]-oldcounts[s] for s in a.STAGES},
                             projected_500_ocr_seconds=seconds*500/len(truths)))
    return rows,changes

def oracle(stages,cohorts):
    return {c:[k for k,t in truths.items() if any(replay.pred(stages[k],i)==t for i in range(4))] for c,truths in cohorts.items()}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'docs/parser_optimization2')
    ap.add_argument('--compare',action='store_true')
    args=ap.parse_args(); out=args.output
    raw,old,cohorts,m=load_inputs()
    current,elapsed=replay.snapshot(raw)
    if not args.compare:
        if current!=old: raise ValueError('Must probe with accepted Optimization 1 parser')
        out.mkdir(parents=True,exist_ok=False)
        rows,changes=[],[]
        snapshots={}
        for name in PROBES:
            with probe(name): new,_=replay.snapshot(raw)
            r,c=metrics(old,new,cohorts,['current_cascade'],name)
            rows+=r;changes+=c;snapshots[name]=new
        a.write_csv(out/'probe_comparison.csv',rows)
        a.write_csv(out/'probe_changed_cases.csv',changes)
        a.write_json(out/'probe_snapshots.json',snapshots)
        a.write_json(out/'manifest.json',dict(accepted_commit='9b14803cd42459d4bcf91104bcb26e4fb161a372',accepted_parser_sha256=m['new_parser_sha256'],source_sha256=m['source_sha256'],protected_sha256=m['protected_sha256'],probe_script_sha256=a.sha(Path(__file__)),ocr_invocations=0))
        print(json.dumps(rows,indent=2));return
    if (out/'summary.json').exists():raise ValueError('Do not overwrite completed comparison')
    probes=a.read(out/'probe_snapshots.json')
    if current!=probes['combined']: raise ValueError('Production implementation differs from combined probe')
    rows,changes=metrics(old,current,cohorts,list(replay.POLICIES),'implemented')
    before,after=oracle(old,cohorts),oracle(current,cohorts)
    oracles={c:dict(old=len(before[c]),new=len(after[c]),gained_ids=sorted(set(after[c])-set(before[c])),lost_ids=sorted(set(before[c])-set(after[c]))) for c in cohorts}
    added_attempts=[k for k in old if not set(a.cascade(current[k])[1])<=set(a.cascade(old[k])[1])]
    assert not added_attempts
    assert all(s['prediction']['final_date']=='NONE' for ss in current.values() for s in ss if all(s['prediction'][f]=='NONE' for f in ('year','month','day')))
    a.write_csv(out/'policy_comparison.csv',rows)
    a.write_csv(out/'changed_cases.csv',changes)
    a.write_json(out/'new_parser_replay.json',dict(stages=current,parser_replay_seconds=elapsed))
    a.write_json(out/'summary.json',dict(baselines=[r for r in rows if r['policy']=='current_cascade'],policies=rows,oracles=oracles,extra_ocr_attempt_images=added_attempts,ocr_invocations=0,probe_equivalence=True))
    manifest=a.read(out/'manifest.json');manifest['new_parser_sha256']=replay.hashes(sorted((ROOT/'date_parser').glob('*.py')))
    manifest['raw_sha256_after']=a.sha(replay.DUMP/'raw_ocr.jsonl');assert manifest['raw_sha256_after']==a.PIN
    a.write_json(out/'manifest.json',manifest)
    print(json.dumps({'baselines':[r for r in rows if r['policy']=='current_cascade'],'oracles':oracles},indent=2))

if __name__=='__main__':main()
