"""usage: int_eval.py CODE_ROOT OUT.json  -- replay CODE_ROOT's stage_result+run_cascade on saved OCR.
cos: s1/allstages.jsonl (all 5 stages, main 0dd7a35 run).  food: run701 dump + missing stages filled from
the old-parser four-stage snapshot (hybrid, as round 6). clahe_1024 for food is not saved: for images that
main 0dd7a35 left NONE before retry (audit), 선우's full-700 run gives the outcome (4 recovered, others no
candidate); any other image reaching clahe_1024 is flagged unknown."""
import sys, json, csv
S='/tmp/claude-0/-home-user-itda3-DScover-LetsOCR/698ed159-bc16-5296-99ef-b44312b54343/scratchpad'
root,out=sys.argv[1],sys.argv[2]; sys.path.insert(0,root)
import ocr_pipeline as op
snap=json.load(open(S+'/ev6/facts_snap2.json'))
audit=json.load(open('/tmp/claude-0/audit700.json'))
MAIN_NONE={e['image_id'] for e in audit['remaining_errors'] if e['prediction']=='NONE'}
GAIN={'000050':'2025-09-30','000994':'NONE-02-18','001979':'2020-11-04','002766':'2021-03-16'}
def fin(fd):
    p=fd.split('-'); return {'year':p[0],'month':p[1],'day':p[2],'final_date':fd}
def facts(stages):
    f={}
    for s in stages:
        f[s['stage']]=op.stage_result(s['detections'])
    return f
res={}
def run(key,f,food):
    unknown=[]
    def rs(n):
        if n in f: return f[n]
        if n=='clahe_1024' and food:
            iid=key.split('|')[1]
            if iid in GAIN: return fin(GAIN[iid]),{'q':0.99,'M':False}
            if iid in MAIN_NONE: return fin('NONE-NONE-NONE') if False else {'year':'NONE','month':'NONE','day':'NONE','final_date':'NONE'},None
            unknown.append(iid); return {'year':'NONE','month':'NONE','day':'NONE','final_date':'NONE'},None
        raise KeyError(n)
    try: p,m,a=op.run_cascade(rs)
    except KeyError as e: return None
    return {'pred':p['final_date'],'method':m,'attempts':a,'unknown':bool(unknown)}
for l in open(S+'/s1/allstages.jsonl',encoding='utf-8'):
    r=json.loads(l); k='cos|'+str(r['image_id']); res[k]=run(k,facts(r['stages']),False)
for l in open(S+'/run701/ocr_dump.jsonl',encoding='utf-8'):
    r=json.loads(l)
    if 'error' in r: continue
    k='food|'+str(int(r['image_id'])).zfill(6)
    f=facts(r['stages'])
    for s,x in snap.get(k,{}).items():
        if s not in f: f[s]=(x['pred'],x['ev'])
    res[k]=run(k,f,True)
json.dump(res,open(out,'w'))
print('done',len(res),sum(v is None for v in res.values()))
