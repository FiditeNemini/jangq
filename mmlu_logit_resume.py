#!/usr/bin/env python3
"""Resumable logit-mode MMLU. Keeps already-scored subjects from OUT json, runs only what's missing.
Usage: mmlu_logit_resume.py <API> <MODEL> <LIMIT> <OUT.json>  (LIMIT=0 -> full; else ~LIMIT/57 per subject)
Subjects already at >= target count in OUT are kept as-is (reused). Partially-done subjects are re-run
to target. Undone subjects are run to target. Saves merged bysub every 500."""
import sys, os, json, time, httpx, pandas as pd
from huggingface_hub import hf_hub_download
API=sys.argv[1]; MODEL=sys.argv[2]; LIMIT=int(sys.argv[3]) if len(sys.argv)>3 else 0
OUT=sys.argv[4] if len(sys.argv)>4 else f"/Users/eric/mmlu_persubj_{MODEL.replace('/','_')}.json"
p=hf_hub_download('cais/mmlu','all/test-00000-of-00001.parquet',repo_type='dataset')
df=pd.read_parquet(p)
subjects=list(dict.fromkeys(df['subject'].tolist()))
target=(LIMIT//len(subjects)) if LIMIT else 10**9   # per-subject target count

# ---- load prior progress (reuse) ----
done={}
if os.path.exists(OUT):
    try:
        prev=json.load(open(OUT)); done={k:list(v) for k,v in prev.get('bysub',{}).items()}
    except Exception: done={}
bysub={}; correct=0; total=0; reused=[]; torun=[]
for s in subjects:
    have=done.get(s,[0,0])
    if have[1]>=target and have[1]>0:      # fully covered already -> keep
        bysub[s]=have; correct+=have[0]; total+=have[1]; reused.append(s)
    else:
        torun.append(s)                     # undone or partial -> (re)run to target
print(f"resume: reusing {len(reused)} subjects ({total} Q), running {len(torun)} subjects to {target}/subj", flush=True)

client=httpx.Client(base_url=API,timeout=120); LET=[" A"," B"," C"," D"]; t0=time.time(); n_new=0
for s in torun:
    rows=df[df['subject']==s].head(target)
    sc=[0,0]
    for _,r in rows.iterrows():
        ch=r['choices']
        prompt=("The following is a multiple choice question. Reply with just the letter (A, B, C, or D).\n\n"
                f"{r['question']}\nA. {ch[0]}\nB. {ch[1]}\nC. {ch[2]}\nD. {ch[3]}\n\nAnswer:")
        try:
            resp=client.post("/completions",json={"model":MODEL,"prompt":prompt,"max_tokens":1,"temperature":0,"logprobs":20}).json()
            tl=(resp['choices'][0].get('logprobs') or {}).get('top_logprobs',[{}])[0]
            best=None; blp=-1e9
            for i,let in enumerate(LET):
                for cand in (let,let.strip()):
                    if cand in tl and tl[cand]>blp: blp=tl[cand]; best=i
            pred="ABCD"[best] if best is not None else "?"
        except Exception: pred="?"
        ok=(pred=="ABCD"[int(r['answer'])]); sc[0]+=ok; sc[1]+=1; correct+=ok; total+=1; n_new+=1
        if n_new%50==0:
            bysub[s]=sc; json.dump({"model":MODEL,"correct":correct,"total":total,"bysub":bysub},open(OUT,"w"))
            print(f"  [{total} total | +{n_new} new] running={100*correct/total:.2f}% {n_new/(time.time()-t0):.1f} q/s (subj {s})",flush=True)
    bysub[s]=sc
    json.dump({"model":MODEL,"correct":correct,"total":total,"bysub":bysub},open(OUT,"w"))
json.dump({"model":MODEL,"correct":correct,"total":total,"bysub":bysub,"done":True},open(OUT,"w"))
print(f"\nMMLU_LOGIT RESULT: {correct}/{total} = {100*correct/total:.2f}%  ({time.time()-t0:.0f}s, {n_new} new)  -> {OUT}")
