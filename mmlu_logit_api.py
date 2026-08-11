#!/usr/bin/env python3
"""Full MMLU in LOGIT mode via vmlx /v1/completions logprobs. Matches GGUF bench_gguf_mmlu_logit."""
import sys, time, json, httpx, pandas as pd
from huggingface_hub import hf_hub_download
API = sys.argv[1]; MODEL = sys.argv[2]; LIMIT = int(sys.argv[3]) if len(sys.argv)>3 else 0
OUT = sys.argv[4] if len(sys.argv)>4 else f"/Users/eric/mmlu_persubj_{MODEL.replace('/','_')}.json"
p = hf_hub_download('cais/mmlu','all/test-00000-of-00001.parquet',repo_type='dataset')
df = pd.read_parquet(p)
if LIMIT: df = df.groupby('subject').head(max(1,LIMIT//57)).reset_index(drop=True)
client = httpx.Client(base_url=API, timeout=120)
LET = [" A"," B"," C"," D"]
correct = 0; total = 0; bysub = {}
t0 = time.time()
for _, r in df.iterrows():
    ch = r['choices']
    prompt = ("The following is a multiple choice question. Reply with just the letter (A, B, C, or D).\n\n"
              f"{r['question']}\nA. {ch[0]}\nB. {ch[1]}\nC. {ch[2]}\nD. {ch[3]}\n\nAnswer:")
    try:
        resp = client.post("/completions", json={"model":MODEL,"prompt":prompt,"max_tokens":1,"temperature":0,"logprobs":20}).json()
        tl = (resp['choices'][0].get('logprobs') or {}).get('top_logprobs',[{}])[0]
        # highest-logprob among A/B/C/D (try with/without leading space)
        best=None; bestlp=-1e9
        for i,let in enumerate(LET):
            for cand in (let, let.strip()):
                if cand in tl and tl[cand]>bestlp: bestlp=tl[cand]; best=i
        pred = "ABCD"[best] if best is not None else "?"
    except Exception:
        pred = "?"
    ans = "ABCD"[int(r['answer'])]
    ok = (pred==ans); correct += ok; total += 1
    s=r['subject']; bysub.setdefault(s,[0,0]); bysub[s][1]+=1; bysub[s][0]+=ok
    if total%500==0:
        print(f"  [{total}/{len(df)}] running={100*correct/total:.2f}% {(total)/(time.time()-t0):.1f} q/s", flush=True)
        json.dump({"model":MODEL,"correct":correct,"total":total,"bysub":bysub}, open(OUT,"w"))
json.dump({"model":MODEL,"correct":correct,"total":total,"bysub":bysub,"done":True}, open(OUT,"w"))
print(f"\nMMLU_LOGIT RESULT: {correct}/{total} = {100*correct/total:.2f}%  ({time.time()-t0:.0f}s)  per-subject -> {OUT}")
