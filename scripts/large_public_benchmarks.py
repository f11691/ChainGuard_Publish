import json, math, sys, urllib.parse, urllib.request
from pathlib import Path
sys.path.insert(0, "cg/chainguard_v0.7.2_code/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pinj.chainguard import ChainGuardV7
from pinj.chainguard.accumulator import Action

def wilson(k,n,z=1.959963985):
    if not n: return (0,0)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (max(0,c-h)*100, min(1,c+h)*100)

def fetch(ds,cfg,split,cache=Path("external_cache")):
    f=cache/f"{ds.replace('/','_')}__{split}.json"
    if f.is_file(): return json.loads(f.read_text(encoding="utf-8"))
    out,off=[],0
    while True:
        u=(f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(ds,safe='')}"
           f"&config={cfg}&split={urllib.parse.quote(split)}&offset={off}&length=100")
        page=None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"r"}),timeout=60) as r:
                    page=json.load(r); break
            except Exception as e:
                if attempt==5: raise
                import time; time.sleep(2**attempt)
        
        rs=[x["row"] for x in page.get("rows",[])]
        out.extend(rs); tot=page.get("num_rows_total",len(out)); off+=len(rs)
        if not rs or off>=tot: break
    cache.mkdir(exist_ok=True); f.write_text(json.dumps(out,ensure_ascii=False),encoding="utf-8")
    return out

def blocked(texts):
    n=0
    for t in texts:
        g=ChainGuardV7()
        if g.step(t, source="user").action is Action.BLOCK: n+=1
    return n

res={}
# 1. Gandalf: 777 real human system-prompt/password extraction attempts (all attacks)
g=[r["text"] for r in fetch("Lakera/gandalf_ignore_instructions","default","train")+
                     fetch("Lakera/gandalf_ignore_instructions","default","validation")+
                     fetch("Lakera/gandalf_ignore_instructions","default","test") if r.get("text")]
k=blocked(g); lo,hi=wilson(k,len(g))
res["Lakera/gandalf_ignore_instructions"]={"n_attack":len(g),"detected":k,
    "detection":k/len(g),"ci":[lo,hi],"texts":g}
print(f"gandalf   detection {k}/{len(g)} = {100*k/len(g):.1f}%  CI {lo:.1f}-{hi:.1f}")

# 2. safe-guard: 8k labelled, large benign half -> low-FPR estimate
sg=fetch("xTRam1/safe-guard-prompt-injection","default","train")+ \
   fetch("xTRam1/safe-guard-prompt-injection","default","test")
atk=[r["text"] for r in sg if str(r.get("label"))=="1" and r.get("text")]
ben=[r["text"] for r in sg if str(r.get("label"))=="0" and r.get("text")]
ka=blocked(atk); kb=blocked(ben)
la,ha=wilson(ka,len(atk)); lb,hb=wilson(kb,len(ben))
res["xTRam1/safe-guard-prompt-injection"]={"n_attack":len(atk),"detected":ka,
    "detection":ka/len(atk),"detection_ci":[la,ha],
    "n_benign":len(ben),"false_blocked":kb,"false_block":kb/len(ben),
    "false_block_ci":[lb,hb],"attack_texts":atk,"benign_texts":ben}
print(f"safeguard detection {ka}/{len(atk)} = {100*ka/len(atk):.1f}%  CI {la:.1f}-{ha:.1f}")
print(f"safeguard false blk {kb}/{len(ben)} = {100*kb/len(ben):.2f}% CI {lb:.2f}-{hb:.2f}")
json.dump(res, open("big_public.json","w"), ensure_ascii=False, indent=1)
print("wrote big_public.json")
