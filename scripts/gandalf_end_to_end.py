import json, math, random, sys
sys.path.insert(0,"cg/chainguard_v0.7.2_code/src"); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from pinj.chainguard import ChainGuardV7
from pinj.chainguard.accumulator import Action
from pinj.chainguard.output_guard import OutputGuard
from end_to_end import Target, SYSTEM_PROMPT, PROTECTED_RULES, render_turn, leaked, CANARY

def wilson(k,n,z=1.959963985):
    if not n: return (0,0)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (max(0,c-h)*100,min(1,c+h)*100)

N=int(sys.argv[1]) if len(sys.argv)>1 else 200
texts=json.load(open("big_public.json"))["Lakera/gandalf_ignore_instructions"]["texts"]
random.Random(1701).shuffle(texts); texts=texts[:N]
guard=OutputGuard(canaries=[CANARY], protected_texts=[PROTECTED_RULES])
tgt=Target("Qwen/Qwen2.5-1.5B-Instruct")
rows=[]
for i,t in enumerate(texts,1):
    g=ChainGuardV7(); blocked = g.step(t, source="user").action is Action.BLOCK
    msgs=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":render_turn(t,"user")}]
    reply=tgt.reply(msgs)
    lk,d=leaked(reply,guard)
    rows.append({"text":t,"cg_blocked":blocked,"leak":lk,"egress_blocked":d["egress_blocked"],"canary":d["canary"]})
    if i%25==0: print(f"  {i}/{len(texts)}", flush=True)
none_c=sum(1 for r in rows if r["leak"])
ing_c=sum(1 for r in rows if r["leak"] and not r["cg_blocked"])
egr_c=sum(1 for r in rows if r["leak"] and not r["cg_blocked"] and not r["egress_blocked"])
det=sum(1 for r in rows if r["cg_blocked"])
res={"n":len(rows),"chainguard_detected":det,"detection_ci":wilson(det,len(rows)),
     "compromise_none":none_c,"compromise_ingress":ing_c,"compromise_ingress_egress":egr_c,
     "compromise_none_ci":wilson(none_c,len(rows)),"rows":rows}
json.dump(res, open("gandalf_e2e.json","w"), ensure_ascii=False, indent=1)
print(f"gandalf end-to-end n={len(rows)} cg_detect={det} compromise none={none_c} ingress={ing_c} +egress={egr_c}")
