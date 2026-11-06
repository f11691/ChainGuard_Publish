#!/usr/bin/env python3
"""Independent evaluation of ChainGuard's provenance layer on InjecAgent.

The four public datasets used in Section~\\ref{sec:external} are single-turn
prompts without provenance labels, so they exercise the lexical, semantic and
classifier layers but not the provenance policy or the cross-turn state. This
script closes that gap using InjecAgent, a third-party benchmark for *indirect*
prompt injection against tool-using agents, in which the attacker's instruction
is delivered inside a tool response rather than by the user.

Each case is replayed as the two-turn session a real agent would see:

    turn 1  the user's legitimate instruction        source = user
    turn 2  the tool response carrying the injection source = tool

Detection is a block on turn 2. A block on turn 1 is a false block, since the
user instruction is a benign request; this gives an independent false-positive
estimate on exactly the traffic a tool-using deployment generates.

Running the same cases with every segment relabelled `user` ablates the
provenance policy on independent data, which is the measurement Section 6.5
could only make on the authored holdout.

InjecAgent played no part in ChainGuard's development. The BIPIA subset already
in the repository did, per THIRD_PARTY_DATA.md, and is deliberately not used.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/"
FILES = {"direct_harm": "test_cases_dh_base.json",
         "data_stealing": "test_cases_ds_base.json"}


def wilson(k, n, z=1.959963985):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h) * 100, min(1.0, c + h) * 100)


def fetch(name, cache: Path):
    f = cache / ("injecagent_" + FILES[name])
    if f.is_file():
        return json.loads(f.read_text(encoding="utf-8"))
    req = urllib.request.Request(BASE + FILES[name], headers={"User-Agent": "research"})
    with urllib.request.urlopen(req, timeout=90) as r:
        d = json.load(r)
    cache.mkdir(exist_ok=True)
    f.write_text(json.dumps(d), encoding="utf-8")
    return d


def run(cases, mods, tool_source="tool"):
    ChainGuardV7, Action = mods["ChainGuardV7"], mods["Action"]
    det = fp = 0
    reasons = {}
    for c in cases:
        g = ChainGuardV7()
        v1 = g.step(str(c["User Instruction"]), source="user")
        if v1.action is Action.BLOCK:
            fp += 1
            continue                      # session already stopped; turn 2 unseen
        v2 = g.step(str(c["Tool Response"]), source=tool_source)
        if v2.action is Action.BLOCK:
            det += 1
            d = v2.details or {}
            r = ((d.get("v7") or {}).get("escalation_reason")
                 or ((d.get("decision") or {}).get("reasons") or ["unknown"])[0])
            reasons[r] = reasons.get(r, 0) + 1
    return det, fp, reasons


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--cache", type=Path, default=Path("external_cache"))
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    sys.path.insert(0, str((a.repo / "src").resolve()))
    from pinj.chainguard import ChainGuardV7
    from pinj.chainguard.accumulator import Action
    mods = {"ChainGuardV7": ChainGuardV7, "Action": Action}

    out = {}
    for name in FILES:
        cases = fetch(name, a.cache)
        det, fp, reasons = run(cases, mods, "tool")
        det_u, fp_u, _ = run(cases, mods, "user")   # provenance ablated
        n = len(cases)
        out[name] = {
            "n": n,
            "detected": det, "detection": det / n, "detection_ci": wilson(det, n),
            "user_turn_false_blocks": fp, "false_block": fp / n,
            "false_block_ci": wilson(fp, n),
            "detected_without_provenance": det_u,
            "provenance_contribution": det - det_u,
            "reason_codes": reasons,
        }
        print(f"{name:14s} n={n}  detection {det}/{n} = {100*det/n:.1f}% "
              f"(CI {wilson(det,n)[0]:.1f}-{wilson(det,n)[1]:.1f})  "
              f"user-turn false blocks {fp}/{n}", file=sys.stderr)
        print(f"{'':14s} without provenance labels: {det_u}/{n} = {100*det_u/n:.1f}% "
              f"-> provenance worth {det-det_u} detections", file=sys.stderr)
        print(f"{'':14s} reasons: {reasons}", file=sys.stderr)

    tot = sum(v["n"] for v in out.values())
    td = sum(v["detected"] for v in out.values())
    tf = sum(v["user_turn_false_blocks"] for v in out.values())
    tu = sum(v["detected_without_provenance"] for v in out.values())
    out["pooled"] = {"n": tot, "detected": td, "detection": td / tot,
                     "detection_ci": wilson(td, tot),
                     "user_turn_false_blocks": tf, "false_block": tf / tot,
                     "false_block_ci": wilson(tf, tot),
                     "detected_without_provenance": tu,
                     "provenance_contribution": td - tu}
    print(f"\npooled: detection {td}/{tot} = {100*td/tot:.1f}% "
          f"(CI {wilson(td,tot)[0]:.1f}-{wilson(td,tot)[1]:.1f}); "
          f"user-turn false blocks {tf}/{tot}; "
          f"without provenance {tu}/{tot} = {100*tu/tot:.1f}%", file=sys.stderr)
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
