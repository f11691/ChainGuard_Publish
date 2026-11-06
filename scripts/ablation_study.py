#!/usr/bin/env python3
"""Leave-one-out component ablation and reason-code analysis for ChainGuard v0.7.2.

Answers Reviewer 1 (#5-#7, #9), Reviewer 3 (Major Point 2), Reviewer 4 (#3) and
Reviewer 5 (#2) on the same 80-attack / 100-benign holdout used for the headline
result, with the frozen detector untouched.

Every arm is implemented by neutralising one layer's *contribution to the
decision* at the single seam where all layers meet, `RiskPolicy.decide`, rather
than by editing the detector.  The full system is re-run in the same process as
a control, so any arm that fails to reproduce 61/80 and 7/100 is a harness bug
rather than a finding.

Arms
----
    full                  unmodified detector (control)
    no_decoding           decoded-payload forced predicate disabled
    no_transparent        transparent/lexical high-confidence + accumulator removed
    no_provenance         provenance risk and its forced predicate removed
    no_state              bounded cross-turn state removed (risk and predicate)
    no_classifier         local intent classifier removed from the ensemble
    no_semantic_risk      semantic risk and its forced predicate removed
                          (benign-scope gating retained, see note below)
    no_forced_predicates  every forced predicate disabled; weighted fusion only
    no_agreement          classifier/semantic agreement override disabled
    stateless             fresh detector per turn (no state carried at all)

Note on `no_semantic_risk`: the semantic layer supplies both a risk score and
the benign-scope gate that suppresses false positives.  This arm removes the
risk score and the forced predicate but retains the gate, so it measures the
semantic layer's detection contribution.  `no_semantic_scope` additionally
removes the gate and is reported separately, because removing both at once
conflates a detection loss with a false-positive increase.

Usage
-----
    python ablation_study.py --repo <checkout> --out ablation.json [--tex out.tex]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


# --------------------------------------------------------------- ablation arms
# ChainGuard is a two-stage detector: ChainGuardV7.step first runs the inner v6
# layered detector (whose decision is made by RiskPolicy.decide) and returns
# immediately if that blocks; only otherwise does it apply its own v7
# escalation rules.  A component ablation must therefore neutralise the layer
# at BOTH stages, or the v7 rule silently compensates for the disabled v6 input
# and the arm reports no effect.  The arms below patch the narrowest seam that
# removes a layer from both stages.

POLICY_ARMS = {"no_transparent", "no_semantic_risk", "no_semantic_scope",
               "no_forced_predicates", "no_agreement"}


def _policy_patch(arm: str, mod):
    RiskPolicy = mod["RiskPolicy"]
    original = RiskPolicy.decide

    def decide(self, *, semantic, ml, state, provenance, legacy_score,
               legacy_high_confidence=False, decoded_high_confidence=False):
        if arm == "no_transparent":
            legacy_high_confidence = False
            legacy_score = 0.0
        elif arm == "no_semantic_risk":
            semantic = dataclasses.replace(
                semantic, semantic_risk=0.0, high_confidence=False)
        elif arm == "no_semantic_scope":
            semantic = dataclasses.replace(
                semantic,
                intent_class=mod["IntentClass"].OTHER,
                execution_mode=mod["ExecutionMode"].EXECUTE,
                target_provenance=mod["TargetProvenance"].UNKNOWN,
            )
        elif arm == "no_forced_predicates":
            decoded_high_confidence = False
            legacy_high_confidence = False
            provenance = dataclasses.replace(provenance, high_confidence=False)
            state = dataclasses.replace(state, high_confidence=False)
            semantic = dataclasses.replace(semantic, high_confidence=False)
        elif arm == "no_agreement":
            backup = self.cfg
            self.cfg = dataclasses.replace(backup, ml_high=1.01)
            try:
                return original(
                    self, semantic=semantic, ml=ml, state=state,
                    provenance=provenance, legacy_score=legacy_score,
                    legacy_high_confidence=legacy_high_confidence,
                    decoded_high_confidence=decoded_high_confidence)
            finally:
                self.cfg = backup
        return original(
            self, semantic=semantic, ml=ml, state=state, provenance=provenance,
            legacy_score=legacy_score,
            legacy_high_confidence=legacy_high_confidence,
            decoded_high_confidence=decoded_high_confidence)

    return decide


@contextmanager
def ablate(arm: str, mod):
    """Neutralise one layer across both detector stages for the duration."""
    if arm in ("full", "stateless"):
        yield
        return

    import pinj.chainguard.guard_v7 as g7
    RiskPolicy = mod["RiskPolicy"]
    ChainGuardV7 = mod["ChainGuardV7"]
    saved = {}

    def save(obj, name):
        # Take the raw class-dict entry so that staticmethod/classmethod
        # wrappers survive the round trip; fall back to getattr for modules.
        raw = obj.__dict__[name] if name in getattr(obj, "__dict__", {}) \
            else getattr(obj, name)
        saved[(id(obj), name)] = (obj, name, raw)

    if arm in POLICY_ARMS:
        save(RiskPolicy, "decide")
        RiskPolicy.decide = _policy_patch(arm, mod)

    elif arm == "no_decoding":
        # stage 1: the inner decoded-payload predicate; stage 2: v7 decoded_execution
        save(g7, "decode_candidates")
        g7.decode_candidates = lambda *a, **k: []
        save(RiskPolicy, "decide")
        inner = RiskPolicy.decide

        def decide(self, **kw):
            kw["decoded_high_confidence"] = False
            return inner(self, **kw)
        RiskPolicy.decide = decide

    elif arm == "no_classifier":
        # stage 1: ml=None into the policy; stage 2: v7 calibrated_ml_attack
        save(ChainGuardV7, "_ml")
        ChainGuardV7._ml = staticmethod(lambda *a, **k: ("", 0.0))
        save(RiskPolicy, "decide")
        inner = RiskPolicy.decide

        def decide(self, **kw):
            kw["ml"] = None
            return inner(self, **kw)
        RiskPolicy.decide = decide

    elif arm == "no_provenance":
        # Remove source awareness entirely: every segment is treated as trusted
        # user input, which disables the provenance risk, its forced predicate,
        # and the v7 untrusted_source_instruction rule together.
        save(ChainGuardV7, "step")
        inner_step = ChainGuardV7.step

        def step(self, turn, **kw):
            kw["source"] = "user"
            return inner_step(self, turn, **kw)
        ChainGuardV7.step = step

    elif arm == "no_state":
        # stage 1: zero the cross-turn assessment; stage 2: the v7 state flags
        save(RiskPolicy, "decide")
        inner = RiskPolicy.decide

        def decide(self, **kw):
            kw["state"] = dataclasses.replace(
                kw["state"], risk=0.0, high_confidence=False, reasons=())
            return inner(self, **kw)
        RiskPolicy.decide = decide

        save(ChainGuardV7, "_update_state")
        inner_state = ChainGuardV7._update_state

        def update_state(self, text, benign_scoped):
            st = dict(inner_state(self, text, benign_scoped))
            for k in ("alias_execution", "backref_execution", "untrusted_execution"):
                if k in st:
                    st[k] = False
            return st
        ChainGuardV7._update_state = update_state

    else:
        raise SystemExit(f"unknown arm {arm}")

    try:
        yield
    finally:
        for obj, name, val in saved.values():
            setattr(obj, name, val)


# -------------------------------------------------------------------- scoring
def run_arm(arm: str, attacks, benign, mod) -> dict:
    ChainGuardV7 = mod["ChainGuardV7"]
    Action = mod["Action"]
    stateless = arm == "stateless"

    def evaluate(items):
        out = []
        for item in items:
            g = ChainGuardV7()
            sources = item.get("sources") or [item.get("source", "user")] * len(item["turns"])
            blocked, first_turn, actions = False, None, []
            block_reasons, forced, out_stage = [], None, []
            for i, (text, src) in enumerate(zip(item["turns"], sources)):
                if stateless:
                    g = ChainGuardV7()
                v = g.step(text, source=src)
                actions.append(v.action.value)
                det = v.details or {}
                if v.action is Action.BLOCK:
                    blocked, first_turn = True, i + 1
                    v7 = det.get("v7") or {}
                    dec = det.get("decision") or {}
                    if v7.get("escalation_reason"):
                        # blocked by a v7 escalation rule, not the inner policy
                        block_reasons = [f"v7:{v7['escalation_reason']}"]
                        forced = True
                        stage = "v7"
                    else:
                        block_reasons = list(dec.get("reasons") or [])
                        forced = bool(dec.get("forced"))
                        stage = "v6"
                    out_stage.append(stage)
                    break
            out.append({"id": item["id"], "category": item["category"],
                        "blocked": blocked, "first_block_turn": first_turn,
                        "reasons": block_reasons, "forced": forced,
                        "stage": (out_stage or [None])[0], "actions": actions})
        return out

    with ablate(arm, mod):
        ao = evaluate(attacks)
        bo = evaluate(benign)

    tp = sum(1 for o in ao if o["blocked"])
    fp = sum(1 for o in bo if o["blocked"])
    challenge_a = sum(1 for o in ao if not o["blocked"] and "soft_challenge" in o["actions"])
    challenge_b = sum(1 for o in bo if not o["blocked"] and "soft_challenge" in o["actions"])
    return {
        "arm": arm,
        "attacks_blocked": tp, "n_attacks": len(attacks),
        "detection": tp / len(attacks),
        "detection_ci": wilson(tp, len(attacks)),
        "benign_blocked": fp, "n_benign": len(benign),
        "false_block": fp / len(benign),
        "false_block_ci": wilson(fp, len(benign)),
        "challenge_attacks": challenge_a,
        "challenge_benign": challenge_b,
        "escape_by_category": dict(Counter(o["category"] for o in ao if not o["blocked"])),
        "fp_by_category": dict(Counter(o["category"] for o in bo if o["blocked"])),
        "reason_codes": dict(Counter(r for o in ao if o["blocked"] for r in o["reasons"])),
        "fp_reason_codes": dict(Counter(r for o in bo if o["blocked"] for r in o["reasons"])),
        "stage_attacks": dict(Counter(o["stage"] for o in ao if o["blocked"])),
        "stage_benign": dict(Counter(o["stage"] for o in bo if o["blocked"])),
        "forced_attacks": sum(1 for o in ao if o["blocked"] and o["forced"]),
        "forced_benign": sum(1 for o in bo if o["blocked"] and o["forced"]),
        "forced_by_turn1": sum(1 for o in ao if o["blocked"] and o["forced"]
                               and o["first_block_turn"] == 1),
        "unforced_turn1": sum(1 for o in ao if o["blocked"] and not o["forced"]
                              and o["first_block_turn"] == 1),
        "ttd": dict(Counter(o["first_block_turn"] for o in ao if o["blocked"])),
        "attack_ids_blocked": sorted(o["id"] for o in ao if o["blocked"]),
        "benign_ids_blocked": sorted(o["id"] for o in bo if o["blocked"]),
    }


ARMS = ["full", "no_decoding", "no_transparent", "no_provenance", "no_state",
        "no_classifier", "no_semantic_risk", "no_semantic_scope",
        "no_forced_predicates", "no_agreement", "stateless"]

LABEL = {
    "full": "Full system (control)",
    "no_decoding": "-- decoding predicate",
    "no_transparent": "-- transparent/lexical layer",
    "no_provenance": "-- provenance policy",
    "no_state": "-- bounded temporal state",
    "no_classifier": "-- local intent classifier",
    "no_semantic_risk": "-- semantic risk",
    "no_semantic_scope": "-- semantic benign-scope gate",
    "no_forced_predicates": "-- all forced predicates",
    "no_agreement": "-- agreement override",
    "stateless": "Stateless (fresh per turn)",
}


def emit_tex(results: dict, full: dict) -> str:
    rows = []
    for arm in ARMS:
        r = results[arm]
        d_lo, d_hi = (x * 100 for x in r["detection_ci"])
        dd = r["attacks_blocked"] - full["attacks_blocked"]
        df = r["benign_blocked"] - full["benign_blocked"]
        p_a = r.get("mcnemar_attacks", float("nan"))
        p_b = r.get("mcnemar_benign", float("nan"))
        star = lambda p: ("$<$0.001" if p < 0.001 else f"{p:.3f}")
        rows.append(
            f"{LABEL[arm]} & {r['attacks_blocked']}/{r['n_attacks']} "
            f"({r['detection']*100:.0f}\\%) & {dd:+d} & {star(p_a)} & "
            f"{r['benign_blocked']}/{r['n_benign']} ({r['false_block']*100:.0f}\\%) & "
            f"{df:+d} & {star(p_b)}\\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table*}}[!t]
\\centering
\\caption{{Leave-one-out component ablation on the post-freeze holdout. Each arm
neutralises one layer's contribution at the decision seam; the detector source is
unmodified. $\\Delta$ columns are the change in blocked sessions relative to the
full system, and $p$ values are exact two-sided McNemar tests on paired
session decisions.}}
\\label{{tab:ablation}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{@{{}}lccccccc@{{}}}}
\\toprule
\\textbf{{Configuration}} & \\textbf{{Attack detection}} & $\\boldsymbol{{\\Delta}}$ & $\\boldsymbol{{p}}$ & \\textbf{{Benign false block}} & $\\boldsymbol{{\\Delta}}$ & $\\boldsymbol{{p}}$\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--dataset", default="data/post_freeze_source_challenge_v2")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tex", type=Path)
    a = ap.parse_args()

    sys.path.insert(0, str((a.repo / "src").resolve()))
    from pinj.chainguard import ChainGuardV7
    from pinj.chainguard.accumulator import Action
    from pinj.chainguard.risk_policy import RiskPolicy
    from pinj.chainguard.semantic_intent import (
        ExecutionMode, IntentClass, TargetProvenance)
    mod = {"ChainGuardV7": ChainGuardV7, "Action": Action, "RiskPolicy": RiskPolicy,
           "IntentClass": IntentClass, "ExecutionMode": ExecutionMode,
           "TargetProvenance": TargetProvenance}

    base = a.repo / a.dataset
    attacks = json.loads((base / "attacks.json").read_text(encoding="utf-8"))
    benign = json.loads((base / "benign.json").read_text(encoding="utf-8"))

    results = {}
    for arm in ARMS:
        results[arm] = run_arm(arm, attacks, benign, mod)
        r = results[arm]
        print(f"{arm:22s} attacks {r['attacks_blocked']:3d}/{r['n_attacks']}  "
              f"benign {r['benign_blocked']:3d}/{r['n_benign']}  "
              f"challenge(a/b) {r['challenge_attacks']}/{r['challenge_benign']}",
              file=sys.stderr)

    full = results["full"]
    if (full["attacks_blocked"], full["benign_blocked"]) != (61, 7):
        print("WARNING: control arm did not reproduce 61/80 and 7/100 -- "
              "treat all arms as suspect", file=sys.stderr)

    # paired McNemar of each arm against the full system
    fa, fb = set(full["attack_ids_blocked"]), set(full["benign_ids_blocked"])
    for arm, r in results.items():
        ra, rb = set(r["attack_ids_blocked"]), set(r["benign_ids_blocked"])
        r["mcnemar_attacks"] = mcnemar_exact(len(fa - ra), len(ra - fa))
        r["mcnemar_benign"] = mcnemar_exact(len(fb - rb), len(rb - fb))

    a.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {a.out}", file=sys.stderr)
    if a.tex:
        a.tex.write_text(emit_tex(results, full), encoding="utf-8")
        print(f"wrote {a.tex}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
