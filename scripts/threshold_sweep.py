#!/usr/bin/env python3
"""Threshold-sensitivity / ROC analysis for ChainGuard v0.7.2.

Answers Reviewer 1 (#8), Reviewer 3 (Major Points 1 and 2), Reviewer 4 (#3) and
Reviewer 5 (#4), which ask how performance varies around the selected operating
point and request ROC/DET-style reporting rather than a single hand-set point.

Two sweeps are reported because they measure different things:

  full      the shipped detector, sweeping RiskPolicy.block_threshold.  Forced
            predicates and the v7 escalation rules bypass the threshold, so this
            curve has a floor: it cannot fall below the detection (or the false
            blocking) those paths produce on their own.
  fusion    the same sweep with every forced predicate and v7 escalation rule
            disabled, isolating the weighted-fusion stage.  This is the curve
            the threshold actually controls.

The gap between the two quantifies how much of the operating point is set by
the predicate definitions rather than by the scalar cut-off.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np


def evaluate(items, mod, threshold: float, disable_forced: bool) -> set[str]:
    ChainGuardV7 = mod["ChainGuardV7"]
    Action = mod["Action"]
    RiskPolicy = mod["RiskPolicy"]

    original = RiskPolicy.decide
    orig_step = ChainGuardV7.step

    def decide(self, **kw):
        backup = self.cfg
        self.cfg = dataclasses.replace(backup, block_threshold=threshold)
        if disable_forced:
            kw["decoded_high_confidence"] = False
            kw["legacy_high_confidence"] = False
            for k in ("provenance", "state", "semantic"):
                kw[k] = dataclasses.replace(kw[k], high_confidence=False)
            self.cfg = dataclasses.replace(self.cfg, ml_block=1.01)
        try:
            return original(self, **kw)
        finally:
            self.cfg = backup

    def step_no_v7(self, turn, **kw):
        # bypass the v7 escalation layer entirely: return the inner verdict
        return self.inner.step(turn, **kw)

    RiskPolicy.decide = decide
    if disable_forced:
        ChainGuardV7.step = step_no_v7
    try:
        blocked = set()
        for item in items:
            g = ChainGuardV7()
            sources = item.get("sources") or [item.get("source", "user")] * len(item["turns"])
            for text, src in zip(item["turns"], sources):
                if g.step(text, source=src).action is Action.BLOCK:
                    blocked.add(item["id"])
                    break
        return blocked
    finally:
        RiskPolicy.decide = original
        ChainGuardV7.step = orig_step


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--dataset", default="data/post_freeze_source_challenge_v2")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=21)
    a = ap.parse_args()

    sys.path.insert(0, str((a.repo / "src").resolve()))
    from pinj.chainguard import ChainGuardV7
    from pinj.chainguard.accumulator import Action
    from pinj.chainguard.risk_policy import RiskPolicy
    mod = {"ChainGuardV7": ChainGuardV7, "Action": Action, "RiskPolicy": RiskPolicy}

    base = a.repo / a.dataset
    attacks = json.loads((base / "attacks.json").read_text(encoding="utf-8"))
    benign = json.loads((base / "benign.json").read_text(encoding="utf-8"))

    out = {"shipped_threshold": 0.78, "curves": {}}
    for mode, disable in (("full", False), ("fusion", True)):
        pts = []
        for t in np.linspace(0.0, 1.0, a.steps):
            ba = evaluate(attacks, mod, float(t), disable)
            bb = evaluate(benign, mod, float(t), disable)
            pts.append({"threshold": round(float(t), 4),
                        "tpr": len(ba) / len(attacks),
                        "fpr": len(bb) / len(benign),
                        "attacks_blocked": len(ba), "benign_blocked": len(bb)})
            print(f"  {mode:7s} thr={t:.2f}  det={len(ba):3d}/{len(attacks)}  "
                  f"fp={len(bb):3d}/{len(benign)}", file=sys.stderr)
        xs = np.array([p["fpr"] for p in pts])
        ys = np.array([p["tpr"] for p in pts])
        o = np.argsort(xs)
        out["curves"][mode] = {
            "points": pts,
            "auc_partial": float(np.trapezoid(ys[o], xs[o])),
            "min_detection": min(p["tpr"] for p in pts),
            "max_detection": max(p["tpr"] for p in pts),
            "min_false_block": min(p["fpr"] for p in pts),
            "max_false_block": max(p["fpr"] for p in pts),
        }
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
