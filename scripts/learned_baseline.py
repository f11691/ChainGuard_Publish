#!/usr/bin/env python3
"""Matched learned-guard baseline for ChainGuard on the post-freeze holdout.

Answers Reviewer 1 (#3, #4), Reviewer 3 (Major Point 1), Reviewer 4 (#2) and
Reviewer 5 (#1), which ask for a comparison against a deployable learned
detector at matched operating points rather than against the eight-pattern
denylist.

Protocol
--------
Each session is scored exactly as ChainGuard is: turns are processed in order
and the session score is the maximum per-turn injection probability, mirroring
ChainGuard's "block on any turn" rule.  Both detectors therefore see the same
sessions, the same turn text and the same session-level endpoint.

One asymmetry is deliberate and is reported rather than corrected: ChainGuard
receives the host's provenance label for each segment and the learned guard does
not, because these classifiers take text alone.  The learned guard is thus
evaluated on the task it was trained for, and the comparison measures whole
detectors, not isolated text classification.

Threshold resolution: with 100 benign sessions the smallest non-zero false
positive rate that can be observed is 1%, so a TPR at 0.5% FPR cannot be
estimated from this holdout and is reported as such.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def score_sessions(items, model_id: str, batch: int = 16) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    mdl.eval()
    inj = next((i for i, l in mdl.config.id2label.items()
                if str(l).upper().startswith("INJECT")), 1)

    texts, owner = [], []
    for i, it in enumerate(items):
        for t in it["turns"]:
            texts.append(t)
            owner.append(i)

    probs = np.zeros(len(texts))
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            enc = tok(texts[s:s + batch], return_tensors="pt", truncation=True,
                      max_length=512, padding=True)
            probs[s:s + batch] = torch.softmax(
                mdl(**enc).logits, dim=-1)[:, inj].numpy()

    out = np.zeros(len(items))
    for p, o in zip(probs, owner):
        out[o] = max(out[o], p)          # session score = max over turns
    return out


def roc(att: np.ndarray, ben: np.ndarray) -> dict:
    thr = np.unique(np.concatenate([att, ben, [0.0, 1.0]]))[::-1]
    pts = []
    for t in thr:
        tpr = float((att >= t).mean())
        fpr = float((ben >= t).mean())
        pts.append({"threshold": float(t), "tpr": tpr, "fpr": fpr})
    order = np.argsort([p["fpr"] for p in pts])
    xs = np.array([pts[i]["fpr"] for i in order])
    ys = np.array([pts[i]["tpr"] for i in order])
    return {"points": pts, "auc": float(np.trapezoid(ys, xs))}


def tpr_at_fpr(curve: dict, target: float) -> dict | None:
    """Highest TPR achievable without exceeding `target` FPR."""
    ok = [p for p in curve["points"] if p["fpr"] <= target + 1e-12]
    if not ok:
        return None
    best = max(ok, key=lambda p: p["tpr"])
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--dataset", default="data/post_freeze_source_challenge_v2")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--chainguard-results", type=Path,
                    help="ablation.json, to pair the decisions for McNemar")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    base = a.repo / a.dataset
    attacks = json.loads((base / "attacks.json").read_text(encoding="utf-8"))
    benign = json.loads((base / "benign.json").read_text(encoding="utf-8"))

    print(f"scoring {len(attacks)} attack and {len(benign)} benign sessions with "
          f"{a.model}", file=sys.stderr)
    sa = score_sessions(attacks, a.model)
    sb = score_sessions(benign, a.model)

    curve = roc(sa, sb)
    n_a, n_b = len(attacks), len(benign)

    res = {
        "model": a.model,
        "n_attacks": n_a, "n_benign": n_b,
        "auc": curve["auc"],
        "min_resolvable_fpr": 1.0 / n_b,
        "operating_points": {},
        "roc": curve["points"],
        "session_scores": {
            "attacks": {it["id"]: float(s) for it, s in zip(attacks, sa)},
            "benign": {it["id"]: float(s) for it, s in zip(benign, sb)},
        },
    }

    # default threshold and fixed-FPR operating points
    for name, thr in [("default_0.5", 0.5)]:
        tp = int((sa >= thr).sum())
        fp = int((sb >= thr).sum())
        res["operating_points"][name] = {
            "threshold": thr,
            "detection": tp / n_a, "attacks_blocked": tp,
            "detection_ci": wilson(tp, n_a),
            "false_block": fp / n_b, "benign_blocked": fp,
            "false_block_ci": wilson(fp, n_b),
        }
    for target in (0.01, 0.005, 0.07):
        pt = tpr_at_fpr(curve, target)
        key = f"fpr<={target:.3f}"
        if pt is None:
            res["operating_points"][key] = None
            continue
        tp = int(round(pt["tpr"] * n_a))
        fp = int(round(pt["fpr"] * n_b))
        res["operating_points"][key] = {
            "threshold": pt["threshold"],
            "detection": pt["tpr"], "attacks_blocked": tp,
            "detection_ci": wilson(tp, n_a),
            "false_block": pt["fpr"], "benign_blocked": fp,
            "false_block_ci": wilson(fp, n_b),
            "note": ("below the resolution of this benign set"
                     if target < 1.0 / n_b else None),
        }

    # paired comparison against ChainGuard at each operating point
    if a.chainguard_results and a.chainguard_results.is_file():
        cg = json.loads(a.chainguard_results.read_text(encoding="utf-8"))["full"]
        cg_a = set(cg["attack_ids_blocked"])
        cg_b = set(cg["benign_ids_blocked"])
        for key, op in res["operating_points"].items():
            if not op:
                continue
            thr = op["threshold"]
            ba = {it["id"] for it, s in zip(attacks, sa) if s >= thr}
            bb = {it["id"] for it, s in zip(benign, sb) if s >= thr}
            op["mcnemar_attacks_p"] = mcnemar_exact(len(cg_a - ba), len(ba - cg_a))
            op["mcnemar_benign_p"] = mcnemar_exact(len(cg_b - bb), len(bb - cg_b))
            op["chainguard_only_attacks"] = len(cg_a - ba)
            op["baseline_only_attacks"] = len(ba - cg_a)
            op["chainguard_only_benign"] = len(cg_b - bb)
            op["baseline_only_benign"] = len(bb - cg_b)

        # per-category detection, to show where each detector is stronger
        thr = res["operating_points"]["default_0.5"]["threshold"]
        cats: dict[str, dict] = {}
        for it, s in zip(attacks, sa):
            c = cats.setdefault(it["category"], {"n": 0, "baseline": 0, "chainguard": 0})
            c["n"] += 1
            c["baseline"] += int(s >= thr)
            c["chainguard"] += int(it["id"] in cg_a)
        res["by_category"] = cats
        bcats: dict[str, dict] = {}
        for it, s in zip(benign, sb):
            c = bcats.setdefault(it["category"], {"n": 0, "baseline_fp": 0, "chainguard_fp": 0})
            c["n"] += 1
            c["baseline_fp"] += int(s >= thr)
            c["chainguard_fp"] += int(it["id"] in cg_b)
        res["benign_by_category"] = bcats

    a.out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"AUC={curve['auc']:.4f}", file=sys.stderr)
    for k, op in res["operating_points"].items():
        if op:
            print(f"  {k:14s} thr={op['threshold']:.4f} "
                  f"det={op['attacks_blocked']:2d}/{n_a} "
                  f"fp={op['benign_blocked']:2d}/{n_b}", file=sys.stderr)
    print(f"wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
