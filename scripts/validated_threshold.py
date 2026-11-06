#!/usr/bin/env python3
"""Confirmatory low-false-block comparison with thresholds frozen on separate data.

Reviewer 1 (third round, comment 4) asks that low-FPR operating points be
selected on a validation set and then evaluated once on an untouched test set.
The authored holdout is too small for this (100 benign sessions), so the
protocol uses the independent Safe-Guard dataset and its own published split:

  validation  xTRam1/safe-guard-prompt-injection, split "train"  (selection)
  test        xTRam1/safe-guard-prompt-injection, split "test"   (evaluated once)

For each detector and each false-block ceiling (0.5%, 1.0%) the threshold with
the highest validation detection whose validation false-block rate does not
exceed the ceiling is chosen, frozen, and applied unchanged to the test split.

ChainGuard: the shipped full policy, sweeping RiskPolicy.block_threshold over a
fixed grid (forced predicates and escalation are kept, as in deployment).
DeBERTa: protectai/deberta-v3-base-prompt-injection-v2, injection probability.

Usage: python3 scripts/validated_threshold.py --repo . --cache external_cache \
           --out results/validated_threshold_v0.7.2.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DATASET = "xTRam1/safe-guard-prompt-injection"
GRID = [0.78, 0.85, 0.90, 0.95, 1.00, 1.01]   # 0.78 is the shipped threshold
CEILINGS = [0.005, 0.010]
MODEL = "protectai/deberta-v3-base-prompt-injection-v2"


def wilson(k: int, n: int, z: float = 1.959963985) -> list[float]:
    if not n:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [100 * max(0.0, c - h), 100 * min(1.0, c + h)]


def load_split(cache: Path, split: str) -> list[dict]:
    """Rows of one split, via the same cache file and download as
    large_public_benchmarks.py (which cannot be imported: it runs on import)."""
    import time, urllib.parse, urllib.request
    f = cache / f"{DATASET.replace('/', '_')}__{split}.json"
    if f.is_file():
        return json.loads(f.read_text(encoding="utf-8"))
    out, off = [], 0
    while True:
        u = ("https://datasets-server.huggingface.co/rows?dataset="
             f"{urllib.parse.quote(DATASET, safe='')}&config=default"
             f"&split={urllib.parse.quote(split)}&offset={off}&length=100")
        for attempt in range(6):
            try:
                req = urllib.request.Request(u, headers={"User-Agent": "r"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    page = json.load(r)
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        rows = [x["row"] for x in page.get("rows", [])]
        out.extend(rows)
        off += len(rows)
        if not rows or off >= page.get("num_rows_total", len(out)):
            break
    cache.mkdir(exist_ok=True)
    f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def chainguard_blocks(args) -> list[bool]:
    repo, threshold, texts = args
    sys.path.insert(0, str((repo / "src").resolve()))
    from pinj.chainguard import ChainGuardV7
    from pinj.chainguard.accumulator import Action
    from pinj.chainguard.risk_policy import RiskPolicy
    original = RiskPolicy.decide

    def decide(self, **kw):
        backup = self.cfg
        self.cfg = dataclasses.replace(backup, block_threshold=threshold)
        try:
            return original(self, **kw)
        finally:
            self.cfg = backup

    RiskPolicy.decide = decide
    try:
        return [ChainGuardV7().step(t, source="user").action is Action.BLOCK for t in texts]
    finally:
        RiskPolicy.decide = original


def deberta_scores(texts: list[str], batch: int = 32) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    inj = next((i for i, l in mdl.config.id2label.items()
                if str(l).upper().startswith("INJECT")), 1)
    # batch texts of similar length together to avoid padding work; scores are
    # written back in the original order
    order = np.argsort([len(t) for t in texts])
    out = np.zeros(len(texts))
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            idx = order[s:s + batch]
            enc = tok([texts[i] for i in idx], return_tensors="pt", truncation=True,
                      max_length=512, padding=True)
            out[idx] = torch.softmax(mdl(**enc).logits, dim=-1)[:, inj].numpy()
            print(f"  deberta {min(s + batch, len(texts))}/{len(texts)}",
                  file=sys.stderr, flush=True)
    return out


def rates(blocked: np.ndarray, y: np.ndarray) -> dict:
    atk, ben = y == 1, y == 0
    det, fb = int(blocked[atk].sum()), int(blocked[ben].sum())
    return {"detected": det, "n_attack": int(atk.sum()),
            "detection": det / int(atk.sum()), "detection_ci": wilson(det, int(atk.sum())),
            "false_blocked": fb, "n_benign": int(ben.sum()),
            "false_block": fb / int(ben.sum()), "false_block_ci": wilson(fb, int(ben.sum()))}


def select(candidates: dict, y: np.ndarray, ceiling: float):
    """Highest validation detection with validation false-block <= ceiling."""
    best = None
    for thr, blocked in candidates.items():
        r = rates(blocked, y)
        if r["false_block"] <= ceiling + 1e-12:
            key = (r["detected"], -r["false_blocked"])
            if best is None or key > best[0]:
                best = (key, thr, r)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--cache", type=Path, default=Path("external_cache"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    split = {name: load_split(a.cache, name) for name in ("train", "test")}
    texts = {k: [str(r["text"]) for r in v] for k, v in split.items()}
    y = {k: np.array([int(r["label"]) for r in v]) for k, v in split.items()}
    allt = texts["train"] + texts["test"]
    ntr = len(texts["train"])

    # ChainGuard: one pass per grid threshold over both splits
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(chainguard_blocks, [(a.repo, t, allt) for t in GRID]))
    cg = {t: np.array(b) for t, b in zip(GRID, res)}
    print("ChainGuard grid done", file=sys.stderr, flush=True)

    # DeBERTa: one score per text; candidate thresholds are the validation scores
    sc = deberta_scores(allt)
    sc_tr, sc_te = sc[:ntr], sc[ntr:]

    out = {"dataset": DATASET, "validation_split": "train", "test_split": "test",
           "grid": GRID, "model": MODEL, "selection": "highest validation detection "
           "with validation false-block rate <= ceiling; frozen, then applied once to test",
           "shipped_check": {}, "chainguard_grid": {}, "results": {}}

    # self-check: the shipped threshold must reproduce the stored Safe-Guard totals
    allb, ally = cg[0.78], np.concatenate([y["train"], y["test"]])
    out["shipped_check"] = rates(allb, ally)

    for t in GRID:
        out["chainguard_grid"][str(t)] = {
            "validation": rates(cg[t][:ntr], y["train"]),
            "test": rates(cg[t][ntr:], y["test"])}

    # DeBERTa candidate thresholds: every distinct validation score, plus one
    # above the maximum (blocks nothing)
    deb_cand = sorted(float(t) for t in np.unique(np.concatenate([sc_tr, [1.0 + 1e-9]])))
    for c in CEILINGS:
        key = f"fpr<={c:.3f}"
        sel = select({t: cg[t][:ntr] for t in GRID}, y["train"], c)
        if sel is None:
            out["results"].setdefault(key, {})["chainguard"] = {
                "threshold": None, "note": "no grid threshold meets the ceiling on validation"}
        else:
            _, thr, vr = sel
            out["results"].setdefault(key, {})["chainguard"] = {
                "threshold": thr, "validation": vr, "test": rates(cg[thr][ntr:], y["test"])}
        # DeBERTa: lowest threshold whose validation FPR <= ceiling
        # detection only falls as the threshold rises, so the lowest threshold
        # meeting the ceiling has the highest validation detection
        best = None
        for thr in deb_cand:
            r = rates(sc_tr >= thr, y["train"])
            if r["false_block"] <= c + 1e-12:
                best = (thr, r)
                break
        thr, vr = best
        out["results"][key]["deberta"] = {
            "threshold": thr, "validation": vr, "test": rates(sc_te >= thr, y["test"])}
        for name in ("chainguard", "deberta"):
            e = out["results"][key][name]
            if e.get("test"):
                t = e["test"]
                print(f"{key} {name:<10} thr={e['threshold']:.6g} "
                      f"val det {e['validation']['detected']}/{e['validation']['n_attack']} "
                      f"fb {e['validation']['false_blocked']}/{e['validation']['n_benign']} | "
                      f"TEST det {t['detected']}/{t['n_attack']} ({100*t['detection']:.1f}%) "
                      f"fb {t['false_blocked']}/{t['n_benign']} ({100*t['false_block']:.2f}%)",
                      file=sys.stderr)
            else:
                print(f"{key} {name:<10} {e['note']}", file=sys.stderr)

    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
