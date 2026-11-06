#!/usr/bin/env python3
"""Independent-benchmark evaluation of ChainGuard on third-party datasets.

Answers Reviewer 1 (#1), Reviewer 3 (Major Point 3), Reviewer 4 (#1) and
Reviewer 5 (#3), which require validation on data the author did not write.

Datasets
--------
Neither dataset below was used at any point in ChainGuard's development, which
is what makes them usable here. The two external sets already in the repository
are *not* usable for this purpose and are excluded by design: BIPIA guided the
strict retrieved/tool provenance rule, and the zachz prompt-injection benchmark
is part of the classifier's training corpus (THIRD_PARTY_DATA.md).

  deepset/prompt-injections  662 labelled prompts, English and German, mixing
                             injections with benign requests. Supplies an
                             independent attack-detection estimate and, because
                             it is bilingual, an independent check on the
                             multilingual weakness reported in Section 6.4.

  leolee99/NotInject         339 benign prompts built specifically to measure
                             over-defense in guardrail models: they contain
                             injection-associated trigger vocabulary while being
                             entirely benign. Supplies an independent
                             false-block estimate on the hardest available
                             negatives.

Because both are single-turn prompt datasets with no provenance labels, every
item is evaluated as a one-turn user-sourced session. That is the setting the
data supports; it exercises ChainGuard's lexical, semantic and classifier
layers but not its provenance or cross-turn state, and the results must be read
as covering that subset of the architecture.

Contamination check
-------------------
Every external prompt is compared against the classifier's 941-turn training
corpus by exact match, character 5-gram MinHash Jaccard and word containment,
so the independence claim is measured rather than asserted.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.parse
import urllib.request
from pathlib import Path

CACHE = "external_cache"


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fetch_rows(dataset: str, config: str, split: str, cache: Path) -> list[dict]:
    f = cache / f"{dataset.replace('/', '_')}__{split}.json"
    if f.is_file():
        return json.loads(f.read_text(encoding="utf-8"))
    out, offset = [], 0
    while True:
        url = ("https://datasets-server.huggingface.co/rows?"
               f"dataset={urllib.parse.quote(dataset, safe='')}"
               f"&config={config}&split={urllib.parse.quote(split)}"
               f"&offset={offset}&length=100")
        req = urllib.request.Request(url, headers={"User-Agent": "research"})
        with urllib.request.urlopen(req, timeout=60) as r:
            page = json.load(r)
        rows = [x["row"] for x in page.get("rows", [])]
        out.extend(rows)
        total = page.get("num_rows_total", len(out))
        offset += len(rows)
        if not rows or offset >= total:
            break
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"  fetched {len(out)} rows -> {f}", file=sys.stderr)
    return out


def text_of(row: dict) -> str:
    for k in ("text", "prompt", "sentence", "content", "input"):
        if k in row and isinstance(row[k], str):
            return row[k]
    raise KeyError(f"no text field in {list(row)}")


def label_of(row: dict) -> int | None:
    for k in ("label", "labels", "is_injection"):
        if k in row:
            v = row[k]
            if isinstance(v, bool):
                return int(v)
            if isinstance(v, (int, float)):
                return int(v)
            if isinstance(v, str) and v.isdigit():
                return int(v)
    return None


def evaluate(texts: list[str], mods) -> list[dict]:
    ChainGuardV7, Action = mods["ChainGuardV7"], mods["Action"]
    out = []
    for t in texts:
        g = ChainGuardV7()
        v = g.step(t, source="user")
        det = v.details or {}
        v7 = (det.get("v7") or {}).get("escalation_reason")
        reasons = ([f"v7:{v7}"] if v7 else
                   list((det.get("decision") or {}).get("reasons") or []))
        out.append({"blocked": v.action is Action.BLOCK,
                    "challenged": v.action.value == "soft_challenge",
                    "reasons": reasons})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache", type=Path, default=Path(CACHE))
    ap.add_argument("--skip-overlap", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, str((a.repo / "src").resolve()))
    from pinj.chainguard import ChainGuardV7
    from pinj.chainguard.accumulator import Action
    mods = {"ChainGuardV7": ChainGuardV7, "Action": Action}

    results: dict = {"datasets": {}}

    # ---- deepset/prompt-injections: labelled attacks + benign, EN/DE ----
    print("deepset/prompt-injections", file=sys.stderr)
    rows = []
    for split in ("train", "test"):
        rows.extend(fetch_rows("deepset/prompt-injections", "default", split, a.cache))
    items = [(text_of(r), label_of(r)) for r in rows]
    items = [(t, l) for t, l in items if l is not None and t.strip()]
    atk = [t for t, l in items if l == 1]
    ben = [t for t, l in items if l == 0]
    ra, rb = evaluate(atk, mods), evaluate(ben, mods)
    tp = sum(x["blocked"] for x in ra)
    fp = sum(x["blocked"] for x in rb)
    results["datasets"]["deepset/prompt-injections"] = {
        "n_attack": len(atk), "n_benign": len(ben),
        "detected": tp, "detection": tp / len(atk) if atk else 0,
        "detection_ci": wilson(tp, len(atk)),
        "false_blocked": fp, "false_block": fp / len(ben) if ben else 0,
        "false_block_ci": wilson(fp, len(ben)),
        "challenge_attack": sum(x["challenged"] for x in ra),
        "challenge_benign": sum(x["challenged"] for x in rb),
        "attack_texts": atk, "benign_texts": ben,
        "attack_results": ra, "benign_results": rb,
    }
    print(f"  detection {tp}/{len(atk)}  false block {fp}/{len(ben)}", file=sys.stderr)

    # ---- NotInject: benign-only over-defense probe ----
    print("leolee99/NotInject", file=sys.stderr)
    ni: list[str] = []
    for split in ("NotInject_one", "NotInject_two", "NotInject_three"):
        ni.extend(text_of(r) for r in fetch_rows("leolee99/NotInject", "default", split, a.cache))
    ni = [t for t in ni if t.strip()]
    rn = evaluate(ni, mods)
    fp_n = sum(x["blocked"] for x in rn)
    results["datasets"]["leolee99/NotInject"] = {
        "n_benign": len(ni), "false_blocked": fp_n,
        "false_block": fp_n / len(ni) if ni else 0,
        "false_block_ci": wilson(fp_n, len(ni)),
        "challenge_benign": sum(x["challenged"] for x in rn),
        "benign_texts": ni, "benign_results": rn,
    }
    print(f"  false block {fp_n}/{len(ni)}", file=sys.stderr)

    # ---- contamination check against the classifier training corpus ----
    if not a.skip_overlap:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from classifier_report import load_corpus, overlap_report
        train = load_corpus(a.repo)
        for name, key in (("deepset/prompt-injections", "attack_texts"),
                          ("leolee99/NotInject", "benign_texts")):
            texts = results["datasets"][name][key]
            probe = [{"text": t, "kind": "attack", "session": f"x{i}",
                      "turn": 0, "category": name} for i, t in enumerate(texts)]
            ov = overlap_report(train, probe, 128, 1701)
            results["datasets"][name]["overlap_vs_training"] = {
                k: v for k, v in ov.items() if k != "worst"}
            print(f"  overlap {name}: exact={ov['exact']} "
                  f"jmax={ov['j_max']:.3f} cmax={ov['c_max']:.3f} "
                  f"c>0.5={ov['n_c50']}", file=sys.stderr)

    a.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
