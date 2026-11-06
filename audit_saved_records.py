#!/usr/bin/env python3
"""Recompute every headline number in the ChainGuard manuscript from the saved
record files in results/ and check it against the value printed in the paper.

This script executes no detector, no target model, and no network request. It
audits stored outcomes only. To re-run the experiments themselves, see README.md;
RESULT_MAP records which results were re-run and found identical.

Usage:  python3 audit_saved_records.py [--results DIR] [--export-turns FILE]
Exit status is 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval in percent."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


class Audit:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, bool]] = []

    def check(self, label: str, got, expected, tol: float = 0.0) -> None:
        if isinstance(expected, float) or isinstance(got, float):
            ok = abs(float(got) - float(expected)) <= tol
        else:
            ok = got == expected
        self.rows.append((label, str(got), str(expected), ok))

    def report(self) -> int:
        w = max(len(r[0]) for r in self.rows)
        failed = 0
        for label, got, exp, ok in self.rows:
            if not ok:
                failed += 1
            print(f"{'PASS' if ok else 'FAIL'}  {label:<{w}}  recomputed={got:<22} manuscript={exp}")
        print(f"\n{len(self.rows) - failed}/{len(self.rows)} checks passed.")
        return 0 if failed == 0 else 1


def load(results: pathlib.Path, name: str):
    with (results / name).open(encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# result-to-script map (Section 7 of the manuscript / Table~\ref{tab:resultmap})
# --------------------------------------------------------------------------
# rerun:   verified   - re-run from this repository on 6 Nov 2026; the output
#                         was identical to the stored file in results/
#           rerunnable - code and inputs are here or public, but the run
#                         was not repeated (local target-model generation)
# records: per-item     - stored per-item records; recomputed by this script
#          summary      - aggregate counts only
RESULT_MAP = [
    # result, script, input data, stored output, rerun, records
    ("Primary holdout detection and false blocks (Sec. 6.1)",
     "end_to_end.py --models", "data/post_freeze_source_challenge_v2",
     "end_to_end_v0.7.2.json", "verified", "per-item"),
    ("Classifier cross-validation, calibration, overlap (Sec. 4.7.1)",
     "classifier_report.py", "src/pinj/chainguard/resources (941 turns)",
     "classifier_report_v0.7.2.json", "verified", "summary"),
    ("Component ablation, eleven arms (Sec. 6.5)",
     "ablation_study.py", "data/post_freeze_source_challenge_v2",
     "ablation_v0.7.2.json", "verified", "per-item"),
    ("Threshold sensitivity sweep (Sec. 6.6)",
     "threshold_sweep.py --steps 11", "data/post_freeze_source_challenge_v2",
     "threshold_sweep_v0.7.2.json", "verified", "per-item"),
    ("deepset and NotInject benchmarks (Sec. 6.10)",
     "external_benchmark.py", "deepset/prompt-injections, leolee99/NotInject",
     "external_benchmark_v0.7.2.json", "verified", "per-item"),
    ("Gandalf and Safe-Guard benchmarks (Sec. 6.10)",
     "large_public_benchmarks.py", "Lakera/gandalf, xTRam1/safe-guard",
     "large_public_benchmarks_v0.7.2.json", "verified", "summary"),
    ("InjecAgent two-turn evaluation (Sec. 6.10)",
     "injecagent_eval.py", "InjecAgent test cases",
     "injecagent_v0.7.2.json", "verified", "summary"),
    ("Matched learned-guard baseline (Sec. 6.3)",
     "learned_baseline.py", "data/post_freeze_source_challenge_v2 + DeBERTa",
     "learned_baseline_v0.7.2.json", "verified", "per-item"),
    ("ROC comparison figure (Fig. 5)",
     "make_roc_figure.py", "learned_baseline_v0.7.2.json",
     "Fig/fig_roc_comparison.pdf", "rerunnable", "per-item"),
    ("DeBERTa comparator on independent data (Sec. 6.10)",
     "external_benchmark.py, large_public_benchmarks.py, injecagent_eval.py",
     "same independent datasets + DeBERTa",
     "*_deberta_v0.7.2.json (3 files)", "rerunnable", "summary"),
    ("End-to-end target-model replay (Sec. 6.9)",
     "end_to_end.py", "holdout + three local target models",
     "end_to_end_v0.7.2.json", "rerunnable", "per-item"),
    ("Gandalf end-to-end replication (Sec. 6.9)",
     "gandalf_end_to_end.py", "Lakera/gandalf + Qwen2.5-1.5B",
     "gandalf_end_to_end_v0.7.2.json", "rerunnable", "summary"),
    ("Query-bounded adaptive evaluation (Sec. 6.11)",
     "adaptive_redteam.py", "61 blocked holdout sessions + target models",
     "adaptive_*_v0.7.2.json (3 files)", "rerunnable", "per-item"),
]


def write_result_map(path: pathlib.Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["result", "script", "input_data", "stored_output", "rerun", "records"])
        w.writerows(RESULT_MAP)


# --------------------------------------------------------------------------
# audits
# --------------------------------------------------------------------------
def audit_primary_holdout(a: Audit, results: pathlib.Path) -> None:
    e2e = load(results, "end_to_end_v0.7.2.json")
    # chainguard_blocks maps a session id to the 1-indexed first blocked turn,
    # so the number of blocked sessions is the number of keys.
    blocked_attacks = len(e2e["chainguard_blocks"]["attacks"])
    blocked_benign = len(e2e["chainguard_blocks"]["benign"])
    a.check("holdout attack sessions blocked", blocked_attacks, 61)
    a.check("holdout attack sessions total", e2e["n_attacks"], 80)
    a.check("holdout benign sessions blocked", blocked_benign, 7)
    a.check("holdout benign sessions total", e2e["n_benign"], 100)
    lo, hi = wilson(blocked_attacks, e2e["n_attacks"])
    a.check("detection CI lower (%)", round(lo, 2), 65.86, tol=0.02)
    a.check("detection CI upper (%)", round(hi, 2), 84.24, tol=0.02)


def audit_end_to_end(a: Audit, results: pathlib.Path) -> None:
    e2e = load(results, "end_to_end_v0.7.2.json")
    blocks = e2e["chainguard_blocks"]
    pairs = {"none": 0, "input": 0, "input_egress": 0}
    answered = 0
    n_pairs = 0
    n_benign_pairs = 0
    for model in e2e["models"].values():
        for rec in model["attacks"]:
            n_pairs += 1
            first_block = blocks["attacks"].get(rec["id"])
            if any(t.get("leak") for t in rec["turns"]):
                pairs["none"] += 1
            # the input guard truncates the session at its first blocked turn
            visible = (rec["turns"] if first_block is None
                       else rec["turns"][:first_block - 1])
            if any(t.get("leak") for t in visible):
                pairs["input"] += 1
            if any(t.get("leak") and not t.get("egress_blocked") for t in visible):
                pairs["input_egress"] += 1
        for rec in model["benign"]:
            n_benign_pairs += 1
            first_block = blocks["benign"].get(rec["id"])
            visible = (rec["turns"] if first_block is None
                       else rec["turns"][:first_block - 1])
            if visible and not any(t.get("egress_blocked") for t in visible):
                answered += 1
    a.check("model-session pairs (attacks)", n_pairs, 240)
    a.check("unguarded exact-canary disclosure", pairs["none"], 39)
    a.check("disclosure after ingress", pairs["input"], 10)
    a.check("disclosure after ingress+egress", pairs["input_egress"], 0)
    a.check("benign model-session pairs", n_benign_pairs, 300)
    a.check("benign replies retained after both filters", answered, 272)

    gandalf = load(results, "gandalf_end_to_end_v0.7.2.json")
    a.check("Gandalf replication n", gandalf["n"], 250)
    a.check("Gandalf unguarded disclosure (%)",
            round(100 * gandalf["compromise_none"] / gandalf["n"], 1), 20.0, tol=0.05)
    a.check("Gandalf disclosure after ingress (%)",
            round(100 * gandalf["compromise_ingress"] / gandalf["n"], 1), 4.4, tol=0.05)
    a.check("Gandalf disclosure after egress",
            gandalf["compromise_ingress_egress"], 0)


def audit_ablation(a: Audit, results: pathlib.Path) -> None:
    abl = load(results, "ablation_v0.7.2.json")
    a.check("ablation arms recorded", len(abl), 11)
    a.check("ablation full arm detections", abl["full"]["attacks_blocked"], 61)
    a.check("ablation no_state detections", abl["no_state"]["attacks_blocked"], 58)
    a.check("ablation stateless detections", abl["stateless"]["attacks_blocked"], 58)
    a.check("ablation no_classifier detections", abl["no_classifier"]["attacks_blocked"], 57)
    a.check("ablation no_provenance detections", abl["no_provenance"]["attacks_blocked"], 59)
    a.check("ablation no_provenance false blocks", abl["no_provenance"]["benign_blocked"], 2)
    a.check("ablation no_decoding detections", abl["no_decoding"]["attacks_blocked"], 61)
    a.check("ablation no_transparent detections", abl["no_transparent"]["attacks_blocked"], 61)
    # exact one-sided McNemar for the bounded-state arm: 61 vs 58, 3 discordant
    # pairs all favouring the full configuration -> p = 0.5**3 = 0.125 one-sided,
    # 0.250 two-sided.
    disc = abl["full"]["attacks_blocked"] - abl["no_state"]["attacks_blocked"]
    a.check("bounded-state two-sided exact McNemar p",
            round(2 * 0.5 ** disc, 3), 0.250, tol=0.001)


def audit_learned(a: Audit, results: pathlib.Path) -> None:
    lb = load(results, "learned_baseline_v0.7.2.json")
    a.check("learned baseline attack sessions", lb["n_attacks"], 80)
    a.check("learned baseline benign sessions", lb["n_benign"], 100)
    a.check("learned baseline AUC", round(lb["auc"], 3), 0.829, tol=0.001)

    # Table 12: highest detection within each false-block ceiling, recomputed
    # from the stored per-session DeBERTa scores.
    atk = list(lb["session_scores"]["attacks"].values())
    ben = list(lb["session_scores"]["benign"].values())
    for ceiling, expected in ((0.005, 1), (0.010, 1), (0.07, 1),
                              (0.10, 38), (0.20, 51), (0.36, 73)):
        best = -1
        for thr in sorted(set(atk + ben)):
            if sum(1 for b in ben if b >= thr) <= ceiling * len(ben):
                best = max(best, sum(1 for x in atk if x >= thr))
        a.check(f"DeBERTa detection at <={ceiling:.1%} false-block budget", best, expected)

    # the ChainGuard column of the same table comes from the stored sweep grid
    sweep = load(results, "threshold_sweep_v0.7.2.json")["curves"]["full"]["points"]
    for ceiling, expected in ((0.07, 61), (0.10, 65), (0.20, 75), (0.36, 75)):
        eligible = [p["attacks_blocked"] for p in sweep if p["fpr"] <= ceiling + 1e-9]
        a.check(f"ChainGuard detection at <={ceiling:.0%} false-block budget",
                max(eligible) if eligible else None, expected)
    a.check("ChainGuard minimum recorded false-block rate",
            min(p["fpr"] for p in sweep), 0.07, tol=1e-9)


def audit_external(a: Audit, results: pathlib.Path) -> None:
    ext = load(results, "external_benchmark_v0.7.2.json")["datasets"]
    dp = ext["deepset/prompt-injections"]
    ni = ext["leolee99/NotInject"]
    a.check("deepset detection (%)", round(100 * dp["detected"] / dp["n_attack"], 1), 17.1, tol=0.05)
    a.check("deepset false blocks", dp["false_blocked"], 0)
    a.check("NotInject false block (%)",
            round(100 * ni["false_blocked"] / ni["n_benign"], 1), 6.8, tol=0.05)

    lg = load(results, "large_public_benchmarks_v0.7.2.json")
    g = lg["Lakera/gandalf_ignore_instructions"]
    sg = lg["xTRam1/safe-guard-prompt-injection"]
    a.check("Gandalf detection (%)", round(100 * g["detected"] / g["n_attack"], 1), 70.4, tol=0.05)
    a.check("Safe-Guard detection (%)",
            round(100 * sg["detected"] / sg["n_attack"], 1), 41.4, tol=0.05)
    a.check("Safe-Guard false block (%)",
            round(100 * sg["false_blocked"] / sg["n_benign"], 1), 1.9, tol=0.05)

    ia = load(results, "injecagent_v0.7.2.json")
    a.check("InjecAgent direct-harm detection (%)",
            round(100 * ia["direct_harm"]["detected"] / ia["direct_harm"]["n"], 1),
            11.4, tol=0.05)
    a.check("InjecAgent data-stealing detection (%)",
            round(100 * ia["data_stealing"]["detected"] / ia["data_stealing"]["n"], 1),
            68.6, tol=0.05)
    a.check("InjecAgent user-turn false blocks",
            ia["pooled"]["user_turn_false_blocks"], 0)

    total = (g["n_attack"] + sg["n_attack"] + sg["n_benign"]
             + dp["n_attack"] + dp["n_benign"] + ni["n_benign"] + ia["pooled"]["n"])
    a.check("independent evaluation items", total, 13351)


def audit_adaptive(a: Audit, results: pathlib.Path) -> None:
    no_fa = load(results, "adaptive_nofa_v0.7.2.json")
    fa = load(results, "adaptive_fa_tiny_v0.7.2.json")
    a.check("adaptive targets", no_fa["n_targets"], 61)
    a.check("English-rewrite evasions (no Persian)", no_fa["evaded"], 9)
    a.check("evasions with Persian switching", fa["evaded"], 61)
    a.check("adaptive query budget per target", no_fa["budget"], 20)


def export_turns(results: pathlib.Path, out: pathlib.Path) -> int:
    """Export every recorded target-model turn with its ingress/egress decision."""
    e2e = load(results, "end_to_end_v0.7.2.json")
    blocks = e2e["chainguard_blocks"]
    n = 0
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "session", "kind", "category", "turn", "source",
                    "ingress_blocked", "leak", "egress_risk", "egress_blocked",
                    "egress_reasons"])
        for mk, model in e2e["models"].items():
            for kind in ("attacks", "benign"):
                for rec in model[kind]:
                    ingress = rec["id"] in blocks[kind]
                    for i, t in enumerate(rec["turns"], 1):
                        w.writerow([mk, rec["id"], kind, rec.get("category", ""), i,
                                    t.get("source", ""), int(ingress),
                                    int(bool(t.get("leak"))), t.get("egress_risk", ""),
                                    int(bool(t.get("egress_blocked"))),
                                    ";".join(t.get("egress_reasons") or [])])
                        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=pathlib.Path, default=HERE / "results",
                    help="directory holding the *_v0.7.2.json record files")
    ap.add_argument("--export-turns", type=pathlib.Path,
                    help="also write every recorded target-model turn to this CSV")
    ap.add_argument("--write-result-map", type=pathlib.Path,
                    help="write the result-to-script map to this CSV")
    args = ap.parse_args()

    if not args.results.is_dir():
        print(f"results directory not found: {args.results}", file=sys.stderr)
        return 2

    a = Audit()
    audit_primary_holdout(a, args.results)
    audit_end_to_end(a, args.results)
    audit_ablation(a, args.results)
    audit_learned(a, args.results)
    audit_external(a, args.results)
    audit_adaptive(a, args.results)
    rc = a.report()

    if args.export_turns:
        n = export_turns(args.results, args.export_turns)
        print(f"\nExported {n} recorded target-model turns to {args.export_turns}")
    if args.write_result_map:
        write_result_map(args.write_result_map)
        print(f"Wrote result-to-script map to {args.write_result_map}")

    print("\nRe-run status of each reported result:")
    for result, script, _data, _out, rerun, records in RESULT_MAP:
        print(f"  [{rerun:<10}|{records:<8}] {result}  <-  {script}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
