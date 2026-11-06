#!/usr/bin/env python3
"""Render the ChainGuard vs learned-guard ROC figure for the manuscript.

Reviewer 3 (Major Point 1) asks for ROC/DET curves with TPR reported at fixed,
realistic false-positive rates.  The right-hand panel is the low-FPR zoom, since
that is the only region in which either detector could be deployed as an
autonomous blocking control.

Palette: categorical slots 1 and 2 of the documented default palette
(#2a78d6, #eb6834), validated with scripts/validate_palette.js --mode light
(all six checks pass).  Series are additionally distinguished by line style and
marker shape so the figure survives greyscale printing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CG = "#2a78d6"      # categorical slot 1
LG = "#eb6834"      # categorical slot 2
INK = "#1a1a19"
MUTED = "#6b6a63"
GRID = "#e2e1dc"


def stair(points):
    """Threshold-achievable operating points as a monotone staircase.

    Drawn as a step rather than interpolated: where the score distribution ties,
    the ROC has gaps that no threshold can land in, and a straight line across
    such a gap would imply operating points the detector cannot actually be set
    to. Only achievable points are shown.
    """
    best: dict[float, float] = {}
    for p in points:
        f = round(p["fpr"], 6)
        best[f] = max(best.get(f, 0.0), round(p["tpr"], 6))
    xs, ys, run = [], [], 0.0
    for f in sorted(best):
        run = max(run, best[f])        # enforce monotonicity in TPR
        xs.append(f)
        ys.append(run)
    if xs[0] > 0.0:
        xs.insert(0, 0.0)
        ys.insert(0, 0.0)
    xs.append(1.0)
    ys.append(max(1.0, ys[-1]))
    return xs, ys


def panel(ax, cg, lg, cg_op, lg_op, xlim, title, annotate):
    ax.plot(*lg, color=LG, lw=2.0, ls="--", drawstyle="steps-post",
            zorder=3, clip_on=True)
    ax.plot(*cg, color=CG, lw=2.0, ls="-", drawstyle="steps-post",
            zorder=4, clip_on=True)
    # achievable operating points, marked so the steps are not read as a sweep
    ax.plot(lg[0][:-1], lg[1][:-1], ls="none", marker="^", ms=4.5, color=LG,
            mec="white", mew=0.7, zorder=5, clip_on=True)
    ax.plot(cg[0][:-1], cg[1][:-1], ls="none", marker="o", ms=4.5, color=CG,
            mec="white", mew=0.7, zorder=5, clip_on=True)

    ax.plot([0, 1], [0, 1], color=GRID, lw=1.0, ls=":", zorder=1)

    for (x, y), c in ((cg_op, CG), (lg_op, LG)):
        ax.plot([x], [y], marker="*", ms=13, color=c, mec="white", mew=1.2, zorder=6)

    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False-block rate (benign sessions)", fontsize=8, color=INK)
    ax.set_title(title, fontsize=8.5, color=INK, pad=5, loc="left")
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=7.5, colors=MUTED, length=3)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(MUTED)

    for text, xy, xytext, colour, ha in annotate:
        ax.annotate(text, xy=xy, xytext=xytext, fontsize=7.2, color=colour,
                    ha=ha, va="center", zorder=7,
                    arrowprops=dict(arrowstyle="-", color=colour, lw=0.8,
                                    shrinkA=0, shrinkB=3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, type=Path)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    sweep = json.loads(a.sweep.read_text())
    base = json.loads(a.baseline.read_text())

    cg = stair(sweep["curves"]["full"]["points"])
    lg = stair(base["roc"])
    cg_op = (0.07, 61 / 80)
    op = base["operating_points"]["default_0.5"]
    lg_op = (op["false_block"], op["detection"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    fig.patch.set_facecolor("white")

    panel(ax1, cg, lg, cg_op, lg_op, (0, 1.0), "(a) Full range", [
        ("ChainGuard", (0.20, 0.9375), (0.40, 0.78), CG, "left"),
        ("DeBERTa-v3 prompt-injection-v2", (0.36, 0.9125), (0.46, 0.47), LG, "left"),
    ])
    ax1.set_ylabel("Attack-session detection", fontsize=8, color=INK)

    panel(ax2, cg, lg, cg_op, lg_op, (0, 0.205), "(b) Low false-block zoom", [
        ("shipped point:\n7.0% FB, 76.3% det.", (0.07, 0.7625), (0.088, 0.945),
         CG, "left"),
        ("no learned-guard\nthreshold reaches\n$\\leq$7% FB above\n1.3% detection", (0.004, 0.013),
         (0.004, 0.46), LG, "left"),
    ])
    ax2.set_xticks([0.0, 0.05, 0.10, 0.15, 0.20])

    fig.tight_layout(pad=0.6)
    fig.savefig(a.out, format="pdf", bbox_inches="tight")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
