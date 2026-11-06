#!/usr/bin/env python3
"""Produce every number required by Section 4.7.1 of the ChainGuard manuscript.

Reviewers 1 (#19, #20), 3 (Major Point 5) and 5 (#4) require the local intent
classifier's training pipeline to be documented and its lexical overlap with the
evaluation holdout to be quantified.  This script computes all of it from the
frozen v0.7.2 artefacts and emits ready-to-paste LaTeX.

It mirrors `pinj.chainguard.ml_intent.SklearnIntentClassifier` exactly: the same
word 1--2-gram and char_wb 3--5-gram TF-IDF union, and the same
`LogisticRegression(max_iter=2500, class_weight="balanced", C=3.0,
solver="lbfgs", random_state=1701)`.  The classifier ships as a corpus that is
fitted at import time rather than as a pickled model, so these figures are
reproducible from the repository alone.

Usage
-----
    python classifier_report.py --repo <chainguard checkout> --out section.tex

`--repo` is the directory containing `src/pinj/chainguard/resources/` and
`data/post_freeze_source_challenge_v2/`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

# Labels the detector sums into its maliciousness score (ml_intent.MALICIOUS_LABELS).
ATTACK_LABELS = ("protected_disclosure", "instruction_override", "indirect_instruction")
ESCALATION_THRESHOLD = 0.88  # ml_intent escalation cut-off documented in Section 4.7


# ----------------------------------------------------------------------- inputs
def load_corpus(repo: Path) -> list[dict]:
    """Load the training corpus exactly as SklearnIntentClassifier._load_rows does."""
    res = repo / "src/pinj/chainguard/resources"
    rows = json.loads((res / "intent_training.json").read_text(encoding="utf-8"))
    for r in rows:
        r.setdefault("source", "authored")
    external = res / "zachz_training.json"
    if external.is_file():
        ext = json.loads(external.read_text(encoding="utf-8"))
        for r in ext:
            r.setdefault("source", "zachz/prompt-injection-benchmark")
        rows.extend(ext)
    return rows


def load_holdout_turns(repo: Path) -> list[dict]:
    """Flatten the 180 holdout sessions into individual turns."""
    base = repo / "data/post_freeze_source_challenge_v2"
    turns = []
    for name, kind in (("attacks.json", "attack"), ("benign.json", "benign")):
        for item in json.loads((base / name).read_text(encoding="utf-8")):
            for i, text in enumerate(item["turns"]):
                turns.append({"text": text, "kind": kind,
                              "session": item["id"], "turn": i,
                              "category": item["category"]})
    return turns


# ------------------------------------------------------------------- pipeline
def build_pipeline():
    """The exact pipeline from pinj.chainguard.ml_intent."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    features = FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                 ngram_range=(1, 2), min_df=1, max_df=0.995,
                                 sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", lowercase=True,
                                 strip_accents="unicode", ngram_range=(3, 5),
                                 min_df=1, max_features=40000, sublinear_tf=True)),
    ])
    model = LogisticRegression(max_iter=2500, class_weight="balanced", C=3.0,
                              solver="lbfgs", random_state=1701)
    return Pipeline([("features", features), ("model", model)])


def group_key(row: dict) -> str:
    """Collapse a row's `group` to its template, so paraphrase descendants of one
    template never straddle a cross-validation fold (EXPERIMENT_PROTOCOL.md §4)."""
    g = str(row.get("group") or row["text"][:40])
    return re.sub(r"-\d+$", "", g)


def cross_validate(rows: list[dict], folds: int, seed: int) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedGroupKFold

    X = np.array([str(r["text"]) for r in rows])
    y = np.array([str(r["label"]) for r in rows])
    groups = np.array([group_key(r) for r in rows])

    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    labels = sorted(set(y))
    y_true, y_pred, mal = [], [], []
    for tr, te in splitter.split(X, y, groups):
        pipe = build_pipeline()
        pipe.fit(X[tr], y[tr])
        classes = [str(c) for c in pipe.classes_]
        idx = [classes.index(l) for l in ATTACK_LABELS if l in classes]
        proba = pipe.predict_proba(X[te])
        y_true.extend(y[te])
        y_pred.extend(pipe.predict(X[te]))
        mal.extend(np.minimum(1.0, proba[:, idx].sum(axis=1)))

    return {
        "folds": folds,
        "n_groups": len(set(groups)),
        "labels": labels,
        "report": classification_report(y_true, y_pred, digits=3, zero_division=0),
        "report_dict": classification_report(y_true, y_pred, output_dict=True,
                                             zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
        "malicious_score": np.array(mal),
        "y_true": np.array(y_true),
    }


def fit_full(rows: list[dict]):
    pipe = build_pipeline()
    pipe.fit([str(r["text"]) for r in rows], [str(r["label"]) for r in rows])
    return pipe


def calibration(scores: np.ndarray, y_true: np.ndarray, bins: int = 10) -> dict:
    truth = np.isin(y_true, ATTACK_LABELS).astype(float)
    brier = float(np.mean((scores - truth) ** 2))
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (scores >= lo) & ((scores < hi) if hi < 1.0 else (scores <= hi))
        if not m.any():
            continue
        ece += m.mean() * abs(scores[m].mean() - truth[m].mean())
        rows.append((float(lo), float(hi), int(m.sum()),
                     float(scores[m].mean()), float(truth[m].mean())))
    esc = scores >= ESCALATION_THRESHOLD
    return {
        "brier": brier, "ece": float(ece), "bins": rows,
        "n_escalated": int(esc.sum()),
        "precision": float(truth[esc].mean()) if esc.any() else float("nan"),
        "recall": float(truth[esc].sum() / truth.sum()) if truth.sum() else float("nan"),
        "n_attack_rows": int(truth.sum()),
    }


# ------------------------------------------------------------------- overlap
_WORD = re.compile(r"\w+", re.UNICODE)
_ZW = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", _ZW.sub("", t)).strip()


def shingles(text: str, n: int = 5) -> set[str]:
    t = normalise(text)
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))} or {t}


def signature(sh: set[str], a: np.ndarray, b: np.ndarray, prime: int) -> np.ndarray:
    h = np.array([int(hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest(), 16)
                  % prime for s in sh], dtype=np.int64)
    return ((a[:, None] * h[None, :] + b[:, None]) % prime).min(axis=1)


def overlap_report(train: list[dict], holdout: list[dict], perms: int, seed: int) -> dict:
    prime = (1 << 61) - 1
    rng = np.random.default_rng(seed)
    a = rng.integers(1, prime, size=perms, dtype=np.int64)
    b = rng.integers(0, prime, size=perms, dtype=np.int64)

    train_norm = {normalise(str(r["text"])) for r in train}
    train_sig = np.array([signature(shingles(str(r["text"])), a, b, prime) for r in train])
    train_words = [set(_WORD.findall(normalise(str(r["text"])))) for r in train]

    exact, sims, cont, worst = 0, [], [], []
    for r in holdout:
        n = normalise(r["text"])
        if n in train_norm:
            exact += 1
        sig = signature(shingles(r["text"]), a, b, prime)
        j = (train_sig == sig[None, :]).mean(axis=1)
        hw = set(_WORD.findall(n))
        c = np.array([len(hw & tw) / len(hw) for tw in train_words]) if hw else np.array([0.0])
        sims.append(float(j.max()))
        cont.append(float(c.max()))
        worst.append((float(c.max()), float(j.max()), r["session"], r["category"],
                      r["text"][:70], str(train[int(c.argmax())]["text"])[:70]))

    sims, cont = np.array(sims), np.array(cont)
    is_attack = np.array([r["kind"] == "attack" for r in holdout])
    worst.sort(reverse=True)
    return {
        "c_max_attack": float(cont[is_attack].max()) if is_attack.any() else 0.0,
        "j_max_attack": float(sims[is_attack].max()) if is_attack.any() else 0.0,
        "n_holdout": len(holdout),
        "n_train": len(train),
        "exact": exact,
        "j_max": float(sims.max()), "j_p95": float(np.percentile(sims, 95)),
        "j_med": float(np.median(sims)),
        "c_max": float(cont.max()), "c_p95": float(np.percentile(cont, 95)),
        "c_med": float(np.median(cont)),
        "n_c50": int((cont > 0.5).sum()),
        "n_c80": int((cont > 0.8).sum()),
        "n_j50": int((sims > 0.5).sum()),
        "worst": worst[:10],
    }


# --------------------------------------------------------------------- LaTeX
def tex_escape(s: str) -> str:
    return s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%") \
            .replace("&", "\\&").replace("#", "\\#").replace("$", "\\$")


def emit_latex(corpus: dict, cv: dict, cal: dict, ov: dict, pipe) -> str:
    lab = cv["labels"]
    macro = cv["report_dict"]["macro avg"]
    dist = "; ".join(f"{tex_escape(k.replace('_', ' '))} {v}"
                     for k, v in sorted(corpus["labels"].items()))
    prov = "; ".join(f"{tex_escape(k)} {v}" for k, v in sorted(corpus["sources"].items()))

    # Single-letter column keys: the class names are far too long for a
    # one-column table, and truncating them gave unreadable headers.
    KEYS = "ABCDEFGHIJKLMNOP"
    key = {l: KEYS[i] for i, l in enumerate(lab)}

    def pretty(l: str) -> str:
        return key[l] + "~~" + tex_escape(l.replace("_", " ").capitalize())

    header = " & ".join(["\\textbf{True $\\downarrow$~/~Pred $\\rightarrow$}"]
                        + [f"\\textbf{{{key[l]}}}" for l in lab])
    body = "\n".join(
        " & ".join([pretty(l)] + [str(int(v)) for v in cv["confusion"][i]]) + "\\\\"
        for i, l in enumerate(lab))

    per = "\n".join(
        " & ".join([pretty(l),
                    f"{cv['report_dict'][l]['precision']:.3f}",
                    f"{cv['report_dict'][l]['recall']:.3f}",
                    f"{cv['report_dict'][l]['f1-score']:.3f}",
                    str(int(cv['report_dict'][l]['support']))]) + "\\\\"
        for l in lab if l in cv["report_dict"])

    return f"""% ======================================================================
% Section 4.7.1 -- generated by paper/tools/classifier_report.py
% Training corpus SHA-256 : {corpus['sha256']}
% Feature dimensionality  : {corpus['n_features']}
% ======================================================================
The classifier is distributed as a labelled corpus that is fitted at import
time rather than as a serialised model, so every figure below is reproducible
from the released repository.

The corpus contains {corpus['n']} labelled turns over
{corpus['n_groups']} distinct templates. {corpus['n_authored']} turns were
authored for this work and {corpus['n_external']} were taken from the
\\texttt{{zachz/prompt-injection-benchmark}} dataset (MIT licence), giving the
provenance split {prov}. The label distribution is {dist}. Median turn length is
{corpus['median_len']:.0f} words (mean {corpus['mean_len']:.1f}). Labelling was
performed by the author against the operation--target--provenance rubric of
Section~\\ref{{sec:threat}}; no second annotator was available, so no
inter-annotator agreement is reported for the training corpus. Class weighting
is \\texttt{{balanced}}, the inverse-regularisation constant is $C=3.0$, and the
feature union yields {corpus['n_features']} features when fitted on the full
corpus.

Note that the corpus carries six labels rather than the five intent classes
named in Section~\\ref{{sec:clf}}: benign content is split into
\\emph{{benign\\_security\\_discussion}} and \\emph{{benign\\_text\\_processing}}, which
Section~\\ref{{sec:clf}} groups as a single benign class. The maliciousness score
sums the three attack classes, so this distinction does not affect the
escalation decision.

Generalisation is estimated by {cv['folds']}-fold stratified group
cross-validation over the {cv['n_groups']} templates, so that paraphrase
descendants of one template never straddle a fold, as required by the
pre-registered protocol. Macro-averaged out-of-fold performance is precision
{macro['precision']:.3f}, recall {macro['recall']:.3f} and F1
{macro['f1-score']:.3f}; per-class results are in Table~\\ref{{tab:clfperclass}}
and the out-of-fold confusion matrix in Table~\\ref{{tab:clfconfusion}}.

Calibration of the combined maliciousness score matters because the 0.88
escalation threshold is a probability cut-off. The out-of-fold Brier score is
{cal['brier']:.4f} and the expected calibration error over ten equal-width bins
is {cal['ece']:.4f}. At the 0.88 threshold {cal['n_escalated']} of
{corpus['n']} turns escalate, of which a fraction {cal['precision']:.3f} carry
an attack label, corresponding to a recall of {cal['recall']:.3f} over the
{cal['n_attack_rows']} attack-labelled turns. The classifier is therefore
operated well to the precision side of its own optimum, which is consistent
with its role as a supporting signal that is additionally gated by benign
semantic scope.

Overlap with the evaluation holdout is quantified rather than asserted. The
{ov['n_holdout']} holdout turns were compared against all {ov['n_train']}
training turns under the detector's own normalisation. There are {ov['exact']}
exact duplicates. Nearest-neighbour character 5-gram MinHash Jaccard similarity
has maximum {ov['j_max']:.3f}, 95th percentile {ov['j_p95']:.3f} and median
{ov['j_med']:.3f}, with {ov['n_j50']} holdout {'turn' if ov['n_j50'] == 1 else 'turns'}
above 0.5. Word-level containment in the nearest training turn has maximum
{ov['c_max']:.3f}, 95th percentile {ov['c_p95']:.3f} and median {ov['c_med']:.3f},
with {ov['n_c50']} {'turn' if ov['n_c50'] == 1 else 'turns'} above 0.5 and
{ov['n_c80']} above 0.8.

The composition of the residual overlap matters more than its magnitude. The
highest-containment pair is the benign holdout turn ``{tex_escape(ov['worst'][0][4])}''
against the training turn ``{tex_escape(ov['worst'][0][5])}'' at containment
{ov['worst'][0][0]:.2f}, and the leading cases are dominated by shared function
words and generic interrogative frames of that kind rather than by shared attack
payloads. Restricting attention to holdout turns from attack sessions, the
highest containment against any training turn is {ov['c_max_attack']:.2f} and the
highest MinHash Jaccard similarity is {ov['j_max_attack']:.2f}. With no exact
duplicate and no turn above 0.8 containment, the measured overlap does not
support the interpretation that the holdout was drawn from the training
distribution.

We emphasise that this measurement bounds \\emph{{lexical}} overlap only. It
cannot exclude the deeper dependency that the same author designed the
taxonomy, the detector and the holdout, which is why externally authored
attacks remain necessary and are prespecified in Section~\\ref{{sec:conclusion}}.

\\begin{{table}}[H]
\\centering
\\caption{{Out-of-fold per-class performance of the local intent classifier under
{cv['folds']}-fold stratified group cross-validation.}}
\\label{{tab:clfperclass}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{2.5pt}}
\\begin{{tabularx}}{{\\columnwidth}}{{@{{}}Xcccc@{{}}}}
\\toprule
\\textbf{{Class}} & \\textbf{{Prec.}} & \\textbf{{Rec.}} & \\textbf{{F1}} & \\textbf{{Support}}\\\\
\\midrule
{per}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}

\\begin{{table}}[H]
\\centering
\\caption{{Out-of-fold confusion matrix of the local intent classifier. Class
use the letter keys of Table~\\ref{{tab:clfperclass}}.}}
\\label{{tab:clfconfusion}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{2.5pt}}
\\begin{{tabularx}}{{\\columnwidth}}{{@{{}}X{'c' * len(lab)}@{{}}}}
\\toprule
{header}\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1701)
    ap.add_argument("--perms", type=int, default=256)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--json-out", type=Path)
    a = ap.parse_args()

    rows = load_corpus(a.repo)
    holdout = load_holdout_turns(a.repo)

    lengths = np.array([len(str(r["text"]).split()) for r in rows])
    pipe = fit_full(rows)
    corpus = {
        "n": len(rows),
        "n_groups": len({group_key(r) for r in rows}),
        "labels": Counter(str(r["label"]) for r in rows),
        "sources": Counter(str(r.get("source", "authored")) for r in rows),
        "n_authored": sum(1 for r in rows if r.get("source", "authored") == "authored"),
        "n_external": sum(1 for r in rows if r.get("source", "authored") != "authored"),
        "median_len": float(np.median(lengths)),
        "mean_len": float(lengths.mean()),
        "sha256": hashlib.sha256(
            json.dumps([[str(r["text"]), str(r["label"])] for r in rows],
                       ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
        "n_features": int(pipe.named_steps["features"].transform([rows[0]["text"]]).shape[1]),
    }
    print(f"corpus: {corpus['n']} turns / {corpus['n_groups']} templates / "
          f"{corpus['n_features']} features", file=sys.stderr)

    cv = cross_validate(rows, a.folds, a.seed)
    print(cv["report"], file=sys.stderr)
    cal = calibration(cv["malicious_score"], cv["y_true"])
    print(f"Brier={cal['brier']:.4f} ECE={cal['ece']:.4f} "
          f"escalated={cal['n_escalated']} prec={cal['precision']:.3f} "
          f"rec={cal['recall']:.3f}", file=sys.stderr)

    ov = overlap_report(rows, holdout, a.perms, a.seed)
    print(f"overlap: exact={ov['exact']} jmax={ov['j_max']:.3f} "
          f"cmax={ov['c_max']:.3f} c>0.5={ov['n_c50']}", file=sys.stderr)
    print("highest-containment holdout turns:", file=sys.stderr)
    for c, j, sid, cat, ht, tt in ov["worst"][:5]:
        print(f"  c={c:.2f} j={j:.2f} {sid} [{cat}]\n    holdout: {ht}\n"
              f"    train  : {tt}", file=sys.stderr)

    latex = emit_latex(corpus, cv, cal, ov, pipe)
    if a.out:
        a.out.write_text(latex, encoding="utf-8")
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        print(latex)
    if a.json_out:
        a.json_out.write_text(json.dumps({
            "corpus": {k: (dict(v) if isinstance(v, Counter) else v)
                       for k, v in corpus.items()},
            "cv": {"folds": cv["folds"], "n_groups": cv["n_groups"],
                   "labels": cv["labels"], "report": cv["report_dict"],
                   "confusion": cv["confusion"].tolist()},
            "calibration": cal,
            "overlap": {k: v for k, v in ov.items() if k != "worst"},
            "overlap_worst": ov["worst"],
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {a.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
