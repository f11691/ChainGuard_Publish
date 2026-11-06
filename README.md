# ChainGuard reproducibility package (AISC-12003)

Companion materials for *ChainGuard: A Layered and Provenance-Aware Runtime
Architecture for Local Prompt Injection Screening* (AI and Security Convergence,
manuscript AISC-12003). This repository is the reproducibility package cited in the
article's Data Availability Statement; release `aisc-12003-r3` is the version that
accompanies the third revision.

```
LICENSE                    MIT
audit_saved_records.py     recomputes every headline number from saved records
result_map.csv             result -> script / input / stored output / granularity
MANIFEST.sha256            SHA-256 of every file in this package
requirements-repro.txt     pinned versions
scripts/                   11 experiment scripts
results/                   15 stored result files, all labelled v0.7.2
```

## Audit without running the detector

```
python3 audit_saved_records.py
```

Standard library only, no network, no model download. It recomputes 54 reported
values from `results/` and compares each with the number printed in the
manuscript; exit status is 0 only when all 54 agree. It can also export every
recorded target-model turn with its ingress and egress decision:

```
python3 audit_saved_records.py --export-turns turns.csv   # 612 rows
```

## What this package does and does not establish

`result_map.csv` labels each reported result with its evidential granularity:

| granularity  | meaning |
|--------------|---------|
| `rerunnable` | the script reproduces the number from files in this package |
| `auditable`  | per-item records are stored, so the number is recomputed here, but the original measurement cannot be repeated from this package alone |
| `summary`    | only an aggregate summary was stored; no per-item audit is possible |

Nothing here is `rerunnable` end to end except the ROC figure. The scripts
import the detector package `pinj.chainguard` and read external data paths;
**the archive alone cannot rerun the experiments.**

Not included, and therefore not independently verifiable from this package:

- the core detector source (`pinj.chainguard`);
- the researcher-authored holdout inputs (80 attack, 100 benign sessions);
- the classifier training corpus — `classifier_report_v0.7.2.json` records a
  corpus hash, but with the corpus absent that hash cannot be revalidated;
- full independent-benchmark caches, model commit revisions, an environment
  lockfile, the software test suite, and any dated pre-freeze implementation
  history.

`requirements-repro.txt` lists the versions the reported runs used; it is a
record of the environment, not a resolved lockfile.

## Datasets

Independent data is not redistributed here. Sources:
`Lakera/gandalf_ignore_instructions`, `xTRam1/safe-guard-prompt-injection`,
`deepset/prompt-injections` and `leolee99/NotInject` (Hugging Face; NotInject is
distributed with PIGuard, <https://github.com/leolee99/PIGuard/tree/main/datasets>),
and InjecAgent test cases from the InjecAgent repository.

## Version

Detector version 0.7.2. Every result file carries the `v0.7.2` suffix.
