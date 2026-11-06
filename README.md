# ChainGuard v0.7.2: source code, data and reproducibility package

This repository accompanies the article *ChainGuard: A Layered and Provenance-Aware
Runtime Architecture for Local Prompt Injection Screening* (AI and Security
Convergence, manuscript AISC-12003). Release `aisc-12003-r3` is the version that
accompanies the third revision.

```
src/pinj/chainguard/        the ChainGuard detector (frozen v0.7.2 source)
  resources/                classifier training corpus (941 turns)
tests/                      42 unit and regression tests
data/
  post_freeze_source_challenge_v2/   primary holdout: 80 attack, 100 benign sessions
  attacks/ benign/ regression/       fixtures used by the tests
SOURCE_FREEZE_V0.7.2.json   SHA-256 of all 41 frozen source, test and metadata files
SOURCE_INTEGRITY_V0.7.2.json  recomputed check of that manifest
scripts/                    the 12 experiment scripts behind the paper's results
results/                    the 16 stored result files (all *_v0.7.2.json)
audit_saved_records.py      recomputes 63 reported values from results/
result_map.csv              result -> script -> input -> stored output -> re-run status
MANIFEST.sha256             SHA-256 of every file in this repository
requirements-repro.txt      verified versions
```

## Quick start

Python 3.13; verified versions are in `requirements-repro.txt`.

```
pip install -r requirements-repro.txt
pip install -e .
pytest tests                      # 42 passed
sha256sum -c MANIFEST.sha256      # every file unchanged
python3 audit_saved_records.py    # 63/63 checks against the stored records
```

`audit_saved_records.py` uses the standard library only and runs no detector or
model. `--export-turns turns.csv` writes all 612 recorded target-model turns with
their ingress and egress decisions.

## Re-running the experiments

Every script takes `--repo .` (the repository root):

```
python3 scripts/end_to_end.py --repo . --models --out holdout.json      # holdout decisions, no model
python3 scripts/ablation_study.py --repo . --out ablation.json
python3 scripts/threshold_sweep.py --repo . --steps 11 --out sweep.json
python3 scripts/classifier_report.py --repo . --json-out classifier.json
python3 scripts/external_benchmark.py --repo . --out external.json      # downloads deepset, NotInject
python3 scripts/injecagent_eval.py --repo . --out injecagent.json       # downloads InjecAgent
python3 scripts/large_public_benchmarks.py                              # downloads Gandalf, Safe-Guard; writes big_public.json
python3 scripts/learned_baseline.py --repo . --chainguard-results results/ablation_v0.7.2.json --out learned.json
python3 scripts/validated_threshold.py --repo . --out validated.json   # thresholds selected on Safe-Guard train, tested on test
```

`result_map.csv` records the status of every reported result:

| rerun        | meaning |
|--------------|---------|
| `verified`   | re-run from this repository on 6 November 2026; output identical to the file in `results/` |
| `rerunnable` | code and inputs are available, but the run was not repeated |

`verified`: primary holdout, classifier cross-validation and corpus hash,
ablation, threshold sweep, the DeBERTa baseline on the holdout, the deepset,
NotInject, Gandalf, Safe-Guard and InjecAgent benchmarks, and the
validation-selected threshold test (new in this revision; it reproduces the stored
Safe-Guard totals at the shipped threshold as a self-check).

`rerunnable` only: the three local target-model experiments (end-to-end replay,
Gandalf replication, adaptive evaluation), which need CPU-hours of generation,
the DeBERTa runs on the independent datasets, and the ROC figure. Their per-item
records, where stored, are recomputed by `audit_saved_records.py`.

The classifier cross-validation figures require scikit-learn 1.8 or later: earlier
versions assign different folds. Detector decisions are the same on all tested
versions (1.5.2 to 1.9.1).

## What is not here

- Model commit revisions for the three target models were not recorded in the
  original runs. The DeBERTa revision that reproduces the stored scores is
  given in `requirements-repro.txt`.
- The independent benchmark data is not redistributed; the scripts download it
  from its public sources.
- There is no dated pre-freeze development history. `SOURCE_FREEZE_V0.7.2.json`
  fixes the source by hash, and the holdout manifest names that hash as the
  freeze it was created after, but neither file can prove when it was written.

## Licences

Code: MIT (`LICENSE`). The training corpus includes 303 rows from
`zachz/prompt-injection-benchmark` (MIT); see `THIRD_PARTY_DATA.md`.
