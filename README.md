# Revisiting MARBERT for Arabic Sentiment: Reproducibility and Controlled Evaluation Audit

Code, configs, seeds, saved predictions and scripts for AI504 Assignment 2 (Introduction to Computational Linguistics, The British University in Dubai). The project re-runs the MARBERT vs AraBERT sentiment comparison of

> [1] M. Abdul-Mageed, A. Elmadany and E. M. B. Nagoudi. *ARBERT & MARBERT: Deep Bidirectional Transformers for Arabic.* ACL-IJCNLP 2021, pp. 7088–7105. <https://aclanthology.org/2021.acl-long.551>

and asks one question: **does MARBERT's advantage over AraBERT stay stable when checkpoint selection, training randomness and the sentiment metric are controlled?**

**Every table and figure in the report can be rebuilt from the files in this repository with one command, without a GPU.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zayedd2/marbert-repro-audit/blob/main/notebooks/colab_run_all.ipynb) [![tests](https://github.com/zayedd2/marbert-repro-audit/actions/workflows/tests.yml/badge.svg)](https://github.com/zayedd2/marbert-repro-audit/actions/workflows/tests.yml)

---

## Quick start

| I want to… | Run | Needs |
|---|---|---|
| Rebuild every table, figure and report number from the saved predictions | `pip install -r requirements-analysis.txt` then `python scripts/make_report_assets.py` | CPU, about 1 minute |
| Check that the saved predictions reproduce the logged scores | `python scripts/verify_results.py` | CPU |
| Re-run all experiments from scratch | click **Open in Colab** above (`notebooks/colab_run_all.ipynb`), or run `bash scripts/run_all.sh` on a GPU machine | GPU |
| Test the whole pipeline on fake data (no GPU, no downloads) | `bash scripts/dry_run.sh` or `python -m pytest -q` | CPU, about 30 s |

The outputs land in `results/SUMMARY.md` (overview), `results/tables/` (CSV, Markdown and LaTeX), `results/figures/` (PDF and PNG) and `report/generated/` (what the LaTeX report reads).

## Experimental design

| | |
|---|---|
| **Models** | `UBC-NLP/MARBERT` (original, not v2); `aubmindlab/bert-base-arabertv01` ([1] does not name its AraBERT release; see D1) |
| **Datasets** | ArSarcasm-Sent (3-class), ASTD (3-class; trained on merged CAMelSent, tested on the ASTD test set), AJGT (2-class) |
| **Published targets** | MARBERT / AraBERT: ArSarc F1-PN 71.50 / 63.50 · ASTD F1-PN 78.00 / 72.00 · AJGT accuracy 96.11 / 92.22 |
| **Fine-tuning (as in [1])** | Adam (no weight decay), LR 2e-6 constant, batch 32, max length 256, 25 epochs; dev and test scored after every epoch |
| **Seeds** | 42, 1234, 2026 for every configuration. The dev split uses its own fixed seed (20261113), so all runs share one dev set |
| **Historical protocol** (audit item 1) | Full training data; best epoch **on test** by the published metric, as [1] did |
| **Independent protocol** (audit item 2, main result) | Stratified 10% dev held out; best epoch on **dev macro-F1**; test never used for selection |
| **Checkpoint diagnostic** (item 3) | In each independent run: max test score over the epochs minus the test score of the dev-selected epoch |
| **Reliability** (item 4) | Per-seed paired MARBERT − AraBERT differences (mean ± SD), plus a 95% paired bootstrap over test items (10,000 resamples), reported separately |
| **Metric choice** (item 5) | Rankings and margins under the published metric vs macro-F1 |
| **Reproduced if** | (\|gap\| ≤ 2 pp **or** published score inside our seed range) **and** MARBERT > AraBERT on that dataset |
| **Grid** | 2 models × 3 datasets × 3 seeds × 2 protocols = **36 runs** |

**Metrics.** *F1-PN* is the mean of positive-class and negative-class F1 on the full 3-class test set, with neutral items kept. *Macro-F1* is the mean F1 over all classes. Both are computed from confusion matrices in `src/marbert_audit/metrics.py` and unit-tested against scikit-learn.

Every departure from [1], and every unreported detail we had to choose, is listed in **[`docs/DEPARTURES.md`](docs/DEPARTURES.md)**.

## Where each result comes from

| Report item | RQ | File (all generated) | Built by |
|---|---|---|---|
| Table: reproduction vs published | RQ1 | `tables/reproduction.csv`, `tab_reproduction.tex` | `summary.reproduction_table` |
| Figure: per-seed scores vs published | RQ1 | `figures/fig_reproduction.pdf` | `figures.fig_reproduction` |
| Table: main result (dev-selected) | main | `tables/independent.csv`, `tab_independent.tex` | `summary.independent_table` |
| Table: checkpoint diagnostic | RQ2 | `tables/checkpoint_diagnostic.csv`, `tab_checkpoint.tex` | `summary.checkpoint_table` |
| Figure: learning curves, dev vs test | RQ2 | `figures/fig_learning_curves.pdf` | `figures.fig_learning_curves` |
| Figure: optimism gap per run | RQ2 | `figures/fig_checkpoint_gap.pdf` | `figures.fig_checkpoint_gap` |
| Table: margins, seed SD, bootstrap CI, rank flips | RQ3, RQ4 | `tables/margins.csv`, `tab_margins.tex` | `summary.margins_table`, `stats.paired_bootstrap` |
| Figure: margin forest plot | RQ3, RQ4 | `figures/fig_margins.pdf` | `figures.fig_margins` |
| Appendix: per-class P/R/F1 | RQ4 | `tables/per_class.csv`, `tab_per_class.tex` | `summary.per_class_table` |
| Error analysis: confusion matrices, dialect/sarcasm groups | (M1) | `figures/fig_confusion_matrices.pdf`, `tables/subgroups.csv` | `errors.*` |
| Data audit table, cross-dataset and cross-source overlap | (M1) | `tables/data_audit.csv`, `data_overlap.csv`, `data_source_overlap.csv`, `tab_data_audit.tex` | `scripts/data_audit.py` |
| Every number in the report prose | all | `report/generated/numbers.tex` | `latex.numbers_tex` |

## What each run saves

`results/runs/<protocol>/<dataset>/<model>/seed<seed>/`

| File | Content |
|---|---|
| `run_config.yaml` | resolved settings, split sizes, label counts, SHA-256 of the data files |
| `env.json` | Python and package versions, GPU, CUDA, git commit, the **resolved model commit hash**, runtime |
| `epoch_metrics.csv` | dev and test accuracy, macro-F1 and F1-PN after each of the 25 epochs |
| `preds_test.csv.gz`, `preds_dev.csv.gz` | per-item gold label, prediction and class probabilities **for every epoch** |
| `train.log` | readable log |
| `DONE` | written last; runs without it are ignored |

Because predictions are saved for all epochs, any selection rule, metric or statistic can be recomputed later without retraining. Model checkpoints are not kept, because they are too large for git.

## Repository layout

```
configs/            base.yaml (seeds, hyper-parameters, bootstrap), models, datasets (+ published targets), protocols
data/               README (how to obtain data), splits/ (committed split IDs), MANIFEST.json; raw/processed are git-ignored
src/marbert_audit/  metrics · data · preprocess · backends (HF trainer + synthetic) · train · selection · stats
                    summary (result tables) · errors (error-analysis inputs) · latex · figures · audit · envinfo
scripts/            prepare_data · data_audit · run_experiments · verify_results · make_report_assets
                    run_all.sh (full pipeline) · dry_run.sh (synthetic smoke test)
notebooks/          colab_run_all.ipynb - run everything on a free Colab GPU
results/            runs/ (saved per-run outputs), tables/, figures/, SUMMARY.md
report/             experiments_results.tex + appendix_results.tex (ACL section), preview.tex, generated/
docs/               DEPARTURES.md · PIPELINE_EXPLAINED.md · REVIEW_M1.md
tests/              metric, bootstrap, selection and verdict tests + end-to-end synthetic pipeline test
```

## Reproducibility measures

* **Seeds** fix the classifier-head initialisation, dropout and batch order (`torch.Generator`). cuDNN deterministic mode is on. The dev split and the AJGT test split are committed as ID lists.
* **Environment**: every run records its package versions, GPU and resolved Hugging Face model commit. `requirements-lock.txt` is written from the environment that produced the reported runs.
* **Integrity**: `verify_results.py` recomputes every epoch's scores from the saved predictions, confirms that all runs used the same test file and writes `results/MANIFEST.sha256`.
* **Separation**: synthetic dry-run outputs go only to `results_synthetic/`, are watermarked, and the scripts refuse to mix them with real results.
* **Resumable training**: with `--resume-root`, a checkpoint is saved after each epoch (model, optimizer and RNG states), so a Colab disconnect costs at most one epoch.

## Re-running in parts

```bash
python scripts/run_experiments.py --list                                  # what is done / missing
python scripts/run_experiments.py --protocols historical --datasets ajgt  # a subset
python scripts/run_experiments.py --seeds 42 --models marbert            # one model, one seed
```

Teammates can split the grid between them. Copying their `results/runs/...` folders into one repository is enough, and `make_report_assets.py` uses whatever is complete and lists what is missing.

## Team

| Member | Responsibility | Report section |
|---|---|---|
| Member 1 | Data audit; linguistic error analysis | Introduction; Related Work |
| Member 2 | Historical reproduction (audit item 1) | Methodology |
| Member 3 | Controlled evaluation audit (items 2–5) | Discussion; Limitations |
| Member 4 | This repository; results | Experiments and Results |

## Licence

Code: MIT (see `LICENSE`). The datasets are not redistributed and remain under their owners' terms; see `data/README.md`.
