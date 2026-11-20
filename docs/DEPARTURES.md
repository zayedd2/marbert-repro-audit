# Departures and assumptions log

Every place where our setup differs from [1], or where [1] gives no detail and we had to choose. Each entry is mirrored in a config value, so changing a decision means editing one line in `configs/`. Before results are interpreted, **M2** (historical reproduction) confirms the entries marked *to confirm*. When a gap to a published score exceeds the reproduction criterion, check this list before blaming the model.

[1] Abdul-Mageed, Elmadany & Nagoudi (2021). ARBERT & MARBERT. ACL-IJCNLP, pp. 7088–7105.

| ID | Item | What [1] says | What we do | Where | Possible effect | Status |
|---|---|---|---|---|---|---|
| D1 | AraBERT release | Cites "AraBERT (Antoun et al., 2020)" without naming a release | `aubmindlab/bert-base-arabertv01` (2020 base release, no Farasa pre-segmentation needed) | `configs/models.yaml` | AraBERT scores and the margin; flagged as a likely source of discrepancy | fixed in proposal |
| D2 | MARBERT release | MARBERT | `UBC-NLP/MARBERT` (original, not MARBERTv2) | `configs/models.yaml` | none intended | fixed in proposal |
| D3 | Model revision | n/a | Hub `main`. The resolved commit hash is logged in every `env.json` | `env.json` | protects against silent model updates | automatic |
| D4 | LR schedule and warm-up | Not reported | Constant 2e-6, no warm-up | `training.lr_schedule`, `warmup_steps` | optimisation speed and best epoch | assumption |
| D5 | Optimizer details | "Adam", LR 2e-6 | `torch.optim.Adam`, no weight decay, PyTorch default betas (0.9, 0.999) and eps 1e-8 | `training.*` | small | assumption |
| D6 | Gradient clipping | Not reported | None | `training.max_grad_norm: null` | small at LR 2e-6 | assumption |
| D7 | AJGT split | Follows the 80/20 split of Antoun et al. (2020); Table A.1 gives 1.4K / 361 | That split is not public, so we use a stratified 80/20 split (seed 20261113): 1,440 / 360 items. IDs committed in `data/splits/ajgt_test_ids.txt` | `prepare_data.py` | AJGT accuracy can move by about 0.3 pp per item on a 360-item test set | unavoidable |
| D8 | Dev data | No dev split for any of the three datasets; the best epoch is chosen on test | Historical protocol: the same as [1]. Independent protocol: a stratified 10% of training held out as dev, so these runs train on 90% of the data | `dev_fraction`, `split_seed` | independent runs see less training data | by design (main extension) |
| D9 | CAMelSent / ASTD test | Train + dev of ArSAS, ASTD, SemEval and ArSenTD-LEV (about 24K); ASTD test of 664 items | Rebuilt from source files (`data/camelsent_sources.yaml`); exact counts in `data/MANIFEST.json` and the data audit | `prepare_data.py astd` | any size difference changes the training data | *to confirm* (M1) |
| D10 | Text preprocessing | Not reported for fine-tuning | Whitespace normalisation; URLs replaced by `URL` and @mentions by `USER`; no diacritic or tatweel removal. Identical for both models; each model then uses its own tokenizer | `preprocessing.*` | small; affects both models equally | *to confirm* (M2) |
| D11 | Padding | Max length 256 | Truncate at 256 tokens and pad each batch only to its longest item. With attention masking, outputs are unchanged apart from floating-point noise, and training is several times faster | `backends.py` | none intended | implementation |
| D12 | Seeds | Single runs; seed not reported | 3 seeds (42, 1234, 2026). Means ± SD are reported. The dev split uses its own fixed seed | `seeds`, `split_seed` | required by RQ3 | by design |
| D13 | Precision and determinism | Not reported | FP32 (mixed precision off); cuDNN deterministic mode on. Some GPU kernels can still be nondeterministic | `training.fp16`, `deterministic` | turning fp16 on is a departure and must be logged here | assumption |
| D14 | Software versions | 2020–21 era libraries | Current PyTorch/Transformers; exact versions in `env.json` and `requirements-lock.txt` | `env.json` | small numerical differences | unavoidable |
| D15 | Scope | mBERT and other baselines; LR choices | mBERT and learning-rate sweeps dropped | proposal | none on the reproduced targets | fixed in proposal |
| D16 | Checkpoint metric | Best epoch on test by the reported metric | Historical: the same (F1-PN, or accuracy for AJGT). Independent: dev macro-F1. Ties go to the earlier epoch. All metrics are read from the one selected checkpoint | `configs/protocols.yaml` | defines RQ2 | by design |

## How to add an entry

Add a row with the next ID, change the matching config value, and mention the ID in the commit message (for example `D17: enable fp16 for ASTD runs`).
