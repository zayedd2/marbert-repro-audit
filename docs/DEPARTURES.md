# Departures and assumptions - historical reproduction

Reference [1]: Abdul-Mageed, Elmadany & Nagoudi (2021), *ARBERT & MARBERT: Deep Bidirectional Transformers for Arabic*, ACL-IJCNLP 2021, pages 7088–7105. https://aclanthology.org/2021.acl-long.551/

**Type:** *Forced* = information or data from [1] is unavailable; *Choice* = a detail [1] does not report, so we fixed it ourselves; *Addition* = something we do beyond [1] that does not change its protocol; *Departure* = differs from what [1] reports, found after the runs.

## A. Models and training

| # | Type | Paper [1] | Ours | Why / expected effect |
|---|---|---|---|---|
| 1 | Forced | AraBERT version not named | `aubmindlab/bert-base-arabertv01` @ `01f6b7d2d17afcd0f79498f26f47dfc6207b56ea` | Fixed 2020 base release. Checked on ArSarcasm-Sent (seed 42): AraBERT v1 with Farasa segmentation (67.90) and v0.1 with AraBERT's official preprocessing (68.75) stay close to our v0.1 result (67.63) and well above [1] (63.50), so neither version nor preprocessing explains the AraBERT gap. v1 run used QCRI's public Farasa release (farasapy server unreachable; 97.9% of words segmented) |
| 2 | – | MARBERT | `UBC-NLP/MARBERT` @ `88e1fa192dd723cf0b3563500aec46209762eb22` | Original release, not MARBERTv2; revision pinned for all runs |
| 3 | – | Adam, lr 2e-6, batch 32, max length 256, 25 epochs (Sec. 3.4) | Identical | No departure |
| 4 | Choice | LR scheduler / warm-up not reported | Constant LR 2e-6, no warm-up | Simplest reading of [1]. Checked (seed 42, ArSarcasm-Sent): linear decay + 10% warm-up changes F1-PN by +0.05 (MARBERT) and +0.43 (AraBERT); a 10x higher LR (2e-5) by -0.45 and -0.03. Results are robust to this choice |
| 5 | Choice | Adam betas / epsilon, weight decay not reported | PyTorch defaults (0.9, 0.999 / 1e-8); weight decay 0 | Standard Adam |
| 6 | Choice | Dropout not reported | HF `AutoModelForSequenceClassification` default head: dropout 0.1, linear layer on [CLS] pooler | Standard BERT fine-tuning; full fine-tuning of all weights, as in [1] |
| 7 | Choice | Precision not reported | fp16 mixed precision | Speed; small numeric differences, equal for both models |
| 8 | Choice | Padding not reported | Dynamic padding per batch; truncation at 256 kept | Speed; no effect on outputs with attention masks beyond floating-point noise |
| 9 | Departure | Fine-tuning uses the same light pre-processing as pre-training (Sec. 2 and 3): diacritics removed; URLs, user mentions and hashtags replaced with `URL`, `USER` and `HASHTAG` | @mentions -> `USER`, URLs -> `URL`, whitespace normalised; diacritics and hashtags kept as written; same for both models | Found after the runs, not rerun. Applies to both models equally, so it cannot explain why only AraBERT misses. Diacritics are rare in tweets; keeping hashtags leaves their words visible to both models, which may raise absolute scores slightly. Obeid et al. (2020) instead *removed* URLs, usernames and diacritics for the CAMelSent sources |
| 10 | Addition | Single run per configuration; seeds not reported | 3 seeds: 42, 43, 44 | Estimates run-to-run variance, which [1] does not report; verdict uses mean ± SD and seed range |
| 11 | – | Best epoch selected on **test** for datasets without a dev split (Table A.1: no dev for AJGT, ASTD, ArSarcasm_Sent) | Best epoch on test | Same as [1]. Makes absolute scores optimistic for both [1] and us; quantified by the independent protocol (Member 3) |
| 12 | Choice | Hardware not reported | NVIDIA L4 (driver 580.82.07, 23 GB); all runs on the L4, MARBERT and AraBERT for each seed on the same GPU | GPU non-determinism causes small run-to-run noise, smaller than seed variance and equal for both models |

## B. Data

| # | Type | Paper [1] | Ours | Why / expected effect |
|---|---|---|---|---|
| 13 | – | ArSarcasm_Sent: 8.4K train / 2.1K test | Official ArSarcasm split: 8,437 / 2,110; `sentiment` column | Exact match. Note: many tweets contain literal quote marks in the original release (kept as released) |
| 14 | Forced, then checked | AJGT: 80/20 split of Antoun et al. (2020); 1.4K train / 361 test; the train/test split was not released with the paper | Stratified 80/20 split of `AJGT.xlsx` (1,800 tweets), seed 42: 1,440 train / 360 test. Check runs (6) on the authors' split, later found on their GitHub (github.com/UBC-NLP/marbert, `examples/UBC_AJGT_final_shuffled_train.tsv` / `_test.tsv`; `ajgt_ubc`): 1,440 / 360, test 191 positive / 169 negative | The authors' split reproduces [1] within 0.3 points (MARBERT 95.83 ± 0.28, gap −0.28; AraBERT 92.13 ± 0.32, gap −0.09; both published scores inside the seed range), so the AJGT gap on our split (−1.76 / −3.33) comes from the split, not from the models. The two test sets share only 47 of 360 tweets. [1]'s "361" is the line count of the test file (header + 360 tweets) |
| 15 | Forced | ASTD target trained on CAMelSent: train + dev of ArSAS, ASTD, SemEval-2017, ArSenTD-LEV merged by Obeid et al. (2020), ~24.8K tweets; the authors' CAMelSent split was not released | Rebuilt CAMelSent from its four sources with `build_camelsent.py`: **27,394 tweets** (110% of reported); tested on the official ASTD test set (661 tweets, row 16) | ASTD and SemEval use official splits, so the extra ~2.6K tweets come from the unsplit ArSAS and ArSenTD-LEV (we keep 90% of each). Keeping 80% of each would give 25,018 tweets (100.9% of [1]), suggesting Obeid et al.'s own splits assigned less of these sources to training; not verifiable without their split files. Expected effect: slightly higher ASTD scores for both models. Leakage check (Member 1): 0 exact and 4 near-duplicate test tweets (0.6%) appear in training |
| 16 | – | ASTD test: 664 tweets | Official ASTD `4class-unbalanced` test split (0-based line indices), objective (OBJ) tweets dropped: **661 tweets** | Near-exact. ASTD's `NEUTRAL` label is its *mixed* class (832 tweets); keeping OBJ instead would give 1,833 test tweets, so this mapping matches [1] |
| 17 | – | ASTD train + dev | Official train + validation, OBJ dropped: 2,654 tweets | As in [1] |
| 18 | Choice | ArSAS has no official split | Stratified 90% kept as train+dev (seed 42), per [1]'s 80/10/10 rule for unsplit data: 17,785 tweets; *Mixed* (1,302) dropped | Split items differ from Obeid et al.'s. File: `ArSAS.txt`, 21,064 tweets, MD5 `7fc39a33ba3aa3bde15a5bcaae1cc3f2`; source: homepages.inf.ed.ac.uk/wmagdy/resources.htm |
| 19 | – | SemEval-2017 Task 4A Arabic: train (3,355) + dev (671) | Official GOLD train file: 3,355 tweets; surrounding quote marks removed | No departure: the dev set is distributed only as an ID + label list, and all 671 dev IDs are contained in the train file with identical labels, so the train file already covers train + dev. Train file MD5 `f3115252f25e53b952991929d263b5e5` |
| 20 | Forced | ArSenTD-LEV (4,000 tweets) | Original URL (oma-project.com) and the Hugging Face loader no longer work; file obtained from the hULMonA repository (AUB MIND Lab): https://github.com/aub-mind/hULMonA/blob/master/data/ArSenTD-LEV.tsv. Stratified 90% (seed 42): 3,600 tweets; very_pos/very_neg -> pos/neg | File matches Baly et al. (2019) Table 2 exactly (topic and expression distributions; 3-class sentiment 47.1 / 22.1 / 30.8 %). MD5 `38ade7d93f0ed8f6f4dc4cf9706326bc` |
| 21 | – | Duplicates not discussed | Kept as released: 1,128 duplicate texts in the merged training set, 203 with conflicting labels (mainly ArSenTD-LEV, ArSAS, SemEval); 0 training tweets identical to ASTD test tweets; 4 near-duplicates (0.6% of the test set, Member 1's leakage check) | Faithful to [1]. Reported to Member 1 for the data audit |
| 22 | Addition | No dev split described (Table A.1: no dev for AJGT, ASTD, ArSarcasm_Sent) | Dev split (Member 1): 10% of each training set held out with `StratifiedGroupKFold` (seed 42), grouping copies of the same tweet so they never fall on both sides | Used only by the dev-selected protocol (Member 3) to choose the epoch; the historical runs train on the full training set as in [1] |

## C. Scope

| # | Type | Paper [1] | Ours | Why / expected effect |
|---|---|---|---|---|
| 23 | – | 17 sentiment datasets | 3 targets: ArSarcasm-Sent, ASTD, AJGT (as defined in our proposal) | Conclusions apply to these targets only |
| 24 | – | ARBERT also evaluated | Not run | Outside our proposal's scope (MARBERT vs AraBERT) |

## D. Independent protocol (Member 3), added by Member 4 from Member 3's audit

| # | Type | Proposal | Ours | Why / expected effect |
|---|---|---|---|---|
| 25 | Departure (from the proposal) | Epoch chosen on dev macro-F1 | Epoch chosen on dev F1-PN (ArSarcasm-Sent, ASTD) and dev accuracy (AJGT), i.e. the published metric | Found in Member 3's audit. Re-selecting on dev macro-F1 changes the seed-mean test scores by at most 0.37 points (ASTD MARBERT; up to 1.10 in a single run) and MARBERT still leads everywhere: `results/tables/selection_rule_check.csv` |
| 26 | Note | Dev = stratified 10% of each training set | Grouped stratified 10% (row 22). For ASTD the dev set follows the pooled CAMelSent sources: only 266 of 2,740 dev tweets are ASTD (1,778 ArSAS) | ASTD epochs are chosen mostly on non-Egyptian data; a limitation, not a change of protocol |
