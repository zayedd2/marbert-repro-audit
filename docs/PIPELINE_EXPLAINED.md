# The pipeline in plain English (Member 4's guide)

This guide is for explaining the repository to the group or to the professor. No Python knowledge is needed.

## The idea in one paragraph

The MARBERT paper says MARBERT beats AraBERT on Arabic sentiment. But for these three datasets the authors had no dev set, so they picked their best epoch *by looking at the test set*. That is like choosing your best of 25 exam attempts using the answer key. We re-run their experiment twice: once their way (**historical**), and once the clean way (**independent**: we set aside 10% of the training data as a dev set and pick the epoch on that). We use 3 random seeds each time and measure how much the scores wobble. Then we check whether "MARBERT wins" survives every one of those controls.

## What happens when you press "run"

1. **Prepare the data** (`prepare_data.py`). Downloads the datasets, cleans them into one standard table format, and fixes the dev split and the AJGT test split as lists of IDs, so everybody uses exactly the same items.
2. **Audit the data** (`data_audit.py`). Counts items and labels, and finds duplicate tweets and tweets that appear in both train and test.
3. **Train** (`run_experiments.py`). 36 runs = 2 models × 3 datasets × 3 seeds × 2 protocols. Each run trains for 25 epochs. After every epoch it predicts the dev and test sets and **saves every prediction**.
4. **Verify** (`verify_results.py`). Recomputes every score from the saved predictions to prove nothing was edited by hand, and records a fingerprint (SHA-256) of every file.
5. **Build the report material** (`make_report_assets.py`). Reads the saved predictions and produces every table, figure and number in the Experiments and Results section.

The key design choice is that **we save predictions, not just scores**. Anyone can later change the metric, the selection rule or the statistics and recompute, without the hours of GPU time.

## The concepts you may be asked about

**F1-PN.** The metric the paper used for the 3-class datasets: the average of the F1 for *positive* and the F1 for *negative*. Neutral tweets stay in the test set (a neutral tweet predicted "positive" still counts as a mistake), but neutral's own F1 is not averaged in.

**Macro-F1.** The average F1 over *all* classes, neutral included. RQ4 asks whether the winner changes when we switch from F1-PN to macro-F1.

**Why best-on-test is a problem (RQ2).** If you score the test set 25 times and keep the best, you partly pick the epoch that got lucky on those specific test tweets. The "optimism gap" is how much higher that best score is than the score of the epoch chosen on dev. It can never be negative.

**Seeds (RQ3).** A seed fixes the random parts of training: the starting weights of the classifier layer, dropout and the order of training batches. Different seeds give slightly different models. Three seeds show how much the score moves for reasons that have nothing to do with the model's quality.

**Paired bootstrap (RQ3).** The test set is a sample of tweets, and a different sample would give slightly different scores. The bootstrap re-draws the test set 10,000 times *with replacement* and recomputes MARBERT − AraBERT each time. The middle 95% of those differences is the confidence interval. It is "paired" because both models are always scored on the same re-drawn tweets, so easy or hard tweets affect both models equally.

**Why report SD and CI separately.** They answer different questions. The seed SD asks: if we trained again, how much would the margin move? The bootstrap CI asks: if we had a different test sample, how much would it move? The proposal promises to keep them apart.

**"Reproduced".** A published score counts as reproduced if our average is within 2 points of it, or if it falls inside the range of our three seeds, and MARBERT still beats AraBERT on that dataset.

## Likely questions, short answers

* *Why not just train once?* One run cannot separate a real difference from luck. Three seeds plus the bootstrap measure the luck.
* *Doesn't holding out 10% for dev change the training data?* Yes, slightly. That is why the historical protocol still uses the full training data to reproduce the paper. The independent runs are the clean comparison. This is logged as D8.
* *Why AraBERT v0.1?* The paper does not say which AraBERT release it used. We fixed the 2020 base release, the one available when the paper was written, and flagged it as a possible source of discrepancy (D1).
* *How do you know the numbers in the report are right?* They are not typed by hand. A script writes them into `numbers.tex` from the saved predictions, and `verify_results.py` re-checks every logged score.
* *Can someone else rebuild the tables?* Yes. Clone the repo, install `requirements-analysis.txt`, run `python scripts/make_report_assets.py`. No GPU is needed.

## Your checklist as Member 4

- [x] Repository public on GitHub: https://github.com/zayedd2/marbert-repro-audit
- [ ] Teammates' runs copied into `results/runs/` (all 36 have a `DONE` file)
- [ ] `requirements-lock.txt` committed from the environment that produced the runs
- [ ] `python scripts/verify_results.py` prints OK
- [ ] `python scripts/make_report_assets.py` run; `report/generated/` copied into the report project
- [x] Repository link in `report/experiments_results.tex` (https://github.com/zayedd2/marbert-repro-audit)
- [ ] The two `[TO DO]` notes in the section filled in (M2: checks for any target that was not reproduced; M1: error-analysis observations)
- [ ] M1's error analysis reviewed (see `docs/REVIEW_M1.md`)
- [ ] Contribution statement written

## The repository

<https://github.com/zayedd2/marbert-repro-audit>. Teammates can open the notebook straight from the README's **Open in Colab** button. To give them write access, use *Settings → Collaborators* on GitHub.
