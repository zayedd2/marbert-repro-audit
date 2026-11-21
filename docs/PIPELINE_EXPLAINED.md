# The repository in plain English (Member 4's guide)

This guide is for explaining the repository in the group or in the oral defence. No Python knowledge is needed.

## The idea in one paragraph

The MARBERT paper says MARBERT beats AraBERT on Arabic sentiment. For our three datasets the authors had no dev set, so they picked their best epoch *by looking at the test set*. That is like choosing your best of 25 exam attempts using the answer key. Member 2 re-ran their experiment twice: once their way (**historical**), and once the clean way (**independent**: 10% of the training data set aside as a dev set, and the epoch picked on that). Each was run with 3 seeds. Members 1 and 3 then checked the data, the statistics and the errors. My job (Member 4) is to put all of it in one public place where anyone can check every number.

## What the repository does

1. **Keeps the code that produced the results.** `notebooks/historical_reproduction.ipynb` is Member 2's Colab notebook that trained all runs. `scripts/build_camelsent.py` rebuilt the ASTD training data, and `scripts/make_team_splits.py` is Member 1's split script.
2. **Keeps the raw outputs.** For each of the 36 runs, the model's prediction for every test tweet after every one of the 25 epochs (`results/runs/`). That is 1,650 prediction files.
3. **Proves the outputs are genuine** (`scripts/import_team_runs.py`, `scripts/verify_results.py`). Every logged score is recomputed from the saved predictions. If one number had been typed in by hand, the check would stop.
4. **Rebuilds every table and figure** (`scripts/make_report_assets.py`) from those predictions in under a minute, without a GPU.
5. **Cross-checks against the team.** The rebuilt numbers are compared with the files Members 2 and 3 produced in Colab. All 12 checks match exactly (`results/tables/crosscheck_vs_team.csv`).

The key design choice is that **we keep predictions, not just scores**. Anyone can change the metric, the selection rule or the statistics and recompute, without the 10 GPU-hours of training.

## Concepts you may be asked about

**F1-PN.** The paper's metric for the 3-class datasets: the average of the F1 for *positive* and the F1 for *negative*. Neutral tweets stay in the test set (a neutral tweet predicted "positive" still counts as a mistake), but neutral's own F1 is not averaged in. On ASTD this matters a lot, because the neutral (mixed) class is predicted badly.

**Macro-F1.** The average F1 over *all* classes, neutral included.

**Why best-on-test is a problem.** If you score the test set 25 times and keep the best, you partly pick the epoch that got lucky on those test tweets. The *optimism* is how much higher that best score is than the score of the epoch chosen on dev, within the same run. It can never be negative.

**Seeds.** A seed fixes the random parts of training: the starting weights of the classifier layer, dropout and the order of training batches. Three seeds show how much the score moves for reasons unrelated to model quality.

**Paired bootstrap.** The test set is one sample of tweets. The bootstrap re-draws it 1,000 times with replacement and recomputes MARBERT − AraBERT each time. The middle 95% of those differences is the confidence interval. It is "paired" because both models are always scored on the same re-drawn tweets.

**Why seed SD and bootstrap CI are kept apart.** They answer different questions. If we trained again, how much would the lead move? (seeds). If we had a different test sample, how much would it move? (bootstrap).

**"Reproduced".** Fixed in the proposal: our mean is within 2 points of the published score, or the published score lies inside our three-seed range, and MARBERT still beats AraBERT on that dataset.

## Likely questions, short answers

* *Which code produced the results?* Member 2's notebook (`notebooks/historical_reproduction.ipynb`), on one NVIDIA L4 GPU in Colab. My scripts do not train anything: they import, verify and summarise the saved outputs.
* *How do you know the numbers are right?* Every logged score is recomputed from the saved predictions (1,650 files), and the rebuilt tables match the team's own result files exactly. The repository also has automatic tests that run on GitHub.
* *Can someone else rebuild your tables?* Yes. Clone the repo, `pip install -r requirements.txt`, then `python scripts/make_report_assets.py`. No GPU is needed.
* *Why are the tweets not in the repository?* The datasets belong to their authors. We commit the IDs and checksums of every split file, and `data/README.md` says where to get each source.
* *Why does the repository use the team's exact bootstrap seed?* So the intervals in the report regenerate exactly, not just approximately.
* *What changed from the proposal?* The epoch was chosen on dev F1-PN (accuracy for AJGT), not dev macro-F1. Member 3 checked this: the seed-mean scores change by at most 0.37 points, and MARBERT still leads everywhere (departure 25).

## Re-running the summary yourself (worth doing once before the oral)

```
pip install -r requirements.txt
python scripts/verify_results.py        # "OK: every saved prediction set reproduces its logged scores ..."
python scripts/make_report_assets.py    # prints the cross-check table; all rows say True
```
Then open `results/SUMMARY.md`.
