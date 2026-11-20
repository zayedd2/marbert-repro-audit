# Review checklist: Member 1's error analysis (reviewer: Member 4)

Use this when reviewing the error-analysis part of M1's work. Every point can be checked against a generated file.

**Source of the numbers**
- [ ] The confusion matrices and subgroup scores quoted in the text match `results/tables/confusion_matrices.csv` and `results/tables/subgroups.csv`. Nothing is computed by hand.
- [ ] They come from the **independent** protocol (dev-selected checkpoints), not the best-on-test runs.

**Counts and caution**
- [ ] Every subgroup claim gives the group size *n*. Groups under 50 items (marked † in the table) are described as exploratory, not as findings.
- [ ] Differences between subgroups are compared with the seed SD in the same table. A gap smaller than the seed SD is not presented as a real difference.
- [ ] The sarcasm and dialect labels come from ArSarcasm itself. No new labels were invented.

**Observations vs explanations**
- [ ] The Results section states *what* happens (for example "most AraBERT errors on sarcastic tweets are negative→positive").
- [ ] *Why* it happens (for example "irony reverses polarity") goes in the Discussion and is phrased as a possible explanation.

**Failure examples**
- [ ] Quoted tweets come from `results/analysis/failure_examples.csv` (items both models miss in every seed).
- [ ] Only a few short examples are quoted, with a translation, and publishing them respects the dataset's licence. The CSV itself stays out of the public repo, because it is git-ignored.

**Data audit link**
- [ ] Any duplicates or train/test overlap found by `scripts/data_audit.py` are mentioned where they could affect the errors discussed.
