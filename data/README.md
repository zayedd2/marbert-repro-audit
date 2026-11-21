# Data

The tweets are **not redistributed** here: each dataset belongs to its authors. This folder holds what is needed to identify and rebuild the exact data used:

* `splits/<dataset>_<split>_ids.txt` lists the item IDs of every split file, in file order: `train_full` (historical protocol), `train` / `dev` (independent protocol) and `test`. `ajgt_ubc_test_ids.txt` covers the authors' AJGT split.
* `meta/arsarcasm_test_meta.csv` holds the ArSarcasm test labels, dialect and sarcasm flag (no text), for the subgroup results.
* `MANIFEST.json` gives the MD5 checksum, size and label counts of every split file, plus Member 1's split summary.

## Sources and checksums

| Dataset | Source | File and checksum | Used as |
|---|---|---|---|
| ArSarcasm-Sent | github.com/iabufarha/ArSarcasm (also `iabufarha/ar_sarcasm` on the Hugging Face Hub) | official train/test CSVs, `sentiment` column | 8,437 / 2,110, official split |
| AJGT | github.com/komari6/Arabic-twitter-corpus-AJGT | `AJGT.xlsx`, 1,800 tweets | stratified 80/20 split, seed 42: 1,440 / 360 |
| AJGT, authors' split (check runs) | github.com/UBC-NLP/marbert, `examples/UBC_AJGT_final_shuffled_{train,test}.tsv` | | 1,440 / 360 |
| ASTD | github.com/mahmoudnabil/ASTD | `Tweets.txt` + `4class-unbalanced-{train,validation,test}.txt` | test: official test split, objective tweets dropped (661) |
| ArSAS | homepages.inf.ed.ac.uk/wmagdy/resources.htm | `ArSAS.txt`, 21,064 tweets, MD5 `7fc39a33ba3aa3bde15a5bcaae1cc3f2` | CAMelSent source (stratified 90%, Mixed dropped) |
| SemEval-2017 Task 4A Arabic | SemEval-2017 Task 4 release | GOLD train file, MD5 `f3115252f25e53b952991929d263b5e5` | CAMelSent source (train file already contains all dev tweets) |
| ArSenTD-LEV | github.com/aub-mind/hULMonA, `data/ArSenTD-LEV.tsv` (original link dead) | MD5 `38ade7d93f0ed8f6f4dc4cf9706326bc` | CAMelSent source (stratified 90%) |

## Rebuilding the data

1. **ASTD training corpus (CAMelSent)**, Member 2:
   ```
   python scripts/build_camelsent.py --astd_dir ASTD/data --arsas ArSAS.txt \
       --semeval_train SemEval2017-task4-train.subtask-A.arabic.txt --arsentd ArSenTD-LEV.tsv --out data
   ```
   This writes `astd/train.csv` (27,394 tweets), `astd/test.csv` (661) and a count report; the report from the reported runs is `env/camelsent_build_report.json`.
2. **Shared split files**, Member 1: `python scripts/make_team_splits.py <folder with the train/test CSVs> <out>` adds IDs and lower-case labels, and draws the grouped stratified 10% dev sets (seed 42; copies of a tweet stay on one side).
3. **Check**: compare the MD5 checksums of the resulting files with `MANIFEST.json`, or the IDs with `splits/`.

The data audit itself (label mappings, counts, duplicates, cross-source overlap) is `notebooks/member1_data_audit.ipynb`. The findings are summarised in `docs/DEPARTURES.md`, rows 13–22.
