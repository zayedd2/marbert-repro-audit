# Data

The tweets are **not redistributed** in this repository: each dataset belongs to its authors. What *is* committed:

* `splits/*_dev_ids.txt` lists the IDs of the stratified 10% dev subset of each training set (independent protocol). The same subset is used for every model and every seed.
* `splits/ajgt_test_ids.txt` lists the IDs of our 80/20 AJGT test split (AJGT ships without a split).
* `MANIFEST.json` records the SHA-256 checksums, item counts, sources and dropped labels of the processed files.

With the same source files, `scripts/prepare_data.py` rebuilds byte-identical processed files, and the committed ID files fix the splits. `scripts/verify_results.py` checks that every run used the same test file.

## Processed format

`data/processed/<dataset>/{train,test}.tsv`: UTF-8, tab-separated, with these columns:

| column | meaning |
|---|---|
| `id` | stable item ID, for example `arsarcasm-test-00017` |
| `text` | the tweet (tabs and newlines replaced by spaces) |
| `label` | `negative`, `neutral` or `positive` (AJGT: `negative` or `positive`) |
| optional | `dialect` and `sarcasm` (ArSarcasm), `source` (ASTD/CAMelSent) |

## The three datasets

| Dataset | Used as in [1] | How to get it | Expected sizes ([1], Table A.1) |
|---|---|---|---|
| **ArSarcasm-Sent** | 3-class sentiment, official train/test | `python scripts/prepare_data.py arsarcasm` downloads it from the Hugging Face Hub (`iabufarha/ar_sarcasm`, MIT licence). You can also pass the CSVs from the authors' GitHub repository with `--source csv --train-file ... --test-file ...` | 8,437 / 2,110 |
| **AJGT** | 2-class, 80/20 split | `python scripts/prepare_data.py ajgt` (Hub: `komari6/ajgt_twitter_ar`), or `--source xlsx --train-file AJGT.xlsx` | 1,440 / 360 here (1.4K / 361 in [1]; the exact split of [1] is not public, see DEPARTURES D7) |
| **ASTD (3-class)** | trained on **CAMelSent**: the train and dev data of ArSAS, ASTD, SemEval-2017 and ArSenTD-LEV merged; tested on the held-out ASTD test set. This is **not** ASTD-B | Build per-source files (see below), then run `python scripts/prepare_data.py astd --manifest data/camelsent_sources.yaml` | about 24.8K / 664 |

### Building CAMelSent (Member 1, data audit)

[1] takes CAMelSent from Obeid et al. (2020). No ready-made merged file is distributed, so we rebuild it from the four sources. CAMeL Lab's CAMeLBERT repository (`github.com/CAMeL-Lab/CAMeLBERT`) documents their sentiment preprocessing and is the closest public reference.

1. Obtain each source's official splits and save them as TSV files with a text column and a label column under `data/raw/camelsent/`. Use one file per source and split (train, dev, and the ASTD test).
2. Copy `camelsent_sources.example.yaml` to `camelsent_sources.yaml` and fill in the paths, column names and the label mapping. Labels outside negative/neutral/positive (ASTD `OBJ`, mixed) map to `null` and are dropped. The counts are logged in `MANIFEST.json`.
3. Run `python scripts/prepare_data.py astd`. The script warns if the ASTD test set does not contain 664 items.
4. Run `python scripts/data_audit.py`. It reports duplicates, label conflicts and the overlap between the ASTD test set and every training source **before** results are interpreted. Record any difference from [1] in `docs/DEPARTURES.md`.
