#!/usr/bin/env python3
"""
Rebuild the CAMelSent-style training set used for the ASTD target in
ARBERT & MARBERT (Abdul-Mageed et al., 2021), Appendix A.1:

  "Obeid et al. (2020) merge training and development data from ArSAS, ASTD,
   SemEval-2017 and ArSenTD-LEV ... (~24K) and evaluate on the independent
   test sets from each of these sources."   Table A.1: ASTD 24.8K train, 664 test.

Output (ready for historical_reproduction.ipynb):
  <out>/astd/train.csv   columns: text,label,source   (merged train+dev of all sources found)
  <out>/astd/test.csv    columns: text,label,source   (ASTD official test, 3-class)
  <out>/astd/build_report.json                        (counts per source/label, checks, decisions)

Any source you do not have is skipped with a warning, so the script also covers the
time-boxed fallback (e.g. ASTD + ArSenTD-LEV only). Every decision is written to the report.

Usage (only --astd_dir is required):
  python build_camelsent.py --astd_dir ASTD/data \
      --arsas ArSAS.txt \
      --semeval_train SemEval2017-task4-train.subtask-A.arabic.txt \
      --semeval_dev   SemEval2017-task4-dev.subtask-A.arabic.txt \
      --arsentd ArSenTD-LEV.tsv   (or --arsentd hf  to load ramybaly/arsentd_lev from Hugging Face)
      --out data
"""
import argparse, json, os, sys
import pandas as pd
from sklearn.model_selection import train_test_split

LAB = {"pos": "positive", "positive": "positive", "very_positive": "positive",
       "neg": "negative", "negative": "negative", "very_negative": "negative",
       "neutral": "neutral", "neu": "neutral", "obj": "neutral", "objective": "neutral",
       "mixed": "mixed"}

def norm(x):
    return LAB.get(str(x).strip().lower().replace(" ", "_"))

def report_counts(df):
    return {"n": int(len(df)), "labels": {k: int(v) for k, v in df["label"].value_counts().items()}}

# ---------------------------------------------------------------- ASTD (official 4-class unbalanced split)
def load_astd(d):
    rows = [l.rstrip("\n").rsplit("\t", 1) for l in open(f"{d}/Tweets.txt", encoding="utf-8")]
    df = pd.DataFrame(rows, columns=["text", "raw"])
    idx = {s: [int(x) for x in open(f"{d}/4class-unbalanced-{s}.txt").read().split()]   # 0-based line numbers
           for s in ["train", "validation", "test"]}
    assert sorted(sum(idx.values(), [])) == list(range(len(df))), "ASTD split files do not cover Tweets.txt exactly"
    # ASTD labels: OBJ (objective), POS, NEG, NEUTRAL (= 'subjective mixed' in Nabil et al. 2015).
    # 3-class setting: drop OBJ; keep NEUTRAL as the neutral class. This reproduces the paper's test size
    # (661 here vs 664 in Table A.1); keeping OBJ would give 1,833 test tweets.
    m = {"POS": "positive", "NEG": "negative", "NEUTRAL": "neutral"}
    out = {}
    for s, ix in idx.items():
        sub = df.iloc[ix]
        sub = sub[sub["raw"].isin(m)].assign(label=lambda x: x["raw"].map(m), source="ASTD")
        out[s] = sub[["text", "label", "source"]].reset_index(drop=True)
    return pd.concat([out["train"], out["validation"]], ignore_index=True), out["test"]

# ---------------------------------------------------------------- ArSAS (no official split -> 80/10/10, keep 90%)
def load_arsas(path, seed, keep_mixed):
    df = pd.read_csv(path, sep="\t", dtype=str, quoting=3, encoding="utf-8")
    tcol = next(c for c in df.columns if "text" in c.lower() or c.lower() == "tweet")
    lcol = next(c for c in df.columns if c.lower().startswith("sentiment_label") or c.lower() == "sentiment")
    df = df.assign(label=df[lcol].map(norm), text=df[tcol])
    bad = df["label"].isna().sum(); assert bad == 0, f"ArSAS: {bad} rows with unknown sentiment labels"
    df = df.assign(label=df["label"].replace("mixed", "neutral")) if keep_mixed else df[df["label"] != "mixed"]
    # The paper's rule for datasets without a split: 80% train / 10% dev / 10% test. Train+dev = 90%, stratified.
    trdev, _ = train_test_split(df, test_size=0.10, stratify=df["label"], random_state=seed)
    return trdev.assign(source="ArSAS")[["text", "label", "source"]]

# ---------------------------------------------------------------- SemEval-2017 Task 4A Arabic (official train + dev)
def load_semeval(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 3: continue
        text = p[2] if len(p) == 3 else "\t".join(p[2:])
        if len(text) >= 2 and text[0] == text[-1] == '"': text = text[1:-1]   # SemEval wraps text in quotes
        rows.append((text, norm(p[1])))
    df = pd.DataFrame(rows, columns=["text", "label"])
    bad = df["label"].isna().sum(); assert bad == 0, f"{path}: {bad} rows with unknown labels"
    return df.assign(source="SemEval2017")

# ---------------------------------------------------------------- ArSenTD-LEV (no official split -> 80/10/10, keep 90%)
def load_arsentd(spec, seed):
    if spec == "hf":
        from datasets import load_dataset
        ds = load_dataset("ramybaly/arsentd_lev", split="train")
        names = ds.features["Sentiment"].names
        df = pd.DataFrame({"text": ds["Tweet"], "label": [norm(names[i]) for i in ds["Sentiment"]]})
    else:
        sep = "," if spec.endswith(".csv") else "\t"
        raw = pd.read_csv(spec, sep=sep, dtype=str, encoding="utf-8")
        tcol = next(c for c in raw.columns if c.lower() in ("tweet", "text"))
        lcol = next(c for c in raw.columns if c.lower() == "sentiment")
        df = pd.DataFrame({"text": raw[tcol], "label": raw[lcol].map(norm)})
    bad = df["label"].isna().sum(); assert bad == 0, f"ArSenTD-LEV: {bad} rows with unknown labels"
    trdev, _ = train_test_split(df, test_size=0.10, stratify=df["label"], random_state=seed)
    return trdev.assign(source="ArSenTD-LEV")[["text", "label", "source"]]

# ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--astd_dir", required=True)
    ap.add_argument("--arsas"); ap.add_argument("--semeval_train"); ap.add_argument("--semeval_dev")
    ap.add_argument("--arsentd", help="path to ArSenTD-LEV tsv/csv, or 'hf'")
    ap.add_argument("--out", default="data"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arsas_keep_mixed", action="store_true", help="map ArSAS 'Mixed' to neutral instead of dropping it")
    ap.add_argument("--drop_test_overlap", action="store_true",
                    help="remove training tweets identical to ASTD test tweets (NOT done in [1]; off = faithful)")
    a = ap.parse_args()

    rep = {"decisions": [], "sources": {}, "missing_sources": []}
    astd_tr, astd_te = load_astd(a.astd_dir)
    parts = [astd_tr]; rep["sources"]["ASTD (train+validation)"] = report_counts(astd_tr)
    rep["decisions"].append("ASTD: official 4class-unbalanced split; OBJ dropped; NEUTRAL (=mixed in ASTD) kept as neutral.")

    if a.arsas:
        x = load_arsas(a.arsas, a.seed, a.arsas_keep_mixed); parts.append(x); rep["sources"]["ArSAS (90% train+dev)"] = report_counts(x)
        rep["decisions"].append(f"ArSAS: no official split; stratified 90% kept as train+dev (seed {a.seed}); "
                                f"Mixed {'mapped to neutral' if a.arsas_keep_mixed else 'dropped'}.")
    else: rep["missing_sources"].append("ArSAS")

    sem = [p for p in (a.semeval_train, a.semeval_dev) if p]
    if sem:
        x = pd.concat([load_semeval(p) for p in sem], ignore_index=True); parts.append(x)
        rep["sources"]["SemEval-2017 4A Arabic (" + "+".join("train" if p == a.semeval_train else "dev" for p in sem) + ")"] = report_counts(x)
    else: rep["missing_sources"].append("SemEval-2017")

    if a.arsentd:
        x = load_arsentd(a.arsentd, a.seed); parts.append(x); rep["sources"]["ArSenTD-LEV (90% train+dev)"] = report_counts(x)
        rep["decisions"].append(f"ArSenTD-LEV: no official split; stratified 90% kept (seed {a.seed}); very_pos/very_neg -> pos/neg.")
    else: rep["missing_sources"].append("ArSenTD-LEV")

    train = pd.concat(parts, ignore_index=True)
    train["text"] = train["text"].astype(str).str.strip()
    train = train[train["text"] != ""]
    assert set(train["label"]) <= {"positive", "negative", "neutral"}, set(train["label"])

    # Diagnostics (reported, not silently fixed)
    dup = train.duplicated(subset=["text"]).sum()
    conflict = train.groupby("text")["label"].nunique().gt(1).sum()
    overlap = train["text"].isin(set(astd_te["text"])).sum()
    rep["diagnostics"] = {"duplicate_texts_in_train": int(dup), "texts_with_conflicting_labels": int(conflict),
                          "train_tweets_identical_to_astd_test": int(overlap)}
    if a.drop_test_overlap:
        train = train[~train["text"].isin(set(astd_te["text"]))]
        rep["decisions"].append(f"Removed {overlap} training tweets identical to ASTD test tweets (departure from [1]).")

    rep["train_total"] = report_counts(train); rep["test_total"] = report_counts(astd_te)
    rep["paper_targets"] = {"train": 24800, "test": 664}
    rep["checks"] = {"train_vs_paper_pct": round(100 * len(train) / 24800, 1), "test_vs_paper": f"{len(astd_te)} vs 664"}

    os.makedirs(f"{a.out}/astd", exist_ok=True)
    train.sample(frac=1, random_state=a.seed).to_csv(f"{a.out}/astd/train.csv", index=False)
    astd_te.to_csv(f"{a.out}/astd/test.csv", index=False)
    json.dump(rep, open(f"{a.out}/astd/build_report.json", "w"), indent=2, ensure_ascii=False)

    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if rep["missing_sources"]:
        print(f"\nWARNING: missing sources {rep['missing_sources']} -> log as a forced departure; "
              f"train set is {rep['checks']['train_vs_paper_pct']}% of the paper's 24.8K.", file=sys.stderr)

if __name__ == "__main__":
    main()
