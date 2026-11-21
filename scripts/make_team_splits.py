# Member 1: one shared set of split files built from Member 2's train/test files.
# Adds an id column, lower-case labels, and a stratified 10% dev split (seed 42) taken from each training file.
import re, sys, os, unicodedata, json
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
SEED = 42
A = sys.argv[1]; OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
def norm(t):
    t = unicodedata.normalize("NFKC", str(t)); t = re.sub(r"https?://\S+|www\.\S+", " ", t); t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"\bRT\b", " ", t); t = DIAC.sub("", t).replace("ـ", "")
    t = re.sub("[إأآٱ]", "ا", t).replace("ى", "ي").replace("ة", "ه"); t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t); return re.sub(r"\s+", " ", t).strip().lower()
def build(name, train_f, test_f, strat_cols, extra):
    tr = pd.read_csv(train_f, dtype=str, keep_default_na=False); te = pd.read_csv(test_f, dtype=str, keep_default_na=False)
    for d in (tr, te): d["label"] = d["label"].str.strip().str.lower()
    tr.insert(0, "id", f"{name}_train_" + pd.Series(range(len(tr))).astype(str).str.zfill(5))
    te.insert(0, "id", f"{name}_test_" + pd.Series(range(len(te))).astype(str).str.zfill(5))
    tr["norm"] = tr.text.map(norm); te["norm"] = te.text.map(norm)
    strat = tr[strat_cols].agg("|".join, axis=1)
    groups = tr.norm.where(tr.norm != "", tr.id)
    k, h = next(StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED).split(tr, strat, groups))
    cols = ["id", "text", "label"] + extra
    parts = {"train_full": tr, "train": tr.iloc[k], "dev": tr.iloc[h], "test": te}
    for p, d in parts.items(): d[cols].to_csv(f"{OUT}/{name}_{p}.csv", index=False)
    lab = lambda d: d.label.value_counts().reindex(["negative", "neutral", "positive"], fill_value=0).to_dict()
    leak = int(te.norm.isin(set(tr.norm) - {""}).sum())
    return {p: {"n": len(d), **lab(d)} for p, d in parts.items()} | {"test_tweets_with_copy_in_train_full (normalised)": leak}
rep = {
 "arsarcasm": build("arsarcasm", A + "b4948fa3-train.csv", A + "700c7a1a-test.csv", ["label"], ["dialect", "sarcasm"]),
 "ajgt": build("ajgt", A + "d1a20eba-train.csv", A + "035b4041-test.csv", ["label"], ["ID"]),
 "astd": build("astd", A + "6357e44a-train.csv", A + "cd5a4d00-test.csv", ["source", "label"], ["source"]),
}
json.dump(rep, open(f"{OUT}/split_summary.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
