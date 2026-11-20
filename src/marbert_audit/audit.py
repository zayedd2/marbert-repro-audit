"""Data audit (owned by Member 1; code shared so the audit table is regenerable).

For each dataset and split: item counts, label counts, duplicates within the split (after aggressive
normalisation), duplicate texts with conflicting labels, test items whose text also occurs in the
training data, and overlap between every dataset's test set and every other dataset's training set.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATASET_ORDER
from .data import get_train_dev, read_split, sha256_file, processed_path
from .preprocess import normalise_for_matching


def _norm(df: pd.DataFrame) -> pd.Series:
    return df["text"].map(normalise_for_matching)


def _split_row(cfg, d, split, df, train_norm: set | None) -> dict:
    norm = _norm(df)
    labels = cfg["datasets"][d]["labels"]
    row = {"dataset": d, "dataset_name": cfg["datasets"][d]["display_name"], "split": split, "n": len(df)}
    for lab in ("negative", "neutral", "positive"):
        row[f"n_{lab}"] = int((df["label"] == lab).sum()) if lab in labels else float("nan")
    row["duplicates_within"] = int(len(norm) - norm.nunique())
    grp = pd.DataFrame({"t": norm, "y": df["label"]}).groupby("t")["y"].nunique()
    row["conflicting_label_texts"] = int((grp > 1).sum())
    row["empty_texts"] = int((norm == "").sum())
    row["overlap_with_train"] = int(norm.isin(train_norm).sum()) if train_norm is not None else float("nan")
    if "source" in df.columns:
        row["sources"] = "; ".join(f"{k}={v}" for k, v in df["source"].value_counts().sort_index().items())
    return row


def source_overlap(train: pd.DataFrame, test: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Before pooling: overlap between the training sources, and between the test set and each source."""
    if "source" not in train.columns:
        return pd.DataFrame()
    tr = pd.DataFrame({"source": train["source"], "t": _norm(train)})
    sets = {s: set(g["t"]) for s, g in tr.groupby("source")}
    test_norm = _norm(test)
    rows = []
    for a in sorted(sets):
        for b in sorted(sets):
            if a < b:
                rows.append({"dataset": dataset, "compare": f"train:{a} vs train:{b}", "n_a": len(sets[a]), "n_b": len(sets[b]),
                             "shared_texts": len(sets[a] & sets[b])})
        rows.append({"dataset": dataset, "compare": f"test vs train:{a}", "n_a": len(test_norm), "n_b": len(sets[a]),
                     "shared_texts": int(test_norm.isin(sets[a]).sum())})
    return pd.DataFrame(rows)


def run_audit(cfg: dict, data_root: Path | str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, norms, manifest = [], {}, {}
    src_frames = []
    for d in DATASET_ORDER:
        try:
            train = read_split(data_root, d, "train")
            test = read_split(data_root, d, "test")
        except FileNotFoundError:
            continue
        train_norm = set(_norm(train))
        norms[d] = {"train": train_norm, "test": _norm(test)}
        manifest[d] = {s: {"sha256": sha256_file(processed_path(data_root, d, s)), "n": n}
                       for s, n in (("train", len(train)), ("test", len(test)))}
        src_frames.append(source_overlap(train, test, d))
        rows.append(_split_row(cfg, d, "train (full)", train, None))
        tr, dev = get_train_dev(data_root, d, cfg)
        tr_norm = set(_norm(tr))
        rows.append(_split_row(cfg, d, "train (90%)", tr, None))
        rows.append(_split_row(cfg, d, "dev (10%)", dev, tr_norm))
        rows.append(_split_row(cfg, d, "test", test, train_norm))
        expected = cfg["datasets"][d].get("paper_sizes", {})
        manifest[d]["paper_sizes"] = expected
        manifest[d]["size_differences"] = {s: len(x) - expected[s] for s, x in (("train", train), ("test", test)) if s in expected}
    overlap = []
    for a in norms:
        for b in norms:
            overlap.append({"test_of": a, "train_of": b, "n_test": len(norms[a]["test"]),
                            "test_items_in_train": int(norms[a]["test"].isin(norms[b]["train"]).sum())})
    src = pd.concat(src_frames, ignore_index=True) if any(not f.empty for f in src_frames) else pd.DataFrame()
    manifest["_source_overlap"] = src.to_dict(orient="records")
    return pd.DataFrame(rows), pd.DataFrame(overlap), manifest
