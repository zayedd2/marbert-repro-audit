"""Inputs for Member 1's linguistic error analysis: confusion matrices, subgroup scores, failure examples.

All use the INDEPENDENT protocol (dev-selected checkpoints). Subgroups come only from labels that already
exist in the data (ArSarcasm: dialect, sarcasm). Groups below ``subgroups.min_group_size`` are flagged
as exploratory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASET_ORDER, MODEL_ORDER, labels_for, pos_neg_idx
from .data import read_split
from .metrics import confusion_matrix, scores_from_cm
from .stats import load_predictions, seed_summary
from .summary import _subset


def confusion_matrices(runs: pd.DataFrame, cfg: dict, protocol: str = "independent") -> dict:
    """{(dataset, model): {"sum": KxK counts summed over seeds, "per_seed": [...], "labels": [...]}}"""
    out = {}
    for d in DATASET_ORDER:
        labels = labels_for(cfg, d)
        for m in MODEL_ORDER:
            sub = _subset(runs, protocol=protocol, dataset=d, model=m)
            if sub.empty:
                continue
            cms = []
            for _, r in sub.iterrows():
                pr = load_predictions(r["run_dir"], "test", int(r["selected_epoch"]))
                cms.append(confusion_matrix(pr["gold"], pr["pred"], len(labels)))
            out[(d, m)] = {"sum": np.sum(cms, axis=0), "per_seed": cms, "labels": labels, "n_seeds": len(cms)}
    return out


def confusion_long(cms: dict, cfg: dict) -> pd.DataFrame:
    rows = []
    for (d, m), v in cms.items():
        total = v["sum"].sum(axis=1, keepdims=True)
        for i, g in enumerate(v["labels"]):
            for j, p in enumerate(v["labels"]):
                rows.append({"dataset": d, "model": m, "gold": g, "pred": p, "count_summed_over_seeds": int(v["sum"][i, j]),
                             "row_share": float(v["sum"][i, j] / total[i, 0]) if total[i, 0] else np.nan,
                             "n_seeds": v["n_seeds"]})
    return pd.DataFrame(rows)


def subgroup_table(runs: pd.DataFrame, cfg: dict, data_root: Path | str, protocol: str = "independent") -> pd.DataFrame:
    min_n = int(cfg.get("subgroups", {}).get("min_group_size", 50))
    rows = []
    for d in DATASET_ORDER:
        cols = cfg["datasets"][d].get("subgroup_columns") or []
        if not cols:
            continue
        try:
            test = read_split(data_root, d, "test")
        except FileNotFoundError:
            continue
        cols = [c for c in cols if c in test.columns]
        labels = labels_for(cfg, d)
        pos, neg = pos_neg_idx(cfg, d)
        hist = cfg["datasets"][d]["historical_metric"]
        meta = test.set_index("id")
        for m in MODEL_ORDER:
            sub = _subset(runs, protocol=protocol, dataset=d, model=m)
            if sub.empty:
                continue
            preds = [load_predictions(r["run_dir"], "test", int(r["selected_epoch"])) for _, r in sub.iterrows()]
            for col in cols:
                values = meta.loc[preds[0]["id"], col].to_numpy()
                for val in sorted(set(values)):
                    mask = values == val
                    scores = {"macro_f1": [], hist: []}
                    for pr in preds:
                        cm = confusion_matrix(pr["gold"].to_numpy()[mask], pr["pred"].to_numpy()[mask], len(labels))
                        sc = scores_from_cm(cm, pos, neg)
                        for k in scores:
                            scores[k].append(float(sc[k]))
                    row = {"dataset": d, "model": m, "group_column": col, "group": val, "n_items": int(mask.sum()),
                           "exploratory": bool(mask.sum() < min_n), "n_seeds": len(preds)}
                    gold_lab = preds[0]["gold"].to_numpy()[mask]
                    for k, lab in enumerate(labels):
                        row[f"n_{lab}"] = int((gold_lab == k).sum())
                    for k, v in scores.items():
                        s = seed_summary(v)
                        row[f"{k}_mean"], row[f"{k}_sd"] = s["mean"], s["sd"]
                    rows.append(row)
    return pd.DataFrame(rows)


def failure_examples(runs: pd.DataFrame, cfg: dict, data_root: Path | str, protocol: str = "independent") -> pd.DataFrame:
    """Test items that BOTH models get wrong in EVERY seed (with the tweet text, for qualitative analysis)."""
    frames = []
    for d in DATASET_ORDER:
        labels = labels_for(cfg, d)
        try:
            test = read_split(data_root, d, "test").set_index("id")
        except FileNotFoundError:
            continue
        wrong_all, pred_cols = None, {}
        for m in MODEL_ORDER:
            sub = _subset(runs, protocol=protocol, dataset=d, model=m)
            for _, r in sub.iterrows():
                pr = load_predictions(r["run_dir"], "test", int(r["selected_epoch"])).set_index("id")
                w = pr["pred"] != pr["gold"]
                wrong_all = w if wrong_all is None else (wrong_all & w)
                pred_cols[f"{m}_seed{int(r['seed'])}"] = pr["pred"].map(dict(enumerate(labels)))
        if wrong_all is None:
            continue
        ids = wrong_all[wrong_all].index
        out = test.loc[ids].copy()
        out.insert(0, "dataset", d)
        for k, v in pred_cols.items():
            out[k] = v.loc[ids]
        frames.append(out.reset_index())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
