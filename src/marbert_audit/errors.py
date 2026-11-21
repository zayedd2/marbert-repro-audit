"""Inputs to Member 1's error analysis: confusion matrices and ArSarcasm subgroup results.

All use the reported checkpoint of each run (independent protocol unless stated). Subgroups come only from
labels that already exist in ArSarcasm (dialect, sarcasm), read from data/meta/arsarcasm_test_meta.csv
(labels only; no tweet text is stored in the repository). Groups below ``subgroups.min_group_size`` items are
flagged as exploratory.

``subgroup_ci_table`` reproduces Member 3's subgroup bootstrap exactly (B = 2,000, generator seeded 2026,
same loop order), so its intervals match scripts/member3/subgroup_boot.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASET_ORDER, MODEL_ORDER, REPO_ROOT, labels_for, pos_neg_idx
from .metrics import confusion_matrix, scores_from_cm
from .stats import load_predictions, seed_summary
from .summary import _subset

META = REPO_ROOT / "data" / "meta" / "arsarcasm_test_meta.csv"


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
                             "mean_count_per_run": float(v["sum"][i, j] / v["n_seeds"]),
                             "row_share": float(v["sum"][i, j] / total[i, 0]) if total[i, 0] else np.nan,
                             "n_seeds": v["n_seeds"]})
    return pd.DataFrame(rows)


def _meta(meta_path: Path | str | None = None) -> pd.DataFrame:
    meta = pd.read_csv(meta_path or META, dtype=str, keep_default_na=False)
    meta["sarcasm"] = meta["sarcasm"].str.lower().map({"true": "sarcastic", "false": "non-sarcastic"})
    return meta.set_index("id")


def subgroup_table(runs: pd.DataFrame, cfg: dict, protocol: str = "independent", meta_path=None) -> pd.DataFrame:
    """Macro-F1 and F1-PN per dialect and sarcasm group (mean +/- SD over seeds), with group sizes."""
    min_n = int(cfg.get("subgroups", {}).get("min_group_size", 100))
    meta = _meta(meta_path)
    labels = labels_for(cfg, "arsarcasm")
    pos, neg = pos_neg_idx(cfg, "arsarcasm")
    rows = []
    for m in MODEL_ORDER:
        sub = _subset(runs, protocol=protocol, dataset="arsarcasm", model=m)
        if sub.empty:
            continue
        preds = [load_predictions(r["run_dir"], "test", int(r["selected_epoch"])) for _, r in sub.iterrows()]
        for col in ("dialect", "sarcasm"):
            values = meta.loc[preds[0]["id"], col].to_numpy()
            for val in sorted(set(values)):
                mask = values == val
                sc = {"macro_f1": [], "f1_pn": []}
                for pr in preds:
                    cm = confusion_matrix(pr["gold"].to_numpy()[mask], pr["pred"].to_numpy()[mask], len(labels))
                    out = scores_from_cm(cm, pos, neg)
                    for k in sc:
                        sc[k].append(float(out[k]))
                row = {"protocol": protocol, "model": m, "group_column": col, "group": val, "n_items": int(mask.sum()),
                       "exploratory": bool(mask.sum() < min_n), "n_seeds": len(preds)}
                for k, v in sc.items():
                    s = seed_summary(v)
                    row[f"{k}_mean"], row[f"{k}_sd"] = s["mean"], s["sd"]
                rows.append(row)
    return pd.DataFrame(rows)


def _f1_mean(G: np.ndarray, P: np.ndarray, classes: list[int]) -> np.ndarray:
    out = []
    for c in classes:
        tp = ((G == c) & (P == c)).sum(1)
        fp = ((G != c) & (P == c)).sum(1)
        fn = ((G == c) & (P != c)).sum(1)
        den = 2 * tp + fp + fn
        out.append(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0))
    return 100 * np.mean(out, 0)


def subgroup_ci_table(runs: pd.DataFrame, cfg: dict, meta_path=None, n_resamples: int = 2000, seed: int = 2026) -> pd.DataFrame:
    """MARBERT - AraBERT per ArSarcasm subgroup with a paired bootstrap CI (Member 3's procedure)."""
    meta = _meta(meta_path)
    rng = np.random.default_rng(seed)
    groups = [("sarcasm", ["sarcastic", "non-sarcastic"]), ("dialect", ["msa", "egypt", "gulf", "levant"])]
    rows = []
    for col, values in groups:
        for p in ("historical", "independent"):
            preds = {}
            for m in MODEL_ORDER:
                sub = _subset(runs, protocol=p, dataset="arsarcasm", model=m)
                preds[m] = {int(r["seed"]): load_predictions(r["run_dir"], "test", int(r["selected_epoch"])).set_index("id")
                            for _, r in sub.iterrows()}
            seeds = sorted(set(preds["marbert"]) & set(preds["arabert"]))
            if not seeds:
                continue
            for gv in values:
                ids = np.sort(meta.index[meta[col] == gv].to_numpy())
                n = len(ids)
                idx = rng.integers(0, n, (n_resamples, n))
                for met, classes in (("macro_f1", [0, 1, 2]), ("f1_pn", [0, 2])):
                    obs, bs = [], []
                    for s in seeds:
                        v = {m: (preds[m][s].loc[ids, "gold"].to_numpy(), preds[m][s].loc[ids, "pred"].to_numpy()) for m in MODEL_ORDER}
                        obs.append(_f1_mean(v["marbert"][0][None], v["marbert"][1][None], classes)[0]
                                   - _f1_mean(v["arabert"][0][None], v["arabert"][1][None], classes)[0])
                        bs.append(_f1_mean(v["marbert"][0][idx], v["marbert"][1][idx], classes)
                                  - _f1_mean(v["arabert"][0][idx], v["arabert"][1][idx], classes))
                    lo, hi = np.percentile(np.mean(bs, 0), [2.5, 97.5])
                    rows.append({"protocol": p, "group_column": col, "group": gv, "n": n, "metric": met,
                                 "seed_diffs": " / ".join(f"{x:+.2f}" for x in obs), "mean_diff": float(np.mean(obs)),
                                 "ci_low": float(lo), "ci_high": float(hi), "excludes_zero": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)
