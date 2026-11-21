"""Checkpoint selection and per-run summaries (audit items 1-3)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import DATASET_ORDER, MODEL_ORDER, PROTOCOL_ORDER, selection_metric

METRICS = ("accuracy", "macro_f1", "f1_pn")


def select_epoch(epochs: pd.DataFrame, split: str, metric: str, tol: float = 1e-9) -> int:
    """Epoch with the highest ``<split>_<metric>``; ties go to the EARLIEST epoch."""
    col = f"{split}_{metric}"
    if col not in epochs.columns:
        raise KeyError(f"column {col} not in epoch metrics (protocol without a {split} split?)")
    ep = epochs.sort_values("epoch").reset_index(drop=True)
    vals = ep[col].to_numpy(dtype=float)
    first = int(np.flatnonzero(vals >= vals.max() - tol)[0])
    return int(ep.loc[first, "epoch"])


def summarize_run(run_dir: Path, cfg: dict) -> dict:
    rc = yaml.safe_load((run_dir / "run_config.yaml").read_text(encoding="utf-8"))
    ep = pd.read_csv(run_dir / "epoch_metrics.csv").sort_values("epoch").reset_index(drop=True)
    env = json.loads((run_dir / "env.json").read_text(encoding="utf-8")) if (run_dir / "env.json").exists() else {}
    protocol, dataset = rc["protocol"], rc["dataset"]
    split, metric = selection_metric(cfg, protocol, dataset)
    e_sel = select_epoch(ep, split, metric)
    sel = ep.loc[ep["epoch"] == e_sel].iloc[0]
    hist = cfg["datasets"][dataset]["historical_metric"]
    out = {
        "protocol": protocol, "dataset": dataset, "model": rc["model"], "seed": int(rc["seed"]),
        "backend": rc.get("backend", "hf"), "run_dir": str(run_dir),
        "selection_split": split, "selection_metric": metric, "selected_epoch": e_sel,
        "n_epochs": int(ep["epoch"].max()),
        "n_train": rc["n_items"]["train"], "n_dev": rc["n_items"].get("dev"), "n_test": rc["n_items"]["test"],
        "hist_metric": hist, "test_hist": float(sel[f"test_{hist}"]),
        "gpu": env.get("gpu"), "train_seconds": env.get("epoch_seconds_total", env.get("wall_clock_seconds")),
        "model_commit": (env.get("model_info") or {}).get("model_commit"),
        "git_commit": env.get("git_commit"),
    }
    rob = cfg["protocols"][protocol].get("robustness_selection")
    if rob:
        e_rob = select_epoch(ep, rob["split"], rob["metric"])
        r_sel = ep.loc[ep["epoch"] == e_rob].iloc[0]
        out["robust_selected_epoch"] = e_rob
        out["robust_test_hist"] = float(r_sel[f"test_{hist}"])
        out["robust_test_macro_f1"] = float(r_sel["test_macro_f1"])
    last = ep.iloc[-1]
    out["last_test_hist"] = float(last[f"test_{hist}"])
    for m in METRICS:
        out[f"test_{m}"] = float(sel[f"test_{m}"])
        out[f"max_test_{m}"] = float(ep[f"test_{m}"].max())
        out[f"argmax_test_{m}_epoch"] = select_epoch(ep, "test", m)
        out[f"ckpt_gap_{m}"] = out[f"max_test_{m}"] - out[f"test_{m}"]  # >= 0 by construction
        if f"dev_{m}" in ep.columns:
            out[f"dev_{m}"] = float(sel[f"dev_{m}"])
    out["max_test_hist"] = out[f"max_test_{hist}"]
    out["ckpt_gap_hist"] = out[f"ckpt_gap_{hist}"]
    return out


def iter_run_dirs(results_root: Path | str, subdir: str = "runs"):
    root = Path(results_root) / subdir
    if not root.exists():
        return
    pattern = "*/*/*/seed*/DONE" if subdir == "runs" else "**/seed*/DONE"
    for done in sorted(root.glob(pattern)):
        yield done.parent


def collect_runs(results_root: Path | str, cfg: dict, subdir: str = "runs", datasets: list[str] | None = None) -> pd.DataFrame:
    """Summaries of all completed runs. By default only the three main datasets (the 36-run grid);
    pass datasets=["ajgt_ubc"] for the AJGT check runs on the authors' split."""
    datasets = datasets or DATASET_ORDER
    rows = [summarize_run(d, cfg) for d in iter_run_dirs(results_root, subdir)]
    rows = [r for r in rows if r["dataset"] in datasets]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["protocol"] = pd.Categorical(df["protocol"], PROTOCOL_ORDER, ordered=True)
    df["dataset"] = pd.Categorical(df["dataset"], datasets, ordered=True)
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    return df.sort_values(["protocol", "dataset", "model", "seed"]).reset_index(drop=True)


def expected_runs(cfg: dict) -> list[tuple[str, str, str, int]]:
    return [(p, d, m, s) for p in PROTOCOL_ORDER for d in DATASET_ORDER for m in MODEL_ORDER for s in cfg["seeds"]]
