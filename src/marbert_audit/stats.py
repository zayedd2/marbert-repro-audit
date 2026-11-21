"""Reliability statistics (audit item 4): seed variation and paired bootstrap over test items.

The two sources of uncertainty are kept separate, as the proposal requires:
  * training randomness  -> per-seed paired differences, mean +/- SD across seeds;
  * finite-test sampling -> 95% paired bootstrap interval, resampling test ITEMS with replacement.
Pairing: MARBERT seed s is paired with AraBERT seed s, and every bootstrap resample uses the same
item indices for both models and all seeds, so item difficulty cancels out of the difference.
The procedure (B = 1,000, one generator seeded 12345, drawn per protocol and dataset, percentile CI) is the
same as in the team's notebook, so the intervals regenerate exactly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import scores_from_cm


def seed_summary(values) -> dict:
    v = np.asarray(list(values), dtype=float)
    return {
        "n": int(v.size),
        "mean": float(v.mean()) if v.size else np.nan,
        "sd": float(v.std(ddof=1)) if v.size > 1 else np.nan,
        "min": float(v.min()) if v.size else np.nan,
        "max": float(v.max()) if v.size else np.nan,
    }


def load_predictions(run_dir: Path | str, split: str, epoch: int) -> pd.DataFrame:
    df = pd.read_csv(Path(run_dir) / f"preds_{split}.csv.gz", dtype={"id": str})
    sel = df[df["epoch"] == epoch]
    if sel.empty:
        raise ValueError(f"no {split} predictions for epoch {epoch} in {run_dir}")
    return sel.sort_values("id").reset_index(drop=True)


def _batched_scores(codes: np.ndarray, idx: np.ndarray, n_labels: int, pos_idx: int, neg_idx: int, metric: str) -> np.ndarray:
    """Metric for each bootstrap resample. ``codes`` = gold*K + pred (n,), ``idx`` = (b, n) item indices."""
    b, n = idx.shape
    kk = n_labels * n_labels
    flat = codes[idx] + (np.arange(b)[:, None] * kk)
    cms = np.bincount(flat.ravel(), minlength=b * kk).reshape(b, n_labels, n_labels)
    return scores_from_cm(cms, pos_idx, neg_idx)[metric]


def bootstrap_indices(rng: np.random.Generator, n_items: int, n_resamples: int) -> np.ndarray:
    """Resampled item indices, shape (n_resamples, n_items). Shared by both models and all seeds (pairing)."""
    return rng.integers(0, n_items, size=(n_resamples, n_items))


def paired_bootstrap(gold: np.ndarray, preds_a: list[np.ndarray], preds_b: list[np.ndarray], n_labels: int,
                     pos_idx: int, neg_idx: int, metric: str, n_resamples: int = 1000, seed: int = 12345,
                     confidence: float = 0.95, chunk_size: int = 500, idx: np.ndarray | None = None) -> dict:
    """Bootstrap the seed-averaged difference metric(A) - metric(B), in percentage points.

    ``idx`` (n_resamples x n_items) can be passed in so several metrics share the same resamples, exactly as
    the team's notebook does; otherwise it is drawn from ``seed``.
    """
    gold = np.asarray(gold, dtype=np.int64)
    n = gold.size
    if len(preds_a) != len(preds_b) or not preds_a:
        raise ValueError("need the same, non-zero number of seeds for both models")
    if idx is None:
        idx = bootstrap_indices(np.random.default_rng(seed), n, n_resamples)
    n_resamples = idx.shape[0]
    codes_a = [gold * n_labels + np.asarray(p, dtype=np.int64) for p in preds_a]
    codes_b = [gold * n_labels + np.asarray(p, dtype=np.int64) for p in preds_b]
    full = np.arange(n)[None, :]
    point_seed = np.array([
        _batched_scores(ca, full, n_labels, pos_idx, neg_idx, metric)[0] - _batched_scores(cb, full, n_labels, pos_idx, neg_idx, metric)[0]
        for ca, cb in zip(codes_a, codes_b)
    ])
    per_seed = np.empty((len(codes_a), n_resamples))
    for start in range(0, n_resamples, chunk_size):
        sl = idx[start:start + chunk_size]
        for s, (ca, cb) in enumerate(zip(codes_a, codes_b)):
            per_seed[s, start:start + len(sl)] = (_batched_scores(ca, sl, n_labels, pos_idx, neg_idx, metric)
                                                  - _batched_scores(cb, sl, n_labels, pos_idx, neg_idx, metric))
    avg = per_seed.mean(axis=0)
    q = [100 * (1.0 - confidence) / 2.0, 100 * (1.0 - (1.0 - confidence) / 2.0)]
    lo, hi = np.percentile(avg, q)
    seed_lo, seed_hi = np.percentile(per_seed, q, axis=1)
    return {
        "point": float(point_seed.mean()),
        "per_seed_point": point_seed.tolist(),
        "ci_low": float(lo), "ci_high": float(hi),
        "p_gt0": float((avg > 0).mean()),
        "boot_sd": float(avg.std(ddof=1)),
        "per_seed_ci": list(zip(seed_lo.tolist(), seed_hi.tolist())),
        "per_seed_p_not_better": (per_seed <= 0).mean(axis=1).tolist(),
        "n_items": int(n), "n_resamples": int(n_resamples),
    }
