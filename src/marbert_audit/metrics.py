"""Sentiment metrics computed from confusion matrices.

Every function accepts confusion matrices with any number of leading batch dimensions
(shape ``(..., K, K)``, rows = gold, columns = predicted). The bootstrap uses the batched form.

Definitions (all reported in percent):
  * accuracy  - share of items predicted correctly.
  * macro_f1  - unweighted mean of the per-class F1 over ALL classes.
  * f1_pn     - mean of the positive-class and negative-class F1, computed on the full test set
                with neutral items kept (a neutral item predicted positive is a false positive for
                the positive class; a positive item predicted neutral is a false negative).
F1 for a class with no gold and no predicted items is 0, matching scikit-learn's zero_division=0.
"""
from __future__ import annotations

import numpy as np

METRIC_LABELS = {"f1_pn": "F1-PN", "macro_f1": "Macro-F1", "accuracy": "Accuracy"}


def confusion_matrix(gold, pred, n_labels: int) -> np.ndarray:
    gold = np.asarray(gold, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    if gold.shape != pred.shape:
        raise ValueError("gold and pred must have the same shape")
    if gold.size and (gold.min() < 0 or gold.max() >= n_labels or pred.min() < 0 or pred.max() >= n_labels):
        raise ValueError("label index out of range")
    codes = gold * n_labels + pred
    return np.bincount(codes, minlength=n_labels * n_labels).reshape(n_labels, n_labels)


def per_class(cm: np.ndarray) -> dict[str, np.ndarray]:
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    pred_tot = cm.sum(axis=-2)
    gold_tot = cm.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_tot > 0, tp / pred_tot, 0.0)
        recall = np.where(gold_tot > 0, tp / gold_tot, 0.0)
        denom = pred_tot + gold_tot  # = 2tp + fp + fn
        f1 = np.where(denom > 0, 2.0 * tp / denom, 0.0)
    return {"precision": precision, "recall": recall, "f1": f1, "support": gold_tot}


def scores_from_cm(cm: np.ndarray, pos_idx: int | None = None, neg_idx: int | None = None) -> dict[str, np.ndarray]:
    cm = np.asarray(cm, dtype=np.float64)
    pc = per_class(cm)
    total = cm.sum(axis=(-2, -1))
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.where(total > 0, np.diagonal(cm, axis1=-2, axis2=-1).sum(axis=-1) / total, 0.0)
    out = {"accuracy": 100.0 * acc, "macro_f1": 100.0 * pc["f1"].mean(axis=-1)}
    if pos_idx is not None and neg_idx is not None:
        out["f1_pn"] = 100.0 * (pc["f1"][..., pos_idx] + pc["f1"][..., neg_idx]) / 2.0
    return out


def compute_metrics(gold, pred, labels: list[str], pos_idx: int, neg_idx: int) -> dict:
    """Scalar metrics plus a per-class report for one set of predictions."""
    cm = confusion_matrix(gold, pred, len(labels))
    s = scores_from_cm(cm, pos_idx, neg_idx)
    pc = per_class(cm)
    report = {
        lab: {
            "precision": 100.0 * float(pc["precision"][k]),
            "recall": 100.0 * float(pc["recall"][k]),
            "f1": 100.0 * float(pc["f1"][k]),
            "support": int(pc["support"][k]),
        }
        for k, lab in enumerate(labels)
    }
    return {
        "accuracy": float(s["accuracy"]),
        "macro_f1": float(s["macro_f1"]),
        "f1_pn": float(s["f1_pn"]),
        "per_class": report,
        "confusion": cm.tolist(),
    }
