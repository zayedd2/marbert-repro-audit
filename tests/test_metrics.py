import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from marbert_audit.metrics import compute_metrics, confusion_matrix, scores_from_cm

LABELS = ["negative", "neutral", "positive"]


def test_matches_sklearn_on_random_predictions():
    rng = np.random.default_rng(0)
    for _ in range(50):
        gold = rng.integers(0, 3, 300)
        pred = np.where(rng.random(300) < 0.6, gold, rng.integers(0, 3, 300))
        m = compute_metrics(gold, pred, LABELS, pos_idx=2, neg_idx=0)
        assert np.isclose(m["accuracy"], 100 * accuracy_score(gold, pred))
        assert np.isclose(m["macro_f1"], 100 * f1_score(gold, pred, average="macro", zero_division=0))
        # F1-PN: macro over the positive and negative classes only, neutral items KEPT in the data
        assert np.isclose(m["f1_pn"], 100 * f1_score(gold, pred, labels=[2, 0], average="macro", zero_division=0))


def test_f1_pn_counts_neutral_confusions():
    gold = np.array([2, 2, 1, 1, 0, 0])
    pred = np.array([2, 1, 2, 1, 0, 0])  # one positive predicted neutral, one neutral predicted positive
    m = compute_metrics(gold, pred, LABELS, pos_idx=2, neg_idx=0)
    assert np.isclose(m["f1_pn"], 100 * (0.5 + 1.0) / 2)


def test_missing_class_gives_zero_f1():
    gold = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1])
    m = compute_metrics(gold, pred, LABELS, pos_idx=2, neg_idx=0)
    assert np.isclose(m["macro_f1"], 100 * 2 / 3)


def test_batched_confusions_equal_single():
    rng = np.random.default_rng(1)
    gold, pred = rng.integers(0, 3, 100), rng.integers(0, 3, 100)
    cm = confusion_matrix(gold, pred, 3)
    batch = np.stack([cm, cm])
    assert np.allclose(scores_from_cm(batch, 2, 0)["f1_pn"], scores_from_cm(cm, 2, 0)["f1_pn"])
