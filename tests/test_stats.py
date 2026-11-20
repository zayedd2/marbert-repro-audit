import numpy as np

from marbert_audit.stats import paired_bootstrap, seed_summary


def test_identical_models_give_zero_interval():
    rng = np.random.default_rng(0)
    gold = rng.integers(0, 3, 200)
    preds = [rng.integers(0, 3, 200) for _ in range(3)]
    r = paired_bootstrap(gold, preds, preds, 3, 2, 0, "macro_f1", n_resamples=300, seed=1)
    assert r["point"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0


def test_better_model_has_positive_interval_and_is_deterministic():
    rng = np.random.default_rng(0)
    gold = rng.integers(0, 3, 1000)
    good = [np.where(rng.random(1000) < 0.85, gold, rng.integers(0, 3, 1000)) for _ in range(3)]
    bad = [np.where(rng.random(1000) < 0.55, gold, rng.integers(0, 3, 1000)) for _ in range(3)]
    r1 = paired_bootstrap(gold, good, bad, 3, 2, 0, "f1_pn", n_resamples=500, seed=7, chunk_size=128)
    r2 = paired_bootstrap(gold, good, bad, 3, 2, 0, "f1_pn", n_resamples=500, seed=7, chunk_size=64)
    assert r1["ci_low"] > 0 and r1["p_gt0"] == 1.0
    assert r1["ci_low"] <= r1["point"] <= r1["ci_high"]
    assert np.isclose(r1["ci_low"], r2["ci_low"]) and np.isclose(r1["ci_high"], r2["ci_high"])  # chunking does not matter


def test_seed_summary_uses_sample_sd():
    s = seed_summary([1.0, 2.0, 3.0])
    assert s["mean"] == 2.0 and np.isclose(s["sd"], 1.0) and s["min"] == 1.0 and s["max"] == 3.0
