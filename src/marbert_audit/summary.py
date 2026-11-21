"""Turn the per-run summaries into the result tables behind every RQ.

RQ1 reproduction_table   historical protocol vs published targets
RQ2 checkpoint_table     dev-selected test score vs best-on-test score (independent runs)
RQ3 margins_table        MARBERT - AraBERT: seed mean +/- SD and paired bootstrap CI
RQ4 margins_table        the same margins under the historical metric and under macro-F1
main independent_table   dev-selected test scores (the main result)
App. per_class_table     per-class precision / recall / F1 / support
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASET_ORDER, MODEL_ORDER, PROTOCOL_ORDER, labels_for, pos_neg_idx
from .metrics import confusion_matrix, per_class
from .stats import bootstrap_indices, load_predictions, paired_bootstrap, seed_summary


def metric_pair(cfg: dict, dataset: str) -> list[str]:
    """[historical metric, macro_f1] - the two metrics compared for RQ4."""
    return [cfg["datasets"][dataset]["historical_metric"], "macro_f1"]


def _subset(runs: pd.DataFrame, **kw) -> pd.DataFrame:
    if runs.empty:
        return runs
    mask = np.ones(len(runs), dtype=bool)
    for k, v in kw.items():
        mask &= (runs[k].astype(str) == str(v)).to_numpy()
    return runs.loc[mask]


# ------------------------------------------------------------------------------ RQ1
def reproduction_table(runs: pd.DataFrame, cfg: dict, datasets: list[str] | None = None) -> pd.DataFrame:
    thr = float(cfg["reproduction"]["gap_threshold_pp"])
    rows = []
    for d in datasets or DATASET_ORDER:
        dcfg = cfg["datasets"][d]
        means = {}
        for m in MODEL_ORDER:
            sub = _subset(runs, protocol="historical", dataset=d, model=m)
            s = seed_summary(sub["test_hist"]) if not sub.empty else seed_summary([])
            means[m] = s["mean"]
            pub = float(dcfg["published"][m])
            rows.append({
                "dataset": d, "dataset_name": dcfg["display_name"], "metric": dcfg["historical_metric"],
                "model": m, "model_name": cfg["models"][m]["display_name"], "published": pub,
                "n_seeds": s["n"], "mean": s["mean"], "sd": s["sd"], "min": s["min"], "max": s["max"],
                "gap": s["mean"] - pub if s["n"] else np.nan,
                "seeds": ", ".join(f"{v:.2f}" for v in sub["test_hist"]) if not sub.empty else "",
                "selected_epochs": ", ".join(str(int(e)) for e in sub["selected_epoch"]) if not sub.empty else "",
            })
        holds = bool(means["marbert"] > means["arabert"]) if not np.isnan([means["marbert"], means["arabert"]]).any() else None
        for r in rows[-len(MODEL_ORDER):]:
            r["ranking_holds"] = holds
    df = pd.DataFrame(rows)
    df["within_threshold"] = df["gap"].abs() <= thr
    df["in_seed_range"] = (df["min"] <= df["published"]) & (df["published"] <= df["max"])
    ok = (df["within_threshold"] | df["in_seed_range"]) & (df["ranking_holds"] == True)  # noqa: E712
    df["verdict"] = np.where(df["n_seeds"] == 0, "not run", np.where(ok, "reproduced", "not reproduced"))
    return df


# ------------------------------------------------------------------------------ main result
def independent_table(runs: pd.DataFrame, cfg: dict, protocol: str = "independent") -> pd.DataFrame:
    rows = []
    for d in DATASET_ORDER:
        for m in MODEL_ORDER:
            sub = _subset(runs, protocol=protocol, dataset=d, model=m)
            row = {"protocol": protocol, "dataset": d, "dataset_name": cfg["datasets"][d]["display_name"],
                   "model": m, "model_name": cfg["models"][m]["display_name"], "n_seeds": len(sub),
                   "hist_metric": cfg["datasets"][d]["historical_metric"],
                   "selected_epochs": ", ".join(str(int(e)) for e in sub["selected_epoch"]) if not sub.empty else ""}
            for key in ("hist", "macro_f1", "accuracy", "f1_pn"):
                s = seed_summary(sub[f"test_{key}"]) if not sub.empty else seed_summary([])
                row[f"{key}_mean"], row[f"{key}_sd"] = s["mean"], s["sd"]
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ RQ2
def checkpoint_table(runs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for d in DATASET_ORDER:
        for m in MODEL_ORDER:
            sub = _subset(runs, protocol="independent", dataset=d, model=m)
            for metric in metric_pair(cfg, d):
                def s(col):
                    return seed_summary(sub[col]) if not sub.empty else seed_summary([])
                sel, mx, gap = s(f"test_{metric}"), s(f"max_test_{metric}"), s(f"ckpt_gap_{metric}")
                rows.append({
                    "dataset": d, "dataset_name": cfg["datasets"][d]["display_name"], "model": m,
                    "model_name": cfg["models"][m]["display_name"], "metric": metric, "n_seeds": sel["n"],
                    "selected_mean": sel["mean"], "selected_sd": sel["sd"],
                    "max_mean": mx["mean"], "max_sd": mx["sd"],
                    "gap_mean": gap["mean"], "gap_sd": gap["sd"], "gap_min": gap["min"], "gap_max": gap["max"],
                    "selected_epochs": ", ".join(str(int(e)) for e in sub["selected_epoch"]) if not sub.empty else "",
                    "argmax_epochs": ", ".join(str(int(e)) for e in sub[f"argmax_test_{metric}_epoch"]) if not sub.empty else "",
                    "same_epoch_share": float((sub["selected_epoch"] == sub[f"argmax_test_{metric}_epoch"]).mean()) if not sub.empty else np.nan,
                })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ RQ3 + RQ4
def margins_table(runs: pd.DataFrame, cfg: dict, results_root: Path | str) -> pd.DataFrame:
    bcfg = cfg["bootstrap"]
    rng = np.random.default_rng(int(bcfg["seed"]))  # ONE generator, drawn in table order (as in the team's notebook)
    rows, per_seed_rows = [], []
    for p in PROTOCOL_ORDER:
        for d in DATASET_ORDER:
            labels = labels_for(cfg, d)
            pos, neg = pos_neg_idx(cfg, d)
            a = _subset(runs, protocol=p, dataset=d, model="marbert").set_index("seed")
            b = _subset(runs, protocol=p, dataset=d, model="arabert").set_index("seed")
            seeds = sorted(set(a.index) & set(b.index)) if not a.empty and not b.empty else []
            preds = None
            if seeds:
                preds = {}
                for mdl, frame in (("marbert", a), ("arabert", b)):
                    preds[mdl] = [load_predictions(frame.loc[s, "run_dir"], "test", int(frame.loc[s, "selected_epoch"])) for s in seeds]
                ref = preds["marbert"][0]
                for lst in preds.values():
                    for df in lst:
                        if not (df["id"].equals(ref["id"]) and np.array_equal(df["gold"], ref["gold"])):
                            raise ValueError(f"test items differ between runs for {p}/{d}; cannot pair")
            idx = bootstrap_indices(rng, len(ref), int(bcfg["n_resamples"])) if seeds else None
            for kind, metric in zip(("hist", "macro"), metric_pair(cfg, d)):
                row = {"protocol": p, "dataset": d, "dataset_name": cfg["datasets"][d]["display_name"],
                       "metric_kind": kind, "metric": metric, "n_pairs": len(seeds)}
                if seeds:
                    diffs = np.array([a.loc[s, f"test_{metric}"] - b.loc[s, f"test_{metric}"] for s in seeds])
                    ss = seed_summary(diffs)
                    boot = paired_bootstrap(
                        ref["gold"].to_numpy(), [x["pred"].to_numpy() for x in preds["marbert"]],
                        [x["pred"].to_numpy() for x in preds["arabert"]], len(labels), pos, neg, metric,
                        confidence=float(bcfg["confidence"]), idx=idx)
                    for s_, pt, (lo_, hi_), pnb in zip(seeds, boot["per_seed_point"], boot["per_seed_ci"], boot["per_seed_p_not_better"]):
                        per_seed_rows.append({"protocol": p, "dataset": d, "metric": metric, "seed": str(s_), "observed_diff": pt,
                                              "ci_low": lo_, "ci_high": hi_, "p_marbert_not_better": pnb,
                                              "excludes_zero": bool(lo_ > 0 or hi_ < 0)})
                    per_seed_rows.append({"protocol": p, "dataset": d, "metric": metric, "seed": "mean", "observed_diff": boot["point"],
                                          "ci_low": boot["ci_low"], "ci_high": boot["ci_high"], "p_marbert_not_better": 1 - boot["p_gt0"],
                                          "excludes_zero": bool(boot["ci_low"] > 0 or boot["ci_high"] < 0)})
                    row.update({
                        "marbert_mean": float(a.loc[seeds, f"test_{metric}"].mean()),
                        "arabert_mean": float(b.loc[seeds, f"test_{metric}"].mean()),
                        "diff_mean": ss["mean"], "diff_sd": ss["sd"], "diff_min": ss["min"], "diff_max": ss["max"],
                        "per_seed_diffs": ", ".join(f"{x:.2f}" for x in diffs),
                        "marbert_wins": int((diffs > 0).sum()),
                        "ci_low": boot["ci_low"], "ci_high": boot["ci_high"], "p_gt0": boot["p_gt0"],
                        "boot_sd": boot["boot_sd"], "n_items": boot["n_items"],
                        "ranking": "MARBERT > AraBERT" if ss["mean"] > 0 else ("tie" if ss["mean"] == 0 else "AraBERT > MARBERT"),
                    })
                    row["ci_excludes_zero"] = bool(row["ci_low"] > 0 or row["ci_high"] < 0)
                    row["sd_over_margin"] = ss["sd"] / abs(ss["mean"]) if ss["mean"] else np.inf
                    row["ci_halfwidth_over_margin"] = ((boot["ci_high"] - boot["ci_low"]) / 2) / abs(ss["mean"]) if ss["mean"] else np.inf
                rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and "ranking" in df.columns:
        flips = {}
        for (p, d), g in df.groupby(["protocol", "dataset"], observed=True):
            r = g["ranking"].dropna().tolist()
            flips[(p, d)] = (len(r) == 2 and r[0] != r[1]) if len(r) == 2 else None
        df["ranking_changes_with_metric"] = [flips.get((p, d)) for p, d in zip(df["protocol"], df["dataset"])]
    df.attrs["per_seed"] = pd.DataFrame(per_seed_rows)
    return df


# ------------------------------------------------------------------------------ appendix
def per_class_table(runs: pd.DataFrame, cfg: dict, protocol: str = "independent") -> pd.DataFrame:
    rows = []
    for d in DATASET_ORDER:
        labels = labels_for(cfg, d)
        for m in MODEL_ORDER:
            sub = _subset(runs, protocol=protocol, dataset=d, model=m)
            if sub.empty:
                continue
            per_seed = []
            for _, r in sub.iterrows():
                pr = load_predictions(r["run_dir"], "test", int(r["selected_epoch"]))
                per_seed.append(per_class(confusion_matrix(pr["gold"], pr["pred"], len(labels))))
            for k, lab in enumerate(labels):
                row = {"protocol": protocol, "dataset": d, "dataset_name": cfg["datasets"][d]["display_name"],
                       "model": m, "model_name": cfg["models"][m]["display_name"], "label": lab,
                       "support": int(per_seed[0]["support"][k])}
                for key in ("precision", "recall", "f1"):
                    s = seed_summary([100.0 * ps[key][k] for ps in per_seed])
                    row[f"{key}_mean"], row[f"{key}_sd"] = s["mean"], s["sd"]
                rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ extra checks
def selection_rule_table(runs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Independent runs: test score (published metric) when the epoch is chosen on dev with the published metric
    (what was run) vs on dev macro-F1 (what the proposal said), plus the best test epoch of the same runs."""
    rows = []
    for d in DATASET_ORDER:
        for m in MODEL_ORDER:
            sub = _subset(runs, protocol="independent", dataset=d, model=m)
            if sub.empty:
                continue
            a, b, best = seed_summary(sub["test_hist"]), seed_summary(sub["robust_test_hist"]), seed_summary(sub["max_test_hist"])
            rows.append({"dataset": d, "dataset_name": cfg["datasets"][d]["display_name"], "model": m,
                         "model_name": cfg["models"][m]["display_name"], "metric": cfg["datasets"][d]["historical_metric"],
                         "dev_paper_metric_mean": a["mean"], "dev_paper_metric_sd": a["sd"],
                         "dev_macro_f1_mean": b["mean"], "dev_macro_f1_sd": b["sd"],
                         "max_abs_change_per_run": float((sub["robust_test_hist"] - sub["test_hist"]).abs().max()),
                         "best_on_test_mean": best["mean"],
                         "epochs_dev_paper_metric": ", ".join(str(int(e)) for e in sub["selected_epoch"]),
                         "epochs_dev_macro_f1": ", ".join(str(int(e)) for e in sub["robust_selected_epoch"])})
    return pd.DataFrame(rows)


def lead_by_selection_table(runs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """MARBERT's lead in the same independent runs under dev selection vs best-on-test selection."""
    rows = []
    for d in DATASET_ORDER:
        a = _subset(runs, protocol="independent", dataset=d, model="marbert")
        b = _subset(runs, protocol="independent", dataset=d, model="arabert")
        if a.empty or b.empty:
            continue
        dev = a["test_hist"].mean() - b["test_hist"].mean()
        best = a["max_test_hist"].mean() - b["max_test_hist"].mean()
        rows.append({"dataset": d, "dataset_name": cfg["datasets"][d]["display_name"],
                     "metric": cfg["datasets"][d]["historical_metric"], "lead_dev_selected": dev,
                     "lead_best_on_test": best, "lead_gained_by_test_selection": best - dev})
    return pd.DataFrame(rows)


def best_minus_last_table(runs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Historical runs: best test epoch minus the last epoch (epoch 25), published metric."""
    rows = []
    for d in DATASET_ORDER:
        for m in MODEL_ORDER:
            sub = _subset(runs, protocol="historical", dataset=d, model=m)
            if sub.empty:
                continue
            diff = sub["test_hist"] - sub["last_test_hist"]
            s = seed_summary(diff)
            rows.append({"dataset": d, "model": m, "metric": cfg["datasets"][d]["historical_metric"],
                         "best_minus_last_mean": s["mean"], "best_minus_last_min": s["min"], "best_minus_last_max": s["max"]})
    return pd.DataFrame(rows)


def sensitivity_table(extra_runs: pd.DataFrame, main_runs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Single-seed follow-up runs (ArSarcasm-Sent, historical protocol) against the matching main run."""
    labels = {"arabert_v01_official_prep": "AraBERT v0.1, AraBERT's official preprocessing",
              "arabert_v1_farasa": "AraBERT v1 with Farasa segmentation",
              "lr_2e-5": "Constant LR 2e-5 (10x)", "sched_linear_warmup": "Linear decay + 10% warm-up, LR 2e-6"}
    rows = []
    if extra_runs.empty:
        return pd.DataFrame(rows)
    for _, r in extra_runs.iterrows():
        variant = Path(r["run_dir"]).parts[-5]
        main = _subset(main_runs, protocol="historical", dataset=r["dataset"], model=r["model"], seed=r["seed"])
        base = float(main["test_hist"].iloc[0]) if not main.empty else np.nan
        rows.append({"variant": variant, "description": labels.get(variant, variant), "dataset": r["dataset"],
                     "model": r["model"], "seed": int(r["seed"]), "metric": r["hist_metric"],
                     "score": r["test_hist"], "best_epoch": int(r["selected_epoch"]),
                     "main_run_score": base, "change_vs_main": r["test_hist"] - base,
                     "vs_published": r["test_hist"] - cfg["datasets"][r["dataset"]]["published"][r["model"]]})
    return pd.DataFrame(rows).sort_values(["variant", "model"]).reset_index(drop=True)


def compute_table(runs: pd.DataFrame) -> pd.DataFrame:
    """GPU time per protocol and dataset (sum of logged epoch times)."""
    if runs.empty:
        return pd.DataFrame()
    g = runs.groupby(["protocol", "dataset"], observed=True)["train_seconds"].sum().div(3600).rename("gpu_hours").reset_index()
    total = pd.DataFrame([{"protocol": "all", "dataset": "all", "gpu_hours": runs["train_seconds"].sum() / 3600}])
    by_d = runs.groupby("dataset", observed=True)["train_seconds"].sum().div(3600).rename("gpu_hours").reset_index()
    by_d.insert(0, "protocol", "both")
    return pd.concat([g, by_d, total], ignore_index=True)
