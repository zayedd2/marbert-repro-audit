"""Static figures for the ACL report (PDF for LaTeX, PNG for the README).

Colour follows the model everywhere: MARBERT = blue, AraBERT = orange (validated two-slot categorical
palette, colour-blind safe). Every figure also uses marker shape or line style, so it still reads in
grayscale print. Text stays in neutral ink, never in a series colour.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import DATASET_ORDER, MODEL_ORDER  # noqa: E402
from .metrics import METRIC_LABELS  # noqa: E402

COLORS = {"marbert": "#2a78d6", "arabert": "#eb6834"}
MARKERS = {"marbert": "o", "arabert": "s"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ_BLUE = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SINGLE_COL, DOUBLE_COL = 3.15, 6.5  # inches (ACL column / page text width)


def _style():
    plt.rcParams.update({
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 150,
    })


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _watermark(fig, synthetic: bool):
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC DRY RUN\nNOT REAL RESULTS", ha="center", va="center", fontsize=16,
                 color="#d03b3b", alpha=0.25, rotation=20, fontweight="bold", zorder=100)


def _save(fig, out_dir: Path, name: str, synthetic: bool):
    _watermark(fig, synthetic)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _name(cfg, d):
    return cfg["datasets"][d]["short_name"]


# ------------------------------------------------------------------------------ RQ1
def fig_reproduction(repro: pd.DataFrame, runs: pd.DataFrame, cfg: dict, out_dir: Path, synthetic=False):
    _style()
    ds_list = list(dict.fromkeys(repro["dataset"].astype(str)))
    fig, axes = plt.subplots(1, len(ds_list), figsize=(min(DOUBLE_COL, 1.35 * len(ds_list)), 1.7))
    for ax, d in zip(np.atleast_1d(axes), ds_list):
        g = repro[repro["dataset"] == d].set_index("model")
        for i, m in enumerate(MODEL_ORDER):
            sub = runs[(runs["protocol"].astype(str) == "historical") & (runs["dataset"].astype(str) == d) & (runs["model"].astype(str) == m)]
            vals = sub["test_hist"].to_numpy()
            if vals.size:
                jitter = np.linspace(-0.12, 0.12, vals.size) if vals.size > 1 else np.zeros(1)
                ax.scatter(i + jitter, vals, s=14, marker=MARKERS[m], color=COLORS[m], edgecolor="white", linewidth=0.5, zorder=3)
                ax.hlines(vals.mean(), i - 0.25, i + 0.25, color=COLORS[m], linewidth=1.6, zorder=2)
            if m in g.index:
                ax.scatter(i + 0.36, g.loc[m, "published"], s=22, marker="D", facecolor="none", edgecolor=INK, linewidth=0.8, zorder=4)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.margins(y=0.1)
        ax.set_xlim(-0.45, 1.6)
        ax.set_title(f"{_name(cfg, d)}", pad=3)
        ax.set_ylabel(METRIC_LABELS[cfg["datasets"][d]["historical_metric"]], labelpad=1)
        _grid(ax)
    handles = [plt.Line2D([], [], marker=MARKERS[m], color=COLORS[m], linestyle="", markersize=3.5,
                          label=f"{cfg['models'][m]['display_name']} (our seeds)") for m in MODEL_ORDER]
    handles += [plt.Line2D([], [], color=INK2, linewidth=1.6, label="mean of our seeds"),
                plt.Line2D([], [], marker="D", markerfacecolor="none", markeredgecolor=INK, linestyle="", markersize=4,
                           label="published [1]")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.1), columnspacing=1.0)
    fig.tight_layout(w_pad=0.6)
    _save(fig, out_dir, "fig_reproduction", synthetic)


# ------------------------------------------------------------------------------ RQ2
def _epoch_frames(runs: pd.DataFrame, protocol: str) -> pd.DataFrame:
    frames = []
    for _, r in runs[runs["protocol"].astype(str) == protocol].iterrows():
        ep = pd.read_csv(Path(r["run_dir"]) / "epoch_metrics.csv")
        ep["dataset"], ep["model"], ep["seed"] = str(r["dataset"]), str(r["model"]), int(r["seed"])
        frames.append(ep)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fig_learning_curves(runs: pd.DataFrame, cfg: dict, out_dir: Path, synthetic=False):
    _style()
    ep = _epoch_frames(runs, "independent")
    if ep.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 1.9))
    for ax, d in zip(axes, DATASET_ORDER):
        for m in MODEL_ORDER:
            g = ep[(ep["dataset"] == d) & (ep["model"] == m)]
            if g.empty:
                continue
            metric = cfg["datasets"][d]["historical_metric"]
            for split, ls, lw in (("test", "-", 1.6), ("dev", "--", 1.1)):
                agg = g.groupby("epoch")[f"{split}_{metric}"].agg(["mean", "min", "max"])
                ax.plot(agg.index, agg["mean"], linestyle=ls, linewidth=lw, color=COLORS[m])
                if split == "test":
                    ax.fill_between(agg.index, agg["min"], agg["max"], color=COLORS[m], alpha=0.15, linewidth=0)
            sel = runs[(runs["protocol"].astype(str) == "independent") & (runs["dataset"].astype(str) == d) & (runs["model"].astype(str) == m)]
            ax.scatter(sel["selected_epoch"], sel["test_hist"], marker=MARKERS[m], s=16, color=COLORS[m],
                       edgecolor="white", linewidth=0.6, zorder=4)
        ax.set_title(_name(cfg, d), pad=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel(METRIC_LABELS[cfg["datasets"][d]["historical_metric"]], labelpad=1)
        ax.set_xlim(0.5, cfg["training"]["epochs"] + 0.5)
        _grid(ax)
    handles = [plt.Line2D([], [], color=COLORS[m], marker=MARKERS[m], markersize=3.5, linewidth=1.6,
                          label=cfg["models"][m]["display_name"]) for m in MODEL_ORDER]
    handles += [plt.Line2D([], [], color=INK2, linewidth=1.6, label="test (mean, band = seed range)"),
                plt.Line2D([], [], color=INK2, linewidth=1.1, linestyle="--", label="dev (mean)"),
                plt.Line2D([], [], color=INK2, marker="o", linestyle="", markersize=3.5, label="dev-selected epoch, per seed")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=1.0)
    _save(fig, out_dir, "fig_learning_curves", synthetic)


def fig_checkpoint_gap(runs: pd.DataFrame, cfg: dict, out_dir: Path, synthetic=False):
    _style()
    ind = runs[runs["protocol"].astype(str) == "independent"]
    if ind.empty:
        return
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 1.7))
    for i, d in enumerate(DATASET_ORDER):
        for j, m in enumerate(MODEL_ORDER):
            vals = ind[(ind["dataset"].astype(str) == d) & (ind["model"].astype(str) == m)]["ckpt_gap_hist"].to_numpy()
            x = i + (j - 0.5) * 0.34
            if vals.size:
                ax.scatter(np.full(vals.size, x), vals, s=12, marker=MARKERS[m], color=COLORS[m], edgecolor="white", linewidth=0.5, zorder=3)
                ax.hlines(vals.mean(), x - 0.13, x + 0.13, color=COLORS[m], linewidth=1.6, zorder=2)
    ax.set_xticks(range(3), [f"{_name(cfg, d)}\n({METRIC_LABELS[cfg['datasets'][d]['historical_metric']]})" for d in DATASET_ORDER])
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_ylabel("optimism (pp)")
    _grid(ax)
    handles = [plt.Line2D([], [], color=COLORS[m], marker=MARKERS[m], linestyle="", markersize=3.5,
                          label=cfg["models"][m]["display_name"]) for m in MODEL_ORDER]
    handles.append(plt.Line2D([], [], color=INK2, linewidth=1.6, label="mean of seeds"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    _save(fig, out_dir, "fig_checkpoint_gap", synthetic)


# ------------------------------------------------------------------------------ RQ3 + RQ4
def fig_margins(margins: pd.DataFrame, cfg: dict, out_dir: Path, synthetic=False):
    _style()
    if margins.empty or "diff_mean" not in margins:
        return
    rows = [(d, k) for d in DATASET_ORDER for k in ("hist", "macro")]
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.3))
    style = {"independent": dict(face=INK, off=-0.17, label="Independent (dev-selected)"),
             "historical": dict(face="white", off=0.17, label="Historical (best-on-test)")}
    for p, st in style.items():
        for yi, (d, k) in enumerate(rows):
            r = margins[(margins["protocol"] == p) & (margins["dataset"] == d) & (margins["metric_kind"] == k)]
            if r.empty or pd.isna(r["diff_mean"].iloc[0]):
                continue
            r = r.iloc[0]
            y = yi + st["off"]
            ax.hlines(y, r["ci_low"], r["ci_high"], color=INK2, linewidth=1.0, zorder=2)
            seeds = [float(x) for x in str(r["per_seed_diffs"]).split(",") if x.strip()]
            ax.scatter(seeds, np.full(len(seeds), y), marker="|", s=26, color=INK2, linewidth=0.9, zorder=3)
            ax.scatter([r["diff_mean"]], [y], s=22, marker="o", facecolor=st["face"], edgecolor=INK, linewidth=0.8, zorder=4)
    ax.axvline(0, color=INK2, linewidth=0.7, linestyle=":")
    ax.set_yticks(range(len(rows)), [f"{_name(cfg, d)} · {METRIC_LABELS[cfg['datasets'][d]['historical_metric']] if k == 'hist' else 'Macro-F1'}" for d, k in rows])
    ax.invert_yaxis()
    ax.set_xlabel("MARBERT $-$ AraBERT (percentage points)")
    _grid(ax, "x")
    handles = [plt.Line2D([], [], marker="o", markerfacecolor=st["face"], markeredgecolor=INK, linestyle="", markersize=4, label=st["label"])
               for st in style.values()]
    handles += [plt.Line2D([], [], color=INK2, linewidth=1.0, label="95% paired bootstrap CI"),
                plt.Line2D([], [], color=INK2, marker="|", linestyle="", markersize=6, label="per-seed difference")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.13))
    fig.tight_layout()
    _save(fig, out_dir, "fig_margins", synthetic)


# ------------------------------------------------------------------------------ error analysis (M1)
def fig_confusions(cms: dict, cfg: dict, out_dir: Path, synthetic=False):
    _style()
    if not cms:
        return
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
    datasets = [d for d in DATASET_ORDER if any((d, m) in cms for m in MODEL_ORDER)]
    fig, axes = plt.subplots(len(datasets), 2, figsize=(DOUBLE_COL * 0.62, 1.75 * len(datasets)), squeeze=False)
    for i, d in enumerate(datasets):
        for j, m in enumerate(MODEL_ORDER):
            ax = axes[i, j]
            if (d, m) not in cms:
                ax.axis("off")
                continue
            v = cms[(d, m)]
            cm = v["sum"].astype(float)
            share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
            k = len(v["labels"])
            for a in range(k):
                for b in range(k):
                    ax.text(b, a, f"{100 * share[a, b]:.0f}%\n({int(cm[a, b])})", ha="center", va="center", fontsize=6,
                            color="white" if share[a, b] > 0.55 else INK)
            short = [lab[:3] for lab in v["labels"]]
            ax.set_xticks(range(k), short)
            ax.set_yticks(range(k), short)
            ax.set_xlabel("predicted", labelpad=1)
            if j == 0:
                ax.set_ylabel(f"{_name(cfg, d)}\ngold", labelpad=1)
            ax.set_title(f"{cfg['models'][m]['display_name']} (sum of {v['n_seeds']} seeds)", pad=3)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(length=0)
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    _save(fig, out_dir, "fig_confusion_matrices", synthetic)
