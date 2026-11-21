"""Render result tables as LaTeX (booktabs) and Markdown, and write the number macros used in the report.

Generated .tex tables contain only the ``tabular``; captions and labels stay in report/experiments_results.tex
so writers can edit them. Every number quoted in the report prose comes from report/generated/numbers.tex.
Required LaTeX packages: booktabs, multirow (both in the ACL template's usual preamble).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import DATASET_ORDER, MODEL_ORDER, PROTOCOL_ORDER
from .metrics import METRIC_LABELS

DS_MACRO = {"arsarcasm": "ArSarc", "astd": "Astd", "ajgt": "Ajgt", "ajgt_ubc": "AjgtUbc"}
MODEL_MACRO = {"marbert": "Marbert", "arabert": "Arabert"}
PROT_MACRO = {"historical": "Hist", "independent": "Ind"}
KIND_MACRO = {"hist": "Hm", "macro": "Mf"}


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def f(x, d: int = 2) -> str:
    return "--" if _nan(x) else f"{x:.{d}f}"


def signed(x, d: int = 2, tex: bool = True) -> str:
    if _nan(x):
        return "--"
    s = f"{abs(x):.{d}f}"
    if round(x, d) == 0:
        return s
    return ("+" if x > 0 else ("$-$" if tex else "−")) + s


def pm(mean, sd, d: int = 2, tex: bool = True) -> str:
    if _nan(mean):
        return "--"
    if _nan(sd):
        return f(mean, d)
    return f"{mean:.{d}f}{{\\scriptsize$\\,\\pm\\,${sd:.{d}f}}}" if tex else f"{mean:.{d}f} ± {sd:.{d}f}"


def spm(mean, sd, d: int = 2) -> str:
    """Signed mean with SD, e.g. +7.10 (0.85) for margins."""
    if _nan(mean):
        return "--"
    return signed(mean, d) + ("" if _nan(sd) else f"{{\\scriptsize\\,({sd:.{d}f})}}")


def ci(lo, hi, d: int = 2, tex: bool = True) -> str:
    if _nan(lo):
        return "--"
    return f"[{signed(lo, d, tex)}, {signed(hi, d, tex)}]"


def metric_label(m: str) -> str:
    return METRIC_LABELS.get(m, m)


def _short(cfg, d):
    return cfg["datasets"][d]["short_name"]


# ------------------------------------------------------------------------------ tables
def _group_header(text: str, ncols: int) -> str:
    return f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{text}}}}} \\\\"


def _ds_header(cfg, d, metric=None) -> str:
    name = cfg["datasets"][d]["display_name"]
    return f"{name}, {metric_label(metric)}" if metric else name


def tex_reproduction(df: pd.DataFrame, cfg: dict) -> str:
    lines = [r"\begin{tabular}{lrrrc}", r"\toprule",
             r"Model & Published & Ours & Gap & Rep. \\"]
    for d in list(dict.fromkeys(df["dataset"].astype(str))):
        g = df[df["dataset"].astype(str) == d]
        if g.empty:
            continue
        head = _ds_header(cfg, d, g["metric"].iloc[0]) + (", check runs" if cfg["datasets"][d].get("check_only") else "")
        lines += [r"\midrule", _group_header(head, 5)]
        for _, r in g.iterrows():
            verdict = {"reproduced": "Yes", "not reproduced": "No"}.get(r["verdict"], "--")
            lines.append(f"{r['model_name']} & {f(r['published'])} & {pm(r['mean'], r['sd'])} & {signed(r['gap'])} & {verdict} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def tex_independent(df: pd.DataFrame, cfg: dict) -> str:
    lines = [r"\begin{tabular}{lrrl}", r"\toprule",
             r"Model & Published metric & Macro-F1 & Epochs \\"]
    for d in DATASET_ORDER:
        g = df[df["dataset"] == d]
        if g.empty:
            continue
        lines += [r"\midrule", _group_header(_ds_header(cfg, d, g["hist_metric"].iloc[0]), 4)]
        for _, r in g.iterrows():
            lines.append(f"{r['model_name']} & {pm(r['hist_mean'], r['hist_sd'])} & {pm(r['macro_f1_mean'], r['macro_f1_sd'])} & "
                         f"{r['selected_epochs'] or '--'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def tex_checkpoint(ckpt: pd.DataFrame, selrule: pd.DataFrame, cfg: dict) -> str:
    """Published metric only: dev-selected (as run), dev macro-F1 rule (robustness), best-on-test, optimism."""
    lines = [r"\begin{tabular}{l rrrrc}", r"\toprule",
             r"Model & Dev-sel. & Dev macro-F1 & Best on test & Optimism & Same epoch \\"]
    for d in DATASET_ORDER:
        g = ckpt[(ckpt["dataset"] == d) & (ckpt["metric"] != "macro_f1")]
        if g.empty:
            continue
        lines += [r"\midrule", _group_header(_ds_header(cfg, d, g["metric"].iloc[0]), 6)]
        for _, r in g.iterrows():
            sr = selrule[(selrule["dataset"] == d) & (selrule["model"] == r["model"])]
            rob = pm(sr["dev_macro_f1_mean"].iloc[0], sr["dev_macro_f1_sd"].iloc[0]) if not sr.empty else "--"
            same = int(round(r["same_epoch_share"] * r["n_seeds"])) if not _nan(r["same_epoch_share"]) else None
            same_s = "--" if same is None else f"{same}/{int(r['n_seeds'])}"
            lines.append(f"{r['model_name']} & {pm(r['selected_mean'], r['selected_sd'])} & {rob} & {f(r['max_mean'])} & "
                         f"{pm(r['gap_mean'], r['gap_sd'])} & {same_s} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def tex_margins(df: pd.DataFrame, cfg: dict) -> str:
    lines = [r"\begin{tabular}{ll rrc rrc c}", r"\toprule",
             r" & & \multicolumn{3}{c}{Published metric} & \multicolumn{3}{c}{Macro-F1} & \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"Protocol & Dataset & $\Delta$ (SD) & 95\% CI & Wins & $\Delta$ (SD) & 95\% CI & Wins & Rank flip \\", r"\midrule"]
    for i, p in enumerate(PROTOCOL_ORDER):
        if i:
            lines.append(r"\midrule")
        for j, d in enumerate(DATASET_ORDER):
            g = df[(df["protocol"] == p) & (df["dataset"] == d)].set_index("metric_kind")
            if g.empty:
                continue
            first = f"\\multirow{{3}}{{*}}{{{cfg['protocols'][p]['display_name']}}}" if j == 0 else ""
            cells = []
            for kind in ("hist", "macro"):
                r = g.loc[kind] if kind in g.index else None
                if r is None or r.get("n_pairs", 0) == 0:
                    cells += ["--", "--", "--"]
                else:
                    cells += [spm(r["diff_mean"], r["diff_sd"]), ci(r["ci_low"], r["ci_high"]),
                              f"{int(r['marbert_wins'])}/{int(r['n_pairs'])}"]
            flip = g["ranking_changes_with_metric"].iloc[0] if "ranking_changes_with_metric" in g.columns else None
            flip_s = "--" if flip is None or (isinstance(flip, float) and math.isnan(flip)) else ("Yes" if flip else "No")
            lines.append(f"{first} & {_short(cfg, d)} ({metric_label(g['metric'].iloc[0])}) & " + " & ".join(cells) + f" & {flip_s} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def tex_per_class(df: pd.DataFrame, cfg: dict) -> str:
    lines = [r"\begin{tabular}{ll rrr r}", r"\toprule",
             r"Model & Class & P & R & F1 & $n$ \\"]
    for d in DATASET_ORDER:
        g = df[df["dataset"] == d] if not df.empty else df
        if g.empty:
            continue
        lines += [r"\midrule", _group_header(_ds_header(cfg, d), 6)]
        for m, gm in g.groupby("model", sort=False, observed=True):
            for j, (_, r) in enumerate(gm.iterrows()):
                first = f"\\multirow{{{len(gm)}}}{{*}}{{{r['model_name']}}}" if j == 0 else ""
                lines.append(f"{first} & {r['label'][:3]} & {pm(r['precision_mean'], r['precision_sd'])} & "
                             f"{pm(r['recall_mean'], r['recall_sd'])} & {pm(r['f1_mean'], r['f1_sd'])} & {r['support']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def tex_subgroups(ci_df: pd.DataFrame, means: pd.DataFrame, cfg: dict) -> str:
    """ArSarcasm subgroups: n, model macro-F1 means (independent), MARBERT - AraBERT with paired bootstrap CIs."""
    min_n = int(cfg.get("subgroups", {}).get("min_group_size", 100))
    lines = [r"\begin{tabular}{l r rr rr}", r"\toprule",
             r" & & \multicolumn{2}{c}{Macro-F1 (independent)} & \multicolumn{2}{c}{MARBERT $-$ AraBERT, 95\% CI} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Group & $n$ & MARBERT & AraBERT & Macro-F1 & F1-PN \\"]
    if ci_df.empty:
        return "\n".join(lines + [r"\midrule", r"\multicolumn{6}{c}{no subgroup results} \\", r"\bottomrule", r"\end{tabular}"]) + "\n"
    names = {"sarcastic": "Sarcastic", "non-sarcastic": "Non-sarcastic", "msa": "MSA", "egypt": "Egyptian",
             "gulf": "Gulf", "levant": "Levantine"}
    for col, title in (("sarcasm", "Sarcasm"), ("dialect", "Dialect")):
        g = ci_df[(ci_df["protocol"] == "independent") & (ci_df["group_column"] == col)]
        if g.empty:
            continue
        lines += [r"\midrule", _group_header(title, 6)]
        for gv in dict.fromkeys(g["group"]):
            rows = g[g["group"] == gv].set_index("metric")
            n = int(rows["n"].iloc[0])
            mm = means[(means["group_column"] == col) & (means["group"] == gv)].set_index("model")
            mvals = [pm(mm.loc[m, "macro_f1_mean"], mm.loc[m, "macro_f1_sd"]) if m in mm.index else "--" for m in MODEL_ORDER]
            cells = []
            for met in ("macro_f1", "f1_pn"):
                r = rows.loc[met]
                cells.append(f"{signed(r['mean_diff'])} {{\\scriptsize{ci(r['ci_low'], r['ci_high'])}}}")
            star = "$^{\\dagger}$" if n < min_n else ""
            lines.append(f"{names.get(gv, gv)}{star} & {n} & {mvals[0]} & {mvals[1]} & {cells[0]} & {cells[1]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def tex_sensitivity(sens: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{ll rrr}", r"\toprule", r"Variant & Model & F1-PN & vs main run & vs [1] \\", r"\midrule"]
    if sens.empty:
        return "\n".join(lines + [r"\multicolumn{5}{c}{no follow-up runs} \\", r"\bottomrule", r"\end{tabular}"]) + "\n"
    for _, r in sens.iterrows():
        model = {"marbert": "MARBERT", "arabert": "AraBERT"}.get(r["model"], r["model"])
        desc = str(r["description"]).replace("%", "\\%")
        lines.append(f"{desc} & {model} & {f(r['score'])} & {signed(r['change_vs_main'])} & {signed(r['vs_published'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------ markdown
def markdown(df: pd.DataFrame, columns: dict[str, str]) -> str:
    """columns: {source column -> header}. Floats are shown with 2 decimals."""
    head = "| " + " | ".join(columns.values()) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in columns:
            v = r.get(c)
            cells.append(f"{v:.2f}" if isinstance(v, (float, np.floating)) and not _nan(float(v)) else ("–" if _nan(v) else str(v)))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows]) + "\n"


# ------------------------------------------------------------------------------ number macros
def _macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\rs{name}}}{{{value}}}"


def numbers_tex(runs: pd.DataFrame, repro: pd.DataFrame, indep: pd.DataFrame, hist_tbl: pd.DataFrame,
                ckpt: pd.DataFrame, margins: pd.DataFrame, cfg: dict, n_expected: int, synthetic: bool,
                extras: dict | None = None) -> str:
    out = ["% AUTO-GENERATED by scripts/make_report_assets.py - do not edit by hand.",
           "% Every number in report/experiments_results.tex comes from here."]
    banner = r"\textbf{[SYNTHETIC DRY RUN -- NOT REAL RESULTS]} " if synthetic else ""
    out.append(_macro("SyntheticBanner", banner))
    out.append(_macro("NRunsDone", str(len(runs))))
    out.append(_macro("NRunsExpected", str(n_expected)))
    out.append(_macro("Seeds", ", ".join(map(str, cfg["seeds"]))))
    out.append(_macro("NSeeds", str(len(cfg["seeds"]))))
    out.append(_macro("BootB", f"{int(cfg['bootstrap']['n_resamples']):,}".replace(",", "{,}")))
    out.append(_macro("DevPct", f"{100 * cfg['dev_fraction']:.0f}"))
    out.append(_macro("GapThreshold", f(cfg["reproduction"]["gap_threshold_pp"], 0)))
    gpus = sorted({g for g in runs.get("gpu", pd.Series(dtype=str)).dropna().unique()}) if not runs.empty else []
    out.append(_macro("GPU", ", ".join(gpus) if gpus else "--"))
    secs = runs["train_seconds"].dropna().sum() if not runs.empty and "train_seconds" in runs else float("nan")
    out.append(_macro("GPUHours", f(secs / 3600.0, 1) if secs and not _nan(secs) and secs > 0 else "--"))
    commits = sorted({c[:7] for c in runs.get("git_commit", pd.Series(dtype=str)).dropna().unique()}) if not runs.empty else []
    out.append(_macro("GitCommit", ", ".join(commits) if commits else "--"))

    # split sizes (independent protocol)
    for d in DATASET_ORDER:
        sub = runs[(runs["dataset"].astype(str) == d)] if not runs.empty else runs
        ind = sub[sub["protocol"].astype(str) == "independent"] if not sub.empty else sub
        hist = sub[sub["protocol"].astype(str) == "historical"] if not sub.empty else sub
        out.append(_macro(f"NTrainInd{DS_MACRO[d]}", str(int(ind["n_train"].iloc[0])) if not ind.empty else "--"))
        out.append(_macro(f"NDev{DS_MACRO[d]}", str(int(ind["n_dev"].iloc[0])) if not ind.empty and not _nan(ind["n_dev"].iloc[0]) else "--"))
        out.append(_macro(f"NTrainHist{DS_MACRO[d]}", str(int(hist["n_train"].iloc[0])) if not hist.empty else "--"))
        out.append(_macro(f"NTest{DS_MACRO[d]}", str(int(sub["n_test"].iloc[0])) if not sub.empty else "--"))

    # RQ1 reproduction
    for _, r in repro.iterrows():
        k = f"{DS_MACRO[r['dataset']]}{MODEL_MACRO[r['model']]}"
        out.append(_macro(f"HistScore{k}", pm(r["mean"], r["sd"])))
        out.append(_macro(f"HistMean{k}", f(r["mean"])))
        out.append(_macro(f"Pub{k}", f(r["published"])))
        out.append(_macro(f"Gap{k}", signed(r["gap"])))
        out.append(_macro(f"Verdict{k}", r["verdict"]))
    n_rep = int((repro["verdict"] == "reproduced").sum())
    out.append(_macro("NReproduced", str(n_rep)))
    out.append(_macro("NTargets", str(len(repro))))
    out.append(_macro("MaxAbsGap", f(repro["gap"].abs().max())))
    rank = repro.drop_duplicates("dataset").set_index("dataset")["ranking_holds"]
    for d in DATASET_ORDER:
        v = rank.get(d)
        out.append(_macro(f"RankHist{DS_MACRO[d]}", "--" if v is None or (isinstance(v, float) and math.isnan(v)) else ("holds" if v else "does not hold")))
    out.append(_macro("NRankHoldHist", str(int((rank == True).sum()))))  # noqa: E712

    # main result (independent)
    for _, r in indep.iterrows():
        k = f"{DS_MACRO[r['dataset']]}{MODEL_MACRO[r['model']]}"
        out.append(_macro(f"IndHm{k}", pm(r["hist_mean"], r["hist_sd"])))
        out.append(_macro(f"IndMf{k}", pm(r["macro_f1_mean"], r["macro_f1_sd"])))
    for _, r in hist_tbl.iterrows():
        k = f"{DS_MACRO[r['dataset']]}{MODEL_MACRO[r['model']]}"
        out.append(_macro(f"HistMf{k}", pm(r["macro_f1_mean"], r["macro_f1_sd"])))

    # RQ2 checkpoint diagnostic
    for _, r in ckpt.iterrows():
        kind = "Mf" if r["metric"] == "macro_f1" else "Hm"
        k = f"{DS_MACRO[r['dataset']]}{MODEL_MACRO[r['model']]}{kind}"
        out.append(_macro(f"CkptGap{k}", f(r["gap_mean"])))
        out.append(_macro(f"CkptGapMax{k}", f(r["gap_max"])))
    ind_runs = runs[runs["protocol"].astype(str) == "independent"] if not runs.empty else runs
    for kind in ("Mf", "Hm"):
        if ind_runs.empty:
            vals = pd.Series(dtype=float)
        elif kind == "Mf":
            vals = ind_runs["ckpt_gap_macro_f1"]
        else:
            vals = pd.Series([r[f"ckpt_gap_{r['hist_metric']}"] for _, r in ind_runs.iterrows()], dtype=float)
        out.append(_macro(f"CkptGapAll{kind}", f(vals.mean())))
        out.append(_macro(f"CkptGapAllMax{kind}", f(vals.max())))
    if not ind_runs.empty:
        same = sum(int(r["selected_epoch"] == r[f"argmax_test_{r['hist_metric']}_epoch"]) for _, r in ind_runs.iterrows())
        out.append(_macro("CkptSameEpoch", f"{int(same)} of {len(ind_runs)}"))
    else:
        out.append(_macro("CkptSameEpoch", "--"))

    # RQ3/RQ4 margins
    for _, r in margins.iterrows():
        k = f"{PROT_MACRO[r['protocol']]}{DS_MACRO[r['dataset']]}{KIND_MACRO[r['metric_kind']]}"
        has = r.get("n_pairs", 0) and not _nan(r.get("diff_mean"))
        out.append(_macro(f"Margin{k}", signed(r["diff_mean"]) if has else "--"))
        out.append(_macro(f"MarginSD{k}", f(r["diff_sd"]) if has else "--"))
        out.append(_macro(f"MarginCI{k}", ci(r["ci_low"], r["ci_high"]) if has else "--"))
        out.append(_macro(f"MarginWins{k}", f"{int(r['marbert_wins'])}/{int(r['n_pairs'])}" if has else "--"))
        out.append(_macro(f"MarginCIZero{k}", ("excludes" if r["ci_excludes_zero"] else "includes") if has else "--"))
    for p in PROTOCOL_ORDER:
        g = margins[margins["protocol"] == p] if not margins.empty else margins
        if g.empty or "ranking_changes_with_metric" not in g:
            out.append(_macro(f"NRankFlip{PROT_MACRO[p]}", "--"))
            for d in DATASET_ORDER:
                out.append(_macro(f"RankFlip{PROT_MACRO[p]}{DS_MACRO[d]}", "--"))
            continue
        per_d = g.drop_duplicates("dataset").set_index("dataset")["ranking_changes_with_metric"]
        out.append(_macro(f"NRankFlip{PROT_MACRO[p]}", str(int((per_d == True).sum()))))  # noqa: E712
        for d in DATASET_ORDER:
            v = per_d.get(d)
            out.append(_macro(f"RankFlip{PROT_MACRO[p]}{DS_MACRO[d]}", "--" if v is None or (isinstance(v, float) and math.isnan(v)) else ("changes" if v else "does not change")))
        excl = g.dropna(subset=["diff_mean"]) if "diff_mean" in g else g.iloc[0:0]
        out.append(_macro(f"NCIExclZero{PROT_MACRO[p]}", f"{int(excl['ci_excludes_zero'].sum())} of {len(excl)}" if len(excl) else "--"))
        sd_ratio = excl["sd_over_margin"].replace([np.inf], np.nan).max() if len(excl) else float("nan")
        out.append(_macro(f"MaxSDRatio{PROT_MACRO[p]}", f(sd_ratio)))
    out += _extra_macros(runs, ckpt, extras or {})
    return "\n".join(out) + "\n"


def _extra_macros(runs: pd.DataFrame, ckpt: pd.DataFrame, ex: dict) -> list[str]:
    out = []
    hm = ckpt[ckpt["metric"] != "macro_f1"] if not ckpt.empty else ckpt
    if not hm.empty:
        out.append(_macro("OptimismMin", f(hm["gap_mean"].min())))
        out.append(_macro("OptimismMax", f(hm["gap_mean"].max())))
    sel = ex.get("selrule", pd.DataFrame())
    if not sel.empty:
        out.append(_macro("SelRuleMaxMeanChange", f((sel["dev_macro_f1_mean"] - sel["dev_paper_metric_mean"]).abs().max())))
        out.append(_macro("SelRuleMaxRunChange", f(sel["max_abs_change_per_run"].max())))
    for _, r in ex.get("lead", pd.DataFrame()).iterrows():
        k = DS_MACRO[r["dataset"]]
        out.append(_macro(f"LeadDev{k}", f(r["lead_dev_selected"])))
        out.append(_macro(f"LeadBest{k}", f(r["lead_best_on_test"])))
        out.append(_macro(f"LeadGain{k}", signed(r["lead_gained_by_test_selection"])))
    bml = ex.get("bml", pd.DataFrame())
    if not bml.empty:
        out.append(_macro("BestMinusLastMin", f(bml["best_minus_last_min"].min())))
        out.append(_macro("BestMinusLastMax", f(bml["best_minus_last_max"].max())))
    comp = ex.get("compute", pd.DataFrame())
    for _, r in comp[comp["protocol"] == "both"].iterrows() if not comp.empty else []:
        out.append(_macro(f"GPUHours{DS_MACRO[str(r['dataset'])]}", f(r["gpu_hours"], 1)))
    for _, r in ex.get("ubc", pd.DataFrame()).iterrows():
        k = f"AjgtUbc{MODEL_MACRO[r['model']]}"
        out.append(_macro(f"HistScore{k}", pm(r["mean"], r["sd"])))
        out.append(_macro(f"Gap{k}", signed(r["gap"])))
        out.append(_macro(f"Verdict{k}", r["verdict"]))
    ps = ex.get("per_seed", pd.DataFrame())
    if not ps.empty:
        seeds = ps[ps["seed"] != "mean"]
        out.append(_macro("NPerSeedCIExclZero", f"{int(seeds['excludes_zero'].sum())} of {len(seeds)}"))
    for _, r in ex.get("subgroup_ci", pd.DataFrame()).iterrows():
        g = {"sarcastic": "Sarc", "non-sarcastic": "NonSarc", "msa": "Msa", "egypt": "Egy", "gulf": "Gulf", "levant": "Lev"}.get(r["group"])
        if g is None:
            continue
        k = f"Sub{PROT_MACRO[r['protocol']]}{g}{'Mf' if r['metric'] == 'macro_f1' else 'Fpn'}"
        out.append(_macro(k, signed(r["mean_diff"])))
        out.append(_macro(k + "CI", ci(r["ci_low"], r["ci_high"])))
        out.append(_macro(f"SubN{g}", str(int(r["n"]))))
    for _, r in ex.get("per_cls", pd.DataFrame()).iterrows():
        k = f"Pc{DS_MACRO[r['dataset']]}{MODEL_MACRO[r['model']]}{str(r['label']).capitalize()}"
        out.append(_macro(k, f(r["f1_mean"])))
    return list(dict.fromkeys(out))
