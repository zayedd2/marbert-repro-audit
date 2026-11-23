#!/usr/bin/env python
"""Rebuild EVERY table, figure and report number from the saved predictions in results/runs. No GPU needed.

  python scripts/make_report_assets.py
  python scripts/make_report_assets.py --results-root results_synthetic      # CPU smoke test

Writes
  <results-root>/tables/*.csv|.tex         result tables (CSV = full precision; TeX = report-ready)
  <results-root>/figures/*.pdf|.png        figures
  <results-root>/SUMMARY.md                overview shown on GitHub
  <results-root>/tables/crosscheck_vs_team.csv   regenerated numbers vs the team's own result files
  <report-dir>/numbers.tex, tab_*.tex, fig_*.pdf   what report/experiments_results.tex reads
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from marbert_audit import errors, figures, latex, summary
from marbert_audit.config import REPO_ROOT, load_config
from marbert_audit.selection import collect_runs, expected_runs
from marbert_audit.stats import seed_summary

TEAM = REPO_ROOT / "results" / "team_packages"


def is_done(res: Path, p, d, m, s) -> bool:
    return (res / "runs" / p / d / m / f"seed{s}" / "DONE").exists()


def crosscheck(res_tables: dict, team_dir: Path) -> pd.DataFrame:
    """Compare regenerated numbers with the CSV files the team produced in Colab (Members 2 and 3)."""
    rows = []

    def add(name, mine: pd.Series, theirs: pd.Series, decimals: int = 3):
        """Compare at the precision the team's file was written with (2 or 3 decimals)."""
        diff = (mine.astype(float).round(decimals) - theirs.astype(float).round(decimals)).abs()
        rows.append({"check": name, "decimals": decimals, "values_compared": int(diff.notna().sum()),
                     "max_abs_difference": round(float(diff.max()), 6), "match": bool(diff.max() <= 10 ** -decimals + 1e-9)})

    pkg = team_dir / "member3_package"
    if not pkg.exists():
        return pd.DataFrame(rows)
    rep = res_tables["reproduction"].merge(pd.read_csv(pkg / "reproduction_summary.csv"), on=["dataset", "model"])
    add("reproduction: mean over seeds", rep["mean"], rep["ours_mean"], 2)
    add("reproduction: SD over seeds", rep["sd"], rep["ours_sd"], 2)
    ind = res_tables["independent"].merge(pd.read_csv(pkg / "independent_summary.csv"), on=["dataset", "model"])
    add("independent: dev-selected mean", ind["hist_mean"], ind["dev_selected_mean"], 2)
    add("independent: best-on-test mean (same runs)", ind["best_on_test_mean"], ind["same_runs_best_test_mean"], 2)
    diag = pd.read_csv(pkg / "checkpoint_diagnostic.csv")
    diag = diag[diag["protocol"] == "independent"]
    runs = res_tables["runs"]
    r_ind = runs[runs["protocol"].astype(str) == "independent"].copy()
    r_ind["dataset"], r_ind["model"] = r_ind["dataset"].astype(str), r_ind["model"].astype(str)
    m = r_ind.merge(diag, on=["dataset", "model", "seed"])
    add("checkpoint: dev-selected epoch", m["selected_epoch"], m["dev_selected_epoch"])
    add("checkpoint: selection optimism per run", m["ckpt_gap_hist"], m["selection_optimism"])
    boot = res_tables["per_seed_boot"].merge(pd.read_csv(pkg / "bootstrap_ci.csv", dtype={"seed": str}),
                                             on=["protocol", "dataset", "metric", "seed"], suffixes=("", "_team"))
    for c in ("observed_diff", "ci_low", "ci_high"):
        add(f"paired bootstrap: {c} (per seed and mean)", boot[c], boot[f"{c}_team"])
    sub_f = team_dir / "member3_results" / "subgroup_bootstrap_ci.csv"
    if sub_f.exists() and not res_tables["subgroup_ci"].empty:
        t = pd.read_csv(sub_f)
        t["group"] = t["group"].str.split("=").str[1].replace({"True": "sarcastic", "False": "non-sarcastic"})
        s = res_tables["subgroup_ci"].merge(t, on=["protocol", "group", "metric"], suffixes=("", "_team"))
        for c in ("mean_diff", "ci_low", "ci_high"):
            add(f"subgroup bootstrap: {c}", s[c], s[f"{c}_team"], 2)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default=str(REPO_ROOT / "results"))
    ap.add_argument("--report-dir", default=None, help="default: report/generated (or <results-root>/generated for other roots)")
    args = ap.parse_args()

    cfg = load_config()
    res = Path(args.results_root)
    default_root = res.resolve() == (REPO_ROOT / "results").resolve()
    report_dir = Path(args.report_dir) if args.report_dir else (REPO_ROOT / "report" / "generated" if default_root else res / "generated")
    tables, figs = res / "tables", res / "figures"
    for p in (tables, figs, report_dir):
        p.mkdir(parents=True, exist_ok=True)

    exp = expected_runs(cfg)
    missing = [r for r in exp if not is_done(res, *r)]
    runs = collect_runs(res, cfg)
    print(f"{len(runs)} completed runs found, {len(missing)} of {len(exp)} expected runs missing.")
    for r in missing:
        print("  missing:", "/".join(map(str, r)))
    if runs.empty:
        print("Nothing to summarise yet.")
        return
    synthetic = bool((runs["backend"] == "dummy").any())
    if synthetic and default_root:
        raise SystemExit("Synthetic runs found inside results/ - remove them; they must never mix with real results.")
    try:  # version of this repository's analysis code that built the files
        import subprocess
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        head = ""
    runs["git_commit"] = head or None
    ubc_runs = collect_runs(res, cfg, datasets=["ajgt_ubc"])
    extra_runs = collect_runs(res, cfg, subdir="runs_extra", datasets=["arsarcasm"])

    # ---------------------------------------------------------------- tables
    repro = summary.reproduction_table(runs, cfg)
    ubc = summary.reproduction_table(ubc_runs, cfg, datasets=["ajgt_ubc"]) if not ubc_runs.empty else pd.DataFrame()
    repro_all = pd.concat([repro, ubc], ignore_index=True)
    indep = summary.independent_table(runs, cfg, "independent")
    hist_tbl = summary.independent_table(runs, cfg, "historical")
    ckpt = summary.checkpoint_table(runs, cfg)
    selrule = summary.selection_rule_table(runs, cfg)
    lead = summary.lead_by_selection_table(runs, cfg)
    bml = summary.best_minus_last_table(runs, cfg)
    compute = summary.compute_table(runs)
    print("bootstrapping margins ...")
    margins = summary.margins_table(runs, cfg, res)
    per_seed_boot = margins.attrs.get("per_seed", pd.DataFrame())
    per_cls = summary.per_class_table(runs, cfg)
    cms = errors.confusion_matrices(runs, cfg)
    subgroups = errors.subgroup_table(runs, cfg)
    print("bootstrapping subgroups ...")
    subgroup_ci = errors.subgroup_ci_table(runs, cfg)
    sens = summary.sensitivity_table(extra_runs, runs, cfg)
    for df in (indep,):  # best-on-test mean of the same independent runs, for the cross-check
        df["best_on_test_mean"] = [seed_summary(runs[(runs["protocol"].astype(str) == "independent") & (runs["dataset"].astype(str) == d)
                                                     & (runs["model"].astype(str) == m)]["max_test_hist"])["mean"]
                                   for d, m in zip(df["dataset"], df["model"])]

    out_tables = {"all_runs": runs, "reproduction": repro_all, "independent": indep, "historical_all_metrics": hist_tbl,
                  "checkpoint_diagnostic": ckpt, "selection_rule_check": selrule, "lead_by_selection_rule": lead,
                  "best_minus_last_epoch": bml, "compute": compute, "margins": margins, "bootstrap_ci": per_seed_boot,
                  "per_class": per_cls, "subgroups": subgroups, "subgroup_bootstrap_ci": subgroup_ci,
                  "sensitivity_runs": sens, "confusion_matrices": errors.confusion_long(cms, cfg)}
    for name, df in out_tables.items():
        if "run_dir" in df.columns:  # write paths relative to the repository, not the machine they were built on
            df = df.copy()
            df["run_dir"] = [os.path.relpath(Path(p).resolve(), REPO_ROOT.resolve()) for p in df["run_dir"]]
        df.to_csv(tables / f"{name}.csv", index=False)
    cc = crosscheck({"reproduction": repro, "independent": indep, "runs": runs, "per_seed_boot": per_seed_boot,
                     "subgroup_ci": subgroup_ci}, TEAM) if default_root else pd.DataFrame()
    if not cc.empty:
        cc.to_csv(tables / "crosscheck_vs_team.csv", index=False)
        print("\nCross-check against the team's own result files:")
        print(cc.to_string(index=False))

    tex = {
        "tab_reproduction": latex.tex_reproduction(repro_all, cfg),
        "tab_independent": latex.tex_independent(indep, cfg),
        "tab_checkpoint": latex.tex_checkpoint(ckpt, selrule, cfg),
        "tab_margins": latex.tex_margins(margins, cfg),
        "tab_per_class": latex.tex_per_class(per_cls, cfg),
        "tab_subgroups": latex.tex_subgroups(subgroup_ci, subgroups, cfg),
        "tab_sensitivity": latex.tex_sensitivity(sens),
    }
    extras = {"selrule": selrule, "lead": lead, "bml": bml, "compute": compute, "ubc": ubc,
              "per_seed": per_seed_boot, "subgroup_ci": subgroup_ci, "per_cls": per_cls}
    numbers = latex.numbers_tex(runs, repro, indep, hist_tbl, ckpt, margins, cfg, len(exp), synthetic, extras)
    for name, body in {**tex, "numbers": numbers}.items():
        header = "% AUTO-GENERATED by scripts/make_report_assets.py - edit the code, not this file.\n"
        (tables / f"{name}.tex").write_text(header + body, encoding="utf-8")
        (report_dir / f"{name}.tex").write_text(header + body, encoding="utf-8")

    # ---------------------------------------------------------------- figures
    figures.fig_reproduction(repro_all, pd.concat([runs, ubc_runs], ignore_index=True) if not ubc_runs.empty else runs, cfg, figs, synthetic)
    figures.fig_learning_curves(runs, cfg, figs, synthetic)
    figures.fig_checkpoint_gap(runs, cfg, figs, synthetic)
    figures.fig_margins(margins, cfg, figs, synthetic)
    figures.fig_confusions(cms, cfg, figs, synthetic)
    for pdf in figs.glob("*.pdf"):
        shutil.copy(pdf, report_dir / pdf.name)

    # ---------------------------------------------------------------- markdown summary
    md = ["# Results summary", ""]
    if synthetic:
        md += ["> **SYNTHETIC DRY RUN - NOT REAL RESULTS.**", ""]
    md += [f"Completed runs: **{len(runs)} / {len(exp)}** (plus {len(ubc_runs)} AJGT check runs on the authors' split and "
           f"{len(extra_runs)} single-seed follow-up runs). All numbers are regenerated from the saved predictions by "
           "`scripts/make_report_assets.py`; scores are on a 0-100 scale, mean ± SD over seeds 42-44.", "",
           "## RQ1 - Reproduction (historical protocol: best epoch on test, as in [1])", "",
           latex.markdown(repro_all, {"dataset_name": "Dataset", "metric": "Metric", "model_name": "Model", "published": "[1]",
                                      "mean": "Ours", "sd": "SD", "min": "Min", "max": "Max", "gap": "Gap (pp)", "verdict": "Verdict"}),
           "## Main result (independent protocol: epoch chosen on dev with the published metric)", "",
           latex.markdown(indep, {"dataset_name": "Dataset", "model_name": "Model", "hist_metric": "Metric", "hist_mean": "Mean",
                                  "hist_sd": "SD", "macro_f1_mean": "Macro-F1", "macro_f1_sd": "SD", "selected_epochs": "Dev-selected epochs"}),
           "## RQ2 - Checkpoint selection (independent runs)", "",
           latex.markdown(ckpt[ckpt["metric"] != "macro_f1"], {"dataset_name": "Dataset", "model_name": "Model", "metric": "Metric",
                                                               "selected_mean": "Dev-selected", "max_mean": "Best on test",
                                                               "gap_mean": "Optimism", "gap_max": "Max optimism", "selected_epochs": "Dev epochs",
                                                               "argmax_epochs": "Best test epochs"}),
           latex.markdown(selrule, {"dataset_name": "Dataset", "model_name": "Model", "dev_paper_metric_mean": "Dev published metric (as run)",
                                    "dev_macro_f1_mean": "Dev macro-F1 (proposal)", "max_abs_change_per_run": "Max change in one run"}),
           latex.markdown(lead, {"dataset_name": "Dataset", "lead_dev_selected": "Lead, dev-selected", "lead_best_on_test": "Lead, best-on-test",
                                 "lead_gained_by_test_selection": "Gained by test selection"}),
           "## RQ3/RQ4 - MARBERT minus AraBERT (95% paired bootstrap, B = 1,000)", "",
           latex.markdown(margins, {"protocol": "Protocol", "dataset_name": "Dataset", "metric": "Metric", "diff_mean": "Δ mean",
                                    "diff_sd": "Δ SD (seeds)", "ci_low": "CI low", "ci_high": "CI high", "marbert_wins": "MARBERT wins",
                                    "ranking_changes_with_metric": "Ranking changes with metric"}),
           "## ArSarcasm subgroups (MARBERT minus AraBERT, 95% paired bootstrap, B = 2,000)", "",
           latex.markdown(subgroup_ci, {"protocol": "Protocol", "group": "Group", "n": "n", "metric": "Metric", "mean_diff": "Δ mean",
                                        "ci_low": "CI low", "ci_high": "CI high"}),
           "## Follow-up runs (seed 42, ArSarcasm-Sent, historical protocol)", "",
           latex.markdown(sens, {"description": "Variant", "model": "Model", "score": "F1-PN", "change_vs_main": "vs main run",
                                 "vs_published": "vs [1]"}),
           "## Compute", "", latex.markdown(compute, {"protocol": "Protocol", "dataset": "Dataset", "gpu_hours": "GPU-hours"})]
    if not cc.empty:
        md += ["## Cross-check against the team's own result files", "",
               latex.markdown(cc, {"check": "Check", "values_compared": "Values", "max_abs_difference": "Max |difference|", "match": "Match"})]
    md += ["## Figures", ""]
    for name in ("fig_reproduction", "fig_learning_curves", "fig_checkpoint_gap", "fig_margins", "fig_confusion_matrices"):
        if (figs / f"{name}.png").exists():
            md.append(f"![{name}](figures/{name}.png)\n")
    (res / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    (res / "assets_status.json").write_text(json.dumps({"runs_done": len(runs), "runs_expected": len(exp),
                                                         "missing": ["/".join(map(str, r)) for r in missing],
                                                         "synthetic": synthetic}, indent=2) + "\n", encoding="utf-8")
    print(f"\nTables  -> {tables}\nFigures -> {figs}\nReport  -> {report_dir}\nSummary -> {res / 'SUMMARY.md'}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    np.set_printoptions(suppress=True)
    main()
