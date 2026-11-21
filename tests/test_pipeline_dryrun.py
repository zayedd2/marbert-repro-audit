"""End-to-end smoke test: fake team-format runs -> import (with verification) -> every table, figure and macro."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, check=True, capture_output=True, text=True)


def test_dry_run_builds_all_assets(tmp_path):
    team, res = tmp_path / "team_outputs", tmp_path / "results"
    run("scripts/make_synthetic_runs.py", "--out", str(team), "--epochs", "3")
    run("scripts/import_team_runs.py", "--team-outputs", str(team), "--results-root", str(res))
    run("scripts/verify_results.py", "--results-root", str(res))
    run("scripts/make_report_assets.py", "--results-root", str(res))
    gen = res / "generated"
    for name in ["numbers.tex", "tab_reproduction.tex", "tab_independent.tex", "tab_checkpoint.tex", "tab_margins.tex",
                 "tab_per_class.tex", "tab_subgroups.tex", "fig_reproduction.pdf", "fig_learning_curves.pdf",
                 "fig_checkpoint_gap.pdf", "fig_margins.pdf", "fig_confusion_matrices.pdf"]:
        assert (gen / name).exists(), name
    numbers = (gen / "numbers.tex").read_text()
    assert "SYNTHETIC" in numbers and "\\rsNRunsDone}{36}" in numbers
    assert len(list((res / "runs").glob("*/*/*/seed*/DONE"))) == 42  # 36 grid runs + 6 AJGT check runs


def test_import_rejects_tampered_scores(tmp_path):
    team = tmp_path / "team_outputs"
    run("scripts/make_synthetic_runs.py", "--out", str(team), "--epochs", "2")
    f = team / "historical" / "ajgt" / "marbert" / "seed42" / "metrics_per_epoch.jsonl"
    lines = [json.loads(x) for x in f.read_text().splitlines()]
    lines[0]["accuracy"] += 1.0  # a logged score that the saved predictions do not support
    f.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    p = subprocess.run([sys.executable, "scripts/import_team_runs.py", "--team-outputs", str(team),
                        "--results-root", str(tmp_path / "res")], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode != 0 and "recomputed accuracy" in p.stdout


def test_synthetic_runs_never_enter_results(tmp_path):
    team = tmp_path / "team_outputs"
    run("scripts/make_synthetic_runs.py", "--out", str(team), "--epochs", "1")
    res = tmp_path / "res"
    run("scripts/import_team_runs.py", "--team-outputs", str(team), "--results-root", str(res))
    cfg = (res / "runs" / "historical" / "ajgt" / "marbert" / "seed42" / "run_config.yaml").read_text()
    assert "backend: dummy" in cfg
