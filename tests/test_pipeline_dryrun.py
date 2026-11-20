"""End-to-end smoke test: fake data + fake model -> every table, figure and report macro."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True, capture_output=True, text=True)


def test_dry_run_builds_all_assets(tmp_path):
    data, res = tmp_path / "data", tmp_path / "results"
    run("scripts/make_synthetic_data.py", "--data-root", str(data), "--scale", "0.05")
    run("scripts/run_experiments.py", "--backend", "dummy", "--data-root", str(data), "--results-root", str(res), "--epochs", "4")
    run("scripts/verify_results.py", "--results-root", str(res))
    run("scripts/make_report_assets.py", "--results-root", str(res), "--data-root", str(data), "--skip-bootstrap")
    gen = res / "generated"
    for name in ["numbers.tex", "tab_reproduction.tex", "tab_independent.tex", "tab_checkpoint.tex", "tab_margins.tex",
                 "tab_per_class.tex", "tab_subgroups.tex", "fig_reproduction.pdf", "fig_learning_curves.pdf",
                 "fig_checkpoint_gap.pdf", "fig_margins.pdf", "fig_confusion_matrices.pdf"]:
        assert (gen / name).exists(), name
    numbers = (gen / "numbers.tex").read_text()
    assert "SYNTHETIC" in numbers and "\\rsNRunsDone}{36}" in numbers
    assert len(list((res / "runs").glob("*/*/*/seed*/DONE"))) == 36


def test_dummy_backend_refuses_real_results_folder():
    p = subprocess.run([sys.executable, "scripts/run_experiments.py", "--backend", "dummy"], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode != 0 and "SYNTHETIC" in (p.stderr + p.stdout)
