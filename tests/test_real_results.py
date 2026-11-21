"""Regression test on the committed real runs: the regenerated numbers must equal the team's reported ones."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marbert_audit import summary
from marbert_audit.config import load_config
from marbert_audit.selection import collect_runs

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "results" / "runs" / "historical").exists(), reason="real runs not present")


@pytest.fixture(scope="module")
def runs():
    return collect_runs(ROOT / "results", load_config())


def test_all_36_runs_present(runs):
    assert len(runs) == 36


def test_reproduction_matches_member2(runs):
    t = summary.reproduction_table(runs, load_config()).set_index(["dataset", "model"])["mean"].round(2)
    expected = {("arsarcasm", "marbert"): 72.42, ("arsarcasm", "arabert"): 67.86, ("astd", "marbert"): 80.90,
                ("astd", "arabert"): 75.83, ("ajgt", "marbert"): 94.35, ("ajgt", "arabert"): 88.89}
    for k, v in expected.items():
        assert t[k] == pytest.approx(v, abs=0.005), k


def test_dev_selected_matches_member3(runs):
    t = summary.independent_table(runs, load_config()).set_index(["dataset", "model"])["hist_mean"].round(2)
    expected = {("arsarcasm", "marbert"): 70.80, ("arsarcasm", "arabert"): 66.75, ("astd", "marbert"): 79.85,
                ("astd", "arabert"): 75.55, ("ajgt", "marbert"): 93.52, ("ajgt", "arabert"): 88.06}
    for k, v in expected.items():
        assert t[k] == pytest.approx(v, abs=0.005), k


def test_crosscheck_file_all_match():
    f = ROOT / "results" / "tables" / "crosscheck_vs_team.csv"
    if not f.exists():
        pytest.skip("run scripts/make_report_assets.py first")
    assert pd.read_csv(f)["match"].all()
