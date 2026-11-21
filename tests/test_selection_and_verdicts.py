import numpy as np
import pandas as pd

from marbert_audit.config import load_config
from marbert_audit.selection import select_epoch
from marbert_audit.summary import reproduction_table


def test_ties_go_to_earliest_epoch():
    ep = pd.DataFrame({"epoch": [1, 2, 3, 4], "dev_macro_f1": [50.0, 70.0, 70.0, 60.0], "test_macro_f1": [1, 2, 3, 4]})
    assert select_epoch(ep, "dev", "macro_f1") == 2
    assert select_epoch(ep, "test", "macro_f1") == 4


def _runs(values):
    rows = []
    for (d, m), vals in values.items():
        for s, v in zip([42, 43, 44], vals):
            rows.append({"protocol": "historical", "dataset": d, "model": m, "seed": s, "test_hist": v, "selected_epoch": 10})
    return pd.DataFrame(rows)


def test_reproduction_verdict_rules():
    cfg = load_config()
    runs = _runs({
        ("arsarcasm", "marbert"): [70.0, 70.5, 71.0],   # gap -0.95 -> within 2 pp
        ("arsarcasm", "arabert"): [60.0, 66.0, 67.0],   # gap -2.5 but 63.5 inside seed range [60, 67]
        ("astd", "marbert"): [74.0, 74.5, 75.0],        # gap -3.5, outside range -> not reproduced
        ("astd", "arabert"): [71.0, 72.0, 73.0],
        ("ajgt", "marbert"): [90.0, 90.5, 91.0],        # ranking reversed -> neither model counts as reproduced
        ("ajgt", "arabert"): [92.0, 92.2, 92.4],
    })
    t = reproduction_table(runs, cfg).set_index(["dataset", "model"])
    assert t.loc[("arsarcasm", "marbert"), "verdict"] == "reproduced"
    assert t.loc[("arsarcasm", "arabert"), "verdict"] == "reproduced"
    assert t.loc[("astd", "marbert"), "verdict"] == "not reproduced"
    assert t.loc[("astd", "arabert"), "verdict"] == "reproduced"
    assert t.loc[("ajgt", "arabert"), "verdict"] == "not reproduced"
    assert not t.loc[("ajgt", "marbert"), "ranking_holds"]
    assert np.isclose(t.loc[("arsarcasm", "marbert"), "gap"], 70.5 - 71.5)
