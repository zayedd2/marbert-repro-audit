#!/usr/bin/env python
"""Write FAKE run folders in the team's Colab format, for the CPU smoke test only (never results).

  python scripts/make_synthetic_runs.py --out results_synthetic/team_outputs --epochs 4

The fake predictions use the real test IDs (data/splits) and random but fixed labels, so the importer, the
statistics, every table, every figure and the report numbers can be exercised end to end in seconds.
"""
from __future__ import annotations

import argparse
import json
import math
import zlib
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from marbert_audit.config import DATASET_ORDER, MODEL_ORDER, REPO_ROOT, load_config, pos_neg_idx
from marbert_audit.metrics import compute_metrics

SKILL = {"marbert": 1.25, "arabert": 0.85}


def rng_for(*parts) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def ids(name: str, split: str) -> list[str]:
    return (REPO_ROOT / "data" / "splits" / f"{name}_{split}_ids.txt").read_text().split()


def fake_run(out: Path, protocol: str, d: str, m: str, s: int, cfg: dict, epochs: int, gold: dict) -> None:
    labels = cfg["datasets"][d]["labels"]
    pos, neg = pos_neg_idx(cfg, d)
    k = len(labels)
    run = out / protocol / d / m / f"seed{s}"
    (run / "preds").mkdir(parents=True, exist_ok=True)
    splits = ["test"] if protocol == "historical" else ["dev", "test"]
    offset = rng_for("off", d, m, s, protocol).normal(0, 0.1)
    persistent = {sp: (rng_for("diff", d, sp).normal(0, 1, len(gold[sp])), rng_for("u", d, m, s, sp, protocol).random(len(gold[sp])),
                       rng_for("sh", d, m, s, sp, protocol).integers(1, k, len(gold[sp]))) for sp in splits}
    lines = []
    for e in range(1, epochs + 1):
        skill = SKILL[m] * (1 - math.exp(-e / 2.0)) + offset + rng_for("ep", d, m, s, e, protocol).normal(0, 0.06)
        rec = {"epoch": e, "train_loss": 1.1 * math.exp(-e / 6) + 0.3}
        for sp in splits:
            diff, u, shift = persistent[sp]
            p_ok = 1 / (1 + np.exp(-(1.6 * (skill + rng_for("sp", d, m, s, e, sp, protocol).normal(0, 0.05)) - diff + 0.4)))
            g = gold[sp]
            pred = np.where(u < p_ok, g, (g + shift) % k)
            probs = np.full((len(g), k), 0.0) + ((1 - (0.45 + 0.5 * p_ok)) / (k - 1))[:, None]
            probs[np.arange(len(g)), pred] = 0.45 + 0.5 * p_ok
            met = {key: v for key, v in compute_metrics(g, pred, labels, pos, neg).items() if key != "confusion"}
            frame = pd.DataFrame({"gold": g, "pred": pred})
            if protocol == "historical":
                frame.insert(0, "idx", np.arange(len(g)))
            else:
                frame.insert(0, "id", ids(d, sp))
            for i, lab in enumerate(labels):
                frame[f"p_{lab}"] = probs[:, i].round(6)
            frame.to_csv(run / "preds" / f"epoch_{e:02d}_{sp}_preds.csv", index=False)
            if protocol == "historical":
                rec.update(met)
            else:
                rec[sp] = met
        rec["seconds"] = 1.0
        lines.append(json.dumps(rec))
    (run / "metrics_per_epoch.jsonl").write_text("\n".join(lines) + "\n")
    info = {"protocol": protocol, "dataset": d, "model_key": m, "model_id": cfg["models"][m]["hf_id"],
            "model_revision": cfg["models"][m]["revision"], "seed": s, "labels": labels,
            "n_train": 0, "n_test": len(gold["test"]), "hp": {"lr": 2e-6, "epochs": epochs}, "epochs": epochs,
            "use_amp": True, "gpu": "SYNTHETIC", "torch": "none", "transformers": "none"}
    if protocol == "independent":
        info["n_dev"] = len(gold["dev"])
    (run / "run_info.json").write_text(json.dumps(info, indent=2))
    (run / "DONE").write_text("synthetic\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO_ROOT / "results_synthetic" / "team_outputs"))
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config()
    out = Path(args.out)
    for d in DATASET_ORDER + ["ajgt_ubc"]:
        k = len(cfg["datasets"][d]["labels"])
        gold = {sp: rng_for("gold", d, sp).integers(0, k, len(ids(d, sp))) for sp in (["test"] if d == "ajgt_ubc" else ["dev", "test"])}
        for protocol in (["historical"] if d == "ajgt_ubc" else ["historical", "independent"]):
            for m in MODEL_ORDER:
                for s in cfg["seeds"]:
                    fake_run(out, protocol, d, m, s, cfg, args.epochs, gold)
    print(f"synthetic team-format runs written to {out}")


if __name__ == "__main__":
    main()
