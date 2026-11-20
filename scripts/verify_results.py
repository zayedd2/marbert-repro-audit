#!/usr/bin/env python
"""Integrity check: recompute every epoch's scores from the saved predictions and compare with
epoch_metrics.csv, then write a SHA-256 manifest of all run files.

  python scripts/verify_results.py
Exit code 1 if any run disagrees, is incomplete, or used a different test file from the others.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
import yaml

from marbert_audit.config import REPO_ROOT, labels_for, load_config, pos_neg_idx
from marbert_audit.data import sha256_file
from marbert_audit.metrics import compute_metrics
from marbert_audit.selection import iter_run_dirs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default=str(REPO_ROOT / "results"))
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()
    cfg = load_config()
    res = Path(args.results_root)
    problems, n_runs, test_hashes = [], 0, {}
    manifest_lines = []
    for run in iter_run_dirs(res):
        n_runs += 1
        rc = yaml.safe_load((run / "run_config.yaml").read_text(encoding="utf-8"))
        d = rc["dataset"]
        labels, (pos, neg) = labels_for(cfg, d), pos_neg_idx(cfg, d)
        test_hashes.setdefault(d, set()).add(rc["data_sha256"]["test"])
        ep = pd.read_csv(run / "epoch_metrics.csv").set_index("epoch")
        if len(ep) != int(rc["training"]["epochs"]):
            problems.append(f"{run}: {len(ep)} epochs, expected {rc['training']['epochs']}")
        for split in ("dev", "test"):
            f = run / f"preds_{split}.csv.gz"
            if not f.exists():
                continue
            preds = pd.read_csv(f, dtype={"id": str})
            for e, g in preds.groupby("epoch"):
                m = compute_metrics(g["gold"], g["pred"], labels, pos, neg)
                for key in ("accuracy", "macro_f1", "f1_pn"):
                    if abs(m[key] - ep.loc[e, f"{split}_{key}"]) > args.tol:
                        problems.append(f"{run}: epoch {e} {split}_{key} {m[key]:.4f} != {ep.loc[e, f'{split}_{key}']:.4f}")
        for f in sorted(run.iterdir()):
            if f.is_file() and f.name != "train.log":
                manifest_lines.append(f"{sha256_file(f)}  {f.relative_to(res)}")
    for d, hashes in test_hashes.items():
        if len(hashes) > 1:
            problems.append(f"{d}: runs used {len(hashes)} different test files")
    (res / "MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"checked {n_runs} runs; manifest -> {res / 'MANIFEST.sha256'}")
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("OK: all saved predictions reproduce the logged scores.")


if __name__ == "__main__":
    main()
