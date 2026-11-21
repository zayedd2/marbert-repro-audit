#!/usr/bin/env python
"""Integrity check of results/runs: recompute every epoch's scores from the saved predictions, compare them with
epoch_metrics.csv, check that all runs of a dataset used the same test items, and write a SHA-256 manifest.

  python scripts/verify_results.py
Exit code 1 if anything disagrees.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
import yaml

from marbert_audit.config import REPO_ROOT, labels_for, load_config, pos_neg_idx
from marbert_audit.importer import sha256_file
from marbert_audit.metrics import compute_metrics
from marbert_audit.selection import iter_run_dirs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default=str(REPO_ROOT / "results"))
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()
    cfg = load_config()
    res = Path(args.results_root)
    problems, n_runs, n_files, test_ids, manifest = [], 0, 0, {}, []
    for sub in ("runs", "runs_extra"):
        for run in iter_run_dirs(res, sub):
            n_runs += 1
            rc = yaml.safe_load((run / "run_config.yaml").read_text(encoding="utf-8"))
            d = rc["dataset"]
            labels, (pos, neg) = labels_for(cfg, d), pos_neg_idx(cfg, d)
            ep = pd.read_csv(run / "epoch_metrics.csv").set_index("epoch")
            for split in ("dev", "test"):
                f = run / f"preds_{split}.csv.gz"
                if not f.exists():
                    continue
                preds = pd.read_csv(f, dtype={"id": str})
                if split == "test":
                    first = tuple(preds[preds["epoch"] == preds["epoch"].min()]["id"])
                    test_ids.setdefault(d, set()).add(first)
                for e, g in preds.groupby("epoch"):
                    n_files += 1
                    m = compute_metrics(g["gold"], g["pred"], labels, pos, neg)
                    for key in ("accuracy", "macro_f1", "f1_pn"):
                        if abs(m[key] - ep.loc[e, f"{split}_{key}"]) > args.tol:
                            problems.append(f"{run}: epoch {e} {split}_{key} {m[key]:.6f} != {ep.loc[e, f'{split}_{key}']:.6f}")
            for f in sorted(run.rglob("*")):
                if f.is_file():
                    manifest.append(f"{sha256_file(f)}  {f.relative_to(res)}")
    for d, variants in test_ids.items():
        if len(variants) > 1:
            problems.append(f"{d}: runs used {len(variants)} different test item lists")
    (res / "MANIFEST.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"checked {n_runs} runs, {n_files} epoch prediction sets; manifest -> {res / 'MANIFEST.sha256'}")
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems[:50]))
        sys.exit(1)
    print("OK: every saved prediction set reproduces its logged scores, and all runs of a dataset share one test set.")


if __name__ == "__main__":
    main()
