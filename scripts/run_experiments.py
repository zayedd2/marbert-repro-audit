#!/usr/bin/env python
"""Run the experiment grid: protocols x datasets x models x seeds (36 runs in total).

Completed runs (with a DONE file) are skipped, so you can stop and restart at any time.

Examples
  python scripts/run_experiments.py                                     # everything
  python scripts/run_experiments.py --protocols historical --datasets ajgt
  python scripts/run_experiments.py --resume-root /content/drive/MyDrive/marbert_ckpt   # Colab: survive disconnects
  python scripts/run_experiments.py --list                              # show what is done / missing
  python scripts/run_experiments.py --backend dummy --data-root data_synthetic --results-root results_synthetic
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import _bootstrap  # noqa: F401

from marbert_audit.config import DATASET_ORDER, MODEL_ORDER, PROTOCOL_ORDER, REPO_ROOT, load_config
from marbert_audit.train import is_done, run_dir_for, run_single


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocols", nargs="+", default=PROTOCOL_ORDER, choices=PROTOCOL_ORDER)
    ap.add_argument("--datasets", nargs="+", default=DATASET_ORDER, choices=DATASET_ORDER)
    ap.add_argument("--models", nargs="+", default=MODEL_ORDER, choices=MODEL_ORDER)
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="default: seeds in configs/base.yaml")
    ap.add_argument("--backend", default="hf", choices=["hf", "dummy"])
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    ap.add_argument("--results-root", default=str(REPO_ROOT / "results"))
    ap.add_argument("--resume-root", default=None, help="folder for per-epoch checkpoints (resumes interrupted runs)")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs (ONLY for quick tests; logged)")
    ap.add_argument("--force", action="store_true", help="re-run even if DONE")
    ap.add_argument("--keep-going", action="store_true", help="continue with the next run after an error")
    ap.add_argument("--list", action="store_true", help="only list run status")
    args = ap.parse_args()

    overrides = {"training": {"epochs": args.epochs}} if args.epochs else None
    cfg = load_config(overrides=overrides)
    if args.backend == "hf" and Path(args.results_root).resolve() == (REPO_ROOT / "results").resolve() and args.epochs:
        sys.exit("Refusing to write shortened runs into results/. Use a different --results-root for tests.")
    if args.backend == "dummy" and Path(args.results_root).resolve() == (REPO_ROOT / "results").resolve():
        sys.exit("Refusing to write SYNTHETIC runs into results/. Use --results-root results_synthetic.")
    seeds = args.seeds or cfg["seeds"]
    grid = [(p, d, m, s) for p in args.protocols for d in args.datasets for m in args.models for s in seeds]

    if args.list:
        for p, d, m, s in grid:
            print(f"{'done   ' if is_done(run_dir_for(args.results_root, p, d, m, s)) else 'MISSING'}  {p:11s} {d:9s} {m:7s} seed{s}")
        return

    failures = []
    for i, (p, d, m, s) in enumerate(grid, 1):
        print(f"\n===== run {i}/{len(grid)}: {p} / {d} / {m} / seed {s} =====", flush=True)
        try:
            run_single(cfg, p, d, m, s, data_root=args.data_root, results_root=args.results_root,
                       backend=args.backend, resume_root=args.resume_root, force=args.force)
        except Exception:
            traceback.print_exc()
            failures.append((p, d, m, s))
            if not args.keep_going:
                sys.exit(1)
    print(f"\nFinished. {len(grid) - len(failures)}/{len(grid)} runs OK.")
    if failures:
        print("Failed:", failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
