#!/usr/bin/env python
"""Import the team's Colab run folders into results/ and verify every epoch against the saved predictions.

  python scripts/import_team_runs.py --team-outputs path/to/marbert_repro/outputs

Imports
  outputs/historical/{arsarcasm,astd,ajgt}/{marbert,arabert}/seed{42,43,44}   -> results/runs/historical/...
  outputs/independent/{arsarcasm,astd,ajgt}/{marbert,arabert}/seed{42,43,44}  -> results/runs/independent/...
  outputs/historical/ajgt_ubc/...                                              -> results/runs/historical/ajgt_ubc/...
  outputs/extras/<variant>/arsarcasm/<model>/seed42                            -> results/runs_extra/<variant>/historical/arsarcasm/...
Historical prediction files store a row index (idx); it is mapped to item IDs with data/splits/<dataset>_test_ids.txt
(Member 3 verified that Member 1's split files and the training files share IDs and row order).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from marbert_audit.config import DATASET_ORDER, MODEL_ORDER, REPO_ROOT, load_config
from marbert_audit.importer import import_run


def ids(name: str) -> list[str]:
    return (REPO_ROOT / "data" / "splits" / f"{name}_test_ids.txt").read_text(encoding="utf-8").split()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team-outputs", required=True, help="the team's marbert_repro/outputs folder")
    ap.add_argument("--results-root", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()
    cfg = load_config()
    src_root, res = Path(args.team_outputs), Path(args.results_root)
    jobs = []
    for protocol in ("historical", "independent"):
        for d in DATASET_ORDER:
            for m in MODEL_ORDER:
                for s in cfg["seeds"]:
                    jobs.append((src_root / protocol / d / m / f"seed{s}", res / "runs" / protocol / d / m / f"seed{s}", protocol, d, m, s, d))
    for m in MODEL_ORDER:
        for s in cfg["seeds"]:
            jobs.append((src_root / "historical" / "ajgt_ubc" / m / f"seed{s}", res / "runs" / "historical" / "ajgt_ubc" / m / f"seed{s}",
                         "historical", "ajgt_ubc", m, s, "ajgt_ubc"))
    extras = src_root / "extras"
    if extras.exists():
        for run in sorted(extras.glob("*/*/*/seed*")):
            variant, d, m, s = run.parts[-4], run.parts[-3], run.parts[-2], int(run.name.replace("seed", ""))
            jobs.append((run, res / "runs_extra" / variant / "historical" / d / m / f"seed{s}", "historical", d, m, s, d))
    n_files, problems = 0, []
    for src, dst, protocol, d, m, s, id_name in jobs:
        try:
            rep = import_run(src, dst, protocol, d, m, s, cfg, ids(id_name))
            n_files += rep["files_checked"]
            print(f"ok  {dst.relative_to(res)}  ({rep['epochs']} epochs)")
        except Exception as exc:  # report every problem, then fail
            problems.append(f"{src}: {exc}")
            print(f"ERR {src}: {exc}")
    print(f"\n{len(jobs) - len(problems)}/{len(jobs)} runs imported; {n_files} prediction files re-scored, all equal to the logged scores.")
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
