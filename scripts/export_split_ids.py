#!/usr/bin/env python
"""Write the committed split files from Member 1's final team splits (Team_Splits_Final.zip).

The tweets themselves are not redistributed. What is committed:
  data/splits/<dataset>_<split>_ids.txt   item IDs in file order (train_full, train, dev, test)
  data/splits/ajgt_ubc_test_ids.txt       IDs of the authors' AJGT test split (check runs)
  data/meta/arsarcasm_test_meta.csv       id, label, dialect, sarcasm (labels only, no text) for subgroup results
  data/MANIFEST.json                      MD5 checksums and label counts of every split file

  python scripts/export_split_ids.py --splits path/to/member1_outputs/splits --ajgt-ubc-key path/to/ajgt_ubc_test_key.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from marbert_audit.config import REPO_ROOT


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", required=True, help="folder with <dataset>_<split>.csv files from Member 1")
    ap.add_argument("--ajgt-ubc-key", help="ajgt_ubc_test_key.csv from the team's outputs folder")
    args = ap.parse_args()
    src, out = Path(args.splits), REPO_ROOT / "data"
    manifest = {"source": "Member 1, Team_Splits_Final.zip (make_team_splits.py)", "files": {}}
    for ds in ("arsarcasm", "astd", "ajgt"):
        for split in ("train_full", "train", "dev", "test"):
            f = src / f"{ds}_{split}.csv"
            df = pd.read_csv(f, dtype=str, keep_default_na=False)
            (out / "splits" / f"{ds}_{split}_ids.txt").write_text("\n".join(df["id"]) + "\n", encoding="utf-8")
            manifest["files"][f.name] = {"md5": md5(f), "n": len(df), "labels": df["label"].value_counts().sort_index().to_dict()}
            if ds == "arsarcasm" and split == "test":
                df[["id", "label", "dialect", "sarcasm"]].to_csv(out / "meta" / "arsarcasm_test_meta.csv", index=False)
    if args.ajgt_ubc_key:
        key = pd.read_csv(args.ajgt_ubc_key, dtype=str, keep_default_na=False)
        (out / "splits" / "ajgt_ubc_test_ids.txt").write_text("\n".join(key["id"]) + "\n", encoding="utf-8")
        manifest["files"]["ajgt_ubc_test_key.csv"] = {"md5": md5(Path(args.ajgt_ubc_key)), "n": len(key),
                                                      "labels": key["gold"].value_counts().sort_index().to_dict()}
    summary = src / "split_summary.json"
    if summary.exists():
        manifest["split_summary"] = json.loads(summary.read_text(encoding="utf-8"))
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote split IDs for {len(manifest['files'])} files to {out / 'splits'}")


if __name__ == "__main__":
    main()
