#!/usr/bin/env python
"""Create FAKE datasets with the paper's split sizes, for the CPU dry run only (data_synthetic/).

The texts are placeholders; nothing here is real data and nothing here should be reported.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from marbert_audit.config import REPO_ROOT, load_config
from marbert_audit.data import get_train_dev, write_split

# rough label shares, for a realistic shape only
SHARES = {"arsarcasm": {"negative": 0.36, "neutral": 0.46, "positive": 0.18},
          "astd": {"negative": 0.48, "neutral": 0.28, "positive": 0.24},
          "ajgt": {"negative": 0.5, "positive": 0.5}}
DIALECTS = (["msa", "egypt", "gulf", "levant", "magreb"], [0.68, 0.20, 0.05, 0.05, 0.02])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data_synthetic"))
    ap.add_argument("--scale", type=float, default=1.0, help="shrink split sizes (e.g. 0.1) for faster tests")
    args = ap.parse_args()
    cfg = load_config()
    root = Path(args.data_root)
    rng = np.random.default_rng(0)
    for d, dcfg in cfg["datasets"].items():
        labels = dcfg["labels"]
        p = np.array([SHARES[d][lab] for lab in labels])
        for split in ("train", "test"):
            n = max(40, int(dcfg["paper_sizes"][split] * args.scale))
            y = rng.choice(labels, size=n, p=p / p.sum())
            df = pd.DataFrame({"id": [f"{d}-{split}-{i:05d}" for i in range(n)],
                               "text": [f"نص تجريبي {split} رقم {i} @user https://t.co/x" for i in range(n)], "label": y})
            if d == "arsarcasm":
                df["dialect"] = rng.choice(DIALECTS[0], size=n, p=DIALECTS[1])
                df["sarcasm"] = np.where(rng.random(n) < 0.16, "sarcastic", "non-sarcastic")
            if d == "astd":
                df["source"] = rng.choice(["arsas", "astd", "semeval", "arsentd_lev"], size=n) if split == "train" else "astd"
            write_split(df, root, d, split)
        get_train_dev(root, d, cfg)
        print(f"synthetic {d}: written to {root / 'processed' / d}")


if __name__ == "__main__":
    main()
