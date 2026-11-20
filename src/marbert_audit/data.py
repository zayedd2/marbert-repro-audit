"""Read and write the processed data and the committed split-ID files.

Processed format (one file per split): ``<data_root>/processed/<dataset>/<split>.tsv`` with columns
``id``, ``text``, ``label`` and, optionally, metadata columns such as ``source``, ``dialect``, ``sarcasm``.
Labels are stored as names (``negative``/``neutral``/``positive``), never as numbers.

Split-ID files live in ``<data_root>/splits/`` and are committed to git, so the exact dev subsets
(and the AJGT train/test split) are fixed for everyone, independently of library versions.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pandas as pd

LABEL_ALIASES = {
    "positive": "positive", "pos": "positive",
    "negative": "negative", "neg": "negative",
    "neutral": "neutral", "neu": "neutral", "neut": "neutral",
    "objective": "objective", "obj": "objective",
    "mixed": "mixed", "mix": "mixed",
}


def normalize_label(value, label_map: dict | None = None) -> str | None:
    """Map raw label strings to canonical names. Unknown labels return None (and are dropped)."""
    if value is None:
        return None
    raw = str(value).strip()
    if label_map and raw in label_map:
        return label_map[raw]
    key = raw.lower()
    if label_map and key in label_map:
        return label_map[key]
    return LABEL_ALIASES.get(key)


def processed_path(data_root: Path | str, dataset: str, split: str) -> Path:
    return Path(data_root) / "processed" / dataset / f"{split}.tsv"


def write_split(df: pd.DataFrame, data_root: Path | str, dataset: str, split: str) -> Path:
    path = processed_path(data_root, dataset, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = df.copy()
    clean["text"] = clean["text"].astype(str).str.replace(r"[\t\r\n]+", " ", regex=True)
    clean.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
    return path


def read_split(data_root: Path | str, dataset: str, split: str) -> pd.DataFrame:
    path = processed_path(data_root, dataset, split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/prepare_data.py first (see data/README.md)."
        )
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
    missing = {"id", "text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    if df["id"].duplicated().any():
        raise ValueError(f"{path} has duplicate ids")
    return df


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def splits_dir(data_root: Path | str) -> Path:
    return Path(data_root) / "splits"


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_ids(path: Path, ids) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(map(str, ids)) + "\n", encoding="utf-8")


def stratified_holdout(df: pd.DataFrame, fraction: float, seed: int, stratify_col: str = "label") -> tuple[list, list]:
    """Return (keep_ids, holdout_ids) with the holdout stratified by label."""
    from sklearn.model_selection import train_test_split

    keep, hold = train_test_split(
        df["id"].tolist(), test_size=fraction, random_state=seed, stratify=df[stratify_col].tolist()
    )
    return sorted(keep), sorted(hold)


def get_train_dev(data_root: Path | str, dataset: str, cfg: dict, create_if_missing: bool = True):
    """Split the processed training file into train/dev using the committed dev-ID file."""
    train = read_split(data_root, dataset, "train")
    id_file = splits_dir(data_root) / f"{dataset}_dev_ids.txt"
    if not id_file.exists():
        if not create_if_missing:
            raise FileNotFoundError(f"{id_file} missing; run scripts/make_dev_splits.py")
        _, dev_ids = stratified_holdout(train, cfg["dev_fraction"], cfg["split_seed"])
        write_ids(id_file, dev_ids)
    dev_ids = set(read_ids(id_file))
    unknown = dev_ids - set(train["id"])
    if unknown:
        raise ValueError(f"{len(unknown)} dev ids in {id_file} are not in the training file; data changed?")
    is_dev = train["id"].isin(dev_ids)
    return train.loc[~is_dev].reset_index(drop=True), train.loc[is_dev].reset_index(drop=True)
