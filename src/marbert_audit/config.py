"""Load the YAML configs in configs/ into one dictionary."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

MODEL_ORDER = ["marbert", "arabert"]
DATASET_ORDER = ["arsarcasm", "astd", "ajgt"]
PROTOCOL_ORDER = ["historical", "independent"]


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(config_dir: Path | str | None = None, overrides: dict | None = None) -> dict[str, Any]:
    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    cfg = load_yaml(cdir / "base.yaml")
    cfg["models"] = load_yaml(cdir / "models.yaml")
    cfg["datasets"] = load_yaml(cdir / "datasets.yaml")
    cfg["protocols"] = load_yaml(cdir / "protocols.yaml")
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def selection_metric(cfg: dict, protocol: str, dataset: str) -> tuple[str, str]:
    """Return (split, metric) used to pick the checkpoint for a protocol/dataset pair."""
    sel = cfg["protocols"][protocol]["selection"]
    metric = sel["metric"]
    if metric == "historical":
        metric = cfg["datasets"][dataset]["historical_metric"]
    return sel["split"], metric


def labels_for(cfg: dict, dataset: str) -> list[str]:
    return list(cfg["datasets"][dataset]["labels"])


def pos_neg_idx(cfg: dict, dataset: str) -> tuple[int, int]:
    d = cfg["datasets"][dataset]
    labels = d["labels"]
    return labels.index(d["positive_label"]), labels.index(d["negative_label"])
