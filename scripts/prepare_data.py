#!/usr/bin/env python
"""Download / convert the three datasets into data/processed/<dataset>/{train,test}.tsv.

Examples
  python scripts/prepare_data.py arsarcasm                 # from the Hugging Face Hub
  python scripts/prepare_data.py arsarcasm --source csv --train-file ArSarcasm_train.csv --test-file ArSarcasm_test.csv
  python scripts/prepare_data.py ajgt                      # from the Hugging Face Hub
  python scripts/prepare_data.py ajgt --source xlsx --train-file AJGT.xlsx
  python scripts/prepare_data.py astd --manifest data/camelsent_sources.yaml
  python scripts/prepare_data.py all                       # arsarcasm + ajgt (+ astd if the manifest exists)

Also writes the committed split-ID files in data/splits/ (dev subsets for every dataset; the AJGT
80/20 train/test split) and data/MANIFEST.json (checksums, counts, sources). See data/README.md.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
import yaml

from marbert_audit.config import REPO_ROOT, load_config
from marbert_audit.data import (get_train_dev, normalize_label, processed_path, read_ids, sha256_file,
                                splits_dir, stratified_holdout, write_ids, write_split)

ARSARCASM_HF = ["iabufarha/ar_sarcasm", "ar_sarcasm"]
AJGT_HF = ["komari6/ajgt_twitter_ar", "ajgt_twitter_ar"]


def _load_hf(candidates: list[str]):
    from datasets import load_dataset

    errors = []
    for name in candidates:
        try:
            ds = load_dataset(name)
            info = {"hf_dataset": name}
            try:
                any_split = next(iter(ds.values()))
                info["hf_fingerprint"] = getattr(any_split, "_fingerprint", None)
            except Exception:
                pass
            return ds, info
        except Exception as exc:  # try the next mirror
            errors.append(f"{name}: {exc}")
    raise RuntimeError("Could not load any of " + ", ".join(candidates) + "\n" + "\n".join(errors))


def _to_names(split, col):
    feat = split.features[col]
    values = split[col]
    return [feat.int2str(v) for v in values] if hasattr(feat, "int2str") else list(values)


def _pick(df: pd.DataFrame, options: list[str]) -> str:
    for o in options:
        for c in df.columns:
            if c.strip().lower() == o.lower():
                return c
    raise KeyError(f"none of {options} in columns {list(df.columns)}")


def _sarcasm_name(v) -> str:
    s = str(v).strip().lower()
    return "sarcastic" if s in {"true", "1", "sarcastic", "yes"} else "non-sarcastic"


# ------------------------------------------------------------------------------ ArSarcasm
def prepare_arsarcasm(args, cfg, data_root: Path) -> dict:
    allowed = set(cfg["datasets"]["arsarcasm"]["labels"])
    frames, info = {}, {"source": args.source}
    if args.source == "hf":
        ds, hf_info = _load_hf(ARSARCASM_HF)
        info.update(hf_info)
        for split in ("train", "test"):
            s = ds[split]
            frames[split] = pd.DataFrame({
                "text": s["tweet"], "label": _to_names(s, "sentiment"), "dialect": _to_names(s, "dialect"),
                "sarcasm": _to_names(s, "sarcasm"), "original_sentiment": _to_names(s, "original_sentiment"),
                "source": s["source"],
            })
    else:
        for split, path in (("train", args.train_file), ("test", args.test_file)):
            raw = pd.read_csv(path, dtype=str, keep_default_na=False)
            frames[split] = pd.DataFrame({
                "text": raw[_pick(raw, ["tweet", "text"])], "label": raw[_pick(raw, ["sentiment"])],
                "dialect": raw[_pick(raw, ["dialect"])].str.lower(), "sarcasm": raw[_pick(raw, ["sarcasm"])].map(_sarcasm_name),
                "source": raw[_pick(raw, ["source"])] if any(c.lower() == "source" for c in raw.columns) else "",
            })
            info[f"{split}_file_sha256"] = sha256_file(path)
    for split, df in frames.items():
        df["label"] = df["label"].map(normalize_label)
        dropped = int((~df["label"].isin(allowed)).sum())
        df = df[df["label"].isin(allowed)].reset_index(drop=True)
        df.insert(0, "id", [f"arsarcasm-{split}-{i:05d}" for i in range(len(df))])
        write_split(df, data_root, "arsarcasm", split)
        info[f"{split}_n"], info[f"{split}_dropped"] = len(df), dropped
    return info


# ------------------------------------------------------------------------------ AJGT
def prepare_ajgt(args, cfg, data_root: Path) -> dict:
    info = {"source": args.source}
    if args.source == "hf":
        ds, hf_info = _load_hf(AJGT_HF)
        info.update(hf_info)
        s = ds["train"]
        df = pd.DataFrame({"text": s["text"], "label": _to_names(s, "label")})
    else:
        path = args.train_file
        raw = pd.read_excel(path, dtype=str) if str(path).endswith((".xlsx", ".xls")) else pd.read_csv(path, dtype=str)
        df = pd.DataFrame({"text": raw[_pick(raw, ["feed", "tweet", "text"])], "label": raw[_pick(raw, ["sentiment", "label"])]})
        info["file_sha256"] = sha256_file(path)
    df["label"] = df["label"].map(normalize_label)
    df = df[df["label"].isin(cfg["datasets"]["ajgt"]["labels"])].reset_index(drop=True)
    df.insert(0, "id", [f"ajgt-{i:04d}" for i in range(len(df))])
    id_file = splits_dir(data_root) / "ajgt_test_ids.txt"
    if id_file.exists():
        test_ids = set(read_ids(id_file))
        info["test_split"] = "reused committed data/splits/ajgt_test_ids.txt"
    else:
        _, test_ids = stratified_holdout(df, cfg["datasets"]["ajgt"]["test_fraction"], cfg["split_seed"])
        write_ids(id_file, test_ids)
        test_ids = set(test_ids)
        info["test_split"] = f"new stratified {cfg['datasets']['ajgt']['test_fraction']:.0%} split, seed {cfg['split_seed']}"
    write_split(df[~df["id"].isin(test_ids)], data_root, "ajgt", "train")
    write_split(df[df["id"].isin(test_ids)], data_root, "ajgt", "test")
    info["train_n"], info["test_n"] = int((~df["id"].isin(test_ids)).sum()), len(test_ids)
    return info


# ------------------------------------------------------------------------------ ASTD / CAMelSent
def prepare_astd(args, cfg, data_root: Path) -> dict:
    man_path = Path(args.manifest)
    man = yaml.safe_load(man_path.read_text(encoding="utf-8"))
    label_map = man.get("label_map") or {}
    allowed = set(cfg["datasets"]["astd"]["labels"])
    parts = {"train": [], "test": []}
    info = {"source": "manifest", "manifest": str(man_path), "files": []}
    for k, entry in enumerate(man["files"]):
        path = Path(entry["path"])
        if not path.is_absolute():
            path = (man_path.parent / path) if (man_path.parent / path).exists() else REPO_ROOT / path
        sep = entry.get("sep", "\t")
        header = 0 if entry.get("header", True) else None
        raw = pd.read_csv(path, sep=sep, header=header, names=entry.get("columns"), dtype=str, keep_default_na=False,
                          quoting=3 if entry.get("no_quoting") else 0, engine="python")
        df = pd.DataFrame({"text": raw[entry.get("text_col", "text")], "raw_label": raw[entry.get("label_col", "label")]})
        df["label"] = [normalize_label(v, label_map) for v in df["raw_label"]]
        kept = df[df["label"].isin(allowed)].copy()
        kept["source"] = entry["source"]
        kept["id"] = [f"{entry['source']}-{entry['role']}-{k}-{i:06d}" for i in range(len(kept))]
        parts[entry["role"]].append(kept[["id", "text", "label", "source"]])
        info["files"].append({"path": str(entry["path"]), "role": entry["role"], "source": entry["source"],
                              "rows": len(df), "kept": len(kept), "dropped_labels": df.loc[~df["label"].isin(allowed), "raw_label"].value_counts().to_dict(),
                              "sha256": sha256_file(path)})
    for split in ("train", "test"):
        if not parts[split]:
            raise ValueError(f"manifest has no files with role {split!r}")
        df = pd.concat(parts[split], ignore_index=True)
        write_split(df, data_root, "astd", split)
        info[f"{split}_n"] = len(df)
    exp = cfg["datasets"]["astd"]["paper_sizes"]
    if info["test_n"] != exp["test"]:
        print(f"WARNING: ASTD test has {info['test_n']} items; [1] reports {exp['test']}. Log this in docs/DEPARTURES.md.")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["arsarcasm", "ajgt", "astd", "all"])
    ap.add_argument("--source", default="hf", choices=["hf", "csv", "xlsx"])
    ap.add_argument("--train-file")
    ap.add_argument("--test-file")
    ap.add_argument("--manifest", default=str(REPO_ROOT / "data" / "camelsent_sources.yaml"))
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    args = ap.parse_args()
    cfg = load_config()
    data_root = Path(args.data_root)
    if args.dataset == "all":
        todo = ["arsarcasm", "ajgt"] + (["astd"] if Path(args.manifest).exists() else [])
    else:
        todo = [args.dataset]
    man_file = data_root / "MANIFEST.json"
    manifest = json.loads(man_file.read_text(encoding="utf-8")) if man_file.exists() else {}
    for d in todo:
        print(f"== preparing {d}")
        fn = {"arsarcasm": prepare_arsarcasm, "ajgt": prepare_ajgt, "astd": prepare_astd}[d]
        info = fn(args, cfg, data_root)
        get_train_dev(data_root, d, cfg)  # creates data/splits/<d>_dev_ids.txt if missing
        info["dev_ids_sha256"] = sha256_file(splits_dir(data_root) / f"{d}_dev_ids.txt")
        info["processed_sha256"] = {s: sha256_file(processed_path(data_root, d, s)) for s in ("train", "test")}
        info["prepared_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest[d] = info
        print(json.dumps({k: v for k, v in info.items() if k != "files"}, indent=2, ensure_ascii=False))
    man_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.dataset == "all" and "astd" not in todo:
        print("NOTE: ASTD/CAMelSent skipped - create data/camelsent_sources.yaml first (see data/README.md).")


if __name__ == "__main__":
    main()
