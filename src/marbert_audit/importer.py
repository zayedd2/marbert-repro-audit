"""Convert the team's Colab run folders into this repository's run format, verifying every epoch on the way.

Team format (notebooks/historical_reproduction.ipynb), one folder per run:
    outputs/<protocol>/<dataset>/<model>/seed<seed>/
        run_info.json                 settings, model revision, GPU, package versions, file checksums
        metrics_per_epoch.jsonl       one line per epoch (historical: flat; independent: {"dev": .., "test": ..})
        preds/epoch_XX_<split>_preds.csv   idx|id, gold, pred, p_<label>... (gold/pred are class ids)
        DONE

Repository format (what scripts/make_report_assets.py reads):
    results/runs/<protocol>/<dataset>/<model>/seed<seed>/
        run_config.yaml, env.json, epoch_metrics.csv, preds_test.csv.gz, preds_dev.csv.gz, DONE
        original/run_info.json, original/metrics_per_epoch.jsonl   (unchanged copies, for provenance)

Every epoch's accuracy, macro-F1 and F1-PN are recomputed from the saved predictions and compared with the
team's logged values; any disagreement stops the import.
"""
from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .metrics import compute_metrics

TOL = 1e-6


def sha256_file(path: Path | str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv_gz(df: pd.DataFrame, path: Path) -> None:
    """Gzip with a fixed timestamp, so the same data always gives the same bytes (stable checksums)."""
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as gz:
        gz.write(df.to_csv(index=False).encode("utf-8"))


def _epoch_record(rec: dict, split: str, protocol: str) -> dict:
    if split in rec and isinstance(rec[split], dict):
        return rec[split]  # nested record (independent runs and the extra sensitivity runs)
    return rec  # flat record = test split (historical runs)


def import_run(src: Path, dst: Path, protocol: str, dataset: str, model: str, seed: int, cfg: dict,
               test_ids: list[str] | None) -> dict:
    """Convert one run. Returns a small report dict; raises on any mismatch."""
    if not (src / "DONE").exists():
        raise FileNotFoundError(f"{src} has no DONE file (incomplete run)")
    info = json.loads((src / "run_info.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (src / "metrics_per_epoch.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    records.sort(key=lambda r: r["epoch"])
    labels = list(info.get("labels") or cfg["datasets"][dataset]["labels"])
    if labels != list(cfg["datasets"][dataset]["labels"]):
        raise ValueError(f"{src}: labels {labels} differ from configs/datasets.yaml")
    pos, neg = labels.index("positive"), labels.index("negative")
    splits = ["test"] if protocol == "historical" else ["dev", "test"]

    if dst.exists():
        shutil.rmtree(dst)
    (dst / "original").mkdir(parents=True)
    shutil.copy(src / "run_info.json", dst / "original" / "run_info.json")
    shutil.copy(src / "metrics_per_epoch.jsonl", dst / "original" / "metrics_per_epoch.jsonl")

    rows, frames, checked = [], {s: [] for s in splits}, 0
    for rec in records:
        e = int(rec["epoch"])
        row = {"epoch": e, "train_loss": rec.get("train_loss"), "seconds": rec.get("seconds")}
        for split in splits:
            f = src / "preds" / f"epoch_{e:02d}_{split}_preds.csv"
            p = pd.read_csv(f, dtype={"id": str})
            if "id" in p.columns and split == "test" and test_ids is not None and p["id"].str.fullmatch(r"\d+").all():
                p = p.rename(columns={"id": "idx"}).astype({"idx": int})  # extra runs saved the row index as "id"
            if "id" not in p.columns:
                if "idx" not in p.columns or test_ids is None or split != "test":
                    raise ValueError(f"{f}: no id column and no id list to map idx")
                if len(p) != len(test_ids):
                    raise ValueError(f"{f}: {len(p)} rows but {len(test_ids)} test ids")
                p.insert(0, "id", np.asarray(test_ids, dtype=object)[p["idx"].to_numpy()])
                p = p.drop(columns=["idx"])
            m = compute_metrics(p["gold"].to_numpy(), p["pred"].to_numpy(), labels, pos, neg)
            logged = _epoch_record(rec, split, protocol)
            for key in ("accuracy", "macro_f1", "f1_pn"):
                if key in logged and abs(m[key] - float(logged[key])) > TOL:
                    raise ValueError(f"{f}: recomputed {key} {m[key]:.6f} != logged {logged[key]:.6f}")
                row[f"{split}_{key}"] = m[key]
            checked += 1
            p.insert(0, "epoch", e)
            frames[split].append(p[["epoch", "id", "gold", "pred"] + [f"p_{lab}" for lab in labels]])
        rows.append(row)

    epochs = pd.DataFrame(rows)
    epochs.to_csv(dst / "epoch_metrics.csv", index=False)
    for split, parts in frames.items():
        _write_csv_gz(pd.concat(parts, ignore_index=True), dst / f"preds_{split}.csv.gz")

    backend = "dummy" if info.get("gpu") == "SYNTHETIC" else "team-colab"  # synthetic smoke-test runs
    n_items = {"train": info.get("n_train"), "test": info.get("n_test")}
    if "n_dev" in info:
        n_items["dev"] = info["n_dev"]
    md5 = info.get("md5") or {"train": info.get("train_md5"), "test": info.get("test_md5")}
    run_config = {
        "protocol": protocol, "dataset": dataset, "model": model, "seed": int(seed), "backend": backend,
        "protocol_description": info.get("protocol"), "labels": labels,
        "selection": cfg["protocols"][protocol]["selection"] if protocol in cfg["protocols"] else None,
        "model_id": info.get("model_id"), "model_revision": info.get("model_revision"),
        "training": {**(info.get("hp") or {k: info[k] for k in ("lr", "schedule", "warmup") if k in info}),
                     "epochs": info.get("epochs"), "fp16": info.get("use_amp"), "dynamic_padding": info.get("dynamic_padding")},
        "variant": {k: info[k] for k in ("tag", "text_fn", "model_id") if k in info} if "tag" in info else None,
        "preprocessing": info.get("preprocessing"), "n_items": n_items, "data_md5": md5,
        "source_folder": str(Path(protocol) / dataset / model / f"seed{seed}"),
    }
    (dst / "run_config.yaml").write_text(yaml.safe_dump(run_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    env = {
        "gpu": (info.get("gpu") or "").split(",")[0] or None, "gpu_full": info.get("gpu"),
        "torch": info.get("torch"), "transformers": info.get("transformers"),
        "started": info.get("started"), "finished": info.get("finished"),
        "epoch_seconds_total": round(float(epochs["seconds"].fillna(0).sum()), 1),
        "model_info": {"model_commit": info.get("model_revision"), "hf_id": info.get("model_id")},
        "git_commit": None,
    }
    (dst / "env.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    (dst / "DONE").write_text(json.dumps({"epochs": len(epochs), "backend": backend}) + "\n")
    return {"run": str(dst), "epochs": len(epochs), "files_checked": checked}
