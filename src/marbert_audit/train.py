"""Run one (protocol, dataset, model, seed) configuration and save everything needed to rebuild results.

Output folder: ``<results_root>/runs/<protocol>/<dataset>/<model>/seed<seed>/``
  run_config.yaml     resolved settings, data checksums and split sizes
  env.json            package versions, GPU, git commit, model commit, runtime
  epoch_metrics.csv   dev and test scores after every epoch
  preds_test.csv.gz   test predictions + class probabilities for EVERY epoch
  preds_dev.csv.gz    the same for the dev subset (independent protocol only)
  train.log           human-readable log
  DONE                written last; a run without it is incomplete and is ignored by the analysis
"""
from __future__ import annotations

import gzip
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import labels_for, pos_neg_idx
from .data import get_train_dev, processed_path, read_split, sha256_file
from .envinfo import collect_env
from .metrics import compute_metrics
from .preprocess import preprocess


def run_dir_for(results_root: Path | str, protocol: str, dataset: str, model: str, seed: int) -> Path:
    return Path(results_root) / "runs" / protocol / dataset / model / f"seed{seed}"


def is_done(run_dir: Path) -> bool:
    return (run_dir / "DONE").exists()


def _trim_partial(path: Path, keep_upto_epoch: int) -> None:
    if path.exists():
        df = pd.read_csv(path, dtype={"id": str})
        df[df["epoch"] <= keep_upto_epoch].to_csv(path, index=False)


def run_single(cfg: dict, protocol: str, dataset: str, model: str, seed: int,
               data_root: Path | str = "data", results_root: Path | str = "results",
               backend: str = "hf", resume_root: Path | str | None = None, force: bool = False) -> Path:
    run_dir = run_dir_for(results_root, protocol, dataset, model, seed)
    if is_done(run_dir) and not force:
        print(f"[skip] {run_dir} already complete")
        return run_dir
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_fh = open(run_dir / "train.log", "a", encoding="utf-8")

    def log(msg: str):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    labels = labels_for(cfg, dataset)
    lab2id = {lab: i for i, lab in enumerate(labels)}
    pos_idx, neg_idx = pos_neg_idx(cfg, dataset)
    pcfg = cfg["protocols"][protocol]

    # ---------------------------------------------------------------- data
    test = read_split(data_root, dataset, "test")
    if pcfg["use_dev_split"]:
        train, dev = get_train_dev(data_root, dataset, cfg)
    else:
        train, dev = read_split(data_root, dataset, "train"), None
    splits = {"test": test} if dev is None else {"dev": dev, "test": test}
    for name, df in {"train": train, **splits}.items():
        bad = set(df["label"]) - set(labels)
        if bad:
            raise ValueError(f"{dataset}/{name} has labels {bad} outside {labels}")

    def texts(df):
        return [preprocess(t, cfg) for t in df["text"].tolist()]

    gold = {name: df["label"].map(lab2id).to_numpy() for name, df in splits.items()}

    run_config = {
        "protocol": protocol, "dataset": dataset, "model": model, "seed": int(seed), "backend": backend,
        "labels": labels, "selection": pcfg["selection"],
        "model_config": cfg["models"][model], "training": cfg["training"], "preprocessing": cfg["preprocessing"],
        "split_seed": cfg["split_seed"], "dev_fraction": cfg["dev_fraction"] if pcfg["use_dev_split"] else None,
        "n_items": {"train": len(train), **{k: len(v) for k, v in splits.items()}},
        "label_counts": {"train": train["label"].value_counts().to_dict(),
                         **{k: v["label"].value_counts().to_dict() for k, v in splits.items()}},
        "data_sha256": {s: sha256_file(processed_path(data_root, dataset, s)) for s in ("train", "test")},
    }
    dev_ids_file = Path(data_root) / "splits" / f"{dataset}_dev_ids.txt"
    if pcfg["use_dev_split"] and dev_ids_file.exists():
        run_config["data_sha256"]["dev_ids"] = sha256_file(dev_ids_file)
    with open(run_dir / "run_config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(run_config, fh, allow_unicode=True, sort_keys=False)

    env = collect_env()
    log(f"START protocol={protocol} dataset={dataset} model={model} seed={seed} backend={backend} "
        f"train={len(train)} {' '.join(f'{k}={len(v)}' for k, v in splits.items())}")

    # ---------------------------------------------------------------- trainer
    common = dict(tcfg=cfg["training"], labels=labels, seed=int(seed))
    if backend == "hf":
        from .backends import HFTrainer

        resume_dir = Path(resume_root) / protocol / dataset / model / f"seed{seed}" if resume_root else None
        trainer = HFTrainer(model_cfg=cfg["models"][model], train_texts=texts(train),
                            train_labels=train["label"].map(lab2id).tolist(),
                            eval_texts={k: texts(v) for k, v in splits.items()}, resume_dir=resume_dir, log=log, **common)
    elif backend == "dummy":
        from .backends import DummyTrainer

        trainer = DummyTrainer(model_key=model, dataset_key=dataset, eval_labels=gold, protocol=protocol, **common)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    metrics_path = run_dir / "epoch_metrics.csv"
    partial = {name: run_dir / f"preds_{name}.partial.csv" for name in splits}
    if trainer.start_epoch > 1:  # resumed: drop anything written after the checkpoint
        if metrics_path.exists():
            m = pd.read_csv(metrics_path)
            m[m["epoch"] < trainer.start_epoch].to_csv(metrics_path, index=False)
        for p in partial.values():
            _trim_partial(p, trainer.start_epoch - 1)
    else:
        for p in [metrics_path, *partial.values()]:
            if p.exists():
                p.unlink()

    # ---------------------------------------------------------------- epochs
    t_start = time.time()
    for epoch, train_loss, probs, seconds in trainer.epochs():
        row = {"epoch": epoch, "train_loss": round(float(train_loss), 6), "seconds": round(float(seconds), 2)}
        for name, p in probs.items():
            pred = p.argmax(axis=1)
            m = compute_metrics(gold[name], pred, labels, pos_idx, neg_idx)
            for key in ("accuracy", "macro_f1", "f1_pn"):
                row[f"{name}_{key}"] = round(m[key], 6)
            frame = pd.DataFrame({"epoch": epoch, "id": splits[name]["id"].to_numpy(),
                                  "gold": gold[name], "pred": pred})
            for k, lab in enumerate(labels):
                frame[f"p_{lab}"] = np.round(p[:, k], 5)
            frame.to_csv(partial[name], mode="a", header=not partial[name].exists(), index=False)
        pd.DataFrame([row]).to_csv(metrics_path, mode="a", header=not metrics_path.exists(), index=False)
        summary = " ".join(f"{k}={v:.2f}" for k, v in row.items() if k.endswith(("macro_f1", "f1_pn", "accuracy")))
        log(f"epoch {epoch:02d} loss={train_loss:.4f} {summary} ({seconds:.0f}s)")

    # ---------------------------------------------------------------- finalise
    for name, p in partial.items():
        with open(p, "rb") as src, gzip.open(run_dir / f"preds_{name}.csv.gz", "wb", compresslevel=9) as dst:
            shutil.copyfileobj(src, dst)
        p.unlink()
    epoch_table = pd.read_csv(metrics_path)
    env["model_info"] = trainer.info
    env["wall_clock_seconds"] = round(time.time() - t_start, 1)          # this session only
    env["epoch_seconds_total"] = round(float(epoch_table["seconds"].sum()), 1)  # all sessions, incl. resumes
    env["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(run_dir / "env.json", "w", encoding="utf-8") as fh:
        json.dump(env, fh, indent=2, ensure_ascii=False)
    n_epochs = len(epoch_table)
    (run_dir / "DONE").write_text(json.dumps({"epochs": n_epochs, "backend": backend}) + "\n")
    trainer.delete_checkpoint()
    log(f"DONE in {env['wall_clock_seconds']:.0f}s -> {run_dir}")
    log_fh.close()
    return run_dir
