"""Training back-ends.

* ``HFTrainer``    - real fine-tuning with PyTorch + Hugging Face Transformers (needs a GPU in practice).
* ``DummyTrainer`` - produces SYNTHETIC predictions so the whole pipeline (metrics, selection, bootstrap,
                     tables, figures, report numbers) can be smoke-tested on a laptop in seconds.
                     Its output is never a result and is written to a separate ``results_synthetic/`` tree.

Both expose ``.info`` (dict), ``.start_epoch`` (int) and ``.epochs()``, a generator yielding
``(epoch, train_loss, {split_name: probabilities[n_items, n_labels]}, seconds)``.
"""
from __future__ import annotations

import math
import os
import random
import time
import zlib
from pathlib import Path

import numpy as np


def _stable_seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode("utf-8"))


# --------------------------------------------------------------------------------------
# Real fine-tuning
# --------------------------------------------------------------------------------------
class HFTrainer:
    def __init__(self, model_cfg: dict, tcfg: dict, labels: list[str], seed: int,
                 train_texts: list[str], train_labels: list[int], eval_texts: dict[str, list[str]],
                 resume_dir: Path | None = None, log=print, **_ignored):
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.tcfg, self.log = torch, tcfg, log
        self.n_epochs = int(tcfg["epochs"])
        self._seed_everything(seed, tcfg.get("deterministic", True))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            log("WARNING: no GPU found - training will be extremely slow.")

        id2label = dict(enumerate(labels))
        tok = AutoTokenizer.from_pretrained(model_cfg["hf_id"], revision=model_cfg.get("revision"))
        model = AutoModelForSequenceClassification.from_pretrained(
            model_cfg["hf_id"], revision=model_cfg.get("revision"), num_labels=len(labels),
            id2label=id2label, label2id={v: k for k, v in id2label.items()},
        )
        self.model = model.to(self.device)
        self.tok = tok
        max_len = int(tcfg["max_length"])

        def encode(texts):
            full = tok(texts, truncation=False, add_special_tokens=True)["input_ids"]
            enc = tok(texts, truncation=True, max_length=max_len)
            n_trunc = int(sum(len(x) > max_len for x in full))
            lengths = [len(x) for x in enc["input_ids"]]
            return enc, n_trunc, lengths

        enc, n_trunc, lengths = encode(train_texts)
        self.info = {
            "backend": "hf",
            "hf_id": model_cfg["hf_id"],
            "revision_requested": model_cfg.get("revision"),
            "model_commit": getattr(model.config, "_commit_hash", None),
            "tokenizer_class": type(tok).__name__,
            "vocab_size": tok.vocab_size,
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
            "device": str(self.device),
            "truncated_items": {"train": n_trunc},
            "median_tokens": {"train": float(np.median(lengths))},
        }
        keys = list(enc.keys())
        train_items = [dict({k: enc[k][i] for k in keys}, labels=int(train_labels[i])) for i in range(len(train_texts))]

        def collate(batch):
            batch = [dict(b) for b in batch]  # copy: never mutate the stored items
            labels_ = [b.pop("labels") for b in batch] if "labels" in batch[0] else None
            out = tok.pad(batch, return_tensors="pt")
            if labels_ is not None:
                out["labels"] = torch.tensor(labels_, dtype=torch.long)
            return out

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.train_loader = DataLoader(train_items, batch_size=int(tcfg["batch_size"]), shuffle=True,
                                       generator=self.generator, collate_fn=collate)
        self.eval_loaders = {}
        for name, texts in eval_texts.items():
            e_enc, e_trunc, e_len = encode(texts)
            items = [{k: e_enc[k][i] for k in keys} for i in range(len(texts))]
            self.eval_loaders[name] = DataLoader(items, batch_size=int(tcfg["eval_batch_size"]), shuffle=False, collate_fn=collate)
            self.info["truncated_items"][name] = e_trunc
            self.info["median_tokens"][name] = float(np.median(e_len))

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(tcfg["learning_rate"]), betas=tuple(tcfg.get("adam_betas", (0.9, 0.999))),
            eps=float(tcfg.get("adam_eps", 1e-8)), weight_decay=float(tcfg.get("weight_decay", 0.0)),
        )
        self.scheduler = self._make_scheduler(len(self.train_loader))
        self.use_amp = bool(tcfg.get("fp16")) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.start_epoch = 1
        self.ckpt_path = Path(resume_dir) / "last.pt" if resume_dir else None
        if self.ckpt_path and self.ckpt_path.exists():
            self._load_checkpoint()

    # ---------------- helpers
    def _seed_everything(self, seed: int, deterministic: bool):
        torch = self.torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)

    def _make_scheduler(self, steps_per_epoch: int):
        torch = self.torch
        warm = int(self.tcfg.get("warmup_steps", 0) or 0)
        kind = self.tcfg.get("lr_schedule", "constant")
        total = steps_per_epoch * self.n_epochs
        if kind == "constant":
            fn = (lambda s: min(1.0, (s + 1) / warm)) if warm > 0 else (lambda s: 1.0)
        elif kind == "linear":
            def fn(s):
                if warm > 0 and s < warm:
                    return (s + 1) / warm
                return max(0.0, (total - s) / max(1, total - warm))
        else:
            raise ValueError(f"unknown lr_schedule {kind!r}")
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, fn)

    def _predict(self, loader) -> np.ndarray:
        torch = self.torch
        self.model.eval()
        chunks = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(**batch).logits
                chunks.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def _save_checkpoint(self, epoch: int):
        torch = self.torch
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "generator": self.generator.get_state(),
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ckpt_path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, self.ckpt_path)

    def _load_checkpoint(self):
        torch = self.torch
        state = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.generator.set_state(state["generator"])
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"])
        if state.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        self.start_epoch = int(state["epoch"]) + 1
        self.info["resumed_from_epoch"] = int(state["epoch"])
        self.log(f"Resumed from checkpoint after epoch {state['epoch']}")

    def delete_checkpoint(self):
        if self.ckpt_path and self.ckpt_path.exists():
            self.ckpt_path.unlink()

    # ---------------- main loop
    def epochs(self):
        torch = self.torch
        clip = self.tcfg.get("max_grad_norm")
        for epoch in range(self.start_epoch, self.n_epochs + 1):
            t0 = time.time()
            self.model.train()
            loss_sum, n_batches = 0.0, 0
            for batch in self.train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    loss = self.model(**batch).loss
                self.scaler.scale(loss).backward()
                if clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(clip))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                loss_sum += float(loss.detach())
                n_batches += 1
            probs = {name: self._predict(loader) for name, loader in self.eval_loaders.items()}
            if self.ckpt_path:
                self._save_checkpoint(epoch)
            yield epoch, loss_sum / max(1, n_batches), probs, time.time() - t0


# --------------------------------------------------------------------------------------
# Synthetic back-end for smoke tests
# --------------------------------------------------------------------------------------
class DummyTrainer:
    """Generates SYNTHETIC predictions with a plausible shape. NOT a model; never report its output."""

    SKILL = {"marbert": 1.25, "arabert": 0.85}

    def __init__(self, model_key: str, dataset_key: str, tcfg: dict, labels: list[str], seed: int,
                 eval_labels: dict[str, np.ndarray], train_texts=None, protocol: str = "", **_ignored):
        self.model_key, self.dataset_key, self.seed = model_key, dataset_key, seed
        self.n_epochs = int(tcfg["epochs"])
        self.n_labels = len(labels)
        self.eval_labels = {k: np.asarray(v) for k, v in eval_labels.items()}
        self.protocol = protocol
        self.start_epoch = 1
        self.info = {"backend": "dummy", "WARNING": "SYNTHETIC predictions for pipeline testing - not a trained model",
                     "hf_id": None, "model_commit": None, "truncated_items": {}, "median_tokens": {}}

    def delete_checkpoint(self):
        pass

    def epochs(self):
        base = self.SKILL.get(self.model_key, 1.0)
        seed_rng = np.random.default_rng(_stable_seed("seed-offset", self.model_key, self.dataset_key, self.seed, self.protocol))
        seed_offset = seed_rng.normal(0.0, 0.10)
        persistent = {}
        for name, gold in self.eval_labels.items():
            item_rng = np.random.default_rng(_stable_seed("difficulty", self.dataset_key, name))
            difficulty = item_rng.normal(0.0, 1.0, size=gold.size)
            run_rng = np.random.default_rng(_stable_seed("run", self.model_key, self.dataset_key, self.seed, name, self.protocol))
            persistent[name] = (difficulty, run_rng.random(gold.size), run_rng.integers(1, self.n_labels, size=gold.size))
        for epoch in range(1, self.n_epochs + 1):
            ep_rng = np.random.default_rng(_stable_seed("epoch", self.model_key, self.dataset_key, self.seed, epoch, self.protocol))
            skill = base * (1.0 - math.exp(-epoch / 3.5)) + seed_offset + ep_rng.normal(0.0, 0.06)
            probs = {}
            for name, gold in self.eval_labels.items():
                difficulty, u, shift = persistent[name]
                split_skill = skill + ep_rng.normal(0.0, 0.05)  # dev and test do not peak at the same epoch
                p_correct = 1.0 / (1.0 + np.exp(-(1.6 * split_skill - difficulty + 0.4)))
                correct = u < p_correct
                pred = np.where(correct, gold, (gold + shift) % self.n_labels)
                conf = 0.45 + 0.5 * p_correct
                p = np.full((gold.size, self.n_labels), 0.0)
                p += ((1.0 - conf) / max(1, self.n_labels - 1))[:, None]
                p[np.arange(gold.size), pred] = conf
                probs[name] = p / p.sum(axis=1, keepdims=True)
            yield epoch, float(1.1 * math.exp(-epoch / 6.0) + 0.3), probs, 0.0
