"""Record the software and hardware environment of a run (package versions, GPU, git commit)."""
from __future__ import annotations

import datetime as _dt
import platform
import subprocess
import sys
from importlib import metadata

from .config import REPO_ROOT

KEY_PACKAGES = ["torch", "transformers", "tokenizers", "datasets", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "PyYAML"]


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except Exception:
        return None


def installed_packages() -> dict[str, str]:
    pkgs = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            pkgs[name] = dist.version
    return dict(sorted(pkgs.items(), key=lambda kv: kv[0].lower()))


def collect_env() -> dict:
    pkgs = installed_packages()
    lower = {k.lower(): v for k, v in pkgs.items()}
    info = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")) if _git("rev-parse", "HEAD") else None,
        "key_packages": {p: lower.get(p.lower()) for p in KEY_PACKAGES},
        "all_packages": pkgs,
        "gpu": None,
        "cuda": None,
    }
    try:  # torch is optional (not needed to regenerate tables and figures)
        import torch

        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info
