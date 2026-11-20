"""Scripted text preprocessing, applied identically to both models before their own tokenizers."""
from __future__ import annotations

import re

URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"
WS_RE = re.compile(r"\s+")


def preprocess(text: str, cfg: dict) -> str:
    """Apply the steps switched on in ``cfg['preprocessing']``."""
    p = cfg.get("preprocessing", {})
    t = "" if text is None else str(text)
    if p.get("replace_urls"):
        t = URL_RE.sub(f" {p['replace_urls']} ", t)
    if p.get("replace_mentions"):
        t = MENTION_RE.sub(f" {p['replace_mentions']} ", t)
    if p.get("remove_diacritics"):
        t = DIACRITICS_RE.sub("", t)
    if p.get("remove_tatweel"):
        t = t.replace(TATWEEL, "")
    if p.get("normalize_whitespace", True):
        t = WS_RE.sub(" ", t).strip()
    return t


def normalise_for_matching(text: str) -> str:
    """Aggressive normalisation used ONLY by the data audit to find duplicates and overlap."""
    t = URL_RE.sub(" ", str(text))
    t = MENTION_RE.sub(" ", t)
    t = DIACRITICS_RE.sub("", t).replace(TATWEEL, "")
    t = re.sub("[إأآا]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه")
    t = re.sub(r"[^\w\s]", " ", t)
    return WS_RE.sub(" ", t).strip().lower()
