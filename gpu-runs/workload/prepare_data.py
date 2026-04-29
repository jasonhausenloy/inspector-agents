"""Tokenize TinyStories + WikiText to uint16 .npy files for memmap loading.

Idempotent: skips outputs that already exist. Run once per host.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def tokenize(text_iter, enc) -> np.ndarray:
    pieces: list[np.ndarray] = []
    total = 0
    for text in text_iter:
        if not text or not text.strip():
            continue
        ids = enc.encode_ordinary(text)
        ids.append(enc.eot_token)
        pieces.append(np.array(ids, dtype=np.uint16))
        total += len(ids)
        if total > 0 and total % 5_000_000 < 1000:
            print(f"  tokenized ~{total/1e6:.0f}M tokens", flush=True)
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.uint16)


def prepare_tinystories(enc) -> None:
    out = DATA_DIR / "tinystories.npy"
    if out.exists():
        print(f"  exists: {out}, skipping", flush=True)
        return
    print("downloading TinyStories train[:5%]...", flush=True)
    ds = load_dataset("roneneldan/TinyStories", split="train[:5%]")
    print(f"  {len(ds)} stories; tokenizing", flush=True)
    ids = tokenize((x["text"] for x in ds), enc)
    np.save(out, ids)
    print(f"  -> {out} ({len(ids)/1e6:.1f}M tokens)", flush=True)


def prepare_wikitext(enc) -> None:
    out = DATA_DIR / "wikitext.npy"
    if out.exists():
        print(f"  exists: {out}, skipping", flush=True)
        return
    print("downloading WikiText-2-raw-v1 train...", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    print(f"  {len(ds)} lines; tokenizing", flush=True)
    ids = tokenize((x["text"] for x in ds), enc)
    np.save(out, ids)
    print(f"  -> {out} ({len(ids)/1e6:.1f}M tokens)", flush=True)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    prepare_tinystories(enc)
    prepare_wikitext(enc)


if __name__ == "__main__":
    main()
