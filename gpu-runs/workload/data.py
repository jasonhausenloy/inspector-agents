"""Token dataset. Memmaps a uint16 .npy file produced by prepare_data.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class TokenDataset:
    def __init__(self, name: str, block_size: int):
        path = DATA_DIR / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}; run `python workload/prepare_data.py` first"
            )
        # open_memmap respects the .npy header (offset, dtype) so we can stream
        self.data = np.lib.format.open_memmap(str(path), mode="r")
        if self.data.dtype != np.uint16:
            raise ValueError(f"expected uint16 tokens, got {self.data.dtype}")
        self.block_size = block_size

    def get_batch(
        self,
        batch_size: int,
        device: str,
        rng: np.random.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rng = rng if rng is not None else np.random.default_rng()
        max_start = len(self.data) - self.block_size - 1
        ix = rng.integers(0, max_start, size=batch_size)
        x = torch.from_numpy(
            np.stack([self.data[i : i + self.block_size].astype(np.int64) for i in ix])
        )
        y = torch.from_numpy(
            np.stack([self.data[i + 1 : i + 1 + self.block_size].astype(np.int64) for i in ix])
        )
        if device.startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
