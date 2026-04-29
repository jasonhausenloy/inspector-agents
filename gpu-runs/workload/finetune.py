"""Finetune entry. Constant LR, frozen wte/wpe. See _train.run for logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _train import run

if __name__ == "__main__":
    run("finetune")
