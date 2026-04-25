"""Red team — evasion strategies that mutate a clean log trace into a
non-compliant one while trying to look benign.

Each evasion is a function `(records: list[dict]) -> list[dict]` returning a
new list (non-destructive). The Inspector's job is to catch them.
"""

from redteam.split_run import split_run
from redteam.relabel_ops import relabel_ops
from redteam.swap_dataset_hash import swap_dataset_hash
from redteam.pad_batches import pad_batches
from redteam.drop_context import drop_context

EVASIONS = {
    "split_run": split_run,
    "relabel_ops": relabel_ops,
    "swap_dataset_hash": swap_dataset_hash,
    "pad_batches": pad_batches,
    "drop_context": drop_context,
}

__all__ = ["EVASIONS"] + list(EVASIONS.keys())
