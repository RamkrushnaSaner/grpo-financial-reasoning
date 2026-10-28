"""Seeding and small file helpers."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Union

import numpy as np

PathLike = Union[str, Path]


def set_seed(seed: int) -> None:
    """Seed every RNG source.

    Bitwise determinism is not achievable with sampled generation on GPU: this
    makes runs comparable, not identical. Greedy evaluation is deterministic.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj: Any, path: PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)


def load_json(path: PathLike) -> Any:
    with open(path) as f:
        return json.load(f)


def file_sha256(path: PathLike, length: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:length]
