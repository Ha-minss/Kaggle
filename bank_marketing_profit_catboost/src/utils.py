from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Best-effort reproducibility across common libs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
