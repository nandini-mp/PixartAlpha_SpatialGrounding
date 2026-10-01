"""Shared helpers: project root, config loading, seeding, device selection."""

import os
import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    """Load YAML config. Defaults to configs/config.yaml. Resolves paths to absolute."""
    path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["paths"] = {
        k: str(PROJECT_ROOT / v)
        for k, v in cfg["paths"].items()
    }

    return cfg


def set_seed(seed: int):
    """Seed python, numpy and torch (CPU+CUDA) for reproducibility."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return cuda if available, else cpu."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
