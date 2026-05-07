"""Shared utilities: config loading, RNG seeding, logging helpers."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import yaml


PathLike = Union[str, os.PathLike]


def load_config(path: PathLike) -> Dict[str, Any]:
    """Load a YAML config file into a nested dict.

    Parameters
    ----------
    path:
        Path to a YAML file (e.g. ``configs/default.yaml``).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping, got {type(cfg).__name__}")
    return cfg


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if available) PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is a hard dep but be defensive
        pass


def ensure_dir(path: PathLike) -> Path:
    """Create ``path`` (and parents) if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str = "proteinlm_bench", level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger configured with a single stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger
