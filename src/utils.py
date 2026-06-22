"""Miscellaneous helper functions.

This module consolidates small utilities used throughout the
repository.  At present it contains a YAML loader and a simple
directory creator.  Additional helpers can be added here as
necessary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file and return a dictionary.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed configuration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(directory: str | Path) -> Path:
    """Ensure that a directory exists.

    If the directory does not exist it will be created.  The input is
    returned as a :class:`Path` object.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path