"""Experiment configs: YAML files that may inherit from a parent via ``base:``."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a config; ``base: other.yaml`` (same folder) is loaded first and overridden.

    ``name`` defaults to the file stem of the config that was asked for, not of its base.
    """
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    cfg.setdefault("name", path.stem)
    if "base" in cfg:
        cfg = load_config(path.parent / cfg.pop("base")) | cfg
    return cfg
