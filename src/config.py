"""YAML config loading (configs/*.yaml), cached per path."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


@lru_cache(maxsize=None)
def load_yaml(name: str) -> dict:
    """configs/<name> as a dict (empty file -> {})."""
    return yaml.safe_load((CONFIG_DIR / name).read_text()) or {}


def repo_path(relative: str) -> Path:
    """Resolve a path from a config file against the repo root, not the CWD."""
    p = Path(relative)
    return p if p.is_absolute() else REPO_ROOT / p
