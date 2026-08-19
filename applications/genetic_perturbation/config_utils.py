"""YAML loading helpers for the genetic perturbation application."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import yaml


def _deep_merge(base, override):
    out = deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    base = cfg.pop("base_config", None)
    if base:
        base_path = (path.parent / base).resolve()
        cfg = _deep_merge(load_config(base_path), cfg)
    return cfg
