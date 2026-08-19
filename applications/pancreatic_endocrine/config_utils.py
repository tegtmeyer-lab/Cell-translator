"""Configuration helpers for the pancreatic endocrine application."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(path: str) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    common_name = cfg.pop("common_config", None)
    if common_name:
        common_path = (path.parent / common_name).resolve()
        with common_path.open("r", encoding="utf-8") as handle:
            common = yaml.safe_load(handle) or {}
        cfg = _deep_merge(common, cfg)
    return cfg
