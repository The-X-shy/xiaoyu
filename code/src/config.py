"""Configuration loading and merging utilities."""

from __future__ import annotations

import yaml
import copy
from typing import Dict, Any


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Deep merge override into base config."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = merge_configs(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def load_experiment_config(path: str) -> Dict[str, Any]:
    """Load experiment config, merging with base config if specified."""
    cfg = load_config(path)
    base_path = cfg.get("experiment", {}).get("base_config")
    if base_path:
        base_cfg = load_config(base_path)
        cfg = merge_configs(base_cfg, cfg)
    return cfg
