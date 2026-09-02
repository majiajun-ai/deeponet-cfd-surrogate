"""Configuration loading helpers for the v2 experiment matrix."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .common import resolve_path


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path_value: str | Path) -> dict[str, Any]:
    path = resolve_path(path_value)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base_value = payload.pop("base_config", None)
    if not base_value:
        return payload
    base_path = Path(base_value)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    return _merge(load_config(base_path), payload)
