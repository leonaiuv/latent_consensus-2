"""YAML 配置加载器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path) -> dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    inherited = raw_config.pop("inherits", None)

    if not inherited:
        return raw_config

    parent_path = config_path.parent / inherited
    parent_config = load_config(parent_path)
    return _deep_merge(parent_config, raw_config)
