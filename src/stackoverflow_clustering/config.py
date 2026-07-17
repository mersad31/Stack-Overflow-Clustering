from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load YAML configuration and attach the absolute project root."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_project_root"] = str(config_path.parent)
    config["_config_path"] = str(config_path)
    return config


def project_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a configured path relative to the project root."""
    return Path(config["_project_root"]) / config["paths"][key]


def ensure_output_directories(config: dict[str, Any]) -> None:
    """Create every generated-data and report directory."""
    for key in (
        "raw_dir",
        "interim_dir",
        "processed_dir",
        "artifacts_dir",
        "figures_dir",
        "tables_dir",
        "notebooks_dir",
    ):
        project_path(config, key).mkdir(parents=True, exist_ok=True)

