"""YAML configuration loader."""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

LOGGER = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        RuntimeError: If the config cannot be parsed.
    """
    config_file_path = Path(config_path)

    if not config_file_path.exists():
        raise FileNotFoundError(
            f"Config file not found at: {config_file_path}"
        )

    try:
        with open(config_file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to load configuration")
        raise RuntimeError("Unable to parse configuration") from exc
