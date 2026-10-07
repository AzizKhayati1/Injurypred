"""Configuration loading.

A single YAML file (``configs/config.yaml``) drives every stage of the pipeline.
The path can be overridden with the ``PLAYERLOAD_CONFIG`` environment variable.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config. Relative paths inside ``paths`` are resolved against the project root."""
    path = Path(path or os.environ.get("PLAYERLOAD_CONFIG", DEFAULT_CONFIG_PATH))
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg = copy.deepcopy(cfg)
    resolved = {}
    for key, value in cfg.get("paths", {}).items():
        p = Path(value)
        resolved[key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    # Environment overrides used by Docker.
    if os.environ.get("MODEL_DIR"):
        resolved["model_dir"] = os.environ["MODEL_DIR"]
    if os.environ.get("DATA_DIR"):
        resolved["synthetic_dir"] = os.environ["DATA_DIR"]
    cfg["paths"] = resolved
    return cfg


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    return load_config()


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable hash of the configuration (paths excluded, they are machine specific)."""
    cfg = {k: v for k, v in cfg.items() if k != "paths"}
    payload = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:12]
