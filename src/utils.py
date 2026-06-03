import random
from pathlib import Path

import numpy as np
import yaml


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_path(key: str, config: dict | None = None) -> Path:
    if config is None:
        config = load_config()
    root = get_project_root()
    return root / config["paths"][key]
