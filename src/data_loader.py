"""UCI Cleveland Heart Disease loader with local cache."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import get_path, load_config


def load_cleveland(config: dict | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for the Cleveland Heart Disease dataset.

    Downloads via ucimlrepo on first call and caches to data/raw/cleveland.csv.
    Target is binarised: 0 = no disease, 1 = disease (num > 0).
    """
    config = config or load_config()
    raw_dir: Path = get_path("raw_data", config)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / "cleveland.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        from ucimlrepo import fetch_ucirepo  # deferred — optional at import time

        dataset = fetch_ucirepo(id=45)
        df = dataset.data.features.copy()
        df["num"] = dataset.data.targets.values.ravel()
        df.to_csv(cache_path, index=False)

    y = (df["num"] > 0).astype(int).rename("target")
    X = df.drop(columns=["num"])

    return X, y


def feature_names(X: pd.DataFrame) -> dict[str, list[str]]:
    """Return feature name groupings used for ablation studies."""
    continuous = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    binary = ["sex", "fbs", "exang"]
    categorical = ["cp", "restecg", "slope", "thal", "ca"]
    return {"continuous": continuous, "binary": binary, "categorical": categorical}
