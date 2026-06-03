"""Model registry and factory."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


MODEL_REGISTRY: dict = {
    "logreg": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "rf": RandomForestClassifier(n_estimators=200, random_state=42),
    "xgb": XGBClassifier(eval_metric="logloss", random_state=42),
    "lgbm": LGBMClassifier(verbose=-1, random_state=42),
    "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
}

MODEL_NAMES = list(MODEL_REGISTRY.keys())


def build_model(model_name: str, params: dict):
    """Instantiate a model from the registry with the given hyperparameters."""
    seed = 42
    if model_name == "logreg":
        return LogisticRegression(max_iter=1000, random_state=seed, **params)
    elif model_name == "rf":
        return RandomForestClassifier(random_state=seed, **params)
    elif model_name == "xgb":
        return XGBClassifier(eval_metric="logloss", random_state=seed, **params)
    elif model_name == "lgbm":
        return LGBMClassifier(verbose=-1, random_state=seed, **params)
    elif model_name == "mlp":
        return MLPClassifier(max_iter=500, random_state=seed, **params)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from {MODEL_NAMES}")


def get_default_model(model_name: str):
    """Return the default (untuned) instance from the registry."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {MODEL_NAMES}")
    return MODEL_REGISTRY[model_name]
