"""5-fold CV training loop with Optuna tuning and MLflow tracking."""

import warnings
import numpy as np
import pandas as pd
import optuna
import mlflow
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

from src.models import build_model
from src.preprocessor import build_pipeline
from src.utils import load_config

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_search_space(trial: optuna.Trial, model_name: str, config: dict) -> dict:
    space = config["search_spaces"][model_name]

    if model_name == "logreg":
        return {
            "C": trial.suggest_float("C", space["C"][0], space["C"][1], log=True),
        }
    elif model_name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
            "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
            "min_samples_split": trial.suggest_int("min_samples_split", *space["min_samples_split"]),
        }
    elif model_name == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
            "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
            "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
            "subsample": trial.suggest_float("subsample", *space["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *space["colsample_bytree"]),
        }
    elif model_name == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
            "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
            "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
            "num_leaves": trial.suggest_int("num_leaves", *space["num_leaves"]),
        }
    elif model_name == "mlp":
        hl_choices = [tuple(x) for x in space["hidden_layer_sizes"]]
        idx = trial.suggest_categorical("hidden_layer_sizes_idx", list(range(len(hl_choices))))
        return {
            "hidden_layer_sizes": hl_choices[idx],
            "alpha": trial.suggest_float("alpha", *space["alpha"], log=True),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", *space["learning_rate_init"], log=True
            ),
        }
    else:
        raise ValueError(f"No search space for model: {model_name}")


def cross_val_train(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict,
    n_splits: int = 5,
    seed: int = 42,
    use_smote: bool = True,
) -> tuple[list[float], list[float]]:
    """Run stratified k-fold CV and return (auc_scores, f1_scores) per fold."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    smote = SMOTE(random_state=seed)
    auc_scores, f1_scores = [], []

    X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
    y_arr = np.asarray(y)

    for train_idx, val_idx in cv.split(X_df, y_arr):
        X_tr = X_df.iloc[train_idx]
        X_val = X_df.iloc[val_idx]
        y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

        pipe = build_pipeline()
        X_tr_proc = pipe.fit_transform(X_tr)
        X_val_proc = pipe.transform(X_val)

        if use_smote:
            X_tr_res, y_tr_res = smote.fit_resample(X_tr_proc, y_tr)
        else:
            X_tr_res, y_tr_res = X_tr_proc, y_tr

        model = build_model(model_name, params)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*feature names.*")
            model.fit(X_tr_res, y_tr_res)
            y_proba = model.predict_proba(X_val_proc)[:, 1]
            y_pred = model.predict(X_val_proc)

        auc_scores.append(roc_auc_score(y_val, y_proba))
        f1_scores.append(f1_score(y_val, y_pred, average="macro"))

    return auc_scores, f1_scores


def tune_model(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    config: dict,
    n_trials: int = 50,
    use_smote: bool = True,
) -> tuple[dict, float]:
    """Run Optuna study and return (best_params, best_auc)."""
    seed = config["training"]["seed"]
    n_splits = config["training"]["n_splits"]

    def objective(trial: optuna.Trial) -> float:
        params = get_search_space(trial, model_name, config)
        auc_scores, _ = cross_val_train(
            model_name, X, y, params, n_splits=n_splits, seed=seed, use_smote=use_smote
        )
        with mlflow.start_run(run_name=f"{model_name}_trial_{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_auc_roc_mean", float(np.mean(auc_scores)))
            mlflow.log_metric("cv_auc_roc_std", float(np.std(auc_scores)))
        return float(np.mean(auc_scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, study.best_value


def train_final_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: dict,
    seed: int = 42,
    use_smote: bool = True,
):
    """Fit preprocessor + SMOTE + model on the full training set and return (pipe, model)."""
    smote = SMOTE(random_state=seed)
    pipe = build_pipeline()
    X_df = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train
    X_proc = pipe.fit_transform(X_df)

    if use_smote:
        X_res, y_res = smote.fit_resample(X_proc, y_train)
    else:
        X_res, y_res = X_proc, y_train

    model = build_model(model_name, params)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*feature names.*")
        model.fit(X_res, y_res)

    return pipe, model
