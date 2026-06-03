"""Compute metrics and save figures for the held-out test set."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from src.utils import get_path, load_config


def compute_metrics(y_test: np.ndarray, y_proba: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "auc_roc": roc_auc_score(y_test, y_proba),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "pr_auc": average_precision_score(y_test, y_proba),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "brier": brier_score_loss(y_test, y_proba),
    }


def save_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    figures_dir: Path,
) -> None:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    fig.savefig(figures_dir / f"confusion_matrix_{model_name}.png", dpi=150)
    plt.close(fig)


def save_roc_curve(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    figures_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name=model_name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"ROC Curve — {model_name}")
    fig.tight_layout()
    fig.savefig(figures_dir / f"roc_curve_{model_name}.png", dpi=150)
    plt.close(fig)


def save_pr_curve(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    figures_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax, name=model_name)
    ax.set_title(f"PR Curve — {model_name}")
    fig.tight_layout()
    fig.savefig(figures_dir / f"pr_curve_{model_name}.png", dpi=150)
    plt.close(fig)


def evaluate_model(
    model_name: str,
    model,
    pipe,
    X_test,
    y_test: np.ndarray,
    config: dict | None = None,
    cv_auc_mean: float | None = None,
    cv_auc_std: float | None = None,
) -> dict:
    """Run full evaluation: metrics + figures. Returns metrics dict."""
    config = config or load_config()
    figures_dir: Path = get_path("reports_figures", config)
    figures_dir.mkdir(parents=True, exist_ok=True)

    X_proc = pipe.transform(X_test)
    y_proba = model.predict_proba(X_proc)[:, 1]
    y_pred = model.predict(X_proc)

    metrics = compute_metrics(y_test, y_proba, y_pred)
    if cv_auc_mean is not None:
        metrics["cv_auc_mean"] = cv_auc_mean
    if cv_auc_std is not None:
        metrics["cv_auc_std"] = cv_auc_std

    save_confusion_matrix(y_test, y_pred, model_name, figures_dir)
    save_roc_curve(y_test, y_proba, model_name, figures_dir)
    save_pr_curve(y_test, y_proba, model_name, figures_dir)

    print(f"\n{'=' * 40}")
    print(f"  {model_name.upper()} — Test set results")
    print(f"{'=' * 40}")
    for k, v in metrics.items():
        print(f"  {k:<16} {v:.4f}")

    return metrics


def append_results(model_name: str, metrics: dict, config: dict | None = None) -> None:
    """Append a row to reports/tables/results.csv."""
    config = config or load_config()
    tables_dir: Path = get_path("reports_tables", config)
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_path = tables_dir / "results.csv"

    row = {"model": model_name, **metrics}
    df_new = pd.DataFrame([row])

    if results_path.exists():
        df_existing = pd.read_csv(results_path)
        df_existing = df_existing[df_existing["model"] != model_name]
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_csv(results_path, index=False)
    print(f"Results saved → {results_path}")
