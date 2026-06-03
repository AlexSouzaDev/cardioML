"""SHAP-based interpretability: beeswarm, waterfall, feature importance, PDP."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap

from src.utils import get_path, load_config


def get_explainer(model, X_train_proc: np.ndarray):
    name = type(model).__name__
    if name in ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier"):
        return shap.TreeExplainer(model)
    elif name == "LogisticRegression":
        return shap.LinearExplainer(model, X_train_proc)
    else:
        background = shap.sample(X_train_proc, min(100, len(X_train_proc)))
        return shap.KernelExplainer(model.predict_proba, background)


def explain_model(
    model_name: str,
    model,
    pipe,
    X_train,
    X_test,
    feature_names: list[str],
    config: dict | None = None,
) -> None:
    """Generate and save all SHAP figures for a given model."""
    config = config or load_config()
    figures_dir: Path = get_path("reports_figures", config)
    figures_dir.mkdir(parents=True, exist_ok=True)

    X_train_proc = pipe.transform(X_train)
    X_test_proc = pipe.transform(X_test)

    explainer = get_explainer(model, X_train_proc)

    # Compute SHAP values — normalize to 2D (n_samples, n_features) for positive class.
    # SHAP >= 0.46 returns 3D ndarray (n_samples, n_features, n_classes) for tree models;
    # older versions return a list of per-class arrays; KernelExplainer always returns a list.
    shap_values = explainer.shap_values(X_test_proc)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]          # binary: positive class
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_vals = shap_values[:, :, 1]   # 3D → positive class slice
    else:
        shap_vals = shap_values             # already 2D

    _save_summary_plot(shap_vals, X_test_proc, feature_names, model_name, figures_dir)
    _save_importance_plot(shap_vals, feature_names, model_name, figures_dir)
    _save_waterfall_plots(explainer, X_test_proc, model_name, figures_dir)


def _save_summary_plot(
    shap_vals: np.ndarray,
    X_proc: np.ndarray,
    feature_names: list[str],
    model_name: str,
    figures_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_vals,
        X_proc,
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )
    plt.title(f"SHAP Beeswarm — {model_name}")
    plt.tight_layout()
    plt.savefig(figures_dir / f"shap_summary_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def _save_importance_plot(
    shap_vals: np.ndarray,
    feature_names: list[str],
    model_name: str,
    figures_dir: Path,
) -> None:
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:10]
    top_idx_asc = top_idx[::-1]  # ascending for barh (bottom = most important)
    feat_names_arr = np.asarray(feature_names)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        feat_names_arr[top_idx_asc].tolist(),
        mean_abs[top_idx_asc].tolist(),
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top-10 Feature Importance — {model_name}")
    fig.tight_layout()
    fig.savefig(figures_dir / f"shap_importance_{model_name}.png", dpi=150)
    plt.close(fig)


def _save_waterfall_plots(
    explainer,
    X_proc: np.ndarray,
    model_name: str,
    figures_dir: Path,
    n_samples: int = 3,
) -> None:
    # Save waterfall for the first n_samples test instances
    for i in range(min(n_samples, len(X_proc))):
        try:
            explanation = explainer(X_proc[i : i + 1])
            if hasattr(explanation, "values") and explanation.values.ndim == 3:
                # Multi-output: take positive class
                vals = explanation.values[:, :, 1]
                base = explanation.base_values[:, 1]
                exp = shap.Explanation(
                    values=vals[0],
                    base_values=base[0],
                    data=explanation.data[0],
                    feature_names=explanation.feature_names,
                )
            else:
                exp = explanation[0]

            shap.plots.waterfall(exp, show=False)
            plt.title(f"SHAP Waterfall — {model_name} — patient {i}")
            plt.tight_layout()
            plt.savefig(
                figures_dir / f"shap_waterfall_{model_name}_patient_{i}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
        except Exception as e:
            print(f"  Waterfall plot skipped for patient {i}: {e}")
