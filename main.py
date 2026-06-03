"""CLI entrypoint for the Heart Disease ML Research Agent."""

import argparse
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from src.data_loader import load_cleveland
from src.evaluator import append_results, evaluate_model
from src.explainer import explain_model
from src.models import MODEL_NAMES
from src.preprocessor import split_data
from src.trainer import cross_val_train, train_final_model, tune_model
from src.utils import get_path, load_config, set_seed



def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Heart Disease ML Research Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        choices=MODEL_NAMES,
        help="Single model to train/evaluate",
    )
    group.add_argument(
        "--all-models",
        action="store_true",
        help="Run full comparison across all 5 models",
    )
    parser.add_argument("--tune", action="store_true", help="Enable Optuna hyperparameter search")
    parser.add_argument("--explain", action="store_true", help="Generate SHAP outputs after evaluation")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, load best saved model")
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE oversampling")
    parser.add_argument("--n-trials", type=int, default=None, help="Override Optuna trial count")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args(argv)


def get_feature_names_from_pipe(pipe, X_sample) -> list[str]:
    """Extract output feature names from the ColumnTransformer after fitting."""
    try:
        return pipe.named_steps["col_transform"].get_feature_names_out().tolist()
    except Exception:
        n_features = pipe.transform(X_sample[:1]).shape[1]
        return [f"f{i}" for i in range(n_features)]


def run_model(
    model_name: str,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    config: dict,
    tune: bool,
    explain: bool,
    eval_only: bool,
    use_smote: bool,
    n_trials: int,
    seed: int,
    experiments_dir: Path,
) -> dict:
    model_dir = experiments_dir / "saved_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    pipe_path = model_dir / f"pipe_{model_name}.pkl"
    model_path = model_dir / f"model_{model_name}.pkl"

    mlflow.set_experiment("heart_disease_classification")

    if eval_only:
        if not pipe_path.exists() or not model_path.exists():
            print(f"[{model_name}] No saved model found at {model_dir}. Run without --eval-only first.")
            sys.exit(1)
        pipe = joblib.load(pipe_path)
        model = joblib.load(model_path)
        cv_auc_mean, cv_auc_std = None, None
    else:
        X_trainval = pd.concat([X_train, X_val], ignore_index=True)
        y_trainval = np.concatenate([y_train.values, y_val.values], axis=0)

        with mlflow.start_run(run_name=f"{model_name}_run"):
            if tune:
                print(f"\n[{model_name}] Tuning with Optuna ({n_trials} trials) …")
                best_params, best_auc = tune_model(
                    model_name, X_trainval, y_trainval, config, n_trials=n_trials, use_smote=use_smote
                )
                cv_auc_mean = best_auc
                cv_auc_std = None
                print(f"[{model_name}] Best CV AUC-ROC: {best_auc:.4f}")
                mlflow.log_params(best_params)
                mlflow.log_metric("best_cv_auc", best_auc)
            else:
                best_params = {}
                print(f"\n[{model_name}] Running 5-fold CV …")
                auc_scores, f1_scores = cross_val_train(
                    model_name,
                    X_trainval,
                    y_trainval,
                    best_params,
                    n_splits=config["training"]["n_splits"],
                    seed=seed,
                    use_smote=use_smote,
                )
                cv_auc_mean = float(np.mean(auc_scores))
                cv_auc_std = float(np.std(auc_scores))
                print(f"[{model_name}] CV AUC-ROC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")
                mlflow.log_params(best_params)
                mlflow.log_metric("cv_auc_roc_mean", cv_auc_mean)
                mlflow.log_metric("cv_auc_roc_std", cv_auc_std)

            print(f"[{model_name}] Fitting final model on train+val …")
            pipe, model = train_final_model(
                model_name, X_trainval, y_trainval, best_params, seed=seed, use_smote=use_smote
            )
            joblib.dump(pipe, pipe_path)
            joblib.dump(model, model_path)
            mlflow.log_artifact(str(model_path), artifact_path="model")

    metrics = evaluate_model(
        model_name, model, pipe, X_test, y_test.values, config,
        cv_auc_mean=cv_auc_mean, cv_auc_std=cv_auc_std,
    )
    append_results(model_name, metrics, config)

    if explain:
        print(f"[{model_name}] Generating SHAP explanations …")
        feature_names = get_feature_names_from_pipe(pipe, X_train)
        explain_model(
            model_name, model, pipe,
            X_train, X_test,
            feature_names=feature_names,
            config=config,
        )

    return metrics


def main(argv=None) -> None:
    args = parse_args(argv)
    config = load_config()

    seed = args.seed if args.seed is not None else config["training"]["seed"]
    n_trials = args.n_trials if args.n_trials is not None else config["training"]["n_trials"]
    use_smote = not args.no_smote

    set_seed(seed)

    experiments_dir: Path = get_path("experiments", config)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{experiments_dir.resolve()}/mlflow.db")

    print("Loading Cleveland Heart Disease dataset …")
    X, y = load_cleveland(config)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        test_size=config["training"]["test_size"],
        val_size=config["training"]["val_size"],
        seed=seed,
    )
    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    models_to_run = MODEL_NAMES if args.all_models else [args.model]
    all_metrics: dict[str, dict] = {}

    for model_name in models_to_run:
        all_metrics[model_name] = run_model(
            model_name=model_name,
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            config=config,
            tune=args.tune,
            explain=args.explain,
            eval_only=args.eval_only,
            use_smote=use_smote,
            n_trials=n_trials,
            seed=seed,
            experiments_dir=experiments_dir,
        )

    if len(models_to_run) > 1:
        print("\n" + "=" * 60)
        print("  COMPARISON SUMMARY")
        print("=" * 60)
        header = f"{'Model':<10} {'AUC-ROC':>9} {'F1-macro':>9} {'PR-AUC':>9} {'MCC':>7} {'Brier':>7}"
        print(header)
        print("-" * 60)
        for name, m in all_metrics.items():
            print(
                f"{name:<10} {m.get('auc_roc', 0):>9.4f} {m.get('f1_macro', 0):>9.4f} "
                f"{m.get('pr_auc', 0):>9.4f} {m.get('mcc', 0):>7.4f} {m.get('brier', 0):>7.4f}"
            )


if __name__ == "__main__":
    main()
