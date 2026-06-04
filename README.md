# cardioML

cardioML is a machine learning research project for predicting the presence of heart disease from the UCI Cleveland Heart Disease dataset. It trains and compares five classification models, evaluates them on a held-out test set, saves charts and metrics, tracks experiments with MLflow, and can generate SHAP explanations that show which input features influenced the model.

This project is for education, experimentation, and reproducible ML research. It is not a medical device, not a clinical diagnosis tool, and should not be used to make health decisions.

## Public-readiness status

The project is useful and runnable as a research command-line project, but it is not fully ready as a public end-user product.

Ready now:

- Clear Python source structure under `src/`.
- MIT license included.
- Local Cleveland dataset cache included at `data/raw/cleveland.csv`.
- Training, evaluation, model comparison, optional hyperparameter tuning, MLflow tracking, and SHAP explainability are implemented.
- Generated models, figures, and result tables are written to predictable folders.

Needs work before a polished public release:

- Add automated tests for data loading, preprocessing, training, and CLI commands.
- Add continuous integration, for example GitHub Actions.
- Pin exact dependency versions with a lock file for stronger reproducibility.
- Remove committed generated MLflow run artifacts from version control history if publishing a clean repository.
- Add a simple user interface or API if the goal is non-technical public use.
- Add dataset citation and licensing notes from the original UCI dataset source.
- Add clinical validation, privacy review, bias review, monitoring, and regulatory review before any real healthcare use.

## What the project does in plain language

The project takes historical patient measurements, such as age, cholesterol, chest pain type, maximum heart rate, and exercise-related symptoms. It then teaches several machine learning models to recognize patterns associated with heart disease.

After training, the project checks how well each model performs on data it did not train on. This helps estimate whether the model learned useful patterns instead of only memorizing the examples.

The output is not a doctor's answer. It is a research prediction from a small public dataset.

## Dataset

The project uses the Cleveland subset of the UCI Heart Disease dataset.

Current cached file:

- `data/raw/cleveland.csv`
- 303 rows
- 14 columns
- Original target column: `num`

The original `num` target has values from 0 to 4:

- `0` means no diagnosed heart disease in the dataset label.
- `1`, `2`, `3`, and `4` indicate different levels of diagnosed disease.

The code converts this into a binary machine learning target:

- `0`: no heart disease
- `1`: heart disease present, meaning original `num > 0`

Missing values are present in:

- `ca`: 4 missing values
- `thal`: 2 missing values

The preprocessing pipeline imputes these values during training.

## Features used by the models

The input columns are:

| Column | Meaning in simple terms |
| --- | --- |
| `age` | Patient age |
| `sex` | Biological sex encoded as a number |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar indicator |
| `restecg` | Resting electrocardiographic result |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina indicator |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy |
| `thal` | Thalassemia-related encoded result |

## Models

The project can train these models:

| Command name | Model |
| --- | --- |
| `logreg` | Logistic Regression |
| `rf` | Random Forest |
| `xgb` | XGBoost |
| `lgbm` | LightGBM |
| `mlp` | Multi-layer Perceptron neural network |

## How the pipeline works

1. Load the Cleveland dataset from `data/raw/cleveland.csv`.
2. Convert the target into a binary label: no disease or disease.
3. Split the dataset into train, validation, and test sets using stratified sampling.
4. Build a preprocessing pipeline:
   - Fill missing categorical values with the most common value.
   - Fill missing numeric `ca` values with the median.
   - One-hot encode categorical columns.
   - Pass binary and continuous columns through.
   - Scale the final feature matrix with `StandardScaler`.
5. Optionally apply SMOTE to the training fold only.
6. Train one model or all supported models.
7. Evaluate on the held-out test set.
8. Save metrics, plots, trained model files, and MLflow experiment data.
9. Optionally generate SHAP plots to explain model behavior.

SMOTE is applied after preprocessing and only inside training data. This is important because applying oversampling before the split would leak information from training into validation or test data.

## Project structure

```text
cardioML/
|-- config.yaml              # Paths, training settings, and Optuna search spaces
|-- main.py                  # Command-line entrypoint
|-- requirements.txt         # Python dependencies
|-- data/
|   `-- raw/
|       `-- cleveland.csv    # Cached Cleveland dataset
|-- src/
|   |-- data_loader.py       # Loads and prepares the dataset target
|   |-- preprocessor.py      # Train/validation/test split and preprocessing pipeline
|   |-- models.py            # Model registry and model factory
|   |-- trainer.py           # Cross-validation, SMOTE, Optuna tuning, final training
|   |-- evaluator.py         # Metrics and evaluation plots
|   |-- explainer.py         # SHAP explanations
|   `-- utils.py             # Config, paths, and random seed helpers
|-- experiments/             # Generated MLflow DB and saved models
`-- reports/                 # Generated charts and result tables
```

## Install

You need Python 3.10 or newer. Python 3.11 is a good choice for this project.

Open a terminal in the project folder:

```bash
cd /path/to/cardioML
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Quick start for non-technical users

If you only want to try the project, run this command:

```bash
python main.py --all-models
```

That command trains all five models and prints a comparison table. It also saves files you can inspect later:

- Result table: `reports/tables/results.csv`
- Charts: `reports/figures/`
- Saved model files: `experiments/saved_models/`
- MLflow experiment database: `experiments/mlflow.db`

To train only one model, run:

```bash
python main.py --model rf
```

In that example, `rf` means Random Forest.

## Commands

Train one model:

```bash
python main.py --model logreg
python main.py --model rf
python main.py --model xgb
python main.py --model lgbm
python main.py --model mlp
```

Train all models:

```bash
python main.py --all-models
```

Train all models without SMOTE:

```bash
python main.py --all-models --no-smote
```

Tune hyperparameters with Optuna:

```bash
python main.py --model xgb --tune
```

Use fewer tuning trials for a faster test run:

```bash
python main.py --model xgb --tune --n-trials 10
```

Generate SHAP explanations:

```bash
python main.py --model rf --explain
```

Evaluate a previously saved model:

```bash
python main.py --model rf --eval-only
```

Set a different random seed:

```bash
python main.py --all-models --seed 123
```

## Understanding the results

The project reports these metrics:

| Metric | Simple explanation |
| --- | --- |
| `auc_roc` | How well the model separates positive and negative cases across thresholds. Higher is better. |
| `f1_macro` | Balance between precision and recall, treating both classes evenly. Higher is better. |
| `pr_auc` | Precision-recall area, useful when classes are imbalanced. Higher is better. |
| `mcc` | Matthews correlation coefficient, a balanced classification score. Higher is better. |
| `brier` | Measures probability calibration error. Lower is better. |
| `cv_auc_mean` | Average cross-validation AUC before final test evaluation. Higher is better. |
| `cv_auc_std` | Variation in cross-validation AUC. Lower usually means more stable results. |

Example local results from this repository state:

| Model | Test AUC-ROC | F1 Macro | PR-AUC | MCC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9295 | 0.8258 | 0.8974 | 0.6547 | 0.1082 |
| Random Forest | 0.9267 | 0.8693 | 0.8734 | 0.7419 | 0.1168 |
| XGBoost | 0.9048 | 0.8258 | 0.8425 | 0.6547 | 0.1304 |
| LightGBM | 0.9181 | 0.8478 | 0.9009 | 0.7028 | 0.1194 |
| MLP | 0.9048 | 0.7822 | 0.8815 | 0.5674 | 0.1517 |

These numbers can change if dependencies, random seeds, tuning settings, or data change.

## Viewing experiment history with MLflow

After running training, start the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```

MLflow lets you compare runs, parameters, and metrics.

## SHAP explanations

SHAP plots help explain which features influenced the model.

When you run:

```bash
python main.py --model rf --explain
```

the project saves explanation images in `reports/figures/`, including:

- `shap_summary_rf.png`
- `shap_importance_rf.png`
- `shap_waterfall_rf_patient_0.png`
- `shap_waterfall_rf_patient_1.png`
- `shap_waterfall_rf_patient_2.png`

Plain-language interpretation:

- A summary plot shows which features generally matter most.
- An importance plot ranks the strongest features.
- A waterfall plot explains one individual prediction.

These explanations describe model behavior. They do not prove medical causation.

## Configuration

Most settings live in `config.yaml`.

Important fields:

| Field | Meaning |
| --- | --- |
| `paths.raw_data` | Where raw data is stored |
| `paths.experiments` | Where MLflow and saved models are stored |
| `paths.reports_figures` | Where charts are saved |
| `paths.reports_tables` | Where result tables are saved |
| `training.seed` | Random seed for reproducibility |
| `training.n_splits` | Number of cross-validation folds |
| `training.test_size` | Fraction of data used for final test |
| `training.val_size` | Fraction of data used for validation |
| `training.n_trials` | Default number of Optuna tuning trials |
| `search_spaces` | Hyperparameter ranges for each model |

## For developers

The entrypoint is `main.py`. It parses CLI arguments, loads configuration, loads data, creates the train/validation/test split, and calls `run_model()` for each selected model.

Important implementation details:

- `src/data_loader.py` loads `data/raw/cleveland.csv` if it exists.
- If the cache is missing, `src/data_loader.py` tries to fetch the dataset with `ucimlrepo`.
- `src/preprocessor.py` builds a scikit-learn `Pipeline` with a `ColumnTransformer`.
- `src/trainer.py` performs stratified k-fold cross-validation and final model training.
- `src/evaluator.py` computes metrics and saves ROC, precision-recall, and confusion matrix plots.
- `src/explainer.py` uses TreeExplainer, LinearExplainer, or KernelExplainer depending on the model type.
- `src/models.py` centralizes supported model names and constructors.

## Current limitations

- The dataset is small, with only 303 rows.
- The data is historical and may not represent modern populations or clinical workflows.
- Some features are encoded medical variables that may be confusing without the original dataset documentation.
- The project has no automated tests yet.
- The project has no web app, desktop app, or REST API.
- The project does not include clinical validation.
- The model output is not calibrated for real patient care.

## Troubleshooting

If you see `ModuleNotFoundError`, install the dependencies:

```bash
pip install -r requirements.txt
```

If `lightgbm` fails to install, update `pip` first:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `--eval-only` says no saved model was found, train that model first:

```bash
python main.py --model rf
python main.py --model rf --eval-only
```

If SHAP is slow for `mlp`, that is expected. Neural-network explanations use a slower SHAP explainer than tree models.

## License

This project is released under the MIT License. See `LICENSE` for details.

## Responsible use

cardioML is a learning and research project. Do not use it as a substitute for professional medical advice, diagnosis, treatment, triage, or emergency care. If this project is ever adapted for healthcare use, it needs clinical review, validation on representative data, documented risk controls, privacy safeguards, and regulatory assessment.
