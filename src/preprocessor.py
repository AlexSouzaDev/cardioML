"""Preprocessing pipeline: impute → encode → scale.

SMOTE is NOT part of this pipeline — it is applied separately in trainer.py
after the split to prevent leakage into validation/test folds.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_COLS = ["cp", "restecg", "slope", "thal"]
IMPUTED_COLS = ["ca", "thal"]
PASSTHROUGH_BINARY = ["sex", "fbs", "exang"]
PASSTHROUGH_CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def build_pipeline() -> Pipeline:
    """Build the preprocessing pipeline (fit on train fold only)."""

    # Impute missing values in ca and thal before encoding
    imputer = SimpleImputer(strategy="median")

    cat_transformer = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
        ]
    )

    ca_transformer = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
        ]
    )

    passthrough_cols = PASSTHROUGH_BINARY + PASSTHROUGH_CONTINUOUS

    col_transformer = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, CATEGORICAL_COLS),
            ("ca", ca_transformer, ["ca"]),
            ("passthrough", "passthrough", passthrough_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("col_transform", col_transformer),
            ("scaler", StandardScaler()),
        ]
    )

    return pipeline


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple:
    """Stratified train / val / test split.

    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    from sklearn.model_selection import train_test_split

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # val_size relative to the original dataset, so adjust fraction accordingly
    adjusted_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=adjusted_val, stratify=y_trainval, random_state=seed
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
