"""
Model training module.

Implements:
- preprocessing pipeline (impute/encode/scale)
- model training with GridSearchCV
- MLflow experiment logging
- evaluation metrics
"""

import os
from typing import Dict, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42


def build_preprocess_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing pipeline for numeric/categorical columns."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    num_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ]
    )
    return preprocessor


def build_model(model_type: str = "logistic"):
    """Return estimator and parameter grid for hyperparameter search."""
    if model_type == "logistic":
        estimator = LogisticRegression(
            solver="liblinear", max_iter=1000, random_state=RANDOM_STATE
        )
        param_grid = {
            "model__C": [0.01, 0.1, 1.0, 10],
            "model__penalty": ["l1", "l2"],
        }
    elif model_type == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        )
        param_grid = {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
        }
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return estimator, param_grid


def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    """Compute evaluation metrics for classification."""
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probas),
    }
    return metrics


def train_with_mlflow(
    df: pd.DataFrame,
    target_col: str = "is_high_risk",
    experiment_name: str = "credit-risk-model",
    model_type: str = "logistic",
    test_size: float = 0.2,
    tracking_uri: Optional[str] = None,
    save_local_path: str = "./models/best_model",
) -> Tuple[object, Dict[str, float]]:
    """
    Train model with preprocessing, hyperparameter tuning, and MLflow logging.
    Returns trained model and metrics. Also saves a local model artifact for API fallback.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocess_pipeline(X_train)
    estimator, param_grid = build_model(model_type)

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=0,
    )

    with mlflow.start_run():
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        metrics = evaluate_model(best_model, X_test, y_test)

        mlflow.log_params(search.best_params_)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(best_model, artifact_path="model")

        # Save local artifact for API fallback
        os.makedirs(os.path.dirname(save_local_path), exist_ok=True)
        mlflow.sklearn.save_model(best_model, path=save_local_path)

    return best_model, metrics


def train_cli(
    data_path: str,
    target_col: str = "is_high_risk",
    model_type: str = "logistic",
    tracking_uri: Optional[str] = None,
    save_local_path: str = "./models/best_model",
):
    """Convenience entry point for CLI usage."""
    df = pd.read_csv(data_path)
    model, metrics = train_with_mlflow(
        df=df,
        target_col=target_col,
        model_type=model_type,
        tracking_uri=tracking_uri,
        save_local_path=save_local_path,
    )
    print(f"Training complete. Metrics: {metrics}")
    return model, metrics


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/train.py <data_path> [target_col] [model_type]")
        sys.exit(1)

    data_path = sys.argv[1]
    target_col = sys.argv[2] if len(sys.argv) > 2 else "is_high_risk"
    model_type = sys.argv[3] if len(sys.argv) > 3 else "logistic"
    train_cli(data_path, target_col=target_col, model_type=model_type)
