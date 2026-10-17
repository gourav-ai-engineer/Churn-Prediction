"""Train and evaluate a leakage-conscious baseline churn classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             precision_recall_curve, roc_auc_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_prediction.data import load_telco_data, split_features_target


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Build preprocessing and model as one sklearn Pipeline to prevent leakage."""
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])
    return Pipeline([
        ("preprocess", prep),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
    ])


def choose_threshold_by_cost(y_true, probabilities, false_negative_cost=5.0, false_positive_cost=1.0):
    """Choose a threshold minimizing explicit FN/FP business cost on validation data."""
    import numpy as np
    thresholds = np.linspace(0.05, 0.95, 91)
    best = {"threshold": 0.5, "cost": float("inf")}
    y = np.asarray(y_true)
    p = np.asarray(probabilities)
    for threshold in thresholds:
        pred = p >= threshold
        fn = int(((y == 1) & (~pred)).sum())
        fp = int(((y == 0) & pred).sum())
        cost = fn * false_negative_cost + fp * false_positive_cost
        if cost < best["cost"]:
            best = {"threshold": float(threshold), "cost": float(cost), "false_negatives": fn, "false_positives": fp}
    return best


def train(data_path: str | Path, output_dir: str | Path = "artifacts") -> dict:
    frame = load_telco_data(data_path)
    X, y = split_features_target(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Use a validation split from training data for threshold selection; test stays untouched.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    model = build_pipeline(X_fit)
    model.fit(X_fit, y_fit)
    val_probs = model.predict_proba(X_val)[:, 1]
    threshold_info = choose_threshold_by_cost(y_val, val_probs)
    test_probs = model.predict_proba(X_test)[:, 1]
    threshold = threshold_info["threshold"]
    test_pred = (test_probs >= threshold).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, test_probs)),
        "pr_auc": float(average_precision_score(y_test, test_probs)),
        "churn_recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "threshold": threshold,
        "threshold_selection_validation_cost": threshold_info["cost"],
        "classification_report": classification_report(y_test, test_pred, output_dict=True, zero_division=0),
        "note": "Metrics are measured on a held-out test set; targets are not guaranteed.",
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "churn_baseline.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print(classification_report(y_test, test_pred, target_names=["No churn", "Churn"], zero_division=0))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    train(args.data, args.output_dir)


if __name__ == "__main__":
    main()
