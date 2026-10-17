"""Dataset loading and schema validation for IBM Telco Customer Churn."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

TARGET = "Churn"
DROP_COLUMNS = ("customerID",)


def load_telco_data(path: str | Path) -> pd.DataFrame:
    """Load the IBM Telco CSV and validate the columns required for modeling.

    Download the dataset separately and pass its local CSV path. The raw dataset is
    intentionally not committed to this repository.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the IBM Telco Customer Churn CSV "
            "and place it at data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv."
        )
    frame = pd.read_csv(path)
    required = {TARGET, "customerID", "TotalCharges"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame[TARGET] = frame[TARGET].map({"No": 0, "Yes": 1})
    if frame[TARGET].isna().any():
        raise ValueError("Churn target must contain only 'Yes' and 'No'.")
    return frame


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and binary target, excluding customer identifiers."""
    if TARGET not in frame.columns:
        raise ValueError(f"Target column {TARGET!r} is missing.")
    y = frame[TARGET].astype(int)
    X = frame.drop(columns=[TARGET, *[c for c in DROP_COLUMNS if c in frame.columns]])
    return X, y
