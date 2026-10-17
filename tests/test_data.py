import pandas as pd
import pytest
from churn_prediction.data import load_telco_data, split_features_target


def test_loader_converts_target_and_total_charges(tmp_path):
    path = tmp_path / "telco.csv"
    pd.DataFrame({
        "customerID": ["a", "b"], "TotalCharges": [" ", "25.5"],
        "tenure": [1, 2], "Contract": ["Month-to-month", "One year"],
        "Churn": ["Yes", "No"],
    }).to_csv(path, index=False)
    frame = load_telco_data(path)
    assert frame["Churn"].tolist() == [1, 0]
    assert pd.isna(frame.loc[0, "TotalCharges"])
    X, y = split_features_target(frame)
    assert "customerID" not in X.columns
    assert y.tolist() == [1, 0]


def test_loader_rejects_unknown_target(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"customerID": ["a"], "TotalCharges": ["10"], "Churn": ["Maybe"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="only 'Yes' and 'No'"):
        load_telco_data(path)


def test_missing_file_has_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="Download the IBM Telco"):
        load_telco_data(tmp_path / "missing.csv")
