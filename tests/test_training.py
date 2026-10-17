import numpy as np
import pandas as pd
from churn_prediction.train import build_pipeline, choose_threshold_by_cost


def test_pipeline_fits_mixed_types_and_predicts_probabilities():
    X = pd.DataFrame({
        "tenure": [1, 3, 12, 24, 2, 36, 48, 5],
        "MonthlyCharges": [80, 75, 40, 30, 90, 20, 25, 85],
        "Contract": ["Month-to-month", "Month-to-month", "One year", "Two year", "Month-to-month", "Two year", "Two year", "Month-to-month"],
    })
    y = pd.Series([1, 1, 0, 0, 1, 0, 0, 1])
    model = build_pipeline(X)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (8, 2)
    assert np.all((probs[:, 1] >= 0) & (probs[:, 1] <= 1))


def test_cost_threshold_returns_valid_threshold_and_counts():
    result = choose_threshold_by_cost([0, 1, 1, 0], [0.1, 0.8, 0.4, 0.7])
    assert 0.05 <= result["threshold"] <= 0.95
    assert result["false_negatives"] >= 0
    assert result["false_positives"] >= 0
