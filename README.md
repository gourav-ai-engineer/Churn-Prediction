# Customer Churn Prediction

An end-to-end machine-learning project for identifying telecom customers at risk of churn and supporting retention decisions. The repository is being built in milestones; the first milestone provides dataset validation, exploratory analysis, a leakage-conscious scikit-learn baseline, a cost-sensitive validation threshold, and tests.

## Current milestone: EDA + baseline

- IBM Telco Customer Churn CSV loader with target/schema checks.
- Exploratory notebook: missing values, target imbalance, numeric distributions, and segment-level churn rates.
- Preprocessing and Logistic Regression in one sklearn `Pipeline`: median imputation/scaling for numeric columns; most-frequent imputation and one-hot encoding for categorical columns.
- `class_weight="balanced"` to address class imbalance without synthesizing records.
- Stratified train/test split; a validation subset of the training data selects the threshold using an explicit false-negative/false-positive cost ratio. The test split is not used for threshold selection.
- Held-out ROC-AUC, PR-AUC, churn recall and classification report saved to `artifacts/metrics.json`.
- Unit tests for input validation, preprocessing, probability output, and threshold selection.

## Dataset setup

Download the IBM Telco Customer Churn dataset from Kaggle (commonly named `WA_Fn-UseC_-Telco-Customer-Churn.csv`). Do not commit the raw dataset. Place it here:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The loader converts `TotalCharges` to numeric (blank values become missing) and `Churn` from Yes/No to 1/0. The customer ID is excluded from training features.

## Local setup (Windows PowerShell)

Python 3.10–3.12 is recommended.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

If PowerShell blocks activation, use `\.venv\Scripts\python.exe -m pip install -e ".[dev]"` and run commands with `\.venv\Scripts\python.exe`.

## Run EDA and train the baseline

From the repository root:

```powershell
jupyter notebook notebooks/01_eda_baseline.ipynb
python -m churn_prediction.train
```

Or specify a different dataset/output location:

```powershell
python -m churn_prediction.train --data path/to/telco.csv --output-dir artifacts
```

Artifacts created locally:
- `artifacts/churn_baseline.joblib` — fitted sklearn pipeline.
- `artifacts/metrics.json` — held-out metrics and threshold metadata.

## Success criteria and honest reporting

ROC-AUC ≥ 0.84 and churn recall ≥ 0.75 are project targets, not guaranteed outcomes. Report the actual test metrics and the selected threshold. The default business-cost assumption assigns a false negative (missed churner) 5× the cost of a false positive (unnecessarily contacted customer); revise this ratio with retention-team input. Threshold selection is performed on validation data, not the final test set.

## Roadmap

1. **EDA + baseline (current):** data checks, notebook, Logistic Regression, tests, held-out metrics.
2. **Model improvement:** stratified cross-validation, Random Forest/XGBoost comparison, tuning, SHAP global/local explanations, MLflow experiment tracking and model registry.
3. **Productization:** FastAPI `POST /predict`, top-three customer-specific reasons, Streamlit UI, Docker Compose, CI, deployment.
4. **Stretch:** batch scoring and drift monitoring with Evidently.

## Important interpretation note

SHAP and feature attributions describe how a model arrived at a prediction; they do not establish that a feature causes churn. Retention actions should be tested and evaluated for business impact.
