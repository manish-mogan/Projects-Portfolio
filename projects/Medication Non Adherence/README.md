# Medication Non Adherence

Predict medication **non-adherence risk** across patient records using tree-based ML models (XGBoost, LightGBM) and explain predictions with **LIME**.

## Primary deliverable

- Notebook: `Medication Non Adherence.ipynb`

## Data

- File: `Medication_Non_Adherence.csv`
- Rows: 1,152 patient records
- Target column: `ADH` (in {-1, 1})
  - This project models **non-adherence** as `nonadherent = 1 if ADH == -1 else 0`

## What the notebook does

- Loads and cleans the dataset (drops accidental index column)
- Builds a preprocessing pipeline
  - Numeric: median imputation
  - Categorical: most-frequent imputation + one-hot encoding
- Trains two models
  - XGBoost (`xgboost.XGBClassifier`)
  - LightGBM (`lightgbm.LGBMClassifier`)
- Evaluates on a stratified holdout set (ROC AUC, Average Precision, F1, etc.)
- Runs LIME explanations on a sample of test rows and aggregates top predictors

## Output

The notebook can optionally write a JSON summary to:

- `run_summary.json`
