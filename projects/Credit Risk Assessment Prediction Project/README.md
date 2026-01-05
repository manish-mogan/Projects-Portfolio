# Credit Risk Assessment Prediction Project

This project trains a model to predict credit card default (`credit_card_default`).

## Files

- `train.csv`: training data with target column `credit_card_default`
- `test.csv`: test data without the target
- `credit_risk.ipynb`: end-to-end workflow (EDA → model → evaluation → predictions)

## How to run

1. Create/update the Python environment:
   - `./scripts/setup_venv.sh`
2. Open `credit_risk.ipynb` and select the `.venv` kernel.
3. Run cells top-to-bottom.

## Output

Running the notebook generates:
- `submission_predictions.csv` (probabilities + 0.5-threshold predictions)

