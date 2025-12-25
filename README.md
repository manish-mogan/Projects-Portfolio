# Projects Portfolio

A recruiter-friendly portfolio of data analytics and data science work using Python and Jupyter notebooks.

## Repository layout

- `code/` — notebooks (EDA, visualization, modeling, story-driven analyses)
- `code/stellar_mapper/` — multi-file subproject (data + notebook + outputs)
- `data/` — local datasets used by notebooks (including synthetic datasets)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open any notebook under `code/` in VS Code (or Jupyter) and run cells top-to-bottom.

## Synthetic datasets (optional)

To add a couple of clean, reproducible datasets for demo projects:

```bash
python scripts/generate_synthetic_datasets.py
```

This writes:

- `data/synthetic_customer_churn.csv`
- `data/synthetic_retail_sales_daily.csv`

## Added projects

- `code/synthetic_customer_churn.ipynb` — end-to-end churn modeling (EDA → features → baseline models)
- `code/synthetic_retail_sales_forecasting.ipynb` — time series forecasting + anomaly detection on synthetic retail sales

## Notes

- Some notebooks use live data sources (e.g., OpenStreetMap / Overpass) and require an internet connection.
- Large/visual artifacts (e.g., PDFs) are included when they are part of the portfolio output.
