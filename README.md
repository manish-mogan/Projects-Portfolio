# Projects Portfolio

A portfolio of Python + Jupyter projects covering analytics, experimentation, forecasting, NLP, geospatial analysis, and network science.

## Repository layout

- `code/` — each project lives in its own folder with its notebook and any local data files it uses
- `scripts/` — helper scripts (notebook output stripping)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open any notebook under `code/` in VS Code (or Jupyter) and run cells top-to-bottom.

## Notebook outputs

Notebook outputs can be helpful for quickly reviewing results. To keep the repo fast, the helper script removes large widget metadata blobs by default.

If you want to also remove outputs + execution counts (for lightweight diffs), run:

```bash
python scripts/strip_notebook_outputs.py --strip-outputs
```

## Projects

- `code/AB Testing/ab_testing.ipynb` — A/B testing workflow (SRM checks, uplift estimation, CUPED, multiple testing)
- `code/Geospatial Site Selection/geospatial_site_selection.ipynb` — geospatial clustering + candidate site selection
- `code/NLP Sentiment Topics/nlp_sentiment_topics.ipynb` — sentiment scoring + topic modeling on reviews
- `code/Customer Churn/customer_churn.ipynb` — churn modeling (EDA → features → baseline models)
- `code/Retail Sales Forecasting/retail_sales_forecasting.ipynb` — time series forecasting
- `code/Fraud Detection Anomaly/fraud_detection_anomaly.ipynb` — anomaly detection / fraud scoring
- `code/Customer Segmentation RFM/customer_segmentation_rfm.ipynb` — RFM feature engineering + customer segmentation
- `code/Cohort Retention Analysis/cohort_retention_analysis.ipynb` — cohort retention analysis from event logs
- `code/Karate/karate_network.ipynb` — Karate Club network centrality + visualization (see also `code/Karate/karate.html`)
- `code/Stellar Mapper/stellar_map_builder.ipynb` — star catalog mapping (see folder for supporting files)

## Notes

- Some notebooks use live data sources (e.g., OpenStreetMap / Overpass) and require an internet connection.
- If geospatial dependencies are hard to install with pip on your platform, Conda/Mamba can be easier.
