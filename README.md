# Projects Portfolio

A recruiter-friendly portfolio of data analytics and data science work using Python and Jupyter notebooks.

## Repository layout

- `code/` — each project lives in its own folder and includes its notebook + `data/`
- `scripts/` — helper scripts (dataset generation, notebook output stripping)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Open any notebook under `code/` in VS Code (or Jupyter) and run cells top-to-bottom.

## Keep notebooks output-free

This repo keeps notebook outputs (cell results / execution counts) out of version control. After running notebooks locally:

```bash
python scripts/strip_notebook_outputs.py
```

## Demo datasets (optional)

To (re)generate deterministic demo datasets used by a few projects:

```bash
python scripts/generate_demo_datasets.py
```

This writes datasets into project-local folders, for example:

- `code/customer_churn/data/customer_churn.csv`
- `code/retail_sales_forecasting/data/retail_sales_daily.csv`
- `code/ab_testing/data/ab_test_experiment.csv`
- `code/nlp_sentiment_topics/data/product_reviews.csv`
- `code/fraud_detection_anomaly/data/transactions.csv`
- `code/customer_segmentation_rfm/data/orders.csv`
- `code/cohort_retention_analysis/data/events.csv`

## Projects

- `code/ab_testing/ab_testing.ipynb` — A/B testing workflow (SRM checks, uplift, CUPED, multiple testing)
- `code/geospatial_site_selection/geospatial_site_selection.ipynb` — geospatial clustering + candidate site selection
- `code/nlp_sentiment_topics/nlp_sentiment_topics.ipynb` — sentiment scoring + topic modeling on reviews
- `code/customer_churn/customer_churn.ipynb` — churn modeling (EDA → features → baseline models)
- `code/retail_sales_forecasting/retail_sales_forecasting.ipynb` — time series forecasting on generated retail sales
- `code/fraud_detection_anomaly/fraud_detection_anomaly.ipynb` — anomaly detection / fraud scoring on transactions
- `code/customer_segmentation_rfm/customer_segmentation_rfm.ipynb` — RFM feature engineering + customer segmentation
- `code/cohort_retention_analysis/cohort_retention_analysis.ipynb` — cohort retention + survival-style views from events
- `code/karate/karate_network.ipynb` — network centrality + visualization (see also `code/karate/karate.html`)
- `code/stellar_mapper/stellar_map_builder.ipynb` — multi-file project mapping star catalog data (see `code/stellar_mapper/`)

## Notes

- Some notebooks use live data sources (e.g., OpenStreetMap / Overpass) and need an internet connection.
- If geospatial dependencies are difficult to install on your platform, Conda/Mamba can be easier than system pip.
