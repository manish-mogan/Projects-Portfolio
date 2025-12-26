# Projects Portfolio

A portfolio of Python + Jupyter projects covering analytics, experimentation, forecasting, NLP, geospatial analysis, and network science.

## Repository layout

- `code/` — each project lives in its own folder with its notebook and a `data/` subfolder
- `scripts/` — helper scripts (dataset generation, notebook output stripping)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open any notebook under `code/` in VS Code (or Jupyter) and run cells top-to-bottom.

## Keep notebooks output-free

Notebook outputs (execution counts, cell outputs) are kept out of version control. If you ran notebooks locally, strip outputs before committing:

```bash
python scripts/strip_notebook_outputs.py
```

## Demo datasets (optional)

Some projects include deterministic, generated datasets for repeatable results. To (re)generate them:

```bash
python scripts/generate_demo_datasets.py
```

Datasets are written into each project’s local `data/` folder (examples):

- `code/customer_churn/data/customer_churn.csv`
- `code/retail_sales_forecasting/data/retail_sales_daily.csv`
- `code/ab_testing/data/ab_test_experiment.csv`
- `code/nlp_sentiment_topics/data/product_reviews.csv`
- `code/fraud_detection_anomaly/data/transactions.csv`
- `code/customer_segmentation_rfm/data/orders.csv`
- `code/cohort_retention_analysis/data/events.csv`

## Projects

- `code/ab_testing/ab_testing.ipynb` — A/B testing workflow (SRM checks, uplift estimation, CUPED, multiple testing)
- `code/geospatial_site_selection/geospatial_site_selection.ipynb` — geospatial clustering + candidate site selection
- `code/nlp_sentiment_topics/nlp_sentiment_topics.ipynb` — sentiment scoring + topic modeling on reviews
- `code/customer_churn/customer_churn.ipynb` — churn modeling (EDA → features → baseline models)
- `code/retail_sales_forecasting/retail_sales_forecasting.ipynb` — time series forecasting on generated retail sales
- `code/fraud_detection_anomaly/fraud_detection_anomaly.ipynb` — anomaly detection / fraud scoring on transactions
- `code/customer_segmentation_rfm/customer_segmentation_rfm.ipynb` — RFM feature engineering + customer segmentation
- `code/cohort_retention_analysis/cohort_retention_analysis.ipynb` — cohort retention analysis from event logs
- `code/karate/karate_network.ipynb` — Karate Club network centrality + visualization (see also `code/karate/karate.html`)
- `code/stellar_mapper/stellar_map_builder.ipynb` — star catalog mapping (see `code/stellar_mapper/` for supporting files)

## Notes

- Some notebooks use live data sources (e.g., OpenStreetMap / Overpass) and require an internet connection.
- If geospatial dependencies are hard to install with pip on your platform, Conda/Mamba can be easier.
