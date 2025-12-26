# Projects Portfolio

A recruiter-friendly portfolio of data analytics and data science work using Python and Jupyter notebooks.

## Repository layout

- `code/` — projects (each project has its own folder, notebook, and data)
- `code/karate/` — network-analysis project for the Karate Club graph
- `code/stellar_mapper/` — multi-file subproject (data + notebook + outputs)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open any notebook under `code/` in VS Code (or Jupyter) and run cells top-to-bottom.

## Keep notebooks output-free

This repo is intended to keep notebook outputs (cell results / execution counts) out of version control.
If you run notebooks locally, you can strip outputs before committing:

```bash
python scripts/strip_notebook_outputs.py
```

## Demo datasets (optional)

To add a couple of clean, reproducible datasets for demo projects:

```bash
python scripts/generate_demo_datasets.py
```

This writes:

- `code/customer_churn/data/customer_churn.csv`
- `code/retail_sales_forecasting/data/retail_sales_daily.csv`
- `code/ab_testing/data/ab_test_experiment.csv`
- `code/nlp_sentiment_topics/data/product_reviews.csv`

## Added projects

- `code/customer_churn.ipynb` — end-to-end churn modeling (EDA → features → baseline models)
- `code/retail_sales_forecasting.ipynb` — time series forecasting + anomaly detection on generated retail sales
- `code/ab_testing.ipynb` — A/B testing workflow (SRM, conversion uplift, revenue per user, segments)
- `code/geospatial_site_selection.ipynb` — geospatial clustering + candidate site selection with an interactive map
- `code/nlp_sentiment_topics.ipynb` — sentiment scoring + topic modeling (NMF) on reviews + text
- `code/karate/karate_network.ipynb` — quick network centrality + visualization (see also `code/karate/karate.html`)
## Notes

- Some notebooks use live data sources (e.g., OpenStreetMap / Overpass) and require an internet connection.
- Large/visual artifacts (e.g., PDFs) are included when they are part of the portfolio output.

If you run into installation issues with geospatial dependencies (e.g., `geopandas`), consider using Conda/Mamba
or installing system packages required by `pyproj`/`gdal` on your platform.
