# Projects Portfolio

A portfolio of Python + Jupyter projects covering analytics, experimentation, forecasting, NLP, geospatial analysis, and network science.

## Repository layout

- `projects/` — each project lives in its own folder with its notebook and any local data files it uses

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open any notebook under `projects/` in VS Code (or Jupyter) and run cells top-to-bottom.

## Notebook outputs

Notebook outputs are committed so the rendered results are visible.

## Projects

- `projects/AB Testing/AB Testing.ipynb` — A/B testing workflow (SRM checks, uplift estimation, CUPED, multiple testing)
- `projects/Austin Housing Data Analysis/` — Austin housing analysis project (cleaned structure + final dashboard)
-   - Notebooks: `projects/Austin Housing Data Analysis/notebooks/`
-   - Dashboard: `projects/Austin Housing Data Analysis/reports/dashboard_presentation.html` (serve locally for full interactivity)
- `projects/Geospatial Site Selection/Geospatial Site Selection.ipynb` — geospatial clustering + candidate site selection
- `projects/NLP/NLP Sentiment Topics.ipynb` — sentiment scoring + topic modeling on reviews
- `projects/Customer Churn/Customer Churn.ipynb` — churn modeling (EDA → features → baseline models)
- `projects/Retail Sales Forecasting/Retail Sales Forecasting.ipynb` — time series forecasting
- `projects/Fraud Detection Anomaly/Fraud Detection Anomaly.ipynb` — anomaly detection / fraud scoring
- `projects/Customer Segmentation RFM/Customer Segmentation RFM.ipynb` — RFM feature engineering + customer segmentation
- `projects/Cohort Retention Analysis/Cohort Retention Analysis.ipynb` — cohort retention analysis from event logs
- `projects/Karate/Karate.ipynb` — Karate Club network centrality + visualization (see also `projects/Karate/karate.html`)
- `projects/Stellar Mapper/Stellar Mapper.ipynb` — star catalog mapping (see folder for supporting files)

## Notes

- Some notebooks use live data sources (e.g., OpenStreetMap / Overpass) and require an internet connection.
- If geospatial dependencies are hard to install with pip on your platform, Conda/Mamba can be easier.

### Viewing the Austin dashboard

From the repo root:

```bash
./scripts/setup_venv.sh
python -m http.server 8000
```

Then open:

`http://127.0.0.1:8000/projects/Austin%20Housing%20Data%20Analysis/reports/dashboard_presentation.html`
