# Austin Housing Data Analysis

This folder contains a cleaned, reproducible version of the Austin Housing project.

## Structure

- `data/`: Source and cleaned datasets used by the notebooks.
- `notebooks/`: Analysis and visualization notebooks.
- `reports/`: Final deliverables.
  - `dashboard_presentation.html`: The presentation-ready dashboard export.
  - `exports/`: Optional generated chart JSON exports (kept empty by default).

## How to run

1. Create the Python environment (once per Codespace):
  - `./scripts/setup_venv.sh`
2. Open any notebook in `notebooks/` and select the `.venv` kernel.
3. Run from top to bottom.

## View the exported dashboard

The dashboard is a standalone HTML file:

- `reports/dashboard_presentation.html`

To serve it locally in the Codespace:

- `python3 -m http.server 8011`
- Open: `http://127.0.0.1:8011/projects/Austin%20Housing%20Data%20Analysis/reports/dashboard_presentation.html`

Notes:
- Notebooks read inputs from `../data/`.
- If you choose to export charts, they will write to `../reports/exports/`.
- Notebook outputs are stripped to keep the repo light; re-run cells to regenerate visuals.
