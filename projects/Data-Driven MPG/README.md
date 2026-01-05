# Data-Driven MPG

This project explores which vehicle characteristics influence fuel economy (MPG), and includes:
- Data exploration + visualizations in the notebook
- A regression model (linear / ridge) to identify impactful features
- A minimal MySQL schema + Python loader to store/query car model data

## Files

- `cars_clean.csv`: cleaned automobile dataset used for MPG analysis
- `Cars.ipynb`: notebook with exploration and regression modeling
- `mysql/schema.sql`: MySQL table definitions
- `mysql/sample_queries.sql`: sample analytic queries
- `scripts/mysql_ingest.py`: loads the dataset into MySQL

## MySQL (optional)

Set environment variables:

- `MYSQL_HOST` (default `127.0.0.1`)
- `MYSQL_PORT` (default `3306`)
- `MYSQL_USER` (default `root`)
- `MYSQL_PASSWORD` (default empty)
- `MYSQL_DATABASE` (default `cars_db`)

Create the DB + schema, then ingest:

- `python projects/Data-Driven\ MPG/scripts/mysql_ingest.py --create-schema`
