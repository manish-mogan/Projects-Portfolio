# pyright: reportMissingImports=false

import os
from pathlib import Path
import argparse

import pandas as pd
import mysql.connector


def get_connection():
    host = os.environ.get("MYSQL_HOST", "127.0.0.1")
    port = int(os.environ.get("MYSQL_PORT", "3306"))
    user = os.environ.get("MYSQL_USER", "root")
    password = os.environ.get("MYSQL_PASSWORD", "")
    database = os.environ.get("MYSQL_DATABASE", "cars_db")

    return mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        autocommit=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Load cars_clean.csv into MySQL")
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).resolve().parents[1] / "cars_clean.csv"),
        help="Path to cars_clean.csv",
    )
    parser.add_argument(
        "--create-schema",
        action="store_true",
        help="Run mysql/schema.sql before inserting",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Map dataset columns (often hyphenated) to SQL column names (snake_case)
    rename_map = {
        "fuel-type": "fuel_type",
        "num-of-doors": "num_of_doors",
        "body-style": "body_style",
        "drive-wheels": "drive_wheels",
        "engine-location": "engine_location",
        "wheel-base": "wheel_base",
        "curb-weight": "curb_weight",
        "engine-type": "engine_type",
        "num-of-cylinders": "num_of_cylinders",
        "engine-size": "engine_size",
        "fuel-system": "fuel_system",
        "compression-ratio": "compression_ratio",
        "peak-rpm": "peak_rpm",
        "city-mpg": "city_mpg",
        "highway-mpg": "highway_mpg",
    }

    df = df.rename(columns=rename_map)

    cols = [
        "make",
        "fuel_type",
        "aspiration",
        "num_of_doors",
        "body_style",
        "drive_wheels",
        "engine_location",
        "wheel_base",
        "length",
        "width",
        "height",
        "curb_weight",
        "engine_type",
        "num_of_cylinders",
        "engine_size",
        "fuel_system",
        "bore",
        "stroke",
        "compression_ratio",
        "horsepower",
        "peak_rpm",
        "city_mpg",
        "highway_mpg",
        "price",
    ]

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}")

    df = df[cols].copy()

    conn = get_connection()
    try:
        cur = conn.cursor()

        if args.create_schema:
            schema_path = Path(__file__).resolve().parents[1] / "mysql" / "schema.sql"
            schema_sql = schema_path.read_text(encoding="utf-8")
            for stmt in [s.strip() for s in schema_sql.split(";") if s.strip()]:
                cur.execute(stmt)

        placeholders = ", ".join(["%s"] * len(cols))
        insert_sql = f"""
            INSERT INTO cars ({", ".join(cols)})
            VALUES ({placeholders})
        """.strip()

        rows = [tuple(None if pd.isna(v) else v for v in record) for record in df.itertuples(index=False)]

        cur.executemany(insert_sql, rows)
        conn.commit()

        print(f"Inserted {cur.rowcount} rows into cars")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
