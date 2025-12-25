"""Generate small, realistic synthetic datasets for portfolio projects.

Writes CSVs into ./data so notebooks can run without external downloads.

Usage:
  python scripts/generate_synthetic_datasets.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Paths:
    repo_root: Path

    @property
    def data_dir(self) -> Path:
        return self.repo_root / "data"

    @property
    def churn_csv(self) -> Path:
        return self.data_dir / "synthetic_customer_churn.csv"

    @property
    def retail_csv(self) -> Path:
        return self.data_dir / "synthetic_retail_sales_daily.csv"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_customer_churn(rng: np.random.Generator, n: int = 6000) -> pd.DataFrame:
    """Synthetic SaaS churn dataset with a realistic mix of numeric/categorical features."""

    plan = rng.choice(["basic", "pro", "enterprise"], size=n, p=[0.55, 0.35, 0.10])
    region = rng.choice(["NA", "EU", "APAC", "LATAM"], size=n, p=[0.45, 0.25, 0.20, 0.10])
    acquisition = rng.choice(["organic", "paid", "partner", "sales"], size=n, p=[0.50, 0.25, 0.10, 0.15])

    tenure_months = rng.integers(1, 61, size=n)
    seats = np.clip(rng.poisson(lam=8, size=n) + (plan == "enterprise") * rng.integers(10, 40, size=n), 1, None)
    monthly_price = (
        (plan == "basic") * rng.normal(25, 5, size=n)
        + (plan == "pro") * rng.normal(65, 10, size=n)
        + (plan == "enterprise") * rng.normal(220, 35, size=n)
    )
    monthly_price = np.clip(monthly_price, 10, None)

    tickets_last_90d = rng.poisson(lam=1.2, size=n) + (plan == "basic") * rng.poisson(lam=0.5, size=n)
    nps = np.clip(rng.normal(28, 25, size=n), -100, 100)

    weekly_active_days = np.clip(rng.normal(3.6, 1.4, size=n), 0, 7)
    feature_adoption = np.clip(rng.beta(2.2, 2.8, size=n), 0, 1)

    used_discount = rng.random(size=n) < (0.12 + 0.10 * (plan == "basic"))

    # Churn propensity model (logit). Tuned for ~15-30% churn depending on mix.
    plan_effect = np.select(
        [plan == "basic", plan == "pro", plan == "enterprise"],
        [0.35, 0.0, -0.25],
        default=0.0,
    )
    region_effect = np.select(
        [region == "NA", region == "EU", region == "APAC", region == "LATAM"],
        [0.0, 0.05, 0.10, 0.12],
        default=0.0,
    )
    acquisition_effect = np.select(
        [acquisition == "organic", acquisition == "paid", acquisition == "partner", acquisition == "sales"],
        [-0.05, 0.10, 0.00, -0.02],
        default=0.0,
    )

    logit = (
        -1.1
        + plan_effect
        + region_effect
        + acquisition_effect
        + 0.22 * (tickets_last_90d.astype(float) / 3.0)
        - 0.55 * feature_adoption
        - 0.20 * (weekly_active_days / 7.0)
        - 0.35 * (tenure_months / 60.0)
        - 0.18 * (nps / 100.0)
        + 0.18 * used_discount.astype(float)
        + 0.10 * np.log1p(seats) / np.log(50)
    )

    churn_prob = _sigmoid(logit)
    churned = rng.random(size=n) < churn_prob

    df = pd.DataFrame(
        {
            "customer_id": [f"C{idx:06d}" for idx in range(1, n + 1)],
            "plan": plan,
            "region": region,
            "acquisition_channel": acquisition,
            "tenure_months": tenure_months,
            "seats": seats,
            "monthly_price": np.round(monthly_price, 2),
            "tickets_last_90d": tickets_last_90d,
            "nps": np.round(nps, 1),
            "weekly_active_days": np.round(weekly_active_days, 2),
            "feature_adoption": np.round(feature_adoption, 3),
            "used_discount": used_discount.astype(int),
            "churned": churned.astype(int),
        }
    )

    return df


def make_retail_sales_daily(
    rng: np.random.Generator,
    start: str = "2023-01-01",
    end: str = "2025-12-31",
    n_stores: int = 8,
    n_products: int = 12,
) -> pd.DataFrame:
    """Synthetic daily retail sales with seasonality, holidays, promos, and injected anomalies."""

    dates = pd.date_range(start=start, end=end, freq="D")
    store_ids = [f"S{idx:02d}" for idx in range(1, n_stores + 1)]
    product_ids = [f"P{idx:02d}" for idx in range(1, n_products + 1)]

    base_store = rng.normal(1.0, 0.12, size=n_stores)
    base_product = rng.normal(1.0, 0.20, size=n_products)

    rows: list[dict[str, object]] = []

    for d in dates:
        dow = d.dayofweek
        week_factor = 1.0 + (dow >= 5) * 0.10  # weekends

        # yearly seasonality (higher near end of year)
        year_pos = (d.dayofyear - 1) / 365.0
        yearly = 1.0 + 0.18 * np.sin(2 * np.pi * year_pos) + 0.12 * np.cos(2 * np.pi * year_pos)

        # simple "holiday" windows
        is_black_friday_window = (d.month == 11) and (20 <= d.day <= 30)
        is_december = d.month == 12
        holiday_boost = 1.0 + (0.25 if is_black_friday_window else 0.0) + (0.20 if is_december else 0.0)

        for store_i, store_id in enumerate(store_ids):
            store_factor = base_store[store_i]

            # store-level promo days
            promo = rng.random() < 0.08
            promo_uplift = 1.0 + (0.18 if promo else 0.0)

            for prod_i, product_id in enumerate(product_ids):
                prod_factor = base_product[prod_i]

                price = np.clip(rng.normal(20, 7) * (1.0 + 0.3 * (prod_i / max(n_products - 1, 1))), 3, None)

                # units baseline
                lam = 18.0 * store_factor * prod_factor * week_factor * yearly * holiday_boost * promo_uplift
                lam = max(lam, 0.2)
                units = rng.poisson(lam=lam)

                # occasional stockout
                stockout = (rng.random() < 0.01)
                if stockout:
                    units = int(units * rng.uniform(0.0, 0.2))

                # injected anomalies: spikes and dips
                anomaly = "none"
                if rng.random() < 0.0025:
                    anomaly = "spike"
                    units = int(units * rng.uniform(2.2, 4.0))
                elif rng.random() < 0.0025:
                    anomaly = "dip"
                    units = int(units * rng.uniform(0.05, 0.4))

                revenue = units * price

                rows.append(
                    {
                        "date": d.date().isoformat(),
                        "store_id": store_id,
                        "product_id": product_id,
                        "price": round(float(price), 2),
                        "units": int(units),
                        "revenue": round(float(revenue), 2),
                        "promo": int(promo),
                        "stockout": int(stockout),
                        "holiday_window": int(is_black_friday_window or is_december),
                        "anomaly": anomaly,
                    }
                )

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = Paths(repo_root=repo_root)
    paths.data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    churn = make_customer_churn(rng)
    churn.to_csv(paths.churn_csv, index=False)

    retail = make_retail_sales_daily(rng)
    retail.to_csv(paths.retail_csv, index=False)

    print(f"Wrote: {paths.churn_csv} ({len(churn):,} rows)")
    print(f"Wrote: {paths.retail_csv} ({len(retail):,} rows)")


if __name__ == "__main__":
    main()
