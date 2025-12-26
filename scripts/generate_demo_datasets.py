"""Generate small, realistic demo datasets for portfolio projects.

Writes datasets into each project's local folder under:

    code/<project>/data/

so each project can be self-contained and run without external downloads.

Usage:
        python scripts/generate_demo_datasets.py
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
    def code_dir(self) -> Path:
        return self.repo_root / "code"

    @property
    def churn_dir(self) -> Path:
        return self.code_dir / "customer_churn" / "data"

    @property
    def retail_dir(self) -> Path:
        return self.code_dir / "retail_sales_forecasting" / "data"

    @property
    def ab_test_dir(self) -> Path:
        return self.code_dir / "ab_testing" / "data"

    @property
    def reviews_dir(self) -> Path:
        return self.code_dir / "nlp_sentiment_topics" / "data"

    @property
    def churn_csv(self) -> Path:
        return self.churn_dir / "customer_churn.csv"

    @property
    def retail_csv(self) -> Path:
        return self.retail_dir / "retail_sales_daily.csv"

    @property
    def ab_test_csv(self) -> Path:
        return self.ab_test_dir / "ab_test_experiment.csv"

    @property
    def reviews_csv(self) -> Path:
        return self.reviews_dir / "product_reviews.csv"


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


def make_ab_test_experiment(
    rng: np.random.Generator,
    n_users: int = 60000,
    start: str = "2025-01-01",
    days: int = 28,
) -> pd.DataFrame:
    """Synthetic A/B test dataset with conversion + revenue outcomes.

    Includes common covariates so you can demonstrate variance reduction / uplift slices.
    """

    user_id = np.arange(1, n_users + 1)
    assigned = rng.random(n_users) < 0.5
    variant = np.where(assigned, "treatment", "control")

    signup_date = pd.to_datetime(start) + pd.to_timedelta(rng.integers(0, days, size=n_users), unit="D")
    device = rng.choice(["mobile", "desktop", "tablet"], size=n_users, p=[0.62, 0.33, 0.05])
    region = rng.choice(["NA", "EU", "APAC"], size=n_users, p=[0.50, 0.30, 0.20])
    is_new_user = rng.random(n_users) < 0.55

    sessions_7d = np.clip(rng.poisson(lam=3.2, size=n_users) + is_new_user.astype(int), 0, None)
    prior_purchases_90d = np.clip(rng.poisson(lam=0.8, size=n_users) - is_new_user.astype(int), 0, None)

    # Baseline conversion probability
    device_effect = np.select(
        [device == "mobile", device == "desktop", device == "tablet"],
        [-0.10, 0.05, 0.00],
        default=0.0,
    )
    region_effect = np.select(
        [region == "NA", region == "EU", region == "APAC"],
        [0.00, -0.02, -0.04],
        default=0.0,
    )
    engagement = 0.06 * sessions_7d + 0.18 * (prior_purchases_90d > 0).astype(int)

    # Treatment uplift is modest overall, higher for mobile and new users.
    uplift = 0.00
    uplift += (variant == "treatment") * (0.015 + 0.010 * (device == "mobile") + 0.006 * is_new_user)

    logit = -2.15 + device_effect + region_effect + engagement + uplift
    p_conv = _sigmoid(logit)
    converted = rng.random(n_users) < p_conv

    # Revenue only if converted; log-normal purchase amount with some drivers.
    base_amount = rng.lognormal(mean=3.3, sigma=0.55, size=n_users)  # ~ $27 median
    amount_multiplier = 1.0 + 0.12 * (variant == "treatment") + 0.10 * (device == "desktop")
    amount = np.where(converted, base_amount * amount_multiplier, 0.0)

    df = pd.DataFrame(
        {
            "user_id": user_id,
            "signup_date": signup_date.date.astype(str),
            "variant": variant,
            "device": device,
            "region": region,
            "is_new_user": is_new_user.astype(int),
            "sessions_7d": sessions_7d,
            "prior_purchases_90d": prior_purchases_90d,
            "converted": converted.astype(int),
            "revenue": np.round(amount, 2),
        }
    )

    return df


def make_product_reviews(
    rng: np.random.Generator,
    n: int = 12000,
) -> pd.DataFrame:
    """Synthetic product review dataset for sentiment + topic modeling."""

    categories = ["electronics", "home", "fitness", "beauty", "pet"]
    category = rng.choice(categories, size=n, p=[0.25, 0.25, 0.18, 0.17, 0.15])
    verified = rng.random(n) < 0.72

    # Topic-specific phrase pools
    topic_phrases = {
        "electronics": [
            "battery life",
            "bluetooth",
            "screen",
            "charging",
            "sound quality",
            "setup",
        ],
        "home": ["delivery", "packaging", "assembly", "quality", "instructions", "size"],
        "fitness": ["comfort", "fit", "durable", "workout", "grip", "sweat"],
        "beauty": ["scent", "texture", "skin", "irritation", "hydrating", "value"],
        "pet": ["taste", "smell", "ingredients", "my dog", "my cat", "training"],
    }

    positive_templates = [
        "Love it — {phrase} is great and it works as expected.",
        "Surprisingly good. The {phrase} exceeded my expectations.",
        "Great value. Fast shipping and the {phrase} is excellent.",
        "Five stars. Easy to use and the {phrase} is solid.",
    ]
    neutral_templates = [
        "It’s okay. The {phrase} is fine but nothing special.",
        "Decent overall. The {phrase} could be better.",
        "Average. The {phrase} meets the basics.",
    ]
    negative_templates = [
        "Disappointed — the {phrase} is not good and support was unhelpful.",
        "Not worth it. The {phrase} failed after a week.",
        "Poor quality. The {phrase} is frustrating to deal with.",
        "One star. The {phrase} was misleading in the description.",
    ]

    # Star ratings: skewed positive but with category variation
    base = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.07, 0.08, 0.15, 0.30, 0.40])
    # Verified purchases slightly more positive
    stars = np.clip(base + verified.astype(int) * rng.choice([0, 0, 1], size=n, p=[0.75, 0.15, 0.10]), 1, 5)

    sentiment_bucket = np.where(stars >= 4, "pos", np.where(stars == 3, "neu", "neg"))
    text: list[str] = []
    for i in range(n):
        cat = category[i]
        phrase = rng.choice(topic_phrases[cat])
        bucket = sentiment_bucket[i]
        if bucket == "pos":
            template = rng.choice(positive_templates)
        elif bucket == "neu":
            template = rng.choice(neutral_templates)
        else:
            template = rng.choice(negative_templates)
        text.append(template.format(phrase=phrase))

    df = pd.DataFrame(
        {
            "review_id": [f"R{idx:07d}" for idx in range(1, n + 1)],
            "category": category,
            "verified_purchase": verified.astype(int),
            "stars": stars.astype(int),
            "review_text": text,
        }
    )
    return df


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = Paths(repo_root=repo_root)

    for d in (paths.churn_dir, paths.retail_dir, paths.ab_test_dir, paths.reviews_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    churn = make_customer_churn(rng)
    churn.to_csv(paths.churn_csv, index=False)

    retail = make_retail_sales_daily(rng)
    retail.to_csv(paths.retail_csv, index=False)

    ab = make_ab_test_experiment(rng)
    ab.to_csv(paths.ab_test_csv, index=False)

    reviews = make_product_reviews(rng)
    reviews.to_csv(paths.reviews_csv, index=False)

    print(f"Wrote: {paths.churn_csv} ({len(churn):,} rows)")
    print(f"Wrote: {paths.retail_csv} ({len(retail):,} rows)")
    print(f"Wrote: {paths.ab_test_csv} ({len(ab):,} rows)")
    print(f"Wrote: {paths.reviews_csv} ({len(reviews):,} rows)")


if __name__ == "__main__":
    main()
