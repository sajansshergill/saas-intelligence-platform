"""
feature_engineering.py
Loads customer data from DuckDB, engineers features,
computes composite health score, saves processed features.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd


def load_data(db_path: str = "data/processed/saas_crm.duckdb") -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(db_path)
    customers = con.execute("SELECT * FROM customers").df()
    tickets = con.execute("SELECT * FROM support_tickets").df()
    con.close()
    return customers, tickets


def engineer_features(customers: pd.DataFrame, tickets: pd.DataFrame) -> pd.DataFrame:
    # Ticket aggregations per customer
    ticket_agg = (
        tickets.groupby("customer_id")
        .agg(
            total_tickets=("ticket_id", "count"),
            frustrated_tickets=("sentiment_label", lambda x: (x == "frustrated").sum()),
            positive_tickets=("sentiment_label", lambda x: (x == "positive").sum()),
        )
        .reset_index()
    )

    df = customers.merge(ticket_agg, on="customer_id", how="left")
    df[["total_tickets", "frustrated_tickets", "positive_tickets"]] = df[
        ["total_tickets", "frustrated_tickets", "positive_tickets"]
    ].fillna(0)

    # Derived features
    df["frustration_rate"] = df["frustrated_tickets"] / (df["total_tickets"] + 1)
    df["contract_encoded"] = df["contract_type"].map(
        {"Month-to-month": 0, "One year": 1, "Two year": 2}
    )
    df["is_high_value"] = (df["monthly_charges"] > df["monthly_charges"].median()).astype(int)
    df["engagement_score"] = (
        0.4 * (df["login_frequency_30d"] / 30)
        + 0.4 * df["feature_adoption_rate"]
        + 0.2 * (df["num_products"] / 5)
    ).round(4)

    return df


def compute_health_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite Health Score (0-100):
    - Engagement (30%): login freq + feature adoption + products
    - Sentiment (25%): NPS + low frustration rate
    - Stability (25%): tenure + contract type
    - Support burden (20%): low ticket volume
    """
    engagement = df["engagement_score"]
    sentiment = (0.6 * (df["nps_score"] / 10) + 0.4 * (1 - df["frustration_rate"])).clip(0, 1)
    stability = 0.6 * (df["tenure_months"] / 72).clip(0, 1) + 0.4 * (df["contract_encoded"] / 2)
    support_burden = (1 - (df["support_tickets_90d"] / 15).clip(0, 1))

    raw_score = 0.30 * engagement + 0.25 * sentiment + 0.25 * stability + 0.20 * support_burden

    df["health_score"] = (raw_score * 100).round(1)
    df["health_tier"] = pd.cut(
        df["health_score"],
        bins=[0, 40, 65, 85, 100],
        labels=["Critical", "At Risk", "Healthy", "Champion"],
    )
    return df


def save_features(df: pd.DataFrame, db_path: str = "data/processed/saas_crm.duckdb") -> None:
    con = duckdb.connect(db_path)
    con.execute("CREATE OR REPLACE TABLE features AS SELECT * FROM df")
    con.close()
    df.to_csv("data/processed/features.csv", index=False)
    print(f"✅ Features saved: {len(df)} rows")
    print(df[["customer_id", "health_score", "health_tier", "churn"]].head(10))


if __name__ == "__main__":
    customers, tickets = load_data()
    df = engineer_features(customers, tickets)
    df = compute_health_score(df)
    save_features(df)

