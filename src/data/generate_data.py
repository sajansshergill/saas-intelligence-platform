"""
generate_data.py
Generates synthetic SaaS customer dataset simulating:
- Structured usage/engagement data (Telco-style churn features)
- Unstructured support ticket text
Saves to data/raw/customers.csv and data/raw/support_tickets.csv
"""

from __future__ import annotations

import os

import duckdb
import numpy as np
import pandas as pd

np.random.seed(42)
N = 1000


def generate_customers() -> pd.DataFrame:
    tenure = np.random.randint(1, 72, N)
    monthly_charges = np.round(np.random.uniform(20, 120, N), 2)
    total_charges = np.round(tenure * monthly_charges * np.random.uniform(0.85, 1.0, N), 2)
    login_frequency = np.random.randint(0, 30, N)
    feature_adoption = np.round(np.random.uniform(0, 1, N), 2)
    support_tickets = np.random.randint(0, 15, N)
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], N, p=[0.5, 0.3, 0.2])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], N
    )
    num_products = np.random.randint(1, 6, N)
    nps_score = np.random.randint(1, 11, N)

    # Churn logic: high tickets, low login, month-to-month, low NPS → higher churn
    churn_prob = (
        0.3 * (support_tickets / 15)
        + 0.2 * (1 - login_frequency / 30)
        + 0.2 * (contract == "Month-to-month").astype(float)
        + 0.15 * (1 - nps_score / 10)
        + 0.15 * (1 - feature_adoption)
    )
    churn = (np.random.uniform(0, 1, N) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST_{str(i).zfill(4)}" for i in range(N)],
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "login_frequency_30d": login_frequency,
            "feature_adoption_rate": feature_adoption,
            "support_tickets_90d": support_tickets,
            "contract_type": contract,
            "payment_method": payment_method,
            "num_products": num_products,
            "nps_score": nps_score,
            "churn": churn,
        }
    )
    return df


def generate_support_tickets(customer_ids: list[str]) -> pd.DataFrame:
    frustrated_templates = [
        "I've been trying to reach support for days and nobody responds. This is unacceptable.",
        "The platform keeps crashing and I'm losing data. This is a critical issue.",
        "Billing is completely wrong again. I've reported this three times now.",
        "Feature X is broken and has been for weeks. No resolution in sight.",
        "We are seriously considering canceling our subscription if this isn't fixed.",
        "Response times are terrible and our team is blocked. Very disappointed.",
        "The integration keeps failing and support is not helpful at all.",
        "We are paying premium price for a service that doesn't work.",
    ]
    neutral_templates = [
        "How do I configure the API integration with our CRM?",
        "Can you help me export data in CSV format?",
        "I need to add a new user to our account.",
        "Looking for documentation on the reporting module.",
        "How do I reset my password?",
        "What are the data retention policies for our tier?",
        "Need help setting up SSO for our organization.",
        "Can I upgrade my plan mid-cycle?",
    ]
    positive_templates = [
        "Love the new dashboard update! Much easier to use.",
        "Support team was super helpful and resolved my issue quickly.",
        "The onboarding process was smooth and the team is great.",
        "Really impressed with the feature set — exactly what we needed.",
        "The platform has been rock solid since we onboarded. Great work.",
    ]

    rows: list[dict] = []
    for cid in customer_ids:
        n_tickets = np.random.randint(0, 4)
        for _ in range(n_tickets):
            sentiment = np.random.choice(["frustrated", "neutral", "positive"], p=[0.35, 0.45, 0.20])
            if sentiment == "frustrated":
                text = np.random.choice(frustrated_templates)
            elif sentiment == "neutral":
                text = np.random.choice(neutral_templates)
            else:
                text = np.random.choice(positive_templates)
            rows.append(
                {
                    "ticket_id": f"TKT_{np.random.randint(10000, 99999)}",
                    "customer_id": cid,
                    "ticket_text": text,
                    "sentiment_label": sentiment,
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    customers = generate_customers()
    tickets = generate_support_tickets(customers["customer_id"].tolist())

    customers.to_csv("data/raw/customers.csv", index=False)
    tickets.to_csv("data/raw/support_tickets.csv", index=False)

    # Load into DuckDB
    con = duckdb.connect("data/processed/saas_crm.duckdb")
    con.execute(
        "CREATE OR REPLACE TABLE customers AS SELECT * FROM read_csv_auto('data/raw/customers.csv')"
    )
    con.execute(
        "CREATE OR REPLACE TABLE support_tickets AS SELECT * FROM read_csv_auto('data/raw/support_tickets.csv')"
    )

    print(f"✅ Customers: {len(customers)} rows")
    print(f"✅ Support Tickets: {len(tickets)} rows")
    print(f"✅ Churn rate: {customers['churn'].mean():.1%}")
    print("✅ DuckDB saved to data/processed/saas_crm.duckdb")
    con.close()

