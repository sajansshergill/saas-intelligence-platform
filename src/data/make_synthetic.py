from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticConfig:
    n_customers: int = 1500
    n_days: int = 120
    seed: int = 7
    max_tickets_per_customer: int = 8


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def make_customers(cfg: SyntheticConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    tiers = np.array(["SMB", "MidMarket", "Enterprise"])
    tier = rng.choice(tiers, size=cfg.n_customers, p=[0.55, 0.3, 0.15])
    tenure_days = rng.integers(14, 900, size=cfg.n_customers)

    # baseline product fit / adoption propensity
    adoption = rng.normal(0.0, 1.0, size=cfg.n_customers)

    # expansion latent propensity (higher for enterprise, higher adoption)
    expansion_latent = (
        0.5 * adoption
        + (tier == "Enterprise") * 0.8
        + (tier == "MidMarket") * 0.3
        + rng.normal(0, 0.5, size=cfg.n_customers)
    )

    customers = pd.DataFrame(
        {
            "customer_id": [f"C{idx:05d}" for idx in range(cfg.n_customers)],
            "tier": tier,
            "tenure_days": tenure_days,
            "adoption_latent": adoption,
            "expansion_latent": expansion_latent,
        }
    )
    return customers


def make_usage(cfg: SyntheticConfig, customers: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=cfg.n_days - 1)
    days = pd.date_range(start=start_date, end=end_date, freq="D")

    rows = []
    for _, c in customers.iterrows():
        base = 0.6 + 0.25 * float(c.adoption_latent)
        base += 0.15 if c.tier == "Enterprise" else 0.05 if c.tier == "MidMarket" else 0.0
        base = float(np.clip(base, 0.05, 0.95))

        # simulate decays / improving usage
        drift = rng.normal(0, 0.002)

        last_active = None
        for i, d in enumerate(days):
            p_active = float(np.clip(base + drift * i + rng.normal(0, 0.03), 0.01, 0.98))
            active = rng.random() < p_active
            if active:
                last_active = d
            events = int(rng.poisson(4 + 10 * p_active)) if active else 0
            key_features = int(rng.poisson(1 + 4 * p_active)) if active else 0
            seats_active = int(
                np.clip(
                    rng.normal(10 + 30 * p_active + (c.tier == "Enterprise") * 20, 5),
                    1,
                    250,
                )
            )

            rows.append(
                {
                    "customer_id": c.customer_id,
                    "date": d.date(),
                    "active": int(active),
                    "events": events,
                    "key_features_used": key_features,
                    "seats_active": seats_active,
                }
            )

    usage = pd.DataFrame(rows)
    return usage


def make_tickets(cfg: SyntheticConfig, customers: pd.DataFrame, usage: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 2)

    # customer-level aggregates to influence ticket volume and sentiment
    u30 = (
        usage.sort_values(["customer_id", "date"])
        .groupby("customer_id")
        .tail(30)
        .groupby("customer_id")
        .agg(active_days_30=("active", "sum"), events_30=("events", "sum"))
        .reset_index()
    )

    merged = customers.merge(u30, on="customer_id", how="left").fillna(0)

    intents = [
        ("login_issue", -0.6),
        ("billing_question", -0.2),
        ("performance_slow", -0.7),
        ("how_to", -0.1),
        ("bug_report", -0.5),
        ("feature_request", 0.1),
        ("upgrade_pricing", 0.2),
        ("data_sync", -0.4),
        ("permissions", -0.3),
        ("renewal", 0.15),
    ]

    start_date = pd.to_datetime(usage["date"]).min()
    end_date = pd.to_datetime(usage["date"]).max()

    rows = []
    for _, c in merged.iterrows():
        # more tickets when usage is low or performance issues likely
        low_usage = max(0.0, 1.0 - float(c.active_days_30) / 30.0)
        lam = 0.8 + 2.0 * low_usage + (c.tier == "Enterprise") * 0.8
        n_tickets = int(np.clip(rng.poisson(lam), 0, cfg.max_tickets_per_customer))

        for _ in range(n_tickets):
            intent, intent_sent = intents[int(rng.integers(0, len(intents)))]
            created_at = (start_date + (end_date - start_date) * rng.random()).date()
            urgency = int(np.clip(rng.normal(2.5 + 1.5 * low_usage, 1.0), 1, 5))
            resolved = int(rng.random() > (0.15 + 0.35 * low_usage))
            # sentiment: worse when unresolved / urgent / low usage
            sentiment = float(
                np.clip(
                    intent_sent - 0.25 * (1 - resolved) - 0.1 * (urgency - 3) - 0.4 * low_usage
                    + rng.normal(0, 0.15),
                    -1,
                    1,
                )
            )

            text = _render_ticket_text(intent=intent, urgency=urgency, resolved=bool(resolved))
            rows.append(
                {
                    "ticket_id": f"T{random.randint(10_000_000, 99_999_999)}",
                    "customer_id": c.customer_id,
                    "created_at": created_at,
                    "intent": intent,
                    "urgency": urgency,
                    "resolved": resolved,
                    "sentiment": sentiment,
                    "text": text,
                }
            )

    return pd.DataFrame(rows)


def _render_ticket_text(intent: str, urgency: int, resolved: bool) -> str:
    base = {
        "login_issue": "Users cannot sign in; SSO seems to fail intermittently.",
        "billing_question": "Need clarity on invoice line items and proration.",
        "performance_slow": "App is slow; uploads take longer than usual and time out.",
        "how_to": "Requesting guidance on configuring retention policies and access.",
        "bug_report": "Observed unexpected behavior after last update; steps to reproduce attached.",
        "feature_request": "Would like an enhancement for reporting/export and automation.",
        "upgrade_pricing": "Exploring plan upgrade and additional capacity pricing.",
        "data_sync": "Data sync appears delayed; missing recent updates.",
        "permissions": "Permission settings not applying correctly across user groups.",
        "renewal": "Discussing upcoming renewal and potential scope adjustments.",
    }[intent]

    urgency_phrase = {1: "Low impact.", 2: "Minor impact.", 3: "Moderate impact.", 4: "High impact.", 5: "Critical impact."}[urgency]
    status = "Resolved by support." if resolved else "Still unresolved; escalation requested."
    return f"{base} {urgency_phrase} {status}"


def make_labels(customers: pd.DataFrame, usage: pd.DataFrame, tickets: pd.DataFrame) -> pd.DataFrame:
    # label churn based on last-30-days usage + ticket pain + tier + latent factors
    u30 = (
        usage.sort_values(["customer_id", "date"])
        .groupby("customer_id")
        .tail(30)
        .groupby("customer_id")
        .agg(
            active_days_30=("active", "sum"),
            events_30=("events", "sum"),
            features_30=("key_features_used", "sum"),
            seats_avg_30=("seats_active", "mean"),
            last_active_date=("date", "max"),
        )
        .reset_index()
    )
    t60 = (
        tickets.groupby("customer_id")
        .agg(
            tickets_60=("ticket_id", "count"),
            unresolved_60=("resolved", lambda s: int((1 - s).sum()) if len(s) else 0),
            sentiment_mean_60=("sentiment", "mean"),
            urgency_mean_60=("urgency", "mean"),
        )
        .reset_index()
    )

    df = customers.merge(u30, on="customer_id", how="left").merge(t60, on="customer_id", how="left")
    df = df.fillna(
        {
            "active_days_30": 0,
            "events_30": 0,
            "features_30": 0,
            "seats_avg_30": 0,
            "tickets_60": 0,
            "unresolved_60": 0,
            "sentiment_mean_60": 0.0,
            "urgency_mean_60": 0.0,
        }
    )

    end_date = pd.to_datetime(usage["date"]).max()
    last_active = pd.to_datetime(df["last_active_date"])
    df["days_since_last_active"] = (end_date - last_active).dt.days.fillna(cfg_default_days_since_last_active(end_date)).astype(int)

    # churn probability model (synthetic "ground truth")
    tier_bias = df["tier"].map({"SMB": 0.25, "MidMarket": 0.05, "Enterprise": -0.1}).astype(float)
    usage_term = -0.08 * df["active_days_30"].astype(float) - 0.0006 * df["events_30"].astype(float)
    recency_term = 0.12 * df["days_since_last_active"].astype(float)
    support_term = 0.35 * df["unresolved_60"].astype(float) - 0.6 * df["sentiment_mean_60"].astype(float) + 0.1 * df["urgency_mean_60"].astype(float)
    adoption_term = -0.35 * df["adoption_latent"].astype(float)
    noise = np.random.default_rng(42).normal(0, 0.35, size=len(df))

    logit = -3.0 + tier_bias + usage_term + recency_term + support_term + adoption_term + noise
    p = np.vectorize(_sigmoid)(logit)
    churned_30d = (np.random.default_rng(99).random(len(df)) < p).astype(int)

    # expansion flag (high adoption, high usage, low support pain)
    exp_logit = (
        -1.0
        + 0.7 * df["expansion_latent"].astype(float)
        + 0.03 * df["active_days_30"].astype(float)
        + 0.35 * df["sentiment_mean_60"].astype(float)
        - 0.25 * df["unresolved_60"].astype(float)
    )
    exp_p = np.vectorize(_sigmoid)(exp_logit)
    expansion_oppty = (np.random.default_rng(123).random(len(df)) < exp_p).astype(int)

    labels = df[
        [
            "customer_id",
            "tier",
            "tenure_days",
            "active_days_30",
            "events_30",
            "features_30",
            "seats_avg_30",
            "days_since_last_active",
            "tickets_60",
            "unresolved_60",
            "sentiment_mean_60",
            "urgency_mean_60",
        ]
    ].copy()
    labels["churned_30d"] = churned_30d
    labels["expansion_oppty"] = expansion_oppty
    return labels


def cfg_default_days_since_last_active(end_date: pd.Timestamp) -> int:
    # In case a customer has no usage rows (shouldn't happen), set to full range.
    return int(120)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic SaaS CX dataset")
    p.add_argument("--out-dir", type=str, default="data/processed", help="Output directory")
    p.add_argument("--n-customers", type=int, default=1500)
    p.add_argument("--n-days", type=int, default=120)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    cfg = SyntheticConfig(n_customers=args.n_customers, n_days=args.n_days, seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    customers = make_customers(cfg)
    usage = make_usage(cfg, customers)
    tickets = make_tickets(cfg, customers, usage)
    labels = make_labels(customers, usage, tickets)

    customers.to_parquet(out_dir / "customers.parquet", index=False)
    usage.to_parquet(out_dir / "usage_daily.parquet", index=False)
    tickets.to_parquet(out_dir / "tickets.parquet", index=False)
    labels.to_parquet(out_dir / "customer_agg_labels.parquet", index=False)

    print(f"Wrote: {out_dir / 'customers.parquet'}")
    print(f"Wrote: {out_dir / 'usage_daily.parquet'}")
    print(f"Wrote: {out_dir / 'tickets.parquet'}")
    print(f"Wrote: {out_dir / 'customer_agg_labels.parquet'}")


if __name__ == "__main__":
    main()

