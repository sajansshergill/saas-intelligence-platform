"""
app.py — SaaS Customer Health Intelligence Platform
Streamlit CX Stakeholder Dashboard
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.makedirs("outputs", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath("outputs/.mplconfig"))

import duckdb
import pandas as pd
import streamlit as st

import altair as alt

st.set_page_config(
    page_title="SaaS Customer Health Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "data/processed/saas_crm.duckdb"

PIPELINE_STEPS: list[tuple[str, str]] = [
    ("Data Generation", "src/data/generate_data.py"),
    ("Feature Engineering & Health Scoring", "src/features/feature_engineering.py"),
    ("Churn Prediction Model", "src/models/churn_model.py"),
    ("NLP Ticket Analysis (mock without key)", "src/nlp/ticket_analyzer.py"),
]


def db_exists() -> bool:
    return Path(DB_PATH).exists()


def initialize_demo_db() -> None:
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    # Ensure MLflow writes to a writable path on Streamlit Cloud
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"file:{os.path.abspath('data/processed/mlruns')}")
    for _, script in PIPELINE_STEPS:
        # Use Streamlit Cloud's python runtime (sys.executable)
        subprocess.run([sys.executable, script], check=True)


@st.cache_data
def load_data():
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        tables = con.execute("SHOW TABLES").df()["name"].tolist()
        predictions = (
            con.execute("SELECT * FROM predictions").df() if "predictions" in tables else pd.DataFrame()
        )
        tickets_raw = (
            con.execute("SELECT * FROM support_tickets").df()
            if "support_tickets" in tables
            else pd.DataFrame()
        )
        ticket_analysis = (
            con.execute("SELECT * FROM ticket_analysis").df()
            if "ticket_analysis" in tables
            else pd.DataFrame()
        )
        return predictions, tickets_raw, ticket_analysis
    finally:
        con.close()


if not db_exists():
    st.title("🏥 SaaS Customer Health Intelligence Platform")
    st.warning("Demo database not found yet. Initialize data to launch the dashboard.")
    st.markdown(
        "- This will generate synthetic customers + support tickets, compute health scores, train a churn model, and create mock ticket insights.\n"
        "- On Streamlit Cloud this runs once per fresh deployment."
    )

    if st.button("Initialize demo data (recommended)", type="primary"):
        with st.spinner("Initializing demo dataset + models..."):
            initialize_demo_db()
        st.cache_data.clear()
        st.rerun()
    st.stop()

df, tickets_raw, ticket_analysis = load_data()

# Sidebar
st.sidebar.title("🏥 CX Health Platform")
st.sidebar.markdown("---")
health_filter = st.sidebar.multiselect(
    "Filter by Health Tier",
    options=["Critical", "At Risk", "Healthy", "Champion"],
    default=["Critical", "At Risk", "Healthy", "Champion"],
)
risk_filter = st.sidebar.multiselect(
    "Filter by Churn Risk", options=["High", "Medium", "Low"], default=["High", "Medium", "Low"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Built for Nasuni CX Org**")
st.sidebar.markdown("*SaaS Customer Health Intelligence Platform*")

filtered = (
    df[df["health_tier"].astype(str).isin(health_filter) & df["churn_risk_tier"].astype(str).isin(risk_filter)]
    if not df.empty
    else df
)

# Header
st.title("🏥 SaaS Customer Health Intelligence Platform")
st.markdown("*Proactive churn prediction · Health scoring · AI-powered ticket insights*")
st.markdown("---")

# KPIs
if not df.empty:
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Customers", f"{len(filtered):,}")
    k2.metric("Avg Health Score", f"{filtered['health_score'].mean():.1f} / 100")
    k3.metric("High Churn Risk", f"{(filtered['churn_risk_tier'] == 'High').sum():,}")
    k4.metric("Actual Churn Rate", f"{filtered['churn'].mean():.1%}")
    k5.metric("Champions", f"{(filtered['health_tier'] == 'Champion').sum():,}")
    st.markdown("---")

# Row 1: Health Distribution + Churn Risk
col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Health Score Distribution by Tier")
    if not filtered.empty:
        colors = {
            "Critical": "#e74c3c",
            "At Risk": "#f39c12",
            "Healthy": "#2ecc71",
            "Champion": "#3498db",
        }
        chart = (
            alt.Chart(filtered)
            .mark_bar(opacity=0.75)
            .encode(
                x=alt.X("health_score:Q", bin=alt.Bin(maxbins=20), title="Health Score (0–100)"),
                y=alt.Y("count():Q", title="Customer Count"),
                color=alt.Color(
                    "health_tier:N",
                    title="Tier",
                    scale=alt.Scale(domain=list(colors.keys()), range=list(colors.values())),
                ),
                tooltip=[
                    alt.Tooltip("health_tier:N", title="Tier"),
                    alt.Tooltip("count():Q", title="Customers"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

with col2:
    st.subheader("🎯 Churn Risk Tier Breakdown")
    if not filtered.empty:
        risk_colors = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"}
        chart2 = (
            alt.Chart(filtered)
            .mark_bar()
            .encode(
                x=alt.X("churn_risk_tier:N", title="Churn Risk Tier", sort=["High", "Medium", "Low"]),
                y=alt.Y("count():Q", title="Number of Customers"),
                color=alt.Color(
                    "churn_risk_tier:N",
                    title=None,
                    scale=alt.Scale(domain=list(risk_colors.keys()), range=list(risk_colors.values())),
                ),
                tooltip=[alt.Tooltip("churn_risk_tier:N", title="Risk"), alt.Tooltip("count():Q", title="Customers")],
            )
            .properties(height=320)
        )
        st.altair_chart(chart2, use_container_width=True)

st.markdown("---")

# Row 2: Scatter + Heatmap
col3, col4 = st.columns(2)
with col3:
    st.subheader("🔵 Health Score vs Churn Probability")
    if not filtered.empty:
        tier_colors = {"Critical": "#e74c3c", "At Risk": "#f39c12", "Healthy": "#2ecc71", "Champion": "#3498db"}
        chart3 = (
            alt.Chart(filtered)
            .mark_circle(size=60, opacity=0.45)
            .encode(
                x=alt.X("health_score:Q", title="Health Score"),
                y=alt.Y("churn_probability:Q", title="Churn Probability", axis=alt.Axis(format="%")),
                color=alt.Color(
                    "health_tier:N",
                    title="Tier",
                    scale=alt.Scale(domain=list(tier_colors.keys()), range=list(tier_colors.values())),
                ),
                tooltip=[
                    alt.Tooltip("customer_id:N", title="Customer"),
                    alt.Tooltip("health_tier:N", title="Tier"),
                    alt.Tooltip("health_score:Q", title="Health"),
                    alt.Tooltip("churn_probability:Q", title="Churn Prob", format=".1%"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(chart3, use_container_width=True)

with col4:
    st.subheader("📈 Avg Churn Probability by Contract & NPS")
    if not filtered.empty:
        heat = (
            filtered.groupby(["contract_type", "nps_score"], as_index=False)["churn_probability"]
            .mean()
            .rename(columns={"churn_probability": "avg_churn_probability"})
        )
        chart4 = (
            alt.Chart(heat)
            .mark_rect()
            .encode(
                x=alt.X("nps_score:O", title="NPS Score"),
                y=alt.Y("contract_type:N", title="Contract Type"),
                color=alt.Color(
                    "avg_churn_probability:Q",
                    title="Avg Churn Prob",
                    scale=alt.Scale(scheme="redyellowgreen", reverse=True),
                ),
                tooltip=[
                    alt.Tooltip("contract_type:N", title="Contract"),
                    alt.Tooltip("nps_score:O", title="NPS"),
                    alt.Tooltip("avg_churn_probability:Q", title="Avg Churn Prob", format=".1%"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(chart4, use_container_width=True)

st.markdown("---")

# High-Risk Watchlist
st.subheader("🚨 High-Risk Customer Watchlist")
if not filtered.empty:
    at_risk = filtered[filtered["churn_risk_tier"] == "High"].sort_values("churn_probability", ascending=False)
    display_cols = [
        "customer_id",
        "health_score",
        "health_tier",
        "churn_probability",
        "tenure_months",
        "monthly_charges",
        "nps_score",
        "support_tickets_90d",
        "contract_type",
    ]
    st.dataframe(
        at_risk[display_cols]
        .head(20)
        .style.background_gradient(subset=["churn_probability"], cmap="Reds")
        .background_gradient(subset=["health_score"], cmap="RdYlGn")
        .format({"churn_probability": "{:.1%}", "monthly_charges": "${:.2f}", "health_score": "{:.1f}"}),
        width="stretch",
    )

st.markdown("---")

# AI Ticket Insights
st.subheader("🤖 AI-Powered Support Ticket Insights")
if not ticket_analysis.empty:
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Escalation Risk Distribution**")
        esc_counts = ticket_analysis["escalation_risk"].value_counts().rename_axis("escalation_risk").reset_index(name="count")
        esc_colors = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
        esc_chart = (
            alt.Chart(esc_counts)
            .mark_arc(innerRadius=40)
            .encode(
                theta=alt.Theta("count:Q", title="Tickets"),
                color=alt.Color(
                    "escalation_risk:N",
                    title="Escalation Risk",
                    scale=alt.Scale(domain=list(esc_colors.keys()), range=list(esc_colors.values())),
                ),
                tooltip=[alt.Tooltip("escalation_risk:N", title="Risk"), alt.Tooltip("count:Q", title="Tickets")],
            )
            .properties(height=260)
        )
        st.altair_chart(esc_chart, use_container_width=True)

    with t2:
        st.markdown("**Issue Category Breakdown**")
        cat_counts = ticket_analysis["issue_category"].value_counts().rename_axis("issue_category").reset_index(name="count")
        cat_chart = (
            alt.Chart(cat_counts)
            .mark_bar()
            .encode(
                y=alt.Y("issue_category:N", title=None, sort="-x"),
                x=alt.X("count:Q", title="Ticket Count"),
                tooltip=[alt.Tooltip("issue_category:N", title="Category"), alt.Tooltip("count:Q", title="Tickets")],
            )
            .properties(height=260)
        )
        st.altair_chart(cat_chart, use_container_width=True)

    st.markdown("**📋 Sample AI-Analyzed Tickets with CSM Recommendations**")
    churn_signals = ticket_analysis[ticket_analysis["churn_signal"] == True][
        ["customer_id", "escalation_risk", "frustration_level", "issue_category", "recommended_action", "ticket_text"]
    ].head(10)
    st.dataframe(churn_signals, width="stretch")
else:
    st.info("Run `python src/nlp/ticket_analyzer.py` to generate AI ticket insights (mock runs without API key).")

st.markdown("---")

# Expansion Opportunities
st.subheader("🚀 Expansion Opportunities — Champion Accounts")
if not filtered.empty:
    champions = filtered[(filtered["health_tier"] == "Champion") & (filtered["churn_probability"] < 0.2)].sort_values(
        "health_score", ascending=False
    )
    exp_cols = ["customer_id", "health_score", "monthly_charges", "num_products", "feature_adoption_rate", "nps_score", "tenure_months"]
    st.dataframe(
        champions[exp_cols]
        .head(15)
        .style.background_gradient(subset=["health_score"], cmap="Greens")
        .format({"monthly_charges": "${:.2f}", "feature_adoption_rate": "{:.0%}", "health_score": "{:.1f}"}),
        width="stretch",
    )

st.markdown("---")
st.caption("SaaS Customer Health Intelligence Platform · Built by Sajan Singh Shergill · Pace University MS Data Science 2026")

