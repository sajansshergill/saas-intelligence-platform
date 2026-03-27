"""
app.py — SaaS Customer Health Intelligence Platform
Streamlit CX Stakeholder Dashboard
"""

from __future__ import annotations

import os

os.makedirs("outputs", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath("outputs/.mplconfig"))

import duckdb
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

matplotlib.use("Agg")

st.set_page_config(
    page_title="SaaS Customer Health Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "data/processed/saas_crm.duckdb"


@st.cache_data
def load_data():
    con = duckdb.connect(DB_PATH, read_only=True)
    tables = con.execute("SHOW TABLES").df()["name"].tolist()
    predictions = con.execute("SELECT * FROM predictions").df() if "predictions" in tables else pd.DataFrame()
    tickets_raw = (
        con.execute("SELECT * FROM support_tickets").df() if "support_tickets" in tables else pd.DataFrame()
    )
    ticket_analysis = (
        con.execute("SELECT * FROM ticket_analysis").df() if "ticket_analysis" in tables else pd.DataFrame()
    )
    con.close()
    return predictions, tickets_raw, ticket_analysis


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
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = {"Critical": "#e74c3c", "At Risk": "#f39c12", "Healthy": "#2ecc71", "Champion": "#3498db"}
        for tier, grp in filtered.groupby("health_tier", observed=True):
            ax.hist(grp["health_score"], bins=20, alpha=0.7, label=str(tier), color=colors.get(str(tier), "gray"))
        ax.set_xlabel("Health Score (0–100)")
        ax.set_ylabel("Customer Count")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

with col2:
    st.subheader("🎯 Churn Risk Tier Breakdown")
    if not filtered.empty:
        risk_counts = filtered["churn_risk_tier"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        bars = ax2.bar(
            risk_counts.index,
            risk_counts.values,
            color=["#e74c3c", "#f39c12", "#2ecc71"][: len(risk_counts)],
            edgecolor="white",
            linewidth=1.5,
        )
        for bar, val in zip(bars, risk_counts.values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(val), ha="center", fontweight="bold")
        ax2.set_ylabel("Number of Customers")
        st.pyplot(fig2)
        plt.close(fig2)

st.markdown("---")

# Row 2: Scatter + Heatmap
col3, col4 = st.columns(2)
with col3:
    st.subheader("🔵 Health Score vs Churn Probability")
    if not filtered.empty:
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        tier_colors = {"Critical": "#e74c3c", "At Risk": "#f39c12", "Healthy": "#2ecc71", "Champion": "#3498db"}
        for tier, grp in filtered.groupby("health_tier", observed=True):
            ax3.scatter(
                grp["health_score"],
                grp["churn_probability"],
                alpha=0.5,
                s=20,
                label=str(tier),
                color=tier_colors.get(str(tier), "gray"),
            )
        ax3.set_xlabel("Health Score")
        ax3.set_ylabel("Churn Probability")
        ax3.legend(fontsize=8)
        st.pyplot(fig3)
        plt.close(fig3)

with col4:
    st.subheader("📈 Avg Churn Probability by Contract & NPS")
    if not filtered.empty:
        pivot = filtered.groupby(["contract_type", "nps_score"])["churn_probability"].mean().unstack(fill_value=0)
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        sns.heatmap(pivot, annot=False, cmap="RdYlGn_r", ax=ax4, linewidths=0.3, cbar_kws={"label": "Churn Prob"})
        ax4.set_xlabel("NPS Score")
        ax4.set_ylabel("Contract Type")
        st.pyplot(fig4)
        plt.close(fig4)

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
        use_container_width=True,
    )

st.markdown("---")

# AI Ticket Insights
st.subheader("🤖 AI-Powered Support Ticket Insights")
if not ticket_analysis.empty:
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Escalation Risk Distribution**")
        esc_counts = ticket_analysis["escalation_risk"].value_counts()
        fig5, ax5 = plt.subplots(figsize=(5, 3))
        ax5.pie(
            esc_counts.values,
            labels=esc_counts.index,
            colors=["#2ecc71", "#f39c12", "#e74c3c"][: len(esc_counts)],
            autopct="%1.1f%%",
            startangle=140,
        )
        st.pyplot(fig5)
        plt.close(fig5)

    with t2:
        st.markdown("**Issue Category Breakdown**")
        cat_counts = ticket_analysis["issue_category"].value_counts()
        fig6, ax6 = plt.subplots(figsize=(5, 3))
        ax6.barh(cat_counts.index, cat_counts.values, color="#3498db", edgecolor="white")
        ax6.set_xlabel("Ticket Count")
        st.pyplot(fig6)
        plt.close(fig6)

    st.markdown("**📋 Sample AI-Analyzed Tickets with CSM Recommendations**")
    churn_signals = ticket_analysis[ticket_analysis["churn_signal"] == True][
        ["customer_id", "escalation_risk", "frustration_level", "issue_category", "recommended_action", "ticket_text"]
    ].head(10)
    st.dataframe(churn_signals, use_container_width=True)
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
        use_container_width=True,
    )

st.markdown("---")
st.caption("SaaS Customer Health Intelligence Platform · Built by Sajan Singh Shergill · Pace University MS Data Science 2026")

