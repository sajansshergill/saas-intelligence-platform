"""
ticket_analyzer.py
Uses Claude API to extract structured risk signals from support ticket text.
Outputs: escalation_risk, frustration_level, issue_category, recommended_action
"""

from __future__ import annotations

import json
import os
import time

import anthropic
import duckdb
import pandas as pd


def analyze_ticket(client: anthropic.Anthropic, ticket_text: str) -> dict:
    prompt = f"""You are a Customer Success AI analyst. Analyze this support ticket and return ONLY a JSON object with these exact keys:

{{
  "frustration_level": <integer 1-5, where 1=calm, 5=extremely frustrated>,
  "escalation_risk": <"Low" | "Medium" | "High">,
  "issue_category": <"Billing" | "Technical" | "Onboarding" | "Feature Request" | "General">,
  "churn_signal": <true | false>,
  "recommended_action": <one short sentence for the CSM to act on>
}}

Support ticket:
\"\"\"{ticket_text}\"\"\"

Return ONLY the JSON. No explanation, no markdown."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def run_nlp_analysis(db_path: str = "data/processed/saas_crm.duckdb", sample_n: int = 50):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set — running mock NLP analysis instead.")
        return run_mock_analysis(db_path, sample_n)

    client = anthropic.Anthropic(api_key=api_key)

    con = duckdb.connect(db_path)
    tickets = con.execute("SELECT * FROM support_tickets").df()
    con.close()

    sample = tickets.sample(min(sample_n, len(tickets)), random_state=42).copy()
    results = []

    for _, row in sample.iterrows():
        try:
            analysis = analyze_ticket(client, row["ticket_text"])
            analysis["ticket_id"] = row["ticket_id"]
            analysis["customer_id"] = row["customer_id"]
            analysis["ticket_text"] = row["ticket_text"]
            results.append(analysis)
            time.sleep(0.3)
        except Exception as e:
            print(f"  ⚠️  Ticket {row['ticket_id']} failed: {e}")

    df_results = pd.DataFrame(results)
    _save_results(df_results, db_path)
    return df_results


def run_mock_analysis(db_path: str = "data/processed/saas_crm.duckdb", sample_n: int = 50):
    con = duckdb.connect(db_path)
    tickets = con.execute("SELECT * FROM support_tickets").df()
    con.close()

    sample = tickets.sample(min(sample_n, len(tickets)), random_state=42).copy()

    sentiment_map = {
        "frustrated": {
            "frustration_level": 4,
            "escalation_risk": "High",
            "churn_signal": True,
            "issue_category": "Technical",
            "recommended_action": "Escalate to senior CSM immediately and schedule call within 24 hours.",
        },
        "neutral": {
            "frustration_level": 2,
            "escalation_risk": "Low",
            "churn_signal": False,
            "issue_category": "General",
            "recommended_action": "Respond with documentation link and follow up in 3 days.",
        },
        "positive": {
            "frustration_level": 1,
            "escalation_risk": "Low",
            "churn_signal": False,
            "issue_category": "General",
            "recommended_action": "Send thank-you note and explore upsell opportunity.",
        },
    }

    results = []
    for _, row in sample.iterrows():
        base = sentiment_map.get(row["sentiment_label"], sentiment_map["neutral"]).copy()
        base["ticket_id"] = row["ticket_id"]
        base["customer_id"] = row["customer_id"]
        base["ticket_text"] = row["ticket_text"]
        results.append(base)

    df_results = pd.DataFrame(results)
    _save_results(df_results, db_path)
    print(f"✅ Mock NLP analysis complete: {len(df_results)} tickets analyzed")
    return df_results


def _save_results(df_results: pd.DataFrame, db_path: str) -> None:
    con = duckdb.connect(db_path)
    con.execute("CREATE OR REPLACE TABLE ticket_analysis AS SELECT * FROM df_results")
    con.close()
    os.makedirs("data/processed", exist_ok=True)
    df_results.to_csv("data/processed/ticket_analysis.csv", index=False)
    print(f"✅ Ticket analysis saved: {len(df_results)} tickets")


if __name__ == "__main__":
    df = run_nlp_analysis()
    if df is not None and not df.empty:
        print(
            df[
                ["customer_id", "frustration_level", "escalation_risk", "churn_signal", "recommended_action"]
            ].head(10)
        )

