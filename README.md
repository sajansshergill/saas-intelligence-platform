# 🏥 SaaS Customer Health Intelligence Platform

> End-to-end customer health scoring, churn prediction, and AI-powered ticket analysis for SaaS CX teams.

https://saas-intelligence-platform.streamlit.app/

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline (data → features/health score → churn model → ticket analysis):

```bash
python run_pipeline.py
```

Launch the dashboard:

```bash
streamlit run src/dashboard/app.py
```

## Optional: live Claude API ticket analysis

```bash
export ANTHROPIC_API_KEY=your_key_here
python src/nlp/ticket_analyzer.py
```

## View MLflow experiments

```bash
mlflow ui
```

## Folder Structure

```
saas-intelligence-platform/
├── src/
│   ├── data/           # Synthetic data generation
│   ├── features/       # Feature engineering + health scoring
│   ├── models/         # XGBoost + SHAP + MLflow
│   ├── nlp/            # Claude API ticket analyzer (or mock)
│   └── dashboard/      # Streamlit CX dashboard
├── data/raw/           # customers.csv, support_tickets.csv
├── data/processed/     # DuckDB, features.csv, predictions.csv
├── outputs/            # confusion_matrix.png, shap_summary.png
├── run_pipeline.py
└── requirements.txt
```
