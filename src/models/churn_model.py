"""
churn_model.py
Trains XGBoost churn prediction model with SHAP explainability.
Tracks experiments with MLflow.
"""

from __future__ import annotations

import os

os.makedirs("outputs", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath("outputs/.mplconfig"))

import duckdb
import matplotlib
import mlflow
import mlflow.sklearn
import pandas as pd
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FEATURE_COLS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "login_frequency_30d",
    "feature_adoption_rate",
    "support_tickets_90d",
    "num_products",
    "nps_score",
    "total_tickets",
    "frustrated_tickets",
    "positive_tickets",
    "frustration_rate",
    "contract_encoded",
    "is_high_value",
    "engagement_score",
    "health_score",
]
TARGET = "churn"

def configure_mlflow() -> None:
    """
    Streamlit Cloud can mount the repo read-only. Force MLflow to use a writable local
    file store so training doesn't crash.
    """
    default_store = os.path.abspath("data/processed/mlruns")
    os.makedirs(default_store, exist_ok=True)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or f"file:{default_store}"
    mlflow.set_tracking_uri(tracking_uri)


def load_features(db_path: str = "data/processed/saas_crm.duckdb") -> pd.DataFrame:
    con = duckdb.connect(db_path)
    df = con.execute("SELECT * FROM features").df()
    con.close()
    return df


def train_model(df: pd.DataFrame):
    configure_mlflow()

    X = df[FEATURE_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("saas-churn-prediction")

    with mlflow.start_run(run_name="xgboost_churn_v1"):
        params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": (y == 0).sum() / max(1, (y == 1).sum()),
            "random_state": 42,
            "eval_metric": "logloss",
        }
        mlflow.log_params(params)

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("precision", report["1"]["precision"])
        mlflow.log_metric("recall", report["1"]["recall"])
        mlflow.log_metric("f1", report["1"]["f1-score"])
        mlflow.sklearn.log_model(model, "xgboost_churn_model")

        print(f"✅ AUC-ROC: {auc:.4f}")
        print(classification_report(y_test, y_pred))

        os.makedirs("outputs", exist_ok=True)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Retained", "Churned"])
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, colorbar=False)
        ax.set_title("Churn Prediction — Confusion Matrix")
        plt.tight_layout()
        plt.savefig("outputs/confusion_matrix.png", dpi=150)
        mlflow.log_artifact("outputs/confusion_matrix.png")
        plt.close()

        # SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title("SHAP Feature Importance — Churn Drivers")
        plt.tight_layout()
        plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches="tight")
        mlflow.log_artifact("outputs/shap_summary.png")
        plt.close()

        # Save predictions
        X_full = df[FEATURE_COLS]
        df["churn_probability"] = model.predict_proba(X_full)[:, 1].round(4)
        df["churn_risk_tier"] = pd.cut(
            df["churn_probability"], bins=[0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"]
        )

        con = duckdb.connect("data/processed/saas_crm.duckdb")
        con.execute("CREATE OR REPLACE TABLE predictions AS SELECT * FROM df")
        con.close()
        df.to_csv("data/processed/predictions.csv", index=False)
        print("✅ Predictions saved to data/processed/predictions.csv")

        return model, auc


if __name__ == "__main__":
    df = load_features()
    train_model(df)

