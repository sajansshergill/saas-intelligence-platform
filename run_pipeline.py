"""
run_pipeline.py
Master pipeline — runs all steps end-to-end:
1. Generate synthetic data
2. Feature engineering + health scoring
3. Churn prediction model (XGBoost + SHAP + MLflow)
4. NLP ticket analysis (Claude API or mock)
"""

import subprocess
import sys
import os

def run_step(name, script_path):
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    if result.returncode != 0:
        print(f"❌ {name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"✅ {name} complete")


if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)

    run_step("Data Generation", "src/data/generate_data.py")
    run_step("Feature Engineering & Health Scoring", "src/features/feature_engineering.py")
    run_step("Churn Prediction Model", "src/models/churn_model.py")
    run_step("NLP Ticket Analysis", "src/nlp/ticket_analyzer.py")

    print("\n" + "="*60)
    print("  ✅ FULL PIPELINE COMPLETE")
    print("  Run dashboard with:")
    print("  streamlit run src/dashboard/app.py")
    print("  MLflow UI:")
    print("  mlflow ui")
    print("="*60)