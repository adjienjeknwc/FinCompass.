"""
FinCompass - Orchestrator & Execution Pipeline
=============================================

This is the central execution orchestrator for the FinCompass platform. It runs the
entire data engineering and data science pipeline in sequence, ensuring all tables,
models, analyses, and RAG indexes are fully populated before running the web app.

Pipeline execution sequence:
1. Synthetic data generation (generate_data.py)
2. Data cleaning, validation, and Z-score outlier flagging (clean_validate.py)
3. SQLite database initialization and ingestion (load_db.py)
4. Statistical tests: Welch's t-test, regression, Mann-Kendall (stats_analysis.py)
5. Time-series forecasting: SARIMA model training (forecasting.py)
6. Backlog risk engine: DoS risk-scoring (risk_scoring.py)
7. Machine Learning classifier: TF-IDF + Logistic Regression (train_classifier.py)
8. RAG vector index compilation: ChromaDB stores (build_vectorstore.py)
9. Automated reporting: Word Summary + Excel MIS (generate_report.py, excel_mis.py)
"""

import subprocess
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path("/Users/aditi/.gemini/antigravity/scratch/FinCompass")

# Define scripts to run in sequence
PIPELINE_STEPS = [
    ("1. Synthetic Data Generation", PROJECT_ROOT / "etl" / "generate_data.py"),
    ("2. ETL Data Cleaning & Validation", PROJECT_ROOT / "etl" / "clean_validate.py"),
    ("3. Database Ingestion & Summary Aggregations", PROJECT_ROOT / "etl" / "load_db.py"),
    ("4. Statistical Modeling & Hypothesis Tests", PROJECT_ROOT / "analysis" / "stats_analysis.py"),
    ("5. SARIMA Time-Series Forecasting", PROJECT_ROOT / "analysis" / "forecasting.py"),
    ("6. Bank Risk Scoring Engine", PROJECT_ROOT / "analysis" / "risk_scoring.py"),
    ("7. Machine Learning Complaint Classifier", PROJECT_ROOT / "ml" / "train_classifier.py"),
    ("8. AI Policy Assistant RAG Vector Ingestion", PROJECT_ROOT / "rag" / "build_vectorstore.py"),
    ("9. Word Executive Report Generation", PROJECT_ROOT / "automation" / "generate_report.py"),
    ("10. Excel MIS Spreadsheet Generation", PROJECT_ROOT / "automation" / "excel_mis.py"),
]

def run_step(step_name: str, script_path: Path):
    """Executes a single step in the pipeline using subprocess."""
    print(f"\n==================================================")
    print(f"🚀 Running: {step_name}")
    print(f"==================================================")
    
    start_time = time.time()
    
    # Run the script using the current python executable
    res = subprocess.run([sys.executable, str(script_path)], capture_output=False)
    
    if res.returncode != 0:
        print(f"❌ Error: {step_name} failed with exit code {res.returncode}. Aborting pipeline.")
        sys.exit(res.returncode)
        
    elapsed = time.time() - start_time
    print(f"✅ Success: {step_name} completed in {round(elapsed, 2)}s.")

def main():
    """Main pipeline orchestrator entrypoint."""
    start_all = time.time()
    print("🛡️ Starting FinCompass Integrated Analytics Pipeline...")
    
    # Run each script
    for step_name, script_path in PIPELINE_STEPS:
        if not script_path.exists():
            print(f"❌ Error: Script not found at {script_path}")
            sys.exit(1)
        run_step(step_name, script_path)
        
    total_elapsed = time.time() - start_all
    print(f"\n==================================================")
    print(f"🏆 PIPELINE COMPLETED SUCCESSFULLY IN {round(total_elapsed, 2)}s!")
    print(f"==================================================")
    print(f"All database tables, models, forecasts, reports, and vector indexes are up to date.")
    print(f"\n👉 Launch the supervisory dashboard web application with:")
    print(f"   streamlit run app/streamlit_app.py")
    print(f"==================================================")


if __name__ == "__main__":
    main()
