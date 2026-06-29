"""
FinCompass - Complaint Classifier Evaluation Script
=====================================================

This module loads the trained complaint classification pipeline, performs validation
runs against testing datasets, and outputs detailed classification metrics. It allows
supervisors to inspect the model's reliability before rolling out auto-classification.
"""

import pickle
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score

PROJECT_ROOT = Path("/Users/aditi/.gemini/antigravity/scratch/FinCompass")
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "complaint_classifier.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "complaints_processed.csv"

def evaluate():
    """Loads model and prints classification performance report."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train the model first.")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}.")
        
    # Load model pipeline
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        
    # Load complaints
    df = pd.read_csv(DATA_PATH)
    X = df[["complaint_text", "channel"]]
    y = df["complaint_category"]
    
    # Run predictions
    y_pred = model.predict(X)
    
    acc = accuracy_score(y, y_pred)
    print("====================================================")
    print("      SUPERVISORY COMPLAINT CLASSIFIER EVALUATION    ")
    print("====================================================")
    print(f"Overall Accuracy: {round(acc * 100, 2)}%")
    print("\nDetailed Classification Metrics:")
    print(classification_report(y, y_pred))
    print("====================================================")


if __name__ == "__main__":
    evaluate()
