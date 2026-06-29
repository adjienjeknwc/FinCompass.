"""
FinCompass - Supervisory Risk Scoring Module
============================================

This module computes a systemic "Supervisory Risk Score" for each of the 20 banks
under RBI supervision. The risk engine is designed based on the RBI Department of
Supervision (DoS) guidelines, assessing banks across three primary dimensions:
1. Resolution Velocity (Average resolution days)
2. Backlog Volume (Ratio of pending to total complaints)
3. Aging Severity (Ratio of long-pending complaints >60 days to total pending)

Risk Classification:
- Red (High Risk): Score >= 65
- Amber (Medium Risk): 35 <= Score < 65
- Green (Low Risk): Score < 35

The output is exported to `analysis/risk_scores.json` and consumed by the Streamlit
Supervisory Monitoring dashboard.
"""

import pandas as pd
import numpy as np
import sqlite3
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "database" / "fincompass.db"
OUTPUT_JSON = PROJECT_ROOT / "analysis" / "risk_scores.json"

def calculate_risk_scores() -> pd.DataFrame:
    """Calculates bank-level risk scores and classifications."""
    # 1. Fetch required metrics from database
    # Get total complaints, resolved count, and avg resolution days
    with sqlite3.connect(str(DB_PATH)) as conn:
        perf_df = pd.read_sql_query("""
            SELECT 
                b.bank_name,
                b.bank_type,
                COUNT(c.complaint_id) as total_complaints,
                SUM(CASE WHEN c.status = 'Pending' THEN 1 ELSE 0 END) as pending_count,
                AVG(CASE WHEN c.status = 'Resolved' THEN c.resolution_days ELSE NULL END) as avg_res_days
            FROM complaints c
            JOIN banks b ON c.bank_id = b.bank_id
            GROUP BY b.bank_name, b.bank_type
        """, conn)
        
        # Get aging counts for pending complaints (reference date '2024-12-31')
        aging_df = pd.read_sql_query("""
            WITH pending_aging AS (
                SELECT 
                    b.bank_name,
                    c.complaint_id,
                    (julianday('2024-12-31') - julianday(c.date)) as age_days
                FROM complaints c
                JOIN banks b ON c.bank_id = b.bank_id
                WHERE c.status = 'Pending'
            )
            SELECT 
                bank_name,
                SUM(CASE WHEN age_days > 60 THEN 1 ELSE 0 END) as pending_gt_60
            FROM pending_aging
            GROUP BY bank_name
        """, conn)

    # Merge metrics
    df = pd.merge(perf_df, aging_df, on="bank_name", how="left").fillna(0)
    
    # 2. Calculate Dimension Ratios
    # a. Resolution Days Score: Normalized relative to max possible SLA (180 days)
    df["res_days_score"] = (df["avg_res_days"] / 120.0) * 100
    df["res_days_score"] = df["res_days_score"].clip(upper=100) # Cap at 100
    
    # b. Pendency Ratio: pending_count / total_complaints
    df["pendency_ratio"] = (df["pending_count"] / df["total_complaints"]) * 100
    # Normalize: if pendency is >25%, it's maximum risk (100 score)
    df["pendency_score"] = (df["pendency_ratio"] / 25.0) * 100
    df["pendency_score"] = df["pendency_score"].clip(upper=100)
    
    # c. Aging Severity: pending_gt_60 / pending_count
    df["aging_ratio"] = np.where(
        df["pending_count"] > 0,
        (df["pending_gt_60"] / df["pending_count"]) * 100,
        0.0
    )
    # Normalize: if >50% of pending is older than 60 days, maximum risk (100 score)
    df["aging_score"] = (df["aging_ratio"] / 50.0) * 100
    df["aging_score"] = df["aging_score"].clip(upper=100)
    
    # 3. Weighted Risk Score Calculation
    # Weights: 40% Resolution Speed, 30% Backlog Ratio, 30% Aging Severity
    df["risk_score"] = (
        (df["res_days_score"] * 0.40) +
        (df["pendency_score"] * 0.30) +
        (df["aging_score"] * 0.30)
    )
    df["risk_score"] = df["risk_score"].round(2)
    
    # 4. Risk Classification Mapping
    def classify_risk(score):
        if score >= 60:
            return "Red"
        elif score >= 35:
            return "Amber"
        else:
            return "Green"
            
    df["risk_level"] = df["risk_score"].apply(classify_risk)
    
    return df

def main():
    """Main risk engine runner."""
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df = calculate_risk_scores()
        
        # Convert to dictionary for export
        scores_dict = {}
        for _, row in df.iterrows():
            scores_dict[row["bank_name"]] = {
                "bank_type": row["bank_type"],
                "total_complaints": int(row["total_complaints"]),
                "pending_count": int(row["pending_count"]),
                "avg_resolution_days": round(float(row["avg_res_days"]), 1),
                "pending_gt_60": int(row["pending_gt_60"]),
                "risk_score": float(row["risk_score"]),
                "risk_level": row["risk_level"]
            }
            
        with open(OUTPUT_JSON, "w") as f:
            json.dump(scores_dict, f, indent=4)
            
        print("Successfully calculated supervisory risk scores!")
        print(f"Results saved to: {OUTPUT_JSON}")
    except Exception as e:
        print(f"Error during risk scoring: {e}")
        with open(OUTPUT_JSON, "w") as f:
            json.dump({"error": str(e)}, f, indent=4)


if __name__ == "__main__":
    main()
