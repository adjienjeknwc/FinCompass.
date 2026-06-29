"""
FinCompass - Statistical Analysis Module
========================================

This module performs advanced statistical computations on the processed complaints
dataset, mimicking policy research done at the Reserve Bank of India (RBI).

Computations implemented:
1. Descriptive Statistics: Computes mean, median, standard deviation, and skewness of
   complaint resolution times, grouped by bank type.
2. Welch's t-test: Tests the hypothesis that Public Sector banks have significantly
   slower grievance resolution times compared to Private Sector banks.
3. Ordinary Least Squares (OLS) Linear Regression: Analyzes the statistical impact
   of bank type, communication channel, and complaint category on resolution days.
4. Mann-Kendall Trend Test: Evaluates whether Digital Banking Fraud complaints show a
   statistically significant upward trend from 2020 to 2024.
5. JSON Export: Writes all statistical outputs to `analysis/stats_results.json` for
   Streamlit visualization.
"""

import pandas as pd
import numpy as np
import json
import sqlite3
from pathlib import Path
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "database" / "fincompass.db"
OUTPUT_JSON = PROJECT_ROOT / "analysis" / "stats_results.json"

def get_data() -> pd.DataFrame:
    """Fetches resolved complaints data from the SQLite database."""
    query = """
        SELECT 
            c.resolution_days,
            b.bank_type,
            c.channel,
            cat.category_name as complaint_category,
            c.year,
            c.month
        FROM complaints c
        JOIN banks b ON c.bank_id = b.bank_id
        JOIN categories cat ON c.category_id = cat.category_id
    """
    with sqlite3.connect(str(DB_PATH)) as conn:
        return pd.read_sql_query(query, conn)

def run_descriptive_stats(df: pd.DataFrame) -> dict:
    """Computes descriptive stats of resolution days grouped by bank type."""
    # Filter for resolved complaints only
    resolved = df[df["resolution_days"].notnull()].copy()
    resolved["resolution_days"] = resolved["resolution_days"].astype(float)
    
    grouped = resolved.groupby("bank_type")["resolution_days"]
    
    desc_stats = {}
    for name, group in grouped:
        desc_stats[name] = {
            "mean": round(float(group.mean()), 2),
            "median": round(float(group.median()), 2),
            "std": round(float(group.std()), 2),
            "skew": round(float(group.skew()), 2),
            "count": int(group.count())
        }
    return desc_stats

def run_welch_ttest(df: pd.DataFrame) -> dict:
    """Performs Welch's t-test comparing public and private sector resolution times."""
    resolved = df[df["resolution_days"].notnull()]
    
    public_days = resolved[resolved["bank_type"] == "Public Sector"]["resolution_days"].values
    private_days = resolved[resolved["bank_type"] == "Private Sector"]["resolution_days"].values
    
    t_stat, p_val = stats.ttest_ind(public_days, private_days, equal_var=False)
    
    reject_h0 = bool(p_val < 0.05)
    
    interpretation = (
        "Reject Null Hypothesis: Public sector banks take significantly longer to resolve "
        "complaints than private sector banks (p-value < 0.05)."
        if reject_h0 else
        "Fail to Reject Null Hypothesis: There is no statistically significant difference in "
        "resolution speed between public and private sector banks (p-value >= 0.05)."
    )
    
    return {
        "t_statistic": round(float(t_stat), 4),
        "p_value": float(p_val),
        "reject_null": reject_h0,
        "interpretation": interpretation
    }

def run_ols_regression(df: pd.DataFrame) -> dict:
    """Runs OLS regression: resolution_days ~ bank_type + channel + category."""
    resolved = df[df["resolution_days"].notnull()].copy()
    
    # Run regression using statsmodels formula API
    model = smf.ols(
        "resolution_days ~ C(bank_type, Treatment('Private Sector')) + C(channel, Treatment('Online')) + C(complaint_category, Treatment('Other'))",
        data=resolved
    ).fit()
    
    # Extract coefficients, standard errors, t-stats, and p-values
    coef = model.params
    pvals = model.pvalues
    tvals = model.tvalues
    
    results = {}
    for var in coef.index:
        # Clean variable names for front-end rendering
        clean_var = var.replace("C(bank_type, Treatment('Private Sector'))[T.", "Bank Type: ")
        clean_var = clean_var.replace("C(channel, Treatment('Online'))[T.", "Channel: ")
        clean_var = clean_var.replace("C(complaint_category, Treatment('Other'))[T.", "Category: ")
        clean_var = clean_var.replace("]", "")
        
        results[clean_var] = {
            "coefficient": round(float(coef[var]), 4),
            "t_statistic": round(float(tvals[var]), 4),
            "p_value": float(pvals[var]),
            "significant": bool(pvals[var] < 0.05)
        }
        
    return {
        "r_squared": round(float(model.rsquared), 4),
        "adj_r_squared": round(float(model.rsquared_adj), 4),
        "f_statistic": round(float(model.fvalue), 4),
        "f_pvalue": float(model.f_pvalue),
        "coefficients": results
    }

def run_mann_kendall_test(df: pd.DataFrame) -> dict:
    """
    Implements a self-contained Mann-Kendall trend test on Digital Banking Fraud.
    We aggregate fraud complaints monthly and test for a monotonic trend.
    """
    # Filter digital banking fraud and group monthly
    fraud_df = df[df["complaint_category"] == "Digital Banking Fraud"].copy()
    monthly_counts = fraud_df.groupby(["year", "month"]).size().reset_index(name="count")
    
    # Sort chronologically
    monthly_counts = monthly_counts.sort_values(["year", "month"]).reset_index(drop=True)
    x = monthly_counts["count"].values
    n = len(x)
    
    if n < 3:
        return {
            "trend": "Insufficient Data",
            "p_value": 1.0,
            "tau": 0.0,
            "interpretation": "Insufficient data to establish monthly trend."
        }
        
    # Mann-Kendall calculation
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(x[j] - x[i])
            
    # Variance of S
    var_s = (n * (n - 1) * (2 * n + 5)) / 18.0
    
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
        
    # p-value (two-tailed)
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = float(s) / (0.5 * n * (n - 1))
    
    reject_h0 = bool(p_val < 0.05)
    
    if reject_h0 and s > 0:
        trend = "Statistically Significant Upward Trend"
    elif reject_h0 and s < 0:
        trend = "Statistically Significant Downward Trend"
    else:
        trend = "No Statistically Significant Trend"
        
    interpretation = (
        f"The Mann-Kendall test confirms a {trend.lower()} "
        f"in Digital Banking Fraud monthly counts (Tau = {round(tau, 3)}, p-value = {round(p_val, 6)})."
    )
    
    # Convert numpy array/types for JSON serialization
    return {
        "trend": trend,
        "tau": round(tau, 4),
        "s_statistic": int(s),
        "z_score": round(float(z), 4),
        "p_value": float(p_val),
        "reject_null": reject_h0,
        "interpretation": interpretation,
        "monthly_counts": monthly_counts["count"].tolist()
    }

def main():
    """Main statistical analyzer runner."""
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    
    df = get_data()
    
    desc_stats = run_descriptive_stats(df)
    t_test = run_welch_ttest(df)
    regression = run_ols_regression(df)
    mk_test = run_mann_kendall_test(df)
    
    stats_output = {
        "descriptive_statistics": desc_stats,
        "welch_t_test": t_test,
        "ols_regression": regression,
        "mann_kendall_trend_test": mk_test
    }
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(stats_output, f, indent=4)
        
    print("Successfully ran statistical analysis!")
    print(f"Results saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
