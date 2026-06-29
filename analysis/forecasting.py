"""
FinCompass - Complaint Forecasting Module
==========================================

This module implements a monthly complaint volume forecasting system using the
Seasonal Autoregressive Integrated Moving Average (SARIMA) model. It fits the
historical time series of complaint volumes and predicts future volumes for the
next 6 months with 95% confidence intervals, providing policy analysts with a
forward-looking supervisory tool.

Steps:
1. Aggregate historical monthly complaints from `fincompass.db`.
2. Model fit: Train a statsmodels SARIMAX model.
3. Forecast: Generate predictions for the next 6 months.
4. Export: Write forecasted values and confidence bands to `analysis/forecast_results.json`.
"""

import pandas as pd
import numpy as np
import sqlite3
import json
from pathlib import Path
import statsmodels.api as sm

PROJECT_ROOT = Path("/Users/aditi/.gemini/antigravity/scratch/FinCompass")
DB_PATH = PROJECT_ROOT / "database" / "fincompass.db"
OUTPUT_JSON = PROJECT_ROOT / "analysis" / "forecast_results.json"

def get_monthly_series() -> pd.Series:
    """Fetches complaints and aggregates them as a chronological monthly time series."""
    query = """
        SELECT date, COUNT(complaint_id) as complaint_count 
        FROM complaints 
        GROUP BY year, month 
        ORDER BY date ASC
    """
    with sqlite3.connect(str(DB_PATH)) as conn:
        df = pd.read_sql_query(query, conn)
        
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    # Resample to end of month to ensure clean date spacing
    ts = df["complaint_count"].resample("M").sum()
    return ts

def fit_sarima_and_forecast(ts: pd.Series, steps: int = 6) -> dict:
    """Fits SARIMA model and forecasts the next 6 months."""
    # Convert series to float for statsmodels
    ts = ts.astype(float)
    
    # Standard SARIMA configuration (1, 1, 1) x (1, 1, 0, 12)
    # Fit SARIMAX
    try:
        model = sm.tsa.statespace.SARIMAX(
            ts,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 0, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)
        
        # Forecast
        forecast_res = results.get_forecast(steps=steps)
        forecast_mean = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int(alpha=0.05) # 95% confidence interval
        
        # Convert index dates to string
        historical_dates = ts.index.strftime("%Y-%m-%d").tolist()
        historical_values = ts.values.tolist()
        
        forecast_dates = forecast_mean.index.strftime("%Y-%m-%d").tolist()
        forecast_values = forecast_mean.values.tolist()
        
        lower_ci = conf_int.iloc[:, 0].values.tolist()
        upper_ci = conf_int.iloc[:, 1].values.tolist()
        
        # Clean negative forecasted values if any
        forecast_values = [max(0.0, v) for v in forecast_values]
        lower_ci = [max(0.0, v) for v in lower_ci]
        upper_ci = [max(0.0, v) for v in upper_ci]
        
        return {
            "status": "Success",
            "historical_dates": historical_dates,
            "historical_values": historical_values,
            "forecast_dates": forecast_dates,
            "forecast_values": [round(float(v), 2) for v in forecast_values],
            "lower_confidence_interval": [round(float(v), 2) for v in lower_ci],
            "upper_confidence_interval": [round(float(v), 2) for v in upper_ci]
        }
    except Exception as e:
        # Fallback to double exponential smoothing or naive moving average if fitting fails
        # (This acts as a safety measure for small synthetic datasets or model errors)
        print(f"SARIMA fit failed: {e}. Falling back to simple moving average.")
        last_val = ts.iloc[-1]
        mean_growth = ts.pct_change().mean() if len(ts) > 1 else 0.0
        
        forecast_values = []
        forecast_dates = []
        lower_ci = []
        upper_ci = []
        
        last_date = ts.index[-1]
        for i in range(1, steps + 1):
            next_date = last_date + pd.offsets.MonthEnd(i)
            pred = last_val * ((1 + mean_growth) ** i)
            forecast_dates.append(next_date.strftime("%Y-%m-%d"))
            forecast_values.append(pred)
            lower_ci.append(pred * 0.85)
            upper_ci.append(pred * 1.15)
            
        return {
            "status": "Fallback (Moving Average)",
            "historical_dates": ts.index.strftime("%Y-%m-%d").tolist(),
            "historical_values": ts.values.tolist(),
            "forecast_dates": forecast_dates,
            "forecast_values": [round(float(v), 2) for v in forecast_values],
            "lower_confidence_interval": [round(float(v), 2) for v in lower_ci],
            "upper_confidence_interval": [round(float(v), 2) for v in upper_ci]
        }

def main():
    """Main forecasting analyzer runner."""
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        ts = get_monthly_series()
        forecast_results = fit_sarima_and_forecast(ts, steps=6)
        
        with open(OUTPUT_JSON, "w") as f:
            json.dump(forecast_results, f, indent=4)
            
        print("Successfully generated complaint forecasting!")
        print(f"Results saved to: {OUTPUT_JSON}")
    except Exception as e:
        print(f"Error during forecasting: {e}")
        # Save a basic empty structure to ensure downstream dashboards don't crash
        with open(OUTPUT_JSON, "w") as f:
            json.dump({"status": "Failed", "error": str(e)}, f, indent=4)


if __name__ == "__main__":
    main()
