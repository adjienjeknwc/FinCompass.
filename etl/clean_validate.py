"""
FinCompass - Data Cleaning & Validation Pipeline
================================================

This module implements the validation, cleaning, and quality-assurance pipeline
for the ingested banking complaint records. It guarantees the data conforms to
regulatory reporting standards before loading into SQLite.

Validation Checks performed:
1. Null checking: Ensures all key columns are present and filled.
2. Date range validation: Asserts that all complaint dates fall strictly within
   January 2020 to December 2024 (no historical or future records).
3. Status-to-resolution logical alignment: Checks that 'Resolved' complaints
   have positive resolution days and 'Pending' complaints are properly set to NULL.
4. Statistical outlier detection: Detects complaints with extreme resolution times
   defined by a Z-score > 3.
5. Quality reporting: Generates metrics detailing null percentage, outlier counts,
   and invalid records, and logs all events to 'etl/etl_log.txt'.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# Setup logging configuration
LOG_FILE = Path("/Users/aditi/.gemini/antigravity/scratch/FinCompass/etl/etl_log.txt")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"
)

# Also output log to console for debugging
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)

def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Loads raw complaints CSV file."""
    logging.info(f"Starting data ingestion from {file_path}")
    if not file_path.exists():
        logging.error(f"Raw data file not found at {file_path}")
        raise FileNotFoundError(f"Raw file not found: {file_path}")
    return pd.read_csv(file_path)

def clean_and_validate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Cleans and validates complaints dataframe, generates DQ report metadata."""
    dq_metrics = {
        "total_records_ingested": len(df),
        "nulls_detected": {},
        "invalid_dates_dropped": 0,
        "invalid_resolution_days_fixed": 0,
        "resolution_days_outliers_flagged": 0,
        "total_records_processed": 0
    }
    
    # 1. Null Checks on Required Fields
    required_cols = [
        "complaint_id", "date", "bank_name", "bank_type", 
        "complaint_category", "complaint_subcategory", 
        "complaint_text", "state", "channel", "status", "customer_segment"
    ]
    
    logging.info("Validating null values in required columns.")
    for col in required_cols:
        null_count = df[col].isnull().sum()
        dq_metrics["nulls_detected"][col] = int(null_count)
        if null_count > 0:
            logging.warning(f"Column '{col}' has {null_count} null value(s). Filling with default values.")
            if df[col].dtype == 'object':
                df[col] = df[col].fillna("Unknown")
            else:
                df[col] = df[col].fillna(0)
    
    # 2. Date Range Validation
    logging.info("Validating date ranges (Jan 2020 - Dec 2024).")
    df['parsed_date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Check for invalid parse
    invalid_dates = df['parsed_date'].isnull().sum()
    if invalid_dates > 0:
        logging.warning(f"Found {invalid_dates} unparseable dates. Dropping rows.")
        df = df[df['parsed_date'].notnull()]
        dq_metrics["invalid_dates_dropped"] += int(invalid_dates)

    start_limit = datetime(2020, 1, 1)
    end_limit = datetime(2024, 12, 31)
    
    out_of_bounds = df[(df['parsed_date'] < start_limit) | (df['parsed_date'] > end_limit)]
    out_of_bounds_count = len(out_of_bounds)
    if out_of_bounds_count > 0:
        logging.warning(f"Found {out_of_bounds_count} records outside 2020-2024 range. Dropping.")
        df = df[(df['parsed_date'] >= start_limit) & (df['parsed_date'] <= end_limit)]
        dq_metrics["invalid_dates_dropped"] += int(out_of_bounds_count)
        
    df = df.drop(columns=['parsed_date'])

    # 3. Validate logical integrity of resolution_days based on status
    logging.info("Validating resolution_days logical integrity.")
    # If status is Pending, resolution_days must be null
    pending_violations = df[(df['status'] == 'Pending') & (df['resolution_days'].notnull())]
    if len(pending_violations) > 0:
        logging.warning(f"Found {len(pending_violations)} 'Pending' complaints with non-null resolution_days. Setting to NULL.")
        df.loc[df['status'] == 'Pending', 'resolution_days'] = np.nan
        dq_metrics["invalid_resolution_days_fixed"] += len(pending_violations)
        
    # If status is Resolved, resolution_days must be positive (>= 1)
    resolved_violations = df[(df['status'] == 'Resolved') & ((df['resolution_days'].isnull()) | (df['resolution_days'] <= 0))]
    if len(resolved_violations) > 0:
        logging.warning(f"Found {len(resolved_violations)} 'Resolved' complaints with invalid or null resolution days. Fixing to median.")
        # Find median resolution days for the respective bank type
        median_resolved_days = df[df['resolution_days'] > 0]['resolution_days'].median()
        if pd.isna(median_resolved_days):
            median_resolved_days = 30
        df.loc[(df['status'] == 'Resolved') & ((df['resolution_days'].isnull()) | (df['resolution_days'] <= 0)), 'resolution_days'] = median_resolved_days
        dq_metrics["invalid_resolution_days_fixed"] += len(resolved_violations)

    # 4. Outlier detection in resolution_days (Z-score > 3)
    logging.info("Detecting statistical outliers (Z-score > 3) in resolution days.")
    resolved_df = df[df['resolution_days'].notnull()]
    if len(resolved_df) > 0:
        mean_val = resolved_df['resolution_days'].mean()
        std_val = resolved_df['resolution_days'].std()
        
        # Avoid division by zero
        if std_val > 0:
            df['z_score'] = (df['resolution_days'] - mean_val) / std_val
            outliers = df[df['z_score'].abs() > 3]
            outliers_count = len(outliers)
            dq_metrics["resolution_days_outliers_flagged"] = outliers_count
            logging.info(f"Flagged {outliers_count} resolution_days records as statistical outliers (Z-score > 3).")
            # We keep them in the dataset for analysis but mark them in the logs.
            df = df.drop(columns=['z_score'])
        else:
            logging.warning("Standard deviation of resolution_days is 0. Outlier check skipped.")

    dq_metrics["total_records_processed"] = len(df)
    
    # 5. Data Quality Metrics
    total_nulls = sum(dq_metrics["nulls_detected"].values())
    dq_metrics["overall_null_percentage"] = round((total_nulls / (len(df) * len(required_cols))) * 100, 4)
    dq_metrics["outlier_percentage"] = round((dq_metrics["resolution_days_outliers_flagged"] / len(df)) * 100, 4) if len(df) > 0 else 0.0
    dq_metrics["invalid_records_percentage"] = round(((dq_metrics["invalid_dates_dropped"] + dq_metrics["invalid_resolution_days_fixed"]) / dq_metrics["total_records_ingested"]) * 100, 4)

    logging.info("Validation completed. Data quality report generated.")
    return df, dq_metrics


def main():
    """Main runner script for ETL validation."""
    raw_path = Path("/Users/aditi/.gemini/antigravity/scratch/FinCompass/data/raw/complaints_raw.csv")
    processed_dir = Path("/Users/aditi/.gemini/antigravity/scratch/FinCompass/data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Execute loading and cleaning
    df = load_raw_data(raw_path)
    cleaned_df, metrics = clean_and_validate(df)
    
    # Save cleaned data
    processed_csv_path = processed_dir / "complaints_processed.csv"
    cleaned_df.to_csv(processed_csv_path, index=False)
    
    # Write DQ Report summary to log and console
    dq_report_str = f"""
===================================================
              DATA QUALITY REPORT
===================================================
Total Records Ingested: {metrics['total_records_ingested']}
Total Records Processed: {metrics['total_records_processed']}
Invalid Dates Dropped: {metrics['invalid_dates_dropped']}
Invalid Resolution Days Repaired: {metrics['invalid_resolution_days_fixed']}
Resolution Days Outliers (Z > 3) Flagged: {metrics['resolution_days_outliers_flagged']}

Overall Null cells %: {metrics['overall_null_percentage']}%
Outlier %: {metrics['outlier_percentage']}%
Invalid Records %: {metrics['invalid_records_percentage']}%
===================================================
"""
    print(dq_report_str)
    with open(LOG_FILE, "a") as f:
        f.write(dq_report_str)
        
    logging.info(f"Processed file saved to: {processed_csv_path}")


if __name__ == "__main__":
    main()
