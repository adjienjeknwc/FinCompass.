"""
FinCompass - Database Ingestion & ETL Loader
=============================================

This module loads the cleaned and validated CSV complaints dataset into the SQLite
database (`database/fincompass.db`). It executes the schema creation script,
populates lookup tables (`banks`, `categories`), resolves string descriptions in the
complaint records to lookup IDs, and performs aggregations to populate summary
tables (`monthly_summary`, `policy_flags`).
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Paths setup
PROJECT_ROOT = Path("/Users/aditi/.gemini/antigravity/scratch/FinCompass")
DB_PATH = PROJECT_ROOT / "database" / "fincompass.db"
SCHEMA_PATH = PROJECT_ROOT / "database" / "schema.sql"
PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "complaints_processed.csv"

# Re-use the logging setup
logging.basicConfig(
    filename=str(PROJECT_ROOT / "etl" / "etl_log.txt"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Reference definitions
BANKS_DATA = [
    {"name": "SBI", "type": "Public Sector", "license": "LIC-SBI-1001", "hq": "Mumbai"},
    {"name": "PNB", "type": "Public Sector", "license": "LIC-PNB-1002", "hq": "New Delhi"},
    {"name": "Canara Bank", "type": "Public Sector", "license": "LIC-CNB-1003", "hq": "Bengaluru"},
    {"name": "Bank of Baroda", "type": "Public Sector", "license": "LIC-BOB-1004", "hq": "Vadodara"},
    {"name": "Union Bank of India", "type": "Public Sector", "license": "LIC-UBI-1005", "hq": "Mumbai"},
    {"name": "Bank of India", "type": "Public Sector", "license": "LIC-BOI-1006", "hq": "Mumbai"},
    {"name": "UCO Bank", "type": "Public Sector", "license": "LIC-UCO-1007", "hq": "Kolkata"},
    {"name": "Central Bank of India", "type": "Public Sector", "license": "LIC-CBI-1008", "hq": "Mumbai"},
    {"name": "Indian Bank", "type": "Public Sector", "license": "LIC-IDB-1009", "hq": "Chennai"},
    {"name": "HDFC Bank", "type": "Private Sector", "license": "LIC-HDF-2001", "hq": "Mumbai"},
    {"name": "ICICI Bank", "type": "Private Sector", "license": "LIC-ICI-2002", "hq": "Mumbai"},
    {"name": "Axis Bank", "type": "Private Sector", "license": "LIC-AXS-2003", "hq": "Mumbai"},
    {"name": "Kotak Mahindra Bank", "type": "Private Sector", "license": "LIC-KOT-2004", "hq": "Mumbai"},
    {"name": "IndusInd Bank", "type": "Private Sector", "license": "LIC-IND-2005", "hq": "Mumbai"},
    {"name": "Yes Bank", "type": "Private Sector", "license": "LIC-YES-2006", "hq": "Mumbai"},
    {"name": "IDFC First Bank", "type": "Private Sector", "license": "LIC-IDF-2007", "hq": "Mumbai"},
    {"name": "Federal Bank", "type": "Private Sector", "license": "LIC-FED-2008", "hq": "Aluva"},
    {"name": "South Indian Bank", "type": "Private Sector", "license": "LIC-SIB-2009", "hq": "Thrissur"},
    {"name": "RBL Bank", "type": "Private Sector", "license": "LIC-RBL-2010", "hq": "Mumbai"},
    {"name": "Bandhan Bank", "type": "Small Finance Bank", "license": "LIC-BDN-3001", "hq": "Kolkata"}
]

CATEGORIES_DATA = {
    "Digital Banking Fraud": ["Unauthorized UPI Transaction", "Phishing / Vishing Link", "SIM Clone Fraud"],
    "ATM/Debit Card Issues": ["Cash Not Dispensed but Debited", "Card Trapped in Machine", "Unauthorized ATM Withdrawal"],
    "Credit Card Complaints": ["Excessive Annual Fees Charged", "Billing Discrepancies", "Unsolicited Card Issuance"],
    "Loan & EMI Disputes": ["Delay in Loan Sanction", "Incorrect Interest Rate Applied", "Non-closure of Loan Account"],
    "Account Operations": ["Delay in Account Activation", "Failure to Update KYC", "Unauthorized Account Freeze"],
    "Internet Banking": ["Login / OTP Failure", "Funds Transferred to Wrong Account", "Portal Unavailable / Slow"],
    "Mobile Banking": ["App Crash / Login Error", "Mobile Wallet Transfer Issue", "Biometric Authentication Failure"],
    "Mis-selling of Insurance": ["Insurance Bundled with Loan", "Premium Charged without Consent", "False Promises of High Returns"],
    "Pension Complaints": ["Non-disbursal of Pension", "Incorrect Pension Calculation", "Delay in Pension Account Transfer"],
    "Foreign Exchange": ["Delay in Inward Remittance", "Discrepancy in Exchange Rate Applied", "Forex Card Activation Issues"],
    "NRI Services": ["NRE/NRO Account Opening Delay", "Remittance Processing Issue", "NRI KYC Verification Delay"],
    "Other": ["Impolite Staff Behaviour", "Long Wait Times at Branch", "Inadequate Infrastructure at Branch"]
}


def initialize_db(conn: sqlite3.Connection):
    """Executes schema.sql to initialize database tables."""
    logging.info("Initializing database tables...")
    with open(SCHEMA_PATH, "r") as f:
        schema_sql = f.read()
    conn.executescript(schema_sql)
    logging.info("Database schema applied successfully.")


def populate_lookups(conn: sqlite3.Connection) -> tuple[dict, dict]:
    """Populates lookup tables (banks, categories) and returns translation dicts."""
    logging.info("Populating lookup tables...")
    
    # 1. Banks lookup
    banks_df = pd.DataFrame(BANKS_DATA)
    banks_df.columns = ["bank_name", "bank_type", "license_number", "headquarters"]
    
    # Check if banks table is empty before inserting
    curr_banks = pd.read_sql("SELECT * FROM banks", conn)
    if curr_banks.empty:
        banks_df.to_sql("banks", conn, if_exists="append", index=False)
        logging.info("Banks lookup populated.")
        curr_banks = pd.read_sql("SELECT * FROM banks", conn)
        
    bank_map = dict(zip(curr_banks["bank_name"], curr_banks["bank_id"]))

    # 2. Categories lookup
    cat_rows = []
    cat_id = 1
    subcat_id = 1
    for cat_name, subcats in CATEGORIES_DATA.items():
        for subcat_name in subcats:
            cat_rows.append({
                "category_id": cat_id,
                "category_name": cat_name,
                "subcategory_id": subcat_id,
                "subcategory_name": subcat_name
            })
            subcat_id += 1
        cat_id += 1
        
    cat_df = pd.DataFrame(cat_rows)
    curr_categories = pd.read_sql("SELECT * FROM categories", conn)
    if curr_categories.empty:
        cat_df.to_sql("categories", conn, if_exists="append", index=False)
        logging.info("Categories lookup populated.")
        curr_categories = pd.read_sql("SELECT * FROM categories", conn)

    # Dictionary mapping subcategory_name to (category_id, subcategory_id)
    subcat_map = {
        row["subcategory_name"]: (row["category_id"], row["subcategory_id"])
        for _, row in curr_categories.iterrows()
    }
    
    return bank_map, subcat_map


def load_complaints(conn: sqlite3.Connection, bank_map: dict, subcat_map: dict):
    """Loads transaction complaints records after mapping text fields to lookup IDs."""
    logging.info("Loading transaction complaints...")
    df = pd.read_csv(PROCESSED_CSV)
    
    # Map Bank IDs
    df["bank_id"] = df["bank_name"].map(bank_map)
    
    # Map Category and Subcategory IDs
    df["category_id"] = df["complaint_subcategory"].map(lambda x: subcat_map[x][0])
    df["subcategory_id"] = df["complaint_subcategory"].map(lambda x: subcat_map[x][1])
    
    # Keep only target table columns
    cols_to_keep = [
        "complaint_id", "date", "bank_id", "category_id", "subcategory_id",
        "complaint_text", "state", "channel", "status", "resolution_days",
        "customer_segment", "year", "month", "quarter"
    ]
    db_df = df[cols_to_keep]
    
    # In SQLite, NaNs are loaded as NULLs, so we must replace NaN with None for resolution_days
    db_df = db_df.replace({np.nan: None})
    
    # Insert records in chunks
    db_df.to_sql("complaints", conn, if_exists="append", index=False)
    logging.info(f"Loaded {len(db_df)} records into 'complaints' table.")


def populate_monthly_summary(conn: sqlite3.Connection):
    """Calculates aggregates per bank/month and populates monthly_summary table."""
    logging.info("Calculating and populating monthly summaries...")
    
    # Load all complaints from DB
    complaints_df = pd.read_sql("""
        SELECT bank_id, year, month, status, resolution_days 
        FROM complaints
    """, conn)
    
    # Compute aggregates grouping by bank, year, month
    summary_list = []
    grouped = complaints_df.groupby(["bank_id", "year", "month"])
    
    for (bank_id, year, month), group in grouped:
        total = len(group)
        resolved = int(group[group["status"] == "Resolved"].shape[0])
        pending = int(group[group["status"] == "Pending"].shape[0])
        escalated = int(group[group["status"] == "Escalated"].shape[0])
        
        # Calculate mean resolution days (exclude Nulls)
        resolved_group = group[group["resolution_days"].notnull()]
        avg_res = float(resolved_group["resolution_days"].mean()) if len(resolved_group) > 0 else None
        
        summary_list.append({
            "bank_id": bank_id,
            "year": year,
            "month": month,
            "total_complaints": total,
            "resolved_count": resolved,
            "pending_count": pending,
            "escalated_count": escalated,
            "avg_resolution_days": avg_res,
            "complaint_growth_pct": 0.0  # Placeholder, calculated next
        })
        
    summary_df = pd.DataFrame(summary_list)
    
    # Sort chronologically to calculate Month-over-Month growth
    summary_df = summary_df.sort_values(["bank_id", "year", "month"]).reset_index(drop=True)
    
    # Calculate MoM growth percentage
    for bank_id in summary_df["bank_id"].unique():
        bank_subset = summary_df[summary_df["bank_id"] == bank_id]
        shifted_complaints = bank_subset["total_complaints"].shift(1)
        
        growth = ((bank_subset["total_complaints"] - shifted_complaints) / shifted_complaints) * 100
        growth = growth.fillna(0.0).round(2)
        summary_df.loc[summary_df["bank_id"] == bank_id, "complaint_growth_pct"] = growth
        
    # Load into table
    summary_df.to_sql("monthly_summary", conn, if_exists="append", index=False)
    logging.info("Monthly summary metrics table loaded.")


def populate_policy_flags(conn: sqlite3.Connection):
    """Identifies systemic risks using statistical anomalies and records them as policy flags."""
    logging.info("Analyzing and populating policy flags...")
    
    # Fetch aggregates grouped by bank, year, quarter
    complaints_df = pd.read_sql("""
        SELECT bank_id, year, quarter, status, resolution_days 
        FROM complaints
    """, conn)
    
    grouped = complaints_df.groupby(["bank_id", "year", "quarter"])
    
    flags_list = []
    
    # Check each bank/quarter for warning indicators
    for (bank_id, year, quarter), group in grouped:
        total = len(group)
        if total < 50:
            continue  # Ignore small sample quarters to prevent noise
            
        pending_count = len(group[group["status"] == "Pending"])
        escalated_count = len(group[group["status"] == "Escalated"])
        avg_res_days = group[group["resolution_days"].notnull()]["resolution_days"].mean()
        
        # Rule 1: High average resolution days
        if avg_res_days > 75:
            flags_list.append({
                "bank_id": bank_id,
                "year": year,
                "quarter": quarter,
                "flag_type": "Supervisory Delay Alert",
                "flag_description": f"Average resolution days reached {round(avg_res_days, 1)} days, exceeding supervision SLA (60 days).",
                "severity": "High"
            })
            
        # Rule 2: Escalation rate exceeds 15%
        escalation_rate = (escalated_count / total) * 100
        if escalation_rate > 15:
            flags_list.append({
                "bank_id": bank_id,
                "year": year,
                "quarter": quarter,
                "flag_type": "High Escalation Rate Alert",
                "flag_description": f"Complaint escalation rate is {round(escalation_rate, 1)}%, indicating inadequate internal grievance redressal.",
                "severity": "Medium"
            })
            
    # Check for quarterly spike in volume (QoQ growth > 20%)
    # Let's count complaints per bank, year, quarter
    quarterly_counts = complaints_df.groupby(["bank_id", "year", "quarter"]).size().reset_index(name="count")
    quarterly_counts = quarterly_counts.sort_values(["bank_id", "year", "quarter"]).reset_index(drop=True)
    
    for bank_id in quarterly_counts["bank_id"].unique():
        bank_subset = quarterly_counts[quarterly_counts["bank_id"] == bank_id]
        shifted_count = bank_subset["count"].shift(1)
        growth = ((bank_subset["count"] - shifted_count) / shifted_count) * 100
        
        for idx, row in bank_subset.iterrows():
            g_val = growth.loc[idx]
            if pd.notnull(g_val) and g_val > 25:
                flags_list.append({
                    "bank_id": int(row["bank_id"]),
                    "year": int(row["year"]),
                    "quarter": int(row["quarter"]),
                    "flag_type": "Complaint Spike Warning",
                    "flag_description": f"QoQ complaints volume spiked by {round(g_val, 1)}% ({int(row['count'])} total). Investigation recommended.",
                    "severity": "High" if g_val > 40 else "Medium"
                })
                
    if flags_list:
        flags_df = pd.DataFrame(flags_list)
        # Drop duplicates based on unique constraint: bank_id, year, quarter, flag_type
        flags_df = flags_df.drop_duplicates(subset=["bank_id", "year", "quarter", "flag_type"])
        flags_df.to_sql("policy_flags", conn, if_exists="append", index=False)
        logging.info(f"Loaded {len(flags_df)} supervisory policy flags.")
    else:
        logging.info("No policy flags triggered.")


def main():
    """Main database loader orchestrator."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # If DB exists, clean/delete for clean load
    if DB_PATH.exists():
        DB_PATH.unlink()
        logging.info("Existing database file removed for clean load.")
        
    conn = sqlite3.connect(str(DB_PATH))
    try:
        initialize_db(conn)
        bank_map, subcat_map = populate_lookups(conn)
        load_complaints(conn, bank_map, subcat_map)
        populate_monthly_summary(conn)
        populate_policy_flags(conn)
        conn.commit()
        print("Successfully loaded FinCompass database!")
        logging.info("Pipeline load completed successfully.")
    except Exception as e:
        conn.rollback()
        logging.error(f"Error during database loading: {e}", exc_info=True)
        raise e
    finally:
        conn.close()


if __name__ == "__main__":
    main()
