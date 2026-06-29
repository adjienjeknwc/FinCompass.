"""
FinCompass - Database Queries Module
====================================

This module contains analytical SQL query functions wrapping database reads from
`fincompass.db`. These queries implement complex aggregations, window functions,
and joins to provide key metrics for Streamlit dashboard rendering and supervisory
reports.

Implemented Query Functions:
1. get_top_10_banks: Top banks by complaint volume with YoY change (2023 vs 2024).
2. get_category_trends: Quarterly trends of complaints by category.
3. get_resolution_efficiency: Rank banks based on average resolution speed and rates.
4. get_state_heatmap: State vs Complaint Category distribution.
5. get_pendency_analysis: Pending complaints categorized into age buckets (>30, >60, >90 days).
6. get_rising_categories: Categories with QoQ growth greater than 20%.
"""

import sqlite3
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "database" / "fincompass.db"

def get_connection() -> sqlite3.Connection:
    """Helper function to get SQLite connection."""
    return sqlite3.connect(str(DB_PATH))

def get_top_10_banks() -> pd.DataFrame:
    """Returns top 10 banks by complaint volume and their Year-over-Year (YoY) change."""
    query = """
    WITH yearly_counts AS (
        SELECT 
            b.bank_name,
            b.bank_type,
            SUM(CASE WHEN c.year = 2023 THEN 1 ELSE 0 END) as count_2023,
            SUM(CASE WHEN c.year = 2024 THEN 1 ELSE 0 END) as count_2024,
            COUNT(c.complaint_id) as total_volume
        FROM complaints c
        JOIN banks b ON c.bank_id = b.bank_id
        GROUP BY b.bank_id
    )
    SELECT 
        bank_name,
        bank_type,
        count_2023,
        count_2024,
        total_volume,
        CASE 
            WHEN count_2023 = 0 THEN 0.0
            ELSE ROUND(((CAST(count_2024 AS REAL) - count_2023) / count_2023) * 100, 2)
        END as yoy_change_pct
    FROM yearly_counts
    ORDER BY total_volume DESC
    LIMIT 10;
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn)

def get_category_trends() -> pd.DataFrame:
    """Returns complaint counts by category and quarter (chronological trend)."""
    query = """
    SELECT 
        cat.category_name,
        c.year || '-Q' || c.quarter as quarter_label,
        COUNT(c.complaint_id) as complaint_count
    FROM complaints c
    JOIN categories cat ON c.category_id = cat.category_id
    GROUP BY cat.category_name, c.year, c.quarter
    ORDER BY c.year, c.quarter, cat.category_name;
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn)

def get_resolution_efficiency() -> pd.DataFrame:
    """Ranks banks based on average resolution days for resolved complaints."""
    query = """
    SELECT 
        b.bank_name,
        b.bank_type,
        COUNT(c.complaint_id) as total_complaints,
        SUM(CASE WHEN c.status = 'Resolved' THEN 1 ELSE 0 END) as resolved_count,
        ROUND((CAST(SUM(CASE WHEN c.status = 'Resolved' THEN 1 ELSE 0 END) AS REAL) / COUNT(c.complaint_id)) * 100, 2) as resolution_rate_pct,
        ROUND(AVG(CASE WHEN c.status = 'Resolved' THEN c.resolution_days ELSE NULL END), 1) as avg_resolution_days,
        RANK() OVER (
            ORDER BY AVG(CASE WHEN c.status = 'Resolved' THEN c.resolution_days ELSE NULL END) ASC
        ) as efficiency_rank
    FROM complaints c
    JOIN banks b ON c.bank_id = b.bank_id
    GROUP BY b.bank_id
    ORDER BY avg_resolution_days ASC;
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn)

def get_state_heatmap() -> pd.DataFrame:
    """Returns complaints cross-tabulated by state and category."""
    query = """
    SELECT 
        c.state,
        cat.category_name,
        COUNT(c.complaint_id) as complaint_count
    FROM complaints c
    JOIN categories cat ON c.category_id = cat.category_id
    GROUP BY c.state, cat.category_name
    ORDER BY c.state, complaint_count DESC;
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn)

def get_pendency_analysis(ref_date: str = '2024-12-31') -> pd.DataFrame:
    """
    Categorizes pending complaints into aging buckets relative to reference date.
    Buckets: >30 days, >60 days, >90 days pending.
    """
    query = f"""
    WITH pending_aging AS (
        SELECT 
            b.bank_name,
            c.complaint_id,
            (julianday('{ref_date}') - julianday(c.date)) as age_days
        FROM complaints c
        JOIN banks b ON c.bank_id = b.bank_id
        WHERE c.status = 'Pending'
    )
    SELECT 
        bank_name,
        COUNT(complaint_id) as total_pending,
        SUM(CASE WHEN age_days > 30 THEN 1 ELSE 0 END) as pending_30_days,
        SUM(CASE WHEN age_days > 60 THEN 1 ELSE 0 END) as pending_60_days,
        SUM(CASE WHEN age_days > 90 THEN 1 ELSE 0 END) as pending_90_days
    FROM pending_aging
    GROUP BY bank_name
    ORDER BY total_pending DESC;
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn)

def get_rising_categories() -> pd.DataFrame:
    """
    Identifies categories showing QoQ growth > 20% in the latest quarter of the data (2024-Q4).
    Compares 2024-Q4 vs 2024-Q3.
    """
    query = """
    WITH quarterly_cat_counts AS (
        SELECT 
            cat.category_name,
            c.year,
            c.quarter,
            COUNT(c.complaint_id) as count
        FROM complaints c
        JOIN categories cat ON c.category_id = cat.category_id
        WHERE (c.year = 2024 AND c.quarter IN (3, 4))
        GROUP BY cat.category_name, c.year, c.quarter
    ),
    pivoted AS (
        SELECT 
            category_name,
            SUM(CASE WHEN quarter = 3 THEN count ELSE 0 END) as count_q3,
            SUM(CASE WHEN quarter = 4 THEN count ELSE 0 END) as count_q4
        FROM quarterly_cat_counts
        GROUP BY category_name
    )
    SELECT 
        category_name,
        count_q3 as previous_quarter_count,
        count_q4 as latest_quarter_count,
        ROUND(((CAST(count_q4 AS REAL) - count_q3) / count_q3) * 100, 2) as qoq_growth_pct
    FROM pivoted
    WHERE count_q3 > 0 AND ((CAST(count_q4 AS REAL) - count_q3) / count_q3) > 0.20
    ORDER BY qoq_growth_pct DESC;
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn)
