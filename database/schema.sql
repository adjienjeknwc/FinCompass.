-- FinCompass Database Schema
-- SQLite schema for supervisory analytics data

PRAGMA foreign_keys = ON;

-- 1. Banks Lookup Table
CREATE TABLE IF NOT EXISTS banks (
    bank_id INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name TEXT NOT NULL UNIQUE,
    bank_type TEXT CHECK(bank_type IN ('Public Sector', 'Private Sector', 'Small Finance Bank')) NOT NULL,
    license_number TEXT UNIQUE NOT NULL,
    headquarters TEXT NOT NULL
);

-- 2. Categories & Subcategories Lookup Table
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER NOT NULL,
    category_name TEXT NOT NULL,
    subcategory_id INTEGER NOT NULL PRIMARY KEY,
    subcategory_name TEXT NOT NULL UNIQUE
);

-- 3. Complaints Table (Main Transactional Table)
CREATE TABLE IF NOT EXISTS complaints (
    complaint_id TEXT PRIMARY KEY,
    date TEXT NOT NULL,
    bank_id INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    subcategory_id INTEGER NOT NULL,
    complaint_text TEXT NOT NULL,
    state TEXT NOT NULL,
    channel TEXT CHECK(channel IN ('Online', 'Branch', 'Phone', 'Email', 'Ombudsman Portal')) NOT NULL,
    status TEXT CHECK(status IN ('Resolved', 'Pending', 'Escalated')) NOT NULL,
    resolution_days INTEGER, -- NULL for Pending
    customer_segment TEXT CHECK(customer_segment IN ('Retail', 'MSME', 'Corporate')) NOT NULL,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    FOREIGN KEY (bank_id) REFERENCES banks(bank_id),
    FOREIGN KEY (subcategory_id) REFERENCES categories(subcategory_id)
);

-- 4. Monthly Summary Aggregation Table
CREATE TABLE IF NOT EXISTS monthly_summary (
    summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_id INTEGER NOT NULL,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    total_complaints INTEGER DEFAULT 0,
    resolved_count INTEGER DEFAULT 0,
    pending_count INTEGER DEFAULT 0,
    escalated_count INTEGER DEFAULT 0,
    avg_resolution_days REAL,
    complaint_growth_pct REAL,
    FOREIGN KEY (bank_id) REFERENCES banks(bank_id),
    UNIQUE(bank_id, year, month)
);

-- 5. Policy & Supervisory Flags Table
CREATE TABLE IF NOT EXISTS policy_flags (
    flag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_id INTEGER NOT NULL,
    year INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    flag_type TEXT NOT NULL,
    flag_description TEXT NOT NULL,
    severity TEXT CHECK(severity IN ('Low', 'Medium', 'High', 'Critical')) NOT NULL,
    FOREIGN KEY (bank_id) REFERENCES banks(bank_id),
    UNIQUE(bank_id, year, quarter, flag_type)
);

-- Create Indexes for Analytical Queries
CREATE INDEX IF NOT EXISTS idx_complaints_bank ON complaints(bank_id);
CREATE INDEX IF NOT EXISTS idx_complaints_date ON complaints(date);
CREATE INDEX IF NOT EXISTS idx_complaints_category ON complaints(category_id);
CREATE INDEX IF NOT EXISTS idx_monthly_summary_lookup ON monthly_summary(bank_id, year, month);
