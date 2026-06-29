"""
FinCompass - Synthetic Data Generation Module
=============================================

This module generates a highly realistic synthetic dataset of 15,000 banking
complaint records spanning from January 2020 to December 2024. The data is
modeled to mimic consumer complaint patterns received by the Reserve Bank of
India (RBI) under the Integrated Ombudsman Scheme.

Features of this generator:
1. Non-uniform bank distributions (SBI receives the highest volume, public sector
   banks receive more complaints than small finance/private banks).
2. Temporal trends (Digital Banking Fraud rises year-on-year, simulating the rise in
   digital banking adoption and cyber threats).
3. Systemic variations (Public sector banks have ~40% higher average resolution times).
4. Text synthesis: Context-aware templates per subcategory to ensure text-mining
   and classification algorithms (TF-IDF + Logistic Regression) have realistic features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define configurations
NUM_ROWS = 15000

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

# Standard Indian States and UTs list
STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal", "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
    "Lakshadweep", "Delhi", "Puducherry", "Jammu and Kashmir", "Ladakh"
]

CHANNELS = ["Online", "Branch", "Phone", "Email", "Ombudsman Portal"]
CUSTOMER_SEGMENTS = ["Retail", "MSME", "Corporate"]

# Categories and Subcategories mapped to text templates for realistic NLP features
CATEGORIES = {
    "Digital Banking Fraud": [
        "Unauthorized UPI Transaction", "Phishing / Vishing Link", "SIM Clone Fraud"
    ],
    "ATM/Debit Card Issues": [
        "Cash Not Dispensed but Debited", "Card Trapped in Machine", "Unauthorized ATM Withdrawal"
    ],
    "Credit Card Complaints": [
        "Excessive Annual Fees Charged", "Billing Discrepancies", "Unsolicited Card Issuance"
    ],
    "Loan & EMI Disputes": [
        "Delay in Loan Sanction", "Incorrect Interest Rate Applied", "Non-closure of Loan Account"
    ],
    "Account Operations": [
        "Delay in Account Activation", "Failure to Update KYC", "Unauthorized Account Freeze"
    ],
    "Internet Banking": [
        "Login / OTP Failure", "Funds Transferred to Wrong Account", "Portal Unavailable / Slow"
    ],
    "Mobile Banking": [
        "App Crash / Login Error", "Mobile Wallet Transfer Issue", "Biometric Authentication Failure"
    ],
    "Mis-selling of Insurance": [
        "Insurance Bundled with Loan", "Premium Charged without Consent", "False Promises of High Returns"
    ],
    "Pension Complaints": [
        "Non-disbursal of Pension", "Incorrect Pension Calculation", "Delay in Pension Account Transfer"
    ],
    "Foreign Exchange": [
        "Delay in Inward Remittance", "Discrepancy in Exchange Rate Applied", "Forex Card Activation Issues"
    ],
    "NRI Services": [
        "NRE/NRO Account Opening Delay", "Remittance Processing Issue", "NRI KYC Verification Delay"
    ],
    "Other": [
        "Impolite Staff Behaviour", "Long Wait Times at Branch", "Inadequate Infrastructure at Branch"
    ]
}

# Template sentences per subcategory for complaint text synthesis
TEXT_TEMPLATES = {
    "Unauthorized UPI Transaction": [
        "Money debited from account via UPI transaction without my knowledge or OTP sharing.",
        "Lost money due to an unauthorized UPI transfer request which I did not authorize.",
        "Fraudulent UPI debit in my account. I did not initiate this payment.",
        "Unknown UPI transaction of rupees {amount} debited yesterday night from my savings account."
    ],
    "Phishing / Vishing Link": [
        "Clicked a text message link offering rewards and money was stolen from my account.",
        "Received fake SMS for KYC update. Account debited after opening link.",
        "Victim of phishing call where they asked for OTP. Fraudulent transfer occurred.",
        "Fraudster took control of account after I filled details in a spoofed bank website link."
    ],
    "SIM Clone Fraud": [
        "My SIM card stopped working and hackers did unauthorized transactions from my net banking.",
        "Account emptied using cloned SIM to bypass dual-factor OTP authentication. Help needed.",
        "Duplicate SIM card issued without my request, leading to fund theft from bank portal.",
        "SIM swap fraud. Mobile signal went blank and all money was stolen from savings account."
    ],
    "Cash Not Dispensed but Debited": [
        "ATM failed to dispense cash but my bank account was debited. Reversal not done yet.",
        "Withdrew {amount} at ATM. Machine clicked but no cash came. Amount deducted.",
        "Cash not dispensed but message received for successful withdrawal. Please refund money.",
        "ATM machine timed out. Cash was not dispensed but account debited. Dispute raised."
    ],
    "Card Trapped in Machine": [
        "My debit card got stuck inside the ATM slot. Branch refused to help retrieve it.",
        "Debit card trapped in ATM card reader during transaction. Had to block it immediately.",
        "ATM machine swallowed my card. No security guard present to assist. Very poor support.",
        "Debit card not ejected from ATM after transaction. Please replace the card without charge."
    ],
    "Unauthorized ATM Withdrawal": [
        "Cash withdrawn from ATM in another city while my debit card was physically with me.",
        "Cloned debit card used at ATM to steal cash. I did not perform this transaction.",
        "Unauthorized ATM withdrawal in early morning. The card was safely in my wallet.",
        "Fraudulent ATM withdrawal from my account. Requesting CCTV footage check of the booth."
    ],
    "Excessive Annual Fees Charged": [
        "Charged high credit card annual fees despite promise of lifetime free card.",
        "Levied unexpected card maintenance fee without any prior notice. Reverse the charges.",
        "Extorbitant credit card annual charges. Support executive lied during sale.",
        "Excessive annual fees debited on my credit card. Requesting fee waiver as agreed."
    ],
    "Billing Discrepancies": [
        "Credit card statement contains duplicate charges and wrong billing figures this month.",
        "Interest and finance charges added incorrectly on paid balance in credit card account.",
        "Billed twice for a single transaction on credit card. Merchant claims refund but bank didn't post.",
        "Incorrect credit card bill generated. Dispute raised but no resolution for 30 days."
    ],
    "Unsolicited Card Issuance": [
        "Received a credit card which I never applied for. Unsolicited card sent to home address.",
        "Credit card issued and activated without my written consent. Threatening CIBIL impact.",
        "Unsolicited add-on card generated for my account. Stop this malpractice immediately.",
        "Bank issued credit card without my authorization. Please cancel it and reverse card fees."
    ],
    "Delay in Loan Sanction": [
        "Applied for housing loan 2 months ago. Documentation complete but sanction delayed.",
        "Education loan approval taking too long. College admission deadline approaching. Help.",
        "Extreme delay in sanctioning home loan. Documents submitted repeatedly but no update.",
        "Car loan application stuck in processing. Bank staff not responding to emails."
    ],
    "Incorrect Interest Rate Applied": [
        "Applied for loan at {rate_low} interest but bank is charging {rate_high} on my EMI.",
        "Floating rate reduction benefit not passed to my home loan account as per RBI rules.",
        "Loan interest rate increased without giving notice or change in repo rate. Unfair practice.",
        "Bank applied higher interest rate on education loan. Not matching the signed agreement."
    ],
    "Non-closure of Loan Account": [
        "Fully paid home loan. Bank delaying return of property documents and NOC issuance.",
        "EMI still getting debited even after full repayment and closure of personal loan.",
        "Loan account not closed. NOC not provided by branch even after clearing all dues.",
        "Paid off vehicle loan but bank has not updated RTO hypothecation removal status."
    ],
    "Delay in Account Activation": [
        "Submitted account opening form 15 days ago. Activation still pending at branch.",
        "Salary account opening delayed. Unable to receive my first month salary from employer.",
        "Online bank account setup stuck at video KYC phase. Support has no solution.",
        "Branch delayed activation of joint savings account. Form lost by staff."
    ],
    "Failure to Update KYC": [
        "Submitted Re-KYC documents multiple times but account remains partially active.",
        "KYC updation failed repeatedly on online portal. Branch staff asking for physical presence.",
        "KYC records not updated despite submission of Aadhaar and PAN card details.",
        "Account frozen due to pending KYC even though documents were submitted last month."
    ],
    "Unauthorized Account Freeze": [
        "Bank froze my savings account without any prior written notice or court order.",
        "Unable to withdraw cash as my account is frozen. Customer care says audit freeze.",
        "Account blocked without explanation. Direct benefit transfers and salary stuck.",
        "Sudden freeze on corporate account. Business operations halted. Branch not clearing freeze."
    ],
    "Login / OTP Failure": [
        "Not receiving OTP for net banking transactions on my registered mobile number.",
        "Internet banking portal locked. Login credentials showing invalid after system update.",
        "OTP delivery delayed by hours. Transactions getting expired repeatedly.",
        "Net banking login portal showing error code. Unable to access accounts."
    ],
    "Funds Transferred to Wrong Account": [
        "Initiated IMPS transfer. Funds went to wrong account due to bank server glitch.",
        "Transferred funds to beneficiary. Amount debited but not credited to recipient.",
        "Wrong credit due to incorrect account number match. Requesting branch coordination.",
        "IMPS transaction failed but money not refunded back to my account after 7 days."
    ],
    "Portal Unavailable / Slow": [
        "Net banking website down during business hours. Unable to pay GST online.",
        "Bank login portal extremely slow. Gateway timeout error during critical payment.",
        "Internet banking site is throwing server 500 errors. Cannot perform any transfer.",
        "Online banking portal unavailable for past 12 hours. No information from bank."
    ],
    "App Crash / Login Error": [
        "Mobile banking app keeps crashing on startup. Reinstall did not fix it.",
        "Cannot log into mobile banking app. Showing unexpected error. Please resolve.",
        "Mobile app security updates blocking login. Stuck at device binding verification screen.",
        "Mobile banking app down. Unable to check balance or transfer funds."
    ],
    "Mobile Wallet Transfer Issue": [
        "Wallet load transaction failed but money deducted from bank account. No wallet credit.",
        "Cannot transfer money from bank account to linked e-wallet. Transaction blocked.",
        "Wallet balance disappeared after app update. Support team is unresponsive.",
        "Failed wallet payout. Money stuck in transit for past 5 working days."
    ],
    "Biometric Authentication Failure": [
        "Fingerprint scanner integration in mobile app not working. Unable to authenticate.",
        "Biometric login failed. Face ID not recognized after app updates. Forced to use PIN.",
        "Fingerprint sensor error on mobile banking. Cannot authorize transaction.",
        "App keeps asking for password instead of biometric scan. Very annoying experience."
    ],
    "Insurance Bundled with Loan": [
        "Branch forced me to buy expensive life insurance to get my home loan sanctioned.",
        "Personal loan approved only on condition of taking health insurance policy. Unfair bundle.",
        "Insurance premium debited from my account along with loan sanction. Unsolicited buy.",
        "Staff refused to process education loan without purchase of general insurance plan."
    ],
    "Premium Charged without Consent": [
        "Money debited for group health insurance scheme without my enrollment or consent.",
        "Unsolicited insurance premium charged from my savings account this month.",
        "Automated debit for policy premium without prior mandate setup. Stop this auto-debit.",
        "Insurance policy issued in my name without my authorization. Premium deducted."
    ],
    "False Promises of High Returns": [
        "Agent sold insurance policy promising double returns in 5 years. Found it is regular ULIP.",
        "Mis-led by RM to invest in policy claiming guaranteed high returns. Fraudulent sales pitch.",
        "Insurance policy sold as fixed deposit alternative. False promises made by staff.",
        "Tricked into buying high-premium insurance policy instead of savings scheme."
    ],
    "Non-disbursal of Pension": [
        "Pension for this month not credited to my pension account. Senior citizen struggling.",
        "Delay in disbursal of family pension. Branch making me run from desk to desk.",
        "Pension stopped without reason. Submitted life certificate on time last November.",
        "Bank failed to credit monthly pension. No response from branch manager."
    ],
    "Incorrect Pension Calculation": [
        "Pension amount credited is lower than my eligibility. Dearness Relief not calculated.",
        "Wrong deduction from my monthly pension. Branch unable to explain the mismatch.",
        "Discrepancy in pension arrears payout. Pension calculation sheet has errors.",
        "Incorrect pension revision. Less payout than what was sanctioned in PPO."
    ],
    "Delay in Pension Account Transfer": [
        "Requested transfer of pension account to home branch. File pending for 3 months.",
        "Delay in linking PPO with bank branch. Pension disbursal stuck for half a year.",
        "Pension transfer file stuck between two public sector bank branches. Hardship caused.",
        "Delay in processing pension account relocation. Staff claiming lack of records."
    ],
    "Delay in Inward Remittance": [
        "Foreign remittance sent by son from USA not credited to account yet. Delay of 10 days.",
        "Foreign currency inward transfer delayed. Bank asking for unnecessary physical documents.",
        "Inward remittance stuck in transit. Bank claims clearance pending from intermediary.",
        "Foreign remittance delayed. Missing urgent business payment deadlines."
    ],
    "Discrepancy in Exchange Rate Applied": [
        "Very poor exchange rate applied to inward remittance compared to card rates on date.",
        "Charged high conversion fees for forex transfer without transparent disclosure.",
        "Discrepancy in exchange rate for wire transfer. Overcharged compared to standard rates.",
        "Exchange rate applied to international transfer is much lower than market rate."
    ],
    "Forex Card Activation Issues": [
        "Forex card not activated despite loading currency. Stranded in foreign country.",
        "Forex card transactions failing abroad. Customer care numbers unreachable.",
        "Unable to reload forex card online. System showing server validation errors.",
        "Forex card locked due to wrong PIN issue. Unblocking taking more than 48 hours."
    ],
    "NRE/NRO Account Opening Delay": [
        "Documents submitted for NRE account opening. No update from bank for 30 days.",
        "NRO account activation delayed. NRI customer support not responding to emails.",
        "NRE savings account application stuck. KYC validation rejected without proper reason.",
        "Delay in opening joint NRO account. Branch asking for physical signatures again."
    ],
    "Remittance Processing Issue": [
        "Remittance from NRE to overseas account failed. Amount debited but not returned.",
        "International outward remittance rejected by bank without clear explanation.",
        "NRE account remittance transaction showing pending for a week. Funds blocked.",
        "High processing charges levied on outward NRI remittance contrary to fee schedule."
    ],
    "NRI KYC Verification Delay": [
        "KYC verification for NRO account pending. Account operational status limited.",
        "NRI account blocked due to KYC verification delays. Overseas documents ignored.",
        "KYC details updated online but not approved by NRI cell in head office. Delay.",
        "KYC verification stuck. Unable to transact online in NRE account."
    ],
    "Impolite Staff Behaviour": [
        "Branch staff behaved very rudely when I asked for bank statement. Unprofessional.",
        "Bank manager shouted at senior citizen in public. Extremely bad staff behavior.",
        "Cashier refused to accept cash deposit near closing time and used bad language.",
        "Staff was very arrogant and uncooperative during query regarding transaction failure."
    ],
    "Long Wait Times at Branch": [
        "Stuck in bank queue for 2 hours just to deposit draft. Only one counter open.",
        "Long waiting hours at loan department. No seating or water arrangement.",
        "Branch is heavily understaffed. Had to wait for 1.5 hours to update passbook.",
        "Very slow service at branch desk. Long queues and unhelpful executives."
    ],
    "Inadequate Infrastructure at Branch": [
        "No passbook printing machine working at branch for last 3 months. Inconvenience.",
        "Branch has no ramp for wheelchair access. Senior citizens cannot enter branch.",
        "AC not working in crowded branch. Poor drinking water facility at the counter.",
        "No deposit slip booklets available. Computer systems frequently down at branch."
    ]
}


def generate_complaints_data():
    """Generates the main complaints DataFrame with realistic distributions."""
    print("Generating raw complaints data...")
    
    # 1. Base random parameters
    complaint_ids = [f"COMP-{str(i).zfill(5)}" for i in range(1, NUM_ROWS + 1)]
    
    # Generate dates from Jan 1, 2020 to Dec 31, 2024
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)
    delta_days = (end_date - start_date).days
    
    # 2. Bank probabilities (SBI gets most, public sector high, private medium, sfb low)
    bank_weights = []
    for b in BANKS_DATA:
        if b["name"] == "SBI":
            weight = 0.20  # SBI gets 20% of complaints
        elif b["type"] == "Public Sector":
            weight = 0.06  # Public gets higher complaints
        elif b["type"] == "Private Sector":
            weight = 0.025 # Private medium
        else:
            weight = 0.01  # Small Finance Bank low
        bank_weights.append(weight)
    
    # Normalize weights
    bank_weights = np.array(bank_weights)
    bank_weights /= bank_weights.sum()
    
    selected_banks_idx = np.random.choice(len(BANKS_DATA), size=NUM_ROWS, p=bank_weights)
    selected_banks = [BANKS_DATA[i]["name"] for i in selected_banks_idx]
    selected_bank_types = [BANKS_DATA[i]["type"] for i in selected_banks_idx]
    
    # 3. Generate Dates with seasonality & upward trend
    # We want a rising trend of complaints, especially for Digital Banking Fraud
    # Dates are drawn with non-uniform weights to simulate an upward trend over years.
    date_weights = np.linspace(1.0, 3.0, delta_days + 1)
    date_weights /= date_weights.sum()
    
    date_offsets = np.random.choice(delta_days + 1, size=NUM_ROWS, p=date_weights)
    complaint_dates = [start_date + timedelta(days=int(offset)) for offset in date_offsets]
    
    # 4. Generate Categories & Subcategories
    # Digital Banking Fraud should grow YoY
    # Let's define the base probability distribution for categories
    category_list = list(CATEGORIES.keys())
    base_category_probs = {
        "Digital Banking Fraud": 0.12,
        "ATM/Debit Card Issues": 0.15,
        "Credit Card Complaints": 0.10,
        "Loan & EMI Disputes": 0.10,
        "Account Operations": 0.12,
        "Internet Banking": 0.08,
        "Mobile Banking": 0.08,
        "Mis-selling of Insurance": 0.07,
        "Pension Complaints": 0.05,
        "Foreign Exchange": 0.03,
        "NRI Services": 0.03,
        "Other": 0.07
    }
    
    # Normalize base probabilities
    tot_base = sum(base_category_probs.values())
    base_category_probs = {k: v/tot_base for k, v in base_category_probs.items()}
    
    complaint_categories = []
    complaint_subcategories = []
    complaint_texts = []
    
    for dt in complaint_dates:
        year = dt.year
        # Increase probability of Digital Banking Fraud by year (YoY increase)
        # e.g., 2020: base, 2021: +5%, 2022: +10%, 2023: +15%, 2024: +25%
        yoy_prob = base_category_probs.copy()
        inflation_factor = 1.0 + (year - 2020) * 0.25 # up to 2x for digital fraud
        yoy_prob["Digital Banking Fraud"] *= inflation_factor
        
        # Re-normalize
        total_p = sum(yoy_prob.values())
        norm_yoy_prob = {k: v/total_p for k, v in yoy_prob.items()}
        
        # Select category
        cats = list(norm_yoy_prob.keys())
        probs = list(norm_yoy_prob.values())
        cat = np.random.choice(cats, p=probs)
        
        # Select subcategory
        subcat = np.random.choice(CATEGORIES[cat])
        
        # Synthesize complaint text
        templates = TEXT_TEMPLATES[subcat]
        template = random.choice(templates)
        
        # Replace template placeholders with realistic numbers
        amount = random.randint(1000, 150000)
        rate_low = round(random.uniform(6.5, 9.5), 2)
        rate_high = round(rate_low + random.uniform(1.5, 4.0), 2)
        
        text = template.format(amount=amount, rate_low=rate_low, rate_high=rate_high)
        
        complaint_categories.append(cat)
        complaint_subcategories.append(subcat)
        complaint_texts.append(text)
        
    # 5. Generate State & Channel & Customer Segment
    # Standard uniform or slightly weighted random selection
    states_choices = np.random.choice(STATES, size=NUM_ROWS)
    channel_weights = [0.35, 0.20, 0.10, 0.15, 0.20] # Online & Ombudsman Portal are popular
    channels_choices = np.random.choice(CHANNELS, size=NUM_ROWS, p=channel_weights)
    customer_segments = np.random.choice(CUSTOMER_SEGMENTS, size=NUM_ROWS, p=[0.75, 0.20, 0.05])
    
    # 6. Status & Resolution Days
    # public sector banks have worse resolution times than private sector
    # status: Resolved (82%), Pending (10%), Escalated (8%)
    # Let's map resolution days based on bank type
    # Mean resolution days: Public (~45 days), Private (~28 days), SFB (~30 days)
    # Z-score outlier injection: We will intentionally inject a few massive outliers (> 150 days) for Z-score checks.
    status_choices = np.random.choice(["Resolved", "Pending", "Escalated"], size=NUM_ROWS, p=[0.82, 0.10, 0.08])
    resolution_days = []
    
    for i in range(NUM_ROWS):
        status = status_choices[i]
        b_type = selected_bank_types[i]
        
        if status == "Pending":
            resolution_days.append(np.nan)
        else:
            # Resolved or Escalated
            # Scale distribution based on bank type
            if b_type == "Public Sector":
                # Higher mean and standard deviation
                days = int(np.random.gamma(shape=3.5, scale=12.0)) + 1 # mean ~ 43 days
            elif b_type == "Private Sector":
                days = int(np.random.gamma(shape=3.0, scale=9.0)) + 1  # mean ~ 28 days
            else: # Small Finance Bank
                days = int(np.random.gamma(shape=3.0, scale=10.0)) + 1 # mean ~ 31 days
            
            # Bound the days between 1 and 180
            days = min(max(days, 1), 180)
            
            # Inject outliers (1.5% of resolved complaints get >170 days to trigger Z-score outlier checks)
            if random.random() < 0.015:
                days = random.randint(155, 220) # Outlier range
                
            resolution_days.append(days)

    # 7. Build DataFrame
    df = pd.DataFrame({
        "complaint_id": complaint_ids,
        "date": complaint_dates,
        "bank_name": selected_banks,
        "bank_type": selected_bank_types,
        "complaint_category": complaint_categories,
        "complaint_subcategory": complaint_subcategories,
        "complaint_text": complaint_texts,
        "state": states_choices,
        "channel": channels_choices,
        "status": status_choices,
        "resolution_days": resolution_days,
        "customer_segment": customer_segments
    })
    
    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)
    
    # 8. Derived columns
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.to_period("Q").astype(str).str[-1].astype(int)
    
    # Convert dates to string format for CSV raw export
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    
    return df


def main():
    """Main runner script for data generation."""
    # Setup data directories
    raw_dir = Path("/Users/aditi/.gemini/antigravity/scratch/FinCompass/data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    df = generate_complaints_data()
    
    # Save CSV
    raw_csv_path = raw_dir / "complaints_raw.csv"
    df.to_csv(raw_csv_path, index=False)
    print(f"Successfully generated {len(df)} records.")
    print(f"Raw file saved to: {raw_csv_path}")


if __name__ == "__main__":
    main()
