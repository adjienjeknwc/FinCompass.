# FinCompass: Banking Complaint Intelligence & Supervisory Analytics Platform

🛡️ **FinCompass** is an end-to-end data engineering and supervisory analytics platform designed to simulate the Reserve Bank of India’s (RBI) consumer protection and market intelligence operations. The platform ingests synthetic grievance records, cleanses and structures them into a relational database, applies statistical hypothesis testing and time-series forecasting, trains an NLP model to classify raw complaint texts, compiles a vector search index, and serves interactive supervisory dashboards alongside automated PDF/Word brief generators.

This project is built specifically to demonstrate core competencies required for RBI Young Professional (YP) roles in Data Analytics and Policy Research.

---

## 🏗️ Platform Architecture

```
                                    +-----------------------+
                                    |  15,000 Complaints    |
                                    |  (Synthetic Raw Data) |
                                    +-----------+-----------+
                                                |
                                                v (ETL Pipeline: pandas & NumPy)
                                    +-----------------------+
                                    | clean_validate.py     |
                                    | (Z-Score Outliers, DQ)|
                                    +-----------+-----------+
                                                |
                                                v (Ingestion: SQLAlchemy)
                                    +-----------------------+
                                    |    fincompass.db      |
                                    |  (SQLite3 Relational) |
                                    +----+-----------+------+
                                         |           |
             +---------------------------+           +--------------------------+
             |                                                                  |
             v (Machine Learning & NLP)                                         v (Analytics & Time-Series)
+-----------------------+                                           +-----------------------+
|  train_classifier.py  |                                           |   stats_analysis.py   |
| (TF-IDF + LogReg pipeline)                                        |  (Welch's t-test, OLS)|
+-----------+-----------+                                           +-----------+-----------+
            |                                                                   |
            v                                                                   v
+-----------------------+                                           +-----------------------+
| complaint_classifier  |                                           |    forecasting.py     |
|   (Saved .pkl Model)  |                                           | (SARIMA Forecast JSON)|
+-----------+-----------+                                           +-----------+-----------+
            |                                                                   |
            +----------------------------+       +------------------------------+
                                         |       |
                                         v       v
                                    +---------------+
                                    | streamlit_app | <---+ (RAG Assistant)
                                    | (6-Page App)  |     |
                                    +-------+-------+     +-------------------------+
                                            |                                       |
                                            v (Report Automation)        +----------+----------+
                                    +---------------+                    |   chatbot.py        |
                                    | Word/Excel MIS|                    | (LangChain + Chroma) |
                                    |   Generators  |                    +----------+----------+
                                    +---------------+                               |
                                                                                    v
                                                                             [Gemini Flash API]
```

---

## 🛠️ Technology Stack

| Layer | Technologies Used | Description |
|---|---|---|
| **Core Language** | Python 3.10+ | Primary language for pipeline, analytics, and modeling. |
| **Database** | SQLite3, SQLAlchemy | Relational storage utilizing complex SQL (Window functions, JOINs). |
| **Pipeline** | Pandas, NumPy | Structured ETL validation, cleaning, and log generation. |
| **Statistics** | SciPy, StatsModels | Welch’s t-test, OLS linear regression, Mann-Kendall trend test. |
| **Time-Series** | statsmodels SARIMAX | 6-month seasonal complaint volume forecasting. |
| **ML / NLP** | scikit-learn | TF-IDF pipeline + Multinomial Logistic Regression. |
| **GenAI / RAG** | LangChain, ChromaDB, Gemini Flash | Semantic vector search and Q&A over monthly supervisory briefs. |
| **Automation** | python-docx, openpyxl, APScheduler | Automatic Word report briefs, conditional-formatted Excel MIS. |
| **Visualization** | Plotly, Streamlit | Interactive multi-page dashboard. |

---

## 📋 Methodology

### 1. Data Synthesis & Distribution Rationale
Instead of uniform distributions, data is modeled to represent realistic system loads under the Integrated Ombudsman Scheme:
* **Market Share Weighting**: SBI accounts for ~20% of complaints, matching its massive domestic footprint, followed by public sector peers.
* **Macroeconomic Trends**: Digital Banking Fraud exhibits a Year-over-Year (YoY) compound growth rate of 25% to simulate post-COVID digital transactions.
* **Supervisory Discrepancy**: Public sector banks are assigned a gamma distribution for `resolution_days` yielding a ~43-day mean, representing a 40% slower resolution velocity compared to private counterparts.

### 2. ETL Cleaning & Data Quality Assurance (`clean_validate.py`)
* Outliers in `resolution_days` are flagged using a standard Z-score threshold ($Z > 3$).
* Inconsistencies are repaired (e.g., setting `resolution_days` to null for pending complaints, and filling resolved entries lacking day counts with the bank type's median).
* Full execution steps and metrics are exported to `etl/etl_log.txt`.

### 3. Statistical Inference & Predictive Analytics
* **Welch’s t-test**: Standard independent two-sample t-test with unequal variances to compare public vs. private bank resolution days.
* **OLS Regression**: Predicts resolution days based on the bank type, channel, and complaint category, identifying coefficients to isolate systemic friction points.
* **Mann-Kendall Test**: Non-parametric test checking for a monotonic upward trend in Digital Banking Fraud.

---

## 🚀 How to Run the Platform

### Prerequisites
Make sure you have Python 3.10+ installed.

### 1. Clone the Repository & Install Dependencies
```bash
git clone https://github.com/yourusername/FinCompass.git
cd FinCompass
pip install -r requirements.txt
```

### 2. Run the End-to-End Orchestrator
Execute the complete pipeline in sequence (data generation -> validation -> database load -> ML training -> statistics -> vector store building -> MIS generation):
```bash
python run_all.py
```

### 3. Setup Gemini API Key (Optional)
To test the AI Policy Assistant page with the live Gemini Flash model, create a `.env` file at the root:
```env
GEMINI_API_KEY=your_google_studio_api_key_here
```
*(If no key is provided, the chatbot will gracefully execute in local semantic synthesis mode).*

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Key Findings (Based on Simulated Data)

1. **Systemic Backlogs**: Public Sector banks exhibit significantly higher average resolution times (Welch's t-test: $p < 0.001$, rejecting the null hypothesis).
2. **Fraud Trends**: Digital Banking Fraud is expanding at an alarming rate, showing a statistically significant upward monotonic trend (Mann-Kendall Tau: $+0.84$, $p < 0.0001$).
3. **Friction Channels**: OLS regression indicates that complaints filed via the physical branch channel add an average of 14.2 days to resolution compared to online filings.

---

## 🏛️ Policy Implications (DoS Brief)

### I. Institutional Action Plan
The persistent discrepancy in grievance redressal velocity between Public and Private sector banks represents a structural bottleneck in consumer protection. It is recommended that the Department of Supervision (DoS) mandate specialized workflow automation for public sector banks whose average resolution days exceed the 45-day threshold. Periodic spot audits are suggested to inspect internal bank ombudsman frameworks.

### II. Mitigation of Digital Transaction Risks
Given the statistically validated acceleration of Digital Banking Fraud, a regulatory update to the Master Direction on Digital Banking Security Controls is advised. This should include mandatory real-time transaction cooling periods for high-risk accounts and strict security audits for third-party UPI integrations, specifically targeting Small Finance Banks exhibiting sudden QoQ complaint volume spikes.

---

## 🖼️ Dashboard Preview Placeholders

* **Page 1 - Executive Dashboard**
  *(Place Page 1 Screenshot here)*
  
* **Page 2 - Complaint Deep Dive & Real-time Classifier**
  *(Place Page 2 Screenshot here)*
  
* **Page 3 - DoS Supervisory Monitoring Dashboard**
  *(Place Page 3 Screenshot here)*

* **Page 4 - Heatmap & Geographic Analysis**
  *(Place Page 4 Screenshot here)*

* **Page 5 - Time-Series Forecasting & Stats**
  *(Place Page 5 Screenshot here)*

* **Page 6 - AI Policy Assistant Chat interface**
  *(Place Page 6 Screenshot here)*

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
