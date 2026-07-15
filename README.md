<div align="center">

# 🛡️ FinCompass
### Banking Complaint Intelligence & Supervisory Analytics Platform

**An end-to-end data engineering and supervisory analytics platform simulating RBI-grade consumer protection and market intelligence operations.**

[![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![StatsModels](https://img.shields.io/badge/StatsModels-8CAAE6?style=for-the-badge&logo=python&logoColor=white)](https://www.statsmodels.org/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=for-the-badge&logo=sqlalchemy&logoColor=white)](https://www.sqlalchemy.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F00?style=for-the-badge&logo=databricks&logoColor=white)](https://www.trychroma.com/)
[![Gemini](https://img.shields.io/badge/Gemini_Flash-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/python/)

[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](CONTRIBUTING.md)
[![Status](https://img.shields.io/badge/Status-Simulation%20%2F%20Portfolio%20Project-blue?style=flat-square)]()

</div>

---

## 📖 Overview

**FinCompass** is a full-stack data engineering and analytics platform that simulates the **Reserve Bank of India's (RBI)** consumer protection and market intelligence operations end-to-end. It ingests synthetic grievance records, cleans and structures them into a relational database, applies statistical hypothesis testing and time-series forecasting, trains an NLP classifier on raw complaint text, builds a vector search index for semantic Q&A, and serves everything through an interactive **6-page Streamlit supervisory dashboard** — complete with automated PDF/Word/Excel brief generation.

This project was purpose-built to demonstrate the core competencies required for **RBI Young Professional (YP) roles in Data Analytics and Policy Research**: data pipeline engineering, statistical inference, forecasting, applied ML/NLP, and translating quantitative findings into policy-ready supervisory briefs.



---

## 🏗️ Platform Architecture

```
                            ┌───────────────────────────┐
                            │   15,000 Complaints         │
                            │   (Synthetic Raw Data)       │
                            └──────────────┬──────────────┘
                                           │  ETL Pipeline (pandas & NumPy)
                                           ▼
                            ┌───────────────────────────┐
                            │   clean_validate.py          │
                            │   (Z-score outliers, DQ)      │
                            └──────────────┬──────────────┘
                                           │  Ingestion (SQLAlchemy)
                                           ▼
                            ┌───────────────────────────┐
                            │      fincompass.db           │
                            │   (SQLite3 Relational DB)     │
                            └──────┬─────────────┬────────┘
                                  │             │
                  ML & NLP        ▼             ▼        Analytics & Time-Series
              ┌───────────────────────┐   ┌───────────────────────┐
              │  train_classifier.py    │   │   stats_analysis.py     │
              │  (TF-IDF + LogReg)      │   │  (Welch's t-test, OLS)  │
              └────────────┬──────────┘   └────────────┬──────────┘
                           ▼                            ▼
              ┌───────────────────────┐   ┌───────────────────────┐
              │  complaint_classifier   │   │    forecasting.py       │
              │     (Saved .pkl)         │   │  (SARIMA Forecast JSON) │
              └────────────┬──────────┘   └────────────┬──────────┘
                           │                            │
                           └─────────────┬──────────────┘
                                        ▼
                            ┌───────────────────────────┐
                            │      streamlit_app           │◀──┐  RAG Assistant
                            │      (6-Page App)             │   │
                            └──────────────┬──────────────┘   │
                                           │                  │
                                           ▼        ┌──────────┴──────────┐
                            ┌───────────────────────┐   │     chatbot.py         │
                            │   Word / Excel MIS       │   │ (LangChain + Chroma)   │
                            │      Generators            │   └──────────┬──────────┘
                            └───────────────────────┘                 │
                                                                       ▼
                                                          ┌─────────────────────┐
                                                          │   Gemini Flash API    │
                                                          └─────────────────────┘
```

---

## 🛠️ Technology Stack

<div align="center">

| Layer | Technologies | Purpose |
|---|---|---|
| 🐍 **Core Language** | Python 3.10+ | Primary language for pipeline, analytics, and modeling |
| 🗄️ **Database** | SQLite3, SQLAlchemy | Relational storage with window functions, JOINs |
| ⚙️ **Pipeline** | Pandas, NumPy | Structured ETL validation, cleaning, log generation |
| 📊 **Statistics** | SciPy, StatsModels | Welch's t-test, OLS regression, Mann-Kendall trend test |
| 📈 **Time-Series** | statsmodels SARIMAX | 6-month seasonal complaint volume forecasting |
| 🤖 **ML / NLP** | scikit-learn | TF-IDF pipeline + Multinomial Logistic Regression |
| 🧠 **GenAI / RAG** | LangChain, ChromaDB, Gemini Flash | Semantic vector search & Q&A over supervisory briefs |
| 📝 **Automation** | python-docx, openpyxl, APScheduler | Automated Word briefs, conditional-formatted Excel MIS |
| 🎨 **Visualization** | Plotly, Streamlit | Interactive multi-page dashboard |

</div>

---

## 📋 Methodology

### 1. Data Synthesis & Distribution Rationale
Synthetic data is modeled to reflect realistic system loads under the Integrated Ombudsman Scheme, rather than uniform random distributions:
- **Market share weighting** — SBI accounts for ~20% of complaints, matching its domestic footprint, followed by public sector peers.
- **Macroeconomic trends** — Digital Banking Fraud is modeled with a 25% YoY compound growth rate, simulating post-COVID digital transaction growth.
- **Supervisory discrepancy** — Public sector banks are assigned a gamma distribution for `resolution_days`, yielding a ~43-day mean (~40% slower than private counterparts).

### 2. ETL Cleaning & Data Quality Assurance (`clean_validate.py`)
- Outliers in `resolution_days` flagged via a standard **Z-score threshold (Z > 3)**.
- Inconsistencies repaired — e.g., `resolution_days` set to null for pending complaints, and resolved entries missing day counts filled with the bank type's median.
- Full execution steps and metrics exported to `etl/etl_log.txt`.

### 3. Statistical Inference & Predictive Analytics
- **Welch's t-test** — independent two-sample test with unequal variances, comparing public vs. private bank resolution days.
- **OLS Regression** — predicts resolution days from bank type, channel, and complaint category to isolate systemic friction points.
- **Mann-Kendall Test** — non-parametric test for a monotonic upward trend in Digital Banking Fraud.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+

### Installation & Full Pipeline Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/FinCompass.git
cd FinCompass

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the end-to-end orchestrator
# (data generation → validation → DB load → ML training → stats → vector store → MIS generation)
python run_all.py
```

### Optional: Enable the Live AI Policy Assistant

Create a `.env` file at the project root:

```
GEMINI_API_KEY=your_google_studio_api_key_here
```

> If no key is provided, the chatbot gracefully falls back to a local semantic synthesis mode.

### Launch the Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Key Findings (Based on Simulated Data)

| Finding | Statistical Evidence |
|---|---|
| 🏛️ **Systemic backlogs** in public sector banks | Welch's t-test: `p < 0.001` (null hypothesis rejected) |
| 📱 **Digital Banking Fraud** rising sharply | Mann-Kendall Tau: `+0.84`, `p < 0.0001` |
| 🏢 **Branch-filed complaints** add friction | OLS regression: +14.2 days vs. online filings |

---

## 🏛️ Policy Implications (DoS Brief)

**I. Institutional Action Plan**
The persistent gap in grievance redressal velocity between public and private sector banks points to a structural bottleneck in consumer protection. The analysis recommends the Department of Supervision (DoS) mandate specialized workflow automation for public sector banks whose average resolution time exceeds a 45-day threshold, alongside periodic spot audits of internal bank ombudsman frameworks.

**II. Mitigation of Digital Transaction Risks**
Given the statistically validated acceleration in Digital Banking Fraud, the analysis suggests a regulatory update to the Master Direction on Digital Banking Security Controls — including mandatory real-time transaction cooling periods for high-risk accounts and stricter security audits for third-party UPI integrations, with particular focus on Small Finance Banks showing sudden QoQ complaint spikes.

---

## 🖼️ Dashboard Pages

<div align="center">

| # | Page | Description |
|---|---|---|
| 1 | **Executive Dashboard** | High-level KPIs and complaint volume overview |
| 2 | **Complaint Deep Dive** | Real-time NLP classifier on raw complaint text |
| 3 | **DoS Supervisory Monitoring** | Bank-level resolution velocity tracking |
| 4 | **Heatmap & Geographic Analysis** | Regional complaint density visualization |
| 5 | **Time-Series Forecasting & Stats** | SARIMA forecasts and hypothesis test results |
| 6 | **AI Policy Assistant** | RAG-powered chat over monthly supervisory briefs |

</div>

> 📸 *Replace this table with actual screenshots/GIFs per page once available — a 2-column image grid works well here.*

---

## 🗺️ Roadmap

- [ ] Swap synthetic data generator for anonymized real-world grievance datasets
- [ ] Add role-based access control for the supervisory dashboard
- [ ] Expand forecasting to per-bank and per-region granularity
- [ ] CI/CD pipeline for automated model retraining
- [ ] Deploy live demo (Streamlit Cloud / Docker)

---

## ⚠️ Disclaimer

This project uses **entirely synthetic data** and is built as an independent portfolio/demonstration project. It is **not affiliated with, endorsed by, or representative of official RBI systems, data, or policy positions**.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues).

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with 🛡️ for data-driven supervisory intelligence.**

</div>
