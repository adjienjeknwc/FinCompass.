"""
FinCompass - Streamlit Dashboard Application
=============================================

This is the main web application interface for FinCompass, built using Streamlit.
It provides a multi-page, interactive supervisory analytics dashboard with
Plotly visualizations.

Pages:
1. Executive Dashboard: Key Performance Indicators (KPIs) and systemic volume trends.
2. Complaint Deep Dive: Drill down by category, channels, and ML text classification.
3. Supervisory Monitoring: Risk scoring (DoS), pendency analysis, and policy flags.
4. Geographic Analysis: State heatmaps and top-performing regions.
5. Forecasting & Statistics: Welch's t-test, regression results, and SARIMA forecasting.
6. AI Policy Assistant: Chat interface using Gemini Flash RAG pipeline.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import json
import os
import pickle
from pathlib import Path
from datetime import datetime

# Import query functions & chatbot
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from database import queries
from rag.chatbot import get_rag_response

# App Setup
st.set_page_config(
    page_title="FinCompass - Supervisory Analytics Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Colors palette
PRIMARY_COLOR = "#002855" # Navy
SECONDARY_COLOR = "#4B5F7F" # Slate Gray
ACCENT_COLOR = "#00A896" # Teal
LIGHT_BG = "#F8F9FA"

def inject_custom_css():
    """Injects custom style elements to style cards and layouts."""
    st.markdown(f"""
        <style>
        /* General styling */
        .main {{
            background-color: {LIGHT_BG};
            color: #333333;
            font-family: 'Arial', sans-serif;
        }}
        h1, h2, h3 {{
            color: {PRIMARY_COLOR};
        }}
        
        /* Metric card styling */
        .metric-card {{
            background-color: #FFFFFF;
            border-left: 5px solid {PRIMARY_COLOR};
            border-radius: 6px;
            padding: 18px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 15px;
        }}
        .metric-value {{
            font-size: 26px;
            font-weight: bold;
            color: {PRIMARY_COLOR};
            margin-top: 5px;
        }}
        .metric-label {{
            font-size: 13px;
            text-transform: uppercase;
            color: #777777;
            font-weight: 600;
        }}
        
        /* Alert Box */
        .alert-box {{
            background-color: #FFF3CD;
            border-left: 5px solid #FFC107;
            border-radius: 4px;
            padding: 12px;
            margin-bottom: 12px;
            color: #856404;
        }}
        .danger-box {{
            background-color: #F8D7DA;
            border-left: 5px solid #DC3545;
            border-radius: 4px;
            padding: 12px;
            margin-bottom: 12px;
            color: #721C24;
        }}
        </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_full_complaints():
    """Loads all complaints for interactive pandas filtering on dashboard."""
    query = """
        SELECT 
            c.complaint_id, c.date, c.complaint_text, c.state, c.channel, c.status,
            c.resolution_days, c.customer_segment, c.year, c.month, c.quarter,
            b.bank_name, b.bank_type, cat.category_name, cat.subcategory_name
        FROM complaints c
        JOIN banks b ON c.bank_id = b.bank_id
        JOIN categories cat ON c.subcategory_id = cat.subcategory_id
    """
    with sqlite3.connect(str(PROJECT_ROOT / "database" / "fincompass.db")) as conn:
        df = pd.read_sql_query(query, conn)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ----------------------------------------------------
# PAGE 1: EXECUTIVE DASHBOARD
# ----------------------------------------------------
def show_executive_dashboard(df):
    st.title("🛡️ FinCompass - Supervisory Executive Dashboard")
    st.markdown("---")
    
    # Sidebar Page Filters
    st.sidebar.subheader("Dashboard Filters")
    years = sorted(df["year"].unique())
    selected_years = st.sidebar.multiselect("Select Years", years, default=years)
    
    bank_types = df["bank_type"].unique()
    selected_bank_types = st.sidebar.multiselect("Select Bank Types", bank_types, default=list(bank_types))
    
    # Apply filters
    filtered_df = df[
        (df["year"].isin(selected_years)) &
        (df["bank_type"].isin(selected_bank_types))
    ]
    
    if filtered_df.empty:
        st.warning("No data matches selected filters.")
        return
        
    # 1. KPI Cards Row
    col1, col2, col3, col4 = st.columns(4)
    
    # Metrics computations
    total_complaints = len(filtered_df)
    
    # YoY Growth count (for selected years)
    if len(selected_years) > 1:
        prev_year = sorted(selected_years)[-2]
        curr_year = sorted(selected_years)[-1]
        prev_vol = len(df[df["year"] == prev_year])
        curr_vol = len(df[df["year"] == curr_year])
        yoy_growth = round(((curr_vol - prev_vol) / prev_vol) * 100, 1) if prev_vol > 0 else 0.0
        yoy_str = f"{yoy_growth}%"
    else:
        yoy_str = "N/A"
        
    resolved_count = len(filtered_df[filtered_df["status"] == "Resolved"])
    resolution_rate = round((resolved_count / total_complaints) * 100, 1) if total_complaints > 0 else 0.0
    
    avg_res_days = round(filtered_df[filtered_df["status"] == "Resolved"]["resolution_days"].mean(), 1)
    
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Complaints</div><div class="metric-value">{total_complaints:,}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">YoY Growth (Latest)</div><div class="metric-value">{yoy_str}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Resolution Rate</div><div class="metric-value">{resolution_rate}%</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Resolution Time</div><div class="metric-value">{avg_res_days} Days</div></div>', unsafe_allow_html=True)

    st.write("")
    
    # 2. Charts Rows
    row2_col1, row2_col2 = st.columns([2, 1])
    
    with row2_col1:
        st.subheader("📈 Monthly Complaint Volume Trend")
        # Line chart: Monthly complaint volume trend (2020-2024)
        monthly_trend = filtered_df.groupby(filtered_df["date"].dt.to_period("M")).size().reset_index(name="count")
        monthly_trend["date"] = monthly_trend["date"].dt.to_timestamp()
        
        fig = px.line(
            monthly_trend, x="date", y="count", 
            labels={"date": "Date", "count": "Complaints Count"},
            color_discrete_sequence=[PRIMARY_COLOR]
        )
        # Annotation showing COVID Spike in early 2021/late 2020 due to digital transactions rise
        fig.add_annotation(
            x=datetime(2021, 5, 31), y=int(monthly_trend[monthly_trend["date"] == "2021-05-01"]["count"].iloc[0]),
            text="COVID-19 digital spike",
            showarrow=True, arrowhead=1, ax=-40, ay=-30,
            font=dict(color="#DC3545", size=10)
        )
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=10, b=20), plot_bgcolor="rgba(0,0,0,0)")
        fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.2)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.2)")
        st.plotly_chart(fig, use_container_width=True)

    with row2_col2:
        st.subheader("📊 Category Distribution")
        cat_counts = filtered_df["category_name"].value_counts().reset_index()
        fig = px.pie(
            cat_counts, names="category_name", values="count",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=350, showlegend=False, margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Top 10 Banks
    st.subheader("🏦 Top 10 Banks by Complaint Volume")
    top_banks = filtered_df["bank_name"].value_counts().head(10).reset_index()
    # Add bank type metadata for color-coding
    bank_types_map = filtered_df.drop_duplicates("bank_name").set_index("bank_name")["bank_type"].to_dict()
    top_banks["bank_type"] = top_banks["bank_name"].map(bank_types_map)
    
    fig = px.bar(
        top_banks, x="bank_name", y="count", color="bank_type",
        labels={"bank_name": "Bank Name", "count": "Complaint Count", "bank_type": "Bank Type"},
        color_discrete_map={"Public Sector": PRIMARY_COLOR, "Private Sector": SECONDARY_COLOR, "Small Finance Bank": ACCENT_COLOR}
    )
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=10, b=20), plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------
# PAGE 2: COMPLAINT DEEP DIVE & ML CLASSIFICATION
# ----------------------------------------------------
def show_complaint_deep_dive(df):
    st.title("🔍 Grievance Deep Dive & ML Classification")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Grievance Deep Dive Analytics", "🧠 Real-Time ML Classifier"])
    
    with tab1:
        # Search Filters
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            selected_bank = st.selectbox("Filter by Bank", ["All"] + list(df["bank_name"].unique()))
        with col2:
            selected_category = st.selectbox("Filter by Category", ["All"] + list(df["category_name"].unique()))
        with col3:
            search_text = st.text_input("Search complaint description...")

        # Apply Filters
        drill_df = df.copy()
        if selected_bank != "All":
            drill_df = drill_df[drill_df["bank_name"] == selected_bank]
        if selected_category != "All":
            drill_df = drill_df[drill_df["category_name"] == selected_category]
        if search_text:
            drill_df = drill_df[drill_df["complaint_text"].str.contains(search_text, case=False, na=False)]

        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.subheader("📅 Quarterly Category Volumes (Stacked Area)")
            # Pivot table for stacked area chart
            drill_df["quarter_label"] = drill_df["year"].astype(str) + "-Q" + drill_df["quarter"].astype(str)
            trend_df = drill_df.groupby(["quarter_label", "category_name"]).size().reset_index(name="count")
            
            fig = px.area(
                trend_df, x="quarter_label", y="count", color="category_name",
                labels={"quarter_label": "Quarter", "count": "Complaints Volume"},
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=10, b=20), plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            
        with col_right:
            st.subheader("📞 Channels Distribution")
            channel_counts = drill_df["channel"].value_counts().reset_index()
            # Funnel Chart
            fig = go.Figure(go.Funnel(
                y=channel_counts["channel"],
                x=channel_counts["count"],
                marker={"color": [PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR, "#FFC107", "#28A745"]}
            ))
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("👥 Customer Segments & Sub-Category Breakdowns")
        col_seg, col_sub = st.columns([1, 2])
        with col_seg:
            seg_counts = drill_df["customer_segment"].value_counts().reset_index()
            fig = px.pie(seg_counts, names="customer_segment", values="count", hole=0.4,
                         color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR])
            fig.update_layout(height=280, showlegend=True, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
        with col_sub:
            subcat_df = drill_df.groupby("subcategory_name").size().reset_index(name="Count").sort_values("Count", ascending=False).head(8)
            st.dataframe(subcat_df, hide_index=True, use_container_width=True)
            
    with tab2:
        st.subheader("🧠 Real-time Complaint NLP Classification")
        st.markdown(
            "This module uses the trained supervisory model (**TF-IDF + Logistic Regression**) "
            "saved under `ml/models/` to analyze user grievance statements and classify them "
            "into one of the 12 regulatory categories."
        )
        
        user_input = st.text_area(
            "Enter banking grievance description text:",
            "Yesterday, Rs. 5000 was debited from my account via a fraudulent UPI transaction that I did not authorize."
        )
        
        selected_channel = st.selectbox("Filing Channel", ["Online", "Branch", "Phone", "Email", "Ombudsman Portal"])
        
        if st.button("Predict Category"):
            model_path = PROJECT_ROOT / "ml" / "models" / "complaint_classifier.pkl"
            
            if not model_path.exists():
                st.warning("Model file not found. Running synthetic fallback mock classifier...")
                # Simple keyword lookup fallback
                text_lower = user_input.lower()
                pred = "Other"
                if "upi" in text_lower or "fraud" in text_lower or "phishing" in text_lower:
                    pred = "Digital Banking Fraud"
                elif "atm" in text_lower or "cash" in text_lower:
                    pred = "ATM/Debit Card Issues"
                elif "credit card" in text_lower or "annual fee" in text_lower:
                    pred = "Credit Card Complaints"
                elif "loan" in text_lower or "emi" in text_lower:
                    pred = "Loan & EMI Disputes"
                elif "kyc" in text_lower or "freeze" in text_lower:
                    pred = "Account Operations"
                    
                st.success(f"**Predicted Category (Fallback):** {pred}")
            else:
                try:
                    with open(model_path, "rb") as f:
                        pipeline = pickle.load(f)
                        
                    input_df = pd.DataFrame([{
                        "complaint_text": user_input,
                        "channel": selected_channel
                    }])
                    
                    prediction = pipeline.predict(input_df)[0]
                    probs = pipeline.predict_proba(input_df)[0]
                    classes = pipeline.classes_
                    pred_prob = round(max(probs) * 100, 1)
                    
                    st.success(f"**Predicted Category:** {prediction} (Confidence: {pred_prob}%)")
                    
                    # Probabilities chart
                    prob_df = pd.DataFrame({"Category": classes, "Probability": probs}).sort_values("Probability", ascending=True)
                    fig = px.bar(prob_df, x="Probability", y="Category", orientation='h',
                                 title="Model Prediction Probability Breakdown",
                                 color_discrete_sequence=[ACCENT_COLOR])
                    fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error executing model: {e}")


# ----------------------------------------------------
# PAGE 3: SUPERVISORY MONITORING
# ----------------------------------------------------
def show_supervisory_monitoring():
    st.title("🛡️ Supervisory Monitoring & Risk Alerts")
    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("🛑 Rising Complaint Categories Alert Panel")
        st.markdown(
            "Regulatory alerts flagged when a complaint category exhibits Quarter-over-Quarter "
            "(QoQ) growth exceeding **20%** inside 2024-Q4."
        )
        try:
            rising_df = queries.get_rising_categories()
            if not rising_df.empty:
                for _, row in rising_df.iterrows():
                    st.markdown(
                        f"""
                        <div class="danger-box">
                            <strong>🚨 spike alert: {row['category_name']}</strong><br/>
                            QoQ complaints count grew by <strong>{row['qoq_growth_pct']}%</strong> 
                            (from {row['previous_quarter_count']} in Q3 to {row['latest_quarter_count']} in Q4).
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            else:
                st.info("No complaint categories exceeded the QoQ 20% growth alert threshold.")
        except Exception as e:
            st.error(f"Error fetching rising categories: {e}")
            
    with col_right:
        st.subheader("🎖️ Bank Resolution Velocity Ranking")
        try:
            rank_df = queries.get_resolution_efficiency()
            # Sort by avg days
            fig = px.bar(
                rank_df, y="bank_name", x="avg_resolution_days", orientation='h',
                labels={"bank_name": "Bank Name", "avg_resolution_days": "Avg Days to Resolve"},
                title="Resolution Efficiency ranking (Lower is better)",
                color_discrete_sequence=[SECONDARY_COLOR]
            )
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20), plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error fetching ranking: {e}")

    # Risk Scoring Table (DoS Focus)
    st.subheader("📋 Bank-Wise Backlog Pendency & Supervisory Risk Classification")
    try:
        # Load calculated risk scores from JSON if available, otherwise fallback to queries
        scores_path = PROJECT_ROOT / "analysis" / "risk_scores.json"
        
        if scores_path.exists():
            with open(scores_path, "r") as f:
                scores_data = json.load(f)
            
            # Reformat to DataFrame
            rows = []
            for bank, data in scores_data.items():
                rows.append({
                    "Bank Name": bank,
                    "Total Complaints": data["total_complaints"],
                    "Pending Complaints": data["pending_count"],
                    "Avg Resolution Days": data["avg_resolution_days"],
                    "Pending > 60 Days": data["pending_gt_60"],
                    "Risk Score": data["risk_score"],
                    "Risk Level": data["risk_level"]
                })
            table_df = pd.DataFrame(rows).sort_values("Risk Score", ascending=False)
        else:
            # Fallback direct querying
            pend_df = queries.get_pendency_analysis()
            table_df = pend_df.rename(columns={
                "bank_name": "Bank Name",
                "total_pending": "Pending Complaints",
                "pending_30_days": "Pending > 30 Days",
                "pending_60_days": "Pending > 60 Days",
                "pending_90_days": "Pending > 90 Days"
            })
            table_df["Risk Level"] = "Amber" # Default placeholder
            
        def style_risk_level(val):
            if val == "Red":
                return "background-color: #F8D7DA; color: #721C24; font-weight: bold;"
            elif val == "Amber":
                return "background-color: #FFF3CD; color: #856404; font-weight: bold;"
            else:
                return "background-color: #D4EDDA; color: #155724; font-weight: bold;"
                
        # Render stylized table using pandas styler
        st.dataframe(
            table_df.style.applymap(style_risk_level, subset=["Risk Level"]),
            use_container_width=True, hide_index=True
        )
    except Exception as e:
        st.error(f"Error fetching risk scores: {e}")

    # Section 4: Policy Flags Table
    st.subheader("🚩 Policy Flags & Systemic Supervision Notices")
    with sqlite3.connect(str(PROJECT_ROOT / "database" / "fincompass.db")) as conn:
        flags_df = pd.read_sql_query("""
            SELECT b.bank_name, pf.year, pf.quarter, pf.flag_type, pf.flag_description, pf.severity
            FROM policy_flags pf
            JOIN banks b ON pf.bank_id = b.bank_id
            ORDER BY pf.severity DESC, pf.year DESC, pf.quarter DESC
        """, conn)
        
    if not flags_df.empty:
        st.dataframe(flags_df, use_container_width=True, hide_index=True)
    else:
        st.info("No policy flags currently active.")


# ----------------------------------------------------
# PAGE 4: GEOGRAPHIC ANALYSIS
# ----------------------------------------------------
def show_geographic_analysis():
    st.title("🌐 Geographic Complaint Analysis")
    st.markdown("---")
    
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.subheader("📍 State-wise Complaint Volumes")
        try:
            # SQL fetch state counts
            with sqlite3.connect(str(PROJECT_ROOT / "database" / "fincompass.db")) as conn:
                state_df = pd.read_sql_query("""
                    SELECT state as State, COUNT(complaint_id) as Volume 
                    FROM complaints 
                    GROUP BY state 
                    ORDER BY Volume DESC
                """, conn)
            
            fig = px.bar(
                state_df, x="Volume", y="State", orientation='h',
                color="Volume", color_continuous_scale="Purples",
                title="Complaints volume by State"
            )
            fig.update_layout(height=500, showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error fetching state volumes: {e}")

    with col_right:
        st.subheader("🔥 State + Complaint Category Cross-Tabulation Heatmap")
        try:
            heatmap_df = queries.get_state_heatmap()
            # Pivot table to construct heatmap
            pivot_df = heatmap_df.pivot(index="state", columns="category_name", values="complaint_count").fillna(0)
            
            fig = px.imshow(
                pivot_df, 
                labels=dict(x="Complaint Category", y="State", color="Complaints"),
                x=pivot_df.columns, y=pivot_df.index,
                color_continuous_scale="YlOrRd",
                aspect="auto"
            )
            fig.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error building heatmap: {e}")
            
    # Trend for top 5 states
    st.subheader("📈 Top 5 States Volume Trend")
    try:
        with sqlite3.connect(str(PROJECT_ROOT / "database" / "fincompass.db")) as conn:
            top_5_states = pd.read_sql_query("""
                SELECT state, COUNT(*) as count 
                FROM complaints 
                GROUP BY state 
                ORDER BY count DESC 
                LIMIT 5
            """, conn)["state"].tolist()
            
            # Fetch timeline for top 5
            trend_df = pd.read_sql_query(f"""
                SELECT state, year, month, COUNT(complaint_id) as count
                FROM complaints
                WHERE state IN ({",".join([f"'{s}'" for s in top_5_states])})
                GROUP BY state, year, month
                ORDER BY year, month
            """, conn)
            
        trend_df["date"] = pd.to_datetime(trend_df["year"].astype(str) + "-" + trend_df["month"].astype(str) + "-01")
        fig = px.line(
            trend_df, x="date", y="count", color="state",
            labels={"date": "Timeline", "count": "Complaints Count", "state": "State"},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=10, b=20), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error fetching state trends: {e}")


# ----------------------------------------------------
# PAGE 5: FORECASTING & STATISTICS
# ----------------------------------------------------
def show_forecasting_statistics():
    st.title("📊 Time-Series Forecasting & Policy Inference")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["🔮 SARIMA Volume Forecasting", "🔬 Policy Hypothesis Testing & OLS"])
    
    with tab1:
        st.subheader("🔮 6-Month Complaint Volume SARIMA Forecast")
        st.markdown(
            "This model uses a Seasonally Differenced Autoregressive Integrated Moving Average (**SARIMA**) model "
            "fitted on historical monthly aggregate values to project system-wide complaint volumes."
        )
        
        forecast_path = PROJECT_ROOT / "analysis" / "forecast_results.json"
        if not forecast_path.exists():
            st.info("Forecasting JSON results file not found. Fit the model or run run_all.py first.")
        else:
            try:
                with open(forecast_path, "r") as f:
                    f_data = json.load(f)
                    
                if f_data.get("status") in ["Success", "Fallback (Moving Average)"]:
                    hist_dates = pd.to_datetime(f_data["historical_dates"])
                    hist_vals = f_data["historical_values"]
                    
                    fore_dates = pd.to_datetime(f_data["forecast_dates"])
                    fore_vals = f_data["forecast_values"]
                    lower_ci = f_data["lower_confidence_interval"]
                    upper_ci = f_data["upper_confidence_interval"]
                    
                    # Construct Plotly line chart
                    fig = go.Figure()
                    
                    # Historical line
                    fig.add_trace(go.Scatter(
                        x=hist_dates, y=hist_vals,
                        name="Historical Count",
                        line=dict(color=PRIMARY_COLOR, width=2)
                    ))
                    
                    # Forecasted line
                    fig.add_trace(go.Scatter(
                        x=fore_dates, y=fore_vals,
                        name="SARIMA Forecast",
                        line=dict(color=ACCENT_COLOR, width=2, dash="dash")
                    ))
                    
                    # Confidence interval shading
                    fig.add_trace(go.Scatter(
                        x=fore_dates.tolist() + fore_dates.tolist()[::-1],
                        y=upper_ci + lower_ci[::-1],
                        fill='toself',
                        fillcolor='rgba(0, 168, 150, 0.15)',
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=True,
                        name="95% Confidence Band"
                    ))
                    
                    fig.update_layout(
                        height=450,
                        title="Monthly Complaint Volume: Historical vs Projected",
                        xaxis_title="Date",
                        yaxis_title="Monthly Complaints Count",
                        plot_bgcolor="rgba(0,0,0,0)",
                        legend=dict(x=0.01, y=0.99)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error(f"Error in forecasting results: {f_data.get('error')}")
            except Exception as e:
                st.error(f"Error reading forecasting: {e}")

    with tab2:
        st.subheader("🔬 Policy Hypothesis Testing: Welch's t-test")
        st.markdown(
            "**Policy Question:** Are public sector commercial banks significantly slower "
            "at grievance resolution than private sector banks?"
        )
        
        stats_path = PROJECT_ROOT / "analysis" / "stats_results.json"
        
        if not stats_path.exists():
            st.info("Statistical analysis JSON file not found under `analysis/`. Run the statistics script first.")
        else:
            try:
                with open(stats_path, "r") as f:
                    s_data = json.load(f)
                    
                ttest_res = s_data["welch_t_test"]
                
                # Descriptive metrics summary in cards
                desc_res = s_data["descriptive_statistics"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Public Sector Mean</div>
                            <div class="metric-value">{desc_res['Public Sector']['mean']} Days</div>
                            <small>Sample: {desc_res['Public Sector']['count']}</small>
                        </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Private Sector Mean</div>
                            <div class="metric-value">{desc_res['Private Sector']['mean']} Days</div>
                            <small>Sample: {desc_res['Private Sector']['count']}</small>
                        </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">t-statistic</div>
                            <div class="metric-value">{ttest_res['t_statistic']}</div>
                            <small>p-value: {round(ttest_res['p_value'], 6)}</small>
                        </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown(f"**Statistical Interpretation:**")
                if ttest_res["reject_null"]:
                    st.success(ttest_res["interpretation"])
                else:
                    st.warning(ttest_res["interpretation"])
                    
                st.markdown("---")
                
                # OLS Regression Summary
                st.subheader("📈 Ordinary Least Squares (OLS) Linear Regression Results")
                st.markdown(
                    "Predicting **resolution days** from category, channel, and bank type (Baseline = Private Sector, Online Channel, Other Category)."
                )
                
                ols_res = s_data["ols_regression"]
                st.write(f"**Model Fit:** R-squared: `{ols_res['r_squared']}` | Adj. R-squared: `{ols_res['adj_r_squared']}` | F-statistic: `{ols_res['f_statistic']}` (p = `{ols_res['f_pvalue']}`)")
                
                coef_dict = ols_res["coefficients"]
                coef_rows = []
                for var, metrics in coef_dict.items():
                    coef_rows.append({
                        "Variable Impact": var,
                        "Days Coefficient": metrics["coefficient"],
                        "t-statistic": metrics["t_statistic"],
                        "p-value": metrics["p_value"],
                        "Significant (p < 0.05)": "✅ Yes" if metrics["significant"] else "❌ No"
                    })
                st.dataframe(pd.DataFrame(coef_rows), use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                # Mann-Kendall test
                st.subheader("📉 Mann-Kendall Trend Test: Digital Banking Fraud Expansion")
                mk_res = s_data["mann_kendall_trend_test"]
                st.write(f"**Trend Direction:** `{mk_res['trend']}` (S = `{mk_res['s_statistic']}`, Tau = `{mk_res['tau']}`, p = `{round(mk_res['p_value'], 6)}`)")
                st.info(mk_res["interpretation"])
                
            except Exception as e:
                st.error(f"Error reading statistics results: {e}")


# ----------------------------------------------------
# PAGE 6: AI POLICY ASSISTANT (RAG)
# ----------------------------------------------------
def show_ai_policy_assistant():
    st.title("🤖 AI Policy & Supervision Assistant (RAG)")
    st.markdown("---")
    
    st.sidebar.subheader("Gemini Model Settings")
    api_key = st.sidebar.text_input("Enter Gemini API Key (free tier)", type="password")
    if not api_key:
        st.sidebar.info(
            "Providing a Gemini API key enables full LLM responses. "
            "If omitted, the chatbot operates in local fallback mode using local vector database retrieval."
        )
        
    st.markdown(
        "Ask natural language questions about complaint volumes, rising trends, or bank delays. "
        "The assistant queries the **ChromaDB Vector Store** of monthly reports and policy alerts, "
        "and synthesizes the answer using **Google Gemini Flash**."
    )
    
    # Preloaded questions clickables
    st.markdown("**Example Questions (Click to Copy):**")
    ex_q1 = "Which banks had the highest complaint growth in 2024?"
    ex_q2 = "What are the top digital banking issues?"
    ex_q3 = "Which states generate the most complaints?"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.code(ex_q1, language="")
    with col2:
        st.code(ex_q2, language="")
    with col3:
        st.code(ex_q3, language="")
        
    # Text input
    user_query = st.text_input("Enter your policy question:", placeholder="e.g. Which banks were flagged for supervisory delays in 2024?")
    
    if st.button("Query Policy Assistant") and user_query:
        with st.spinner("Retrieving document chunks from ChromaDB and generating answer..."):
            res = get_rag_response(user_query, api_key=api_key)
            
            st.subheader("💡 Assistant Response")
            st.markdown(f"**Execution Mode:** `{res.get('mode', 'Fallback')}`")
            st.markdown(res["answer"])
            
            st.subheader("📁 Retrieved Sources (Context Chunks)")
            for i, src in enumerate(res["sources"], 1):
                with st.expander(f"Chunk {i} - Bank: {src['bank_name']} | Table Source: {src['source']}"):
                    st.write(src["content"])


# ----------------------------------------------------
# MAIN APP ENTRYPOINT
# ----------------------------------------------------
def main():
    inject_custom_css()
    
    db_exists = (PROJECT_ROOT / "database" / "fincompass.db").exists()
    if not db_exists:
        st.warning("🛡️ Welcome to FinCompass! The database is initializing...")
        with st.spinner("Compiling platform files, generating synthetic data, training classification models, and building vector store (takes ~20 seconds)..."):
            import subprocess
            subprocess.run([sys.executable, str(PROJECT_ROOT / "run_all.py")])
        st.success("Platform initialization complete! Reloading page...")
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()
        
    df = load_full_complaints()
    
    # Navigation Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/shield.png", width=96)
    st.sidebar.title("FinCompass Platform")
    
    page = st.sidebar.radio(
        "Supervisory Navigation",
        ["Executive Dashboard", "Complaint Deep Dive", "Supervisory Monitoring", "Geographic Analysis", "Forecasting & Statistics", "AI Policy Assistant (RAG)"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.caption("FinCompass Banking Complaint Analytics System v1.0.0 (RBI Young Professional Portfolio Project)")
    
    # Router
    if page == "Executive Dashboard":
        show_executive_dashboard(df)
    elif page == "Complaint Deep Dive":
        show_complaint_deep_dive(df)
    elif page == "Supervisory Monitoring":
        show_supervisory_monitoring()
    elif page == "Geographic Analysis":
        show_geographic_analysis()
    elif page == "Forecasting & Statistics":
        show_forecasting_statistics()
    elif page == "AI Policy Assistant (RAG)":
        show_ai_policy_assistant()


if __name__ == "__main__":
    main()
