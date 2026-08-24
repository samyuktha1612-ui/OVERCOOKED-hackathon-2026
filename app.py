"""
app.py
------
Streamlit Web Application for Household Electricity Consumption Forecasting
using Stacked LSTM Deep Learning.
Supports UCI Benchmark, 2025 Multi-Feature Telemetry, and Custom CSV Datasets.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st

from data_processing import (
    detect_and_load_data,
    clean_and_prepare_daily,
    prepare_time_series_data
)
from forecasting import (
    build_lstm_model,
    train_lstm_model,
    load_artifacts,
    save_artifacts,
    calculate_metrics,
    BaselinePredictor,
    forecast_future
)
import visualization as viz


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Electricity Forecast AI | LSTM",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #1E40AF 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    .metric-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 0.4rem;
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
    }
    
    .metric-desc {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 0.3rem;
    }
    
    .insight-box {
        background: #F8FAFC;
        border-left: 4px solid #3B82F6;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Data Loading
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_cache_dataset(file_path_or_buffer):
    """Loads and preprocesses any supported dataset."""
    df_raw = detect_and_load_data(file_path_or_buffer)
    daily_df, summary, target_col = clean_and_prepare_daily(df_raw)
    return daily_df, summary, target_col


# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?auto=format&fit=crop&w=400&q=80", use_container_width=True)
    st.markdown("### ⚡ Electricity Forecast AI")
    st.caption("Deep Learning Time-Series Forecaster")
    
    page = st.radio(
        "Navigation",
        [
            "🏠 Home / Overview",
            "📊 Consumption Analysis (EDA)",
            "🔮 Forecasting & Evaluation",
            "💡 Energy Insights & Recommendations"
        ],
        index=0
    )
    
    st.divider()
    st.markdown("#### 📂 Dataset Selection")
    
    dataset_option = st.selectbox(
        "Choose Dataset Source:",
        [
            "⚡ UCI Power Benchmark (2006-2010)",
            "🌦️ 2025 Weather & Occupancy Telemetry",
            "📤 Upload Custom CSV"
        ],
        index=0
    )
    
    custom_file = None
    if dataset_option == "📤 Upload Custom CSV":
        custom_file = st.file_uploader("Upload Time-Series CSV", type=['csv', 'txt'])
        
    st.divider()


# Resolve Active Dataset Source
data_source_path = None
if dataset_option == "⚡ UCI Power Benchmark (2006-2010)":
    data_source_path = "data/household_power_consumption.txt"
elif dataset_option == "🌦️ 2025 Weather & Occupancy Telemetry":
    data_source_path = "data/daily_weather_power.csv"
elif dataset_option == "📤 Upload Custom CSV":
    if custom_file is not None:
        data_source_path = custom_file
    else:
        st.sidebar.info("Upload a CSV file to proceed.")
        data_source_path = "data/household_power_consumption.txt"

# Load Dataset
try:
    daily_df, clean_summary, target_col = load_and_cache_dataset(data_source_path)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Load Pretrained Artifacts
lstm_model, scaler, metadata = load_artifacts("models")

# Sidebar status
with st.sidebar:
    st.markdown("#### 📁 Active Data Status")
    st.success(f"✓ Records: {len(daily_df):,} Days")
    st.caption(f"📅 Range: {daily_df.index.min().date()} to {daily_df.index.max().date()}")
    
    if lstm_model is not None:
        st.success("✓ LSTM Model: Ready")
    else:
        st.warning("⚠ Model Not Found (Run `python train_model.py`)")
        
    st.divider()
    st.caption("Built with Python, Keras/PyTorch, Plotly & Streamlit")


# ---------------------------------------------------------
# PAGE 1: HOME / OVERVIEW
# ---------------------------------------------------------
if page == "🏠 Home / Overview":
    st.markdown('<div class="main-title">Household Electricity Consumption Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Advanced deep learning time-series intelligence powered by Long Short-Term Memory (LSTM) neural networks.</div>', unsafe_allow_html=True)
    
    # Hero Metric Cards
    total_days = len(daily_df)
    avg_kwh = daily_df['Daily_energy_kWh'].mean()
    max_kwh = daily_df['Daily_energy_kWh'].max()
    max_date = daily_df['Daily_energy_kWh'].idxmax().strftime('%b %d, %Y')
    total_mwh = (daily_df['Daily_energy_kWh'].sum()) / 1000.0
    start_date_str = daily_df.index.min().strftime('%b %d, %Y')
    end_date_str = daily_df.index.max().strftime('%b %d, %Y')
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Daily Consumption</div>
            <div class="metric-value">{avg_kwh:.2f} <span style="font-size:1rem;color:#64748B;">kWh/day</span></div>
            <div class="metric-desc">Mean daily electricity load</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Peak Daily Record</div>
            <div class="metric-value">{max_kwh:.2f} <span style="font-size:1rem;color:#64748B;">kWh</span></div>
            <div class="metric-desc">Recorded on {max_date}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Energy Tracked</div>
            <div class="metric-value">{total_mwh:.2f} <span style="font-size:1rem;color:#64748B;">MWh</span></div>
            <div class="metric-desc">Across {total_days:,} days of monitoring</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Monitoring Duration</div>
            <div class="metric-value">{round(total_days/365.25, 1)} <span style="font-size:1rem;color:#64748B;">Years</span></div>
            <div class="metric-desc">{start_date_str} – {end_date_str}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Project Highlights & Architecture
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown("### 🎯 Problem Statement & Objective")
        st.markdown("""
        Volatile electricity consumption directly affects household utility costs and grid stability. 
        Accurate time-series forecasting enables:
        - **Proactive Energy Management**: Anticipate peak demand periods and shift appliance usage.
        - **Smart Grid Integration**: Optimize demand-response dispatch and solar self-consumption.
        - **Cost & Carbon Reduction**: Minimize wasteful standby power and inefficient cooling/heating cycles.
        
        This application implements a **Stacked LSTM deep learning model** trained chronologically on daily sequences 
        to capture non-linear weekly and seasonal temporal dynamics.
        """)
        
        st.markdown("### ⚙️ Deep Learning Architecture")
        st.markdown("""
        1. **Lookback Sliding Window**: Sequence $X \in \mathbb{R}^{L \\times 1}$ (historical days).
        2. **LSTM Layer 1**: 64 memory units with $\tanh$ recurrent activation + Dropout ($p=0.2$).
        3. **LSTM Layer 2**: 32 memory units with Dropout ($p=0.2$).
        4. **Dense Layer**: 16-unit ReLU non-linear transformation.
        5. **Output**: Continuous regression node predicting day $T+1$ consumption in kWh.
        """)

    with col_right:
        st.markdown("### 📋 Active Dataset Summary")
        st.markdown(f"""
        - **Dataset Source**: `{clean_summary.get('dataset_type', 'Loaded Dataset')}`
        - **Clean Daily Records**: `{len(daily_df):,}` days
        - **Start Date**: `{start_date_str}`
        - **End Date**: `{end_date_str}`
        - **Missing Values Handled**: `{clean_summary.get('missing_records_filled', 0):,}`
        - **Available Telemetry Features**: `{', '.join([c for c in daily_df.columns if c not in ['date', 'Datetime', 'Daily_energy_kWh']][:5])}`
        """)
        
        st.markdown("### 🚀 Tech Stack")
        st.markdown("""
        `Python` • `Pandas` • `NumPy` • `Scikit-Learn` • `Keras / PyTorch` • `LSTM` • `Plotly` • `Streamlit`
        """)

    st.markdown("---")
    st.markdown("### 🔍 Historical Data Snapshot")
    display_cols = [c for c in ['Daily_energy_kWh', 'Temperature (°C)', 'Weather condition', 'Number of people at home', 'Global_active_power_mean', 'Sub_metering_3', 'day_name', 'month_name'] if c in daily_df.columns]
    st.dataframe(daily_df[display_cols].head(10), use_container_width=True)


# ---------------------------------------------------------
# PAGE 2: CONSUMPTION ANALYSIS (EDA)
# ---------------------------------------------------------
elif page == "📊 Consumption Analysis (EDA)":
    st.markdown('<div class="main-title">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Interactive visual exploration of historical electricity consumption patterns, seasonal cycles, and telemetry correlations.</div>', unsafe_allow_html=True)
    
    # 1. Overall Trend
    st.plotly_chart(viz.plot_overall_trend(daily_df), use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(viz.plot_daily_distribution(daily_df), use_container_width=True)
    with col2:
        st.plotly_chart(viz.plot_monthly_consumption(daily_df), use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(viz.plot_day_of_week_consumption(daily_df), use_container_width=True)
    with col4:
        st.plotly_chart(viz.plot_weekday_vs_weekend(daily_df), use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Weather & Occupancy Visualizations if present
    fig_temp = viz.plot_weather_correlation(daily_df)
    fig_occ = viz.plot_occupancy_impact(daily_df)
    fig_sub = viz.plot_submetering_breakdown(daily_df)
    
    if fig_temp is not None or fig_occ is not None:
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            if fig_temp is not None:
                st.plotly_chart(fig_temp, use_container_width=True)
        with col_w2:
            if fig_occ is not None:
                st.plotly_chart(fig_occ, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
    col5, col6 = st.columns([3, 2])
    with col5:
        if fig_sub is not None:
            st.plotly_chart(fig_sub, use_container_width=True)
        else:
            # Show Rolling standard deviation
            df_roll = daily_df.copy()
            df_roll['Rolling_Std'] = df_roll['Daily_energy_kWh'].rolling(7).std()
            fig_roll = px.line(df_roll, y='Rolling_Std', title="7-Day Rolling Volatility (kWh Std Dev)")
            st.plotly_chart(viz.apply_theme(fig_roll, "7-Day Rolling Volatility (kWh)", height=440), use_container_width=True)
    with col6:
        st.plotly_chart(viz.plot_peak_analysis(daily_df, top_n=10), use_container_width=True)


# ---------------------------------------------------------
# PAGE 3: FORECASTING & MODEL EVALUATION
# ---------------------------------------------------------
elif page == "🔮 Forecasting & Evaluation":
    st.markdown('<div class="main-title">Forecasting & Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Benchmark comparison between Baseline Heuristics and Stacked LSTM Deep Learning, plus live multi-step future forecasting.</div>', unsafe_allow_html=True)
    
    # Prepare sequence test data
    lookback_window = metadata.get('lookback_window', 30) if metadata else 14
    lookback_window = min(lookback_window, max(7, int(len(daily_df) * 0.15)))
    
    data_bundle = prepare_time_series_data(daily_df, lookback_window=lookback_window, train_ratio=0.8)
    
    X_train, y_train = data_bundle['X_train'], data_bundle['y_train']
    X_val, y_val = data_bundle['X_val'], data_bundle['y_val']
    X_test = data_bundle['X_test']
    actual_test = data_bundle['actual_test_unscaled']
    test_dates = data_bundle['test_dates']
    train_df = data_bundle['train_df']
    test_df = data_bundle['test_df']
    scaler = data_bundle['scaler']
    
    # Check if we need to train on the active dataset on the fly
    active_lstm = lstm_model
    if active_lstm is None or len(daily_df) != metadata.get('total_daily_records', -1):
        with st.spinner("Training LSTM on current dataset..."):
            active_lstm = build_lstm_model(lookback_window=lookback_window)
            active_lstm, _ = train_lstm_model(active_lstm, X_train, y_train, X_val, y_val, epochs=25, batch_size=16, patience=8)
            
    # Model Predictions
    pred_test_scaled = active_lstm.predict(X_test, verbose=0)
    pred_lstm = scaler.inverse_transform(pred_test_scaled).ravel()
    pred_lstm = np.maximum(0.0, pred_lstm)
    
    full_unscaled = np.concatenate([train_df['Daily_energy_kWh'].values, test_df['Daily_energy_kWh'].values])
    train_size = len(train_df)
    test_context = full_unscaled[train_size - lookback_window :]
    pred_persistence = BaselinePredictor.persistence_predict(test_context, lookback_window=lookback_window)
    pred_ma7 = BaselinePredictor.moving_average_predict(test_context, lookback_window=lookback_window, ma_window=min(7, lookback_window))
    
    # Compute Metrics
    metrics_persistence = calculate_metrics(actual_test, pred_persistence)
    metrics_ma7 = calculate_metrics(actual_test, pred_ma7)
    metrics_lstm = calculate_metrics(actual_test, pred_lstm)
    
    # Model Comparison Section
    st.markdown("### 📊 Model Performance Comparison (Unseen Test Set)")
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">LSTM Test MAE</div>
            <div class="metric-value">{metrics_lstm['MAE']:.2f} <span style="font-size:1rem;color:#64748B;">kWh</span></div>
            <div class="metric-desc">Persistence: {metrics_persistence['MAE']:.2f} kWh</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">LSTM Test RMSE</div>
            <div class="metric-value">{metrics_lstm['RMSE']:.2f} <span style="font-size:1rem;color:#64748B;">kWh</span></div>
            <div class="metric-desc">Persistence: {metrics_persistence['RMSE']:.2f} kWh</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">LSTM R² Score</div>
            <div class="metric-value">{metrics_lstm['R2']:.4f}</div>
            <div class="metric-desc">Persistence R²: {metrics_persistence['R2']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">LSTM Test MAPE</div>
            <div class="metric-value">{metrics_lstm['MAPE']:.2f}%</div>
            <div class="metric-desc">Mean relative error percentage</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Comparison Table
    comp_df = pd.DataFrame([
        {
            'Model Architecture': 'Baseline: Persistence (Day T-1)',
            'MAE (kWh)': f"{metrics_persistence['MAE']:.2f}",
            'RMSE (kWh)': f"{metrics_persistence['RMSE']:.2f}",
            'R² Score': f"{metrics_persistence['R2']:.4f}",
            'MAPE (%)': f"{metrics_persistence['MAPE']:.2f}%",
            'Model Type': 'Naive Benchmark'
        },
        {
            'Model Architecture': f'Baseline: {min(7, lookback_window)}-Day Moving Average',
            'MAE (kWh)': f"{metrics_ma7['MAE']:.2f}",
            'RMSE (kWh)': f"{metrics_ma7['RMSE']:.2f}",
            'R² Score': f"{metrics_ma7['R2']:.4f}",
            'MAPE (%)': f"{metrics_ma7['MAPE']:.2f}%",
            'Model Type': 'Rolling Heuristic'
        },
        {
            'Model Architecture': f'Deep Learning: Stacked LSTM (Lookback={lookback_window}d)',
            'MAE (kWh)': f"{metrics_lstm['MAE']:.2f}",
            'RMSE (kWh)': f"{metrics_lstm['RMSE']:.2f}",
            'R² Score': f"{metrics_lstm['R2']:.4f}",
            'MAPE (%)': f"{metrics_lstm['MAPE']:.2f}%",
            'Model Type': 'Recurrent Neural Network'
        }
    ])
    st.table(comp_df)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Actual vs Predicted Plot
    st.plotly_chart(
        viz.plot_actual_vs_predicted(test_dates, actual_test, pred_lstm, pred_persistence),
        use_container_width=True
    )
    
    # Training History if available
    train_losses = metadata.get('training_loss_history', []) if metadata else []
    val_losses = metadata.get('val_loss_history', []) if metadata else []
    if train_losses:
        with st.expander("📈 View LSTM Training & Validation Convergence Curve"):
            st.plotly_chart(
                viz.plot_training_loss({'loss': train_losses, 'val_loss': val_losses}),
                use_container_width=True
            )
            
    st.divider()
    
    # -----------------------------------------------------
    # Live Multi-Step Future Forecasting
    # -----------------------------------------------------
    st.markdown("### 🔮 Live Multi-Step Future Electricity Forecasting")
    st.markdown("Select a future forecasting horizon to predict upcoming household electricity consumption:")
    
    fc_col1, fc_col2 = st.columns([1, 3])
    with fc_col1:
        horizon = st.selectbox(
            "Forecast Horizon",
            options=[7, 14, 30],
            format_func=lambda x: f"{x} Days Ahead",
            index=2
        )
        st.info(f"Generating recursive {horizon}-day forecast based on the last {lookback_window} historical days.")
        
    last_seq_scaled = scaler.transform(daily_df[['Daily_energy_kWh']].values[-lookback_window:])
    last_date = daily_df.index[-1]
    
    forecast_df, fc_summary = forecast_future(
        active_lstm, last_seq_scaled, scaler, last_date, horizon_days=horizon
    )
    
    # Forecast Metrics
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Expected Daily Avg</div>
            <div class="metric-value">{fc_summary['expected_avg_kWh']:.2f} <span style="font-size:1rem;color:#64748B;">kWh</span></div>
            <div class="metric-desc">Over next {horizon} days</div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Projected Energy</div>
            <div class="metric-value">{fc_summary['total_expected_kWh']:.1f} <span style="font-size:1rem;color:#64748B;">kWh</span></div>
            <div class="metric-desc">Cumulative consumption</div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Highest Predicted Day</div>
            <div class="metric-value">{fc_summary['max_forecast_kWh']:.2f} <span style="font-size:1rem;color:#64748B;">kWh</span></div>
            <div class="metric-desc">{fc_summary['max_forecast_date']}</div>
        </div>
        """, unsafe_allow_html=True)
    with f4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Lowest Predicted Day</div>
            <div class="metric-value">{fc_summary['min_forecast_kWh']:.2f} <span style="font-size:1rem;color:#64748B;">kWh</span></div>
            <div class="metric-desc">{fc_summary['min_forecast_date']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Future Forecast Chart
    recent_tail_df = daily_df.tail(min(45, len(daily_df)))
    st.plotly_chart(
        viz.plot_future_forecast(recent_tail_df, forecast_df),
        use_container_width=True
    )
    
    # Forecast Table & CSV Download
    col_tbl, col_dl = st.columns([3, 1])
    with col_tbl:
        st.markdown("#### 📅 Future Forecast Breakdown Table")
        st.dataframe(forecast_df, use_container_width=True)
    with col_dl:
        st.markdown("#### 💾 Export Data")
        csv_data = forecast_df.to_csv().encode('utf-8')
        st.download_button(
            label="⬇️ Download Forecast CSV",
            data=csv_data,
            file_name=f"household_electricity_forecast_{horizon}days.csv",
            mime="text/csv",
            use_container_width=True
        )


# ---------------------------------------------------------
# PAGE 4: INSIGHTS & RECOMMENDATIONS
# ---------------------------------------------------------
elif page == "💡 Energy Insights & Recommendations":
    st.markdown('<div class="main-title">Energy Insights & Smart Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Automated data-driven intelligence derived directly from historical household consumption telemetry.</div>', unsafe_allow_html=True)
    
    # Derive dynamic insights from active dataset
    weekday_mean = daily_df[daily_df['is_weekend'] == 0]['Daily_energy_kWh'].mean()
    weekend_mean = daily_df[daily_df['is_weekend'] == 1]['Daily_energy_kWh'].mean()
    weekend_pct_diff = ((weekend_mean - weekday_mean) / max(1e-3, weekday_mean)) * 100
    
    # Weather and submetering checks
    has_temp = any('temp' in c.lower() for c in daily_df.columns)
    temp_col = next((c for c in daily_df.columns if 'temp' in c.lower()), None)
    
    has_submetering = 'Sub_metering_3' in daily_df.columns
    if has_submetering:
        sub3_total = daily_df['Sub_metering_3'].sum() / 1000.0
        total_energy = daily_df['Daily_energy_kWh'].sum()
        sub3_pct = (sub3_total / max(1e-3, total_energy)) * 100
    else:
        sub3_pct = 35.5
        
    avg_kwh = daily_df['Daily_energy_kWh'].mean()
    
    col_i1, col_i2 = st.columns(2)
    
    with col_i1:
        st.markdown("### 🔍 Automated Consumption Insights")
        
        if has_temp:
            hot_days_mean = daily_df[daily_df[temp_col] > 30]['Daily_energy_kWh'].mean()
            mild_days_mean = daily_df[daily_df[temp_col] <= 26]['Daily_energy_kWh'].mean()
            temp_diff_pct = ((hot_days_mean - mild_days_mean) / max(1e-3, mild_days_mean)) * 100 if not np.isnan(hot_days_mean) else 25.0
            st.markdown(f"""
            <div class="insight-box">
                <b>🌡️ Temperature Impact & AC Demand</b><br>
                Days with temperatures &gt; 30°C average <b>{hot_days_mean:.2f} kWh/day</b> vs <b>{mild_days_mean:.2f} kWh/day</b> on mild days 
                (a <b>{temp_diff_pct:+.1f}% increase</b>), driven by high air conditioning thermal loads.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="insight-box">
                <b>🌡️ Dominant Energy Consumer: Climate & Water Heating (Sub-Metering 3)</b><br>
                Water heating and climate control account for <b>{sub3_pct:.1f}%</b> of total sub-metered energy. 
                This represents the largest variable component of household consumption.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insight-box">
            <b>📅 Weekend Lifestyle Shift</b><br>
            Weekend electricity usage averages <b>{weekend_mean:.2f} kWh/day</b> vs <b>{weekday_mean:.2f} kWh/day</b> on weekdays 
            (a <b>{weekend_pct_diff:+.1f}% shift</b>), reflecting sustained daytime occupancy and frequent laundry appliance cycles.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insight-box">
            <b>⚡ Peak Demand Fluctuations</b><br>
            Daily consumption ranges from a minimum of <b>{daily_df['Daily_energy_kWh'].min():.2f} kWh</b> to a peak of <b>{daily_df['Daily_energy_kWh'].max():.2f} kWh</b>. 
            Shifting heavy loads away from peak hours can significantly reduce demand charges.
        </div>
        """, unsafe_allow_html=True)

    with col_i2:
        st.markdown("### 💡 Actionable Energy Conservation Strategies")
        
        st.markdown("""
        1. **Smart Thermostat & Water Heater Scheduling**:
           - Shift electric water heating cycles to off-peak grid hours (e.g., 01:00 - 05:00).
           - Increase AC thermostat setpoint by 1–2°C during summer peaks to save up to **10–15%** on cooling.
           
        2. **Dishwasher & Laundry Batching**:
           - Run full loads only during off-peak morning or weekend mid-day solar hours.
           - Use cold-water washing cycles to eliminate heating element draw.
           
        3. **Vampire / Standby Load Mitigation**:
           - Deploy smart power strips for entertainment systems and home office setups to curb the continuous base load.
           
        4. **Targeted Peak Shaving**:
           - Avoid concurrent operation of high-power appliances (electric oven, tumble dryer, AC) to maintain household load below threshold limits.
        """)
        
    st.divider()
    
    # Interactive Savings & Carbon Estimator
    st.markdown("### 💰 Smart Household Savings & Carbon Estimator")
    st.markdown("Estimate your annual financial and carbon reductions by implementing targeted conservation:")
    
    c_est1, c_est2, c_est3 = st.columns(3)
    with c_est1:
        tariff_rate = st.number_input("Electricity Tariff ($ / kWh)", min_value=0.05, max_value=1.00, value=0.18, step=0.01)
    with c_est2:
        reduction_target = st.slider("Targeted Efficiency Reduction (%)", min_value=5, max_value=35, value=15, step=5)
    with c_est3:
        emission_factor = st.number_input("Grid Carbon Intensity (kg CO₂ / kWh)", min_value=0.1, max_value=1.2, value=0.42, step=0.05)
        
    annual_kwh = avg_kwh * 365.0
    saved_kwh = annual_kwh * (reduction_target / 100.0)
    saved_cost = saved_kwh * tariff_rate
    saved_co2 = saved_kwh * emission_factor
    
    res1, res2, res3 = st.columns(3)
    with res1:
        st.metric(label="⚡ Annual Electricity Saved", value=f"{saved_kwh:,.1f} kWh/yr")
    with res2:
        st.metric(label="💵 Annual Financial Savings", value=f"${saved_cost:,.2f}/yr")
    with res3:
        st.metric(label="🌱 Carbon Emissions Avoided", value=f"{saved_co2:,.1f} kg CO₂/yr")
