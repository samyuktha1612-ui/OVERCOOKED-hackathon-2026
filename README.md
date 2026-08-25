# ⚡ ENERGY INTELLIGENCE
### *AI-Powered Household Electricity Forecasting & Optimization Platform*

> **Understand your energy. Predict your future. Optimize your consumption.**

An enterprise-grade predictive intelligence system coupling multi-year high-frequency telemetry with deep recurrent neural networks and tree ensembles to deliver high-precision multi-step demand forecasts, identify peak surges, and execute automated conservation action plans.

---

## 🚀 Quickstart & Commands

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
*The web application will automatically open at `http://localhost:8501`.*

### 3. Run Automated End-to-End Test Suite
```bash
python3 test_comprehensive_e2e.py
```

### 4. Run AI Energy Assistant Chatbot Tests
```bash
python3 test_chatbot.py
```

### 5. Run 7-Page Rendering Verification Suite
```bash
python3 test_all_pages_render.py
```

---

## 🧭 Platform Navigation (7 Core Workspaces)

1. **🏠 Overview**: Executive landing workspace featuring real-time KPI telemetry cards, multi-year historical consumption trends, upcoming forecast snapshot, and executive smart energy summary.
2. **📊 Historical Analysis**: Deep exploratory analytics covering consumption overviews, 7 & 30-day moving averages, seasonal monthly boxplots, weekly lifestyle behavior, day-of-week load profiles, top 10 peak days, and 24-hour diurnal intraday power curves.
3. **⚡ Live Forecast**: The core hero workspace. Autoregressive multi-step forecasts across **7, 14, and 30-day horizons** using **Random Forest**, **XGBoost**, or **Stacked LSTM**, with confidence variation bands, out-of-sample test evaluation range sliders, and CSV telemetry export.
4. **💡 Smart Insights**: Automated data-driven intelligence cards structured as **`Insight ➔ Evidence ➔ Action`** for peak surge alerts, weekend lifestyle patterns, climate/temperature sensitivity, and an interactive financial & carbon savings estimator.
5. **💬 AI Energy Assistant**: Domain-specific conversational intelligence grounded in real-time telemetry, model metrics, and active forecasts with 1-click starter prompts and custom natural language querying.
6. **🧠 ML Model**: Comprehensive AI forecasting engine documentation for judges: Why LSTM for sequential time-series, end-to-end architectural pipeline, LSTM neural network layer specifications, benchmark comparisons, and training vs validation loss convergence curve.
7. **ℹ️ About**: Problem statement, solution architecture, core capabilities, and operational intelligence flow (⚡ MONITOR ➔ 📊 UNDERSTAND ➔ 🧠 PREDICT ➔ 💡 ACT).

---

## 📁 Repository Structure & Required Files

| File / Folder | Purpose |
|---|---|
| `app.py` | **Main Application**: Redesigned Streamlit Web App with 7 dedicated pages, dark energy-tech glassmorphic styling, live forecasting, KPI cards, Plotly charts, dynamic insights, AI Chatbot, and report export. |
| `chatbot.py` | **Conversational AI Engine**: Grounded domain-specific chatbot answering queries on forecasts, peak demand, weekend/weekday shifts, ML accuracy, and appliance scheduling. |
| `data_processing.py` | **Data Pipeline**: Intelligent auto-format detection (UCI semicolon or standard CSV), missing value interpolation, daily kilowatt-hour aggregation ($\text{kWh} = \frac{1}{60}\sum \text{kW}$), calendar attributes. |
| `ml_forecasting.py` | **ML Engine**: 36 engineered time-series features (lags, rolling stats, cyclical, ratios), leak-free training, autoregressive multi-step recursive forecasting for Random Forest & XGBoost. |
| `forecasting.py` | **Deep Learning Engine**: Time-series sequencing, MinMaxScaler scaling, Stacked LSTM model architecture, autoregressive recursive multi-step forecasting. |
| `visualization.py` | **Visualization Suite**: Interactive Plotly charts (forecasts with confidence bands, actual vs predicted with range sliders, 24h diurnal profiles, feature importances, peak analyses, weather/occupancy correlations) styled with a dark energy-tech palette. |
| `test_comprehensive_e2e.py` | **Automated Test Suite**: 17-point comprehensive end-to-end verification covering ingestion, features, models, dynamic KPIs, Chatbot engine, and UI safety. |
| `test_chatbot.py` | **Chatbot Unit Tests**: Automated unit testing for all natural language query categories and context grounding. |
| `test_all_pages_render.py` | **Multi-Page Verification**: Automated test confirming all 7 navigation pages execute without exceptions. |
| `data/` | **Data Directory**: `household_power_daily.csv` (1,442 continuous days) and `daily_weather_power.csv` (2025 weather & occupancy dataset). |
| `models/` | **Model Artifacts**: Pre-trained model weights, scalers, feature configurations, and metadata JSON. |
| `requirements.txt` | **Dependencies**: Exact package versions required for zero-friction setup. |

---

## 🧠 Model Artifact Locations

All trained models and metadata are serialized inside the `models/` directory:

- `models/ml_forecast_model.joblib`: Primary trained Random Forest Regressor ($R^2 \approx 0.411$, $\text{MAE} \approx 4.18\text{ kWh}$).
- `models/all_ml_models.joblib`: Serialized candidate models dictionary (`Random Forest`, `Gradient Boosting`, `XGBoost`).
- `models/ml_feature_config.json`: Exact list and order of the 36 engineered feature column names.
- `models/ml_metadata.json`: Model evaluation metrics, training/testing sample counts, and feature importance mappings.
- `models/lstm_weights.pkl` & `models/lstm_model.keras`: Trained 2-layer Stacked LSTM model weights.
- `models/scaler.pkl`: Fitted `MinMaxScaler` for LSTM sequences.
- `models/metadata.json`: LSTM metadata, training loss convergence history, and baseline evaluation scores.
- `models/hourly_profile.json`: 24-hour diurnal power consumption profile extracted from over 2 million minute-level telemetry points.

---

## 🎯 Complete Demo Flow for a Judge

Follow this 5-minute presentation script to deliver a high-impact demo to hackathon judges:

### **Step 1: Welcome to ENERGY INTELLIGENCE (1 min)**
1. Launch `streamlit run app.py`.
2. Land on **🏠 Overview**.
3. **Talking Points:**
   - Household electricity consumption exhibits complex temporal patterns: diurnal peaks, weekend lifestyle shifts, and seasonal weather sensitivity.
   - Explain the end-to-end flow: **⚡ MONITOR (2M+ records) ➔ 📊 UNDERSTAND (36 features) ➔ 🧠 PREDICT (LSTM & Ensembles) ➔ 💡 ACT (Conservation & Savings)**.
   - Highlight the dynamic KPI cards (Baseline Load, Record Peak, Monitored Energy, 30D Forecast Mean, Engine Status).

### **Step 2: Historical Consumption Analysis (1 min)**
1. Switch to **📊 Historical Analysis** in the sidebar.
2. **Key Highlights to Show:**
   - **Overall Trend & 7/30-Day Moving Averages**: Highlight long-term seasonal swings.
   - **24-Hour Diurnal Intraday Profile**: Point out the evening peak window (18:00–22:00) and overnight valley (01:00–05:00).
   - **Sub-Metering Breakdown**: Show how Sub-Metering 3 (Water Heater & Climate Control) represents the dominant thermal load.
   - **Weekend vs. Weekday & Peak Days**: Show how daytime presence on weekends increases baseline power.

### **Step 3: Live Energy Forecast — The Hero Feature (1.5 min)**
1. Navigate to **⚡ Live Forecast**.
2. **Interactive Controls:**
   - Select **Forecast Horizon**: Switch between `7 Days`, `14 Days`, and `30 Days Ahead`.
   - Select **Forecasting Algorithm**: Demonstrate toggling between `Random Forest Regressor (R²: 0.411)`, `XGBoost Regressor (R²: 0.402)`, and `Stacked LSTM (R²: 0.332)`.
   - Click **🚀 Generate Forecast**.
3. **Show Dynamic Outputs:**
   - **Forecast KPI Cards**: Note how `Predicted Average`, `Predicted Peak Surge`, `Predicted Minimum`, and `Forecast Horizon` dynamically update.
   - **Forecast Chart**: Point out the seamless transition from historical actuals into the future horizon with the ±12% expected variation band.
   - **Actual vs Predicted Evaluation Chart**: Use the Plotly range slider to zoom into out-of-sample test predictions.
   - **Model Performance Area**: Highlight the real metrics ($R^2$, MAE, RMSE, MAPE).
   - **Export**: Click **⬇️ Download Forecast CSV** to demonstrate production data export.

### **Step 4: Smart Insights & Action Plans (30 sec)**
1. Switch to **💡 Smart Insights**.
2. Show the structured insight cards (**Insight ➔ Evidence ➔ Action**).
3. **Interactive Savings Estimator:**
   - Adjust the **Electricity Tariff ($/kWh)**, **Targeted Reduction (%)**, and **Carbon Intensity (kg CO₂/kWh)**.
   - Show how the live calculator outputs exact annual dollar savings and metric tons of carbon emissions avoided.

### **Step 5: AI Energy Assistant (1 min)**
1. Switch to **💬 AI Energy Assistant** in the sidebar.
2. Click any of the **Suggested Quick-Question Buttons** (e.g. *"🔮 What is my 30-day forecast?"* or *"⏰ Best time for heavy appliances?"*).
3. Type custom natural language questions (e.g. *"How accurate is the Random Forest model?"* or *"Compare weekend vs weekday load"*).
4. Highlight that the chatbot is **grounded directly in the live telemetry and ML model outputs** in real-time.

### **Step 6: AI Forecasting Engine & Architecture (30 sec)**
1. Switch to **🧠 ML Model**.
2. Show the mathematical justification for LSTM (Recurrent gating for sequential memory).
3. Walk judges through the visual pipeline and the interactive **Training vs Validation Loss Convergence Curve**.
4. Conclude: *"ENERGY INTELLIGENCE is a production-ready, fully verified intelligence system ready for residential and smart-grid deployment."*
