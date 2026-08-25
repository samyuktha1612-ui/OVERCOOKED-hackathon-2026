# ⚡ Household Electricity Consumption Forecasting AI
### *16-Hour Hackathon Prototype: Production Time-Series Predictive Intelligence System*

An end-to-end Machine Learning and Deep Learning system that forecasts household electricity consumption, detects surge periods, correlates weather and occupancy patterns, and delivers automated energy-saving recommendations with financial and carbon savings estimation.

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

### 4. (Optional) Re-Train Models from Scratch
- **Train Random Forest & XGBoost ML Pipeline:**
  ```bash
  python3 train_ml_model.py
  ```
- **Train Stacked LSTM Deep Learning Model:**
  ```bash
  python3 train_model.py
  ```
- **Extract 24-Hour Diurnal Profile:**
  ```bash
  python3 extract_hourly.py
  ```

---

## 📁 Repository Structure & Required Files

| File / Folder | Purpose |
|---|---|
| `app.py` | **Main Application**: Interactive Streamlit Web App with live forecasting, KPI cards, Plotly charts, dynamic insights, AI Chatbot, and report export. |
| `chatbot.py` | **Conversational AI Engine**: Grounded domain-specific chatbot answering queries on forecasts, peak demand, weekend/weekday shifts, ML accuracy, and appliance scheduling. |
| `data_processing.py` | **Data Pipeline**: Intelligent auto-format detection (UCI semicolon or standard CSV), missing value interpolation, daily kilowatt-hour aggregation ($\text{kWh} = \frac{1}{60}\sum \text{kW}$), calendar attributes. |
| `ml_forecasting.py` | **ML Engine**: 36 engineered time-series features (lags, rolling stats, cyclical, ratios), leak-free training, autoregressive multi-step recursive forecasting for Random Forest & XGBoost. |
| `forecasting.py` | **Deep Learning Engine**: Time-series sequencing, MinMaxScaler scaling, Stacked LSTM model architecture, autoregressive recursive multi-step forecasting. |
| `visualization.py` | **Visualization Suite**: Interactive Plotly charts (forecasts with confidence bands, actual vs predicted with range sliders, 24h diurnal profiles, feature importances, peak analyses, weather/occupancy correlations). |
| `test_comprehensive_e2e.py` | **Automated Test Suite**: 17-point comprehensive end-to-end verification covering ingestion, features, models, dynamic KPIs, Chatbot engine, and UI safety. |
| `test_chatbot.py` | **Chatbot Unit Tests**: Automated unit testing for all natural language query categories and context grounding. |
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

### **Step 1: Open the App & Introduce the Problem (1 min)**
1. Launch `streamlit run app.py`.
2. Navigate to **🏠 Home / Overview** via the sidebar.
3. **Talking Points:**
   - Household electricity consumption exhibits complex temporal patterns: diurnal peaks, weekend lifestyle shifts, and seasonal weather sensitivity.
   - Explain the end-to-end flow: **Raw Ingestion (2M+ records) → Leak-Free Feature Engineering (36 features) → Multi-Model Benchmarking (RF, XGBoost, LSTM) → Multi-Step Recursive Forecast → Actionable Conservation Engine**.
   - Show the telemetry stats cards (1,442 days monitored, 26.15 kWh/day baseline mean).

### **Step 2: Historical Consumption Analysis (1 min)**
1. Switch to **📊 Historical Consumption Analysis** in the sidebar.
2. **Key Highlights to Show:**
   - **Overall Trend & 7/30-Day Moving Averages**: Highlight long-term seasonal swings.
   - **24-Hour Diurnal Profile**: Point out the sharp evening peak window (18:00–22:00) and overnight base valley (01:00–05:00).
   - **Sub-Metering Breakdown**: Show how Sub-Metering 3 (Water Heater & Climate Control) represents the dominant thermal load.
   - **Weekend vs. Weekday & Peak Days**: Show how lifestyle changes on weekends increase daytime baseline power.

### **Step 3: Live Forecasting Prototype — The Core Demo (2 min)**
1. Navigate to **🔮 Live Forecasting Prototype** (default page).
2. **Interactive Controls:**
   - Select **Forecast Horizon**: Switch between `7 Days`, `14 Days`, and `30 Days Ahead`.
   - Select **Forecasting Model**: Demonstrate toggling between `Random Forest Regressor (R²: 0.411)`, `XGBoost Regressor (R²: 0.402)`, and `Stacked LSTM (R²: 0.332)`.
   - Click **🚀 Generate Forecast**.
3. **Show Dynamic Outputs:**
   - **KPI Cards**: Note how `Recent Consumption`, `Predicted Consumption`, `Expected Peak Usage`, and `Forecast Change %` dynamically update.
   - **Forecast Chart**: Point out the seamless transition from historical actuals into the future horizon with the ±12% expected variation band.
   - **Actual vs Predicted Evaluation Chart**: Use the Plotly range slider to zoom into out-of-sample test predictions.
   - **Feature Importances**: Highlight how `rolling_mean_7`, `rolling_min_7`, and `lag_1` drive model decisions.
   - **Model-Generated Insights**: Show how the app automatically calculates the trajectory slope (e.g. upward vs downward trend) and flags unusually high surge days (>1.25x historical mean).
   - **Export**: Click **⬇️ Download Forecast CSV** to demonstrate production data export.

### **Step 4: Multi-Dataset Support (30 sec)**
1. In the sidebar under **Dataset Configuration**, switch from *UCI Power Benchmark* to **🌦️ 2025 Weather & Occupancy Telemetry**.
2. Show how the application instantly adapts to the new dataset, extracting temperature and occupancy correlations without code changes.

### **Step 5: Smart Insights & Action Plan (30 sec)**
1. Switch to **💡 Smart Insights & Action Plan**.
2. **Interactive Savings Estimator:**
   - Adjust the **Electricity Tariff ($/kWh)**, **Targeted Reduction (%)**, and **Carbon Intensity (kg CO₂/kWh)**.
   - Show how the live calculator outputs exact annual dollar savings and metric tons of carbon emissions avoided.

### **Step 6: AI Energy Assistant Chatbot (1 min)**
1. Switch to **💬 AI Energy Assistant Chatbot** in the sidebar.
2. Click any of the **Suggested Quick-Question Buttons** (e.g. *"🔮 What is my 30-day forecast?"* or *"⏰ Best time for heavy appliances?"*).
3. Type custom natural language questions (e.g. *"How accurate is the Random Forest model?"* or *"Compare weekend vs weekday load"*).
4. Highlight that the chatbot is **grounded directly in the live telemetry and ML model outputs** in real-time.
5. Conclude: *"This is a production-ready, fully verified intelligence system ready for residential and smart-grid deployment."*
