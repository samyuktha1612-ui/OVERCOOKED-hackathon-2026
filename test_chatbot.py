"""
test_chatbot.py
---------------
Automated test suite for AI Energy Assistant Chatbot engine.
"""

import os
import pandas as pd
import numpy as np
from chatbot import EnergyChatbotEngine
from data_processing import detect_and_load_data, clean_and_prepare_daily
from ml_forecasting import load_ml_artifacts, forecast_future_ml

def test_chatbot_engine():
    print("=" * 70)
    print(" TESTING AI ENERGY ASSISTANT CHATBOT ENGINE")
    print("=" * 70)

    # 1. Load telemetry and artifacts
    daily_df, summary, target_col = clean_and_prepare_daily(detect_and_load_data("data/household_power_daily.csv"))
    ml_model, ml_features, ml_meta = load_ml_artifacts("models")
    fc_df, fc_sum = forecast_future_ml(ml_model, daily_df, ml_features, horizon_days=30)
    
    avg_kwh = float(daily_df['Daily_energy_kWh'].mean())
    recent_kwh = float(daily_df['Daily_energy_kWh'].iloc[-14:].mean())

    context = {
        'daily_df': daily_df,
        'avg_kwh': avg_kwh,
        'recent_kwh': recent_kwh,
        'ml_meta': ml_meta,
        'model_name': ml_meta.get('model_name', 'Random Forest'),
        'model_label': 'Random Forest Regressor',
        'active_horizon': 30,
        'fc_summary': fc_sum,
        'forecast_df': fc_df,
        'predicted_avg_kwh': fc_sum['expected_avg_kWh'],
        'peak_forecast_kwh': fc_sum['max_forecast_kWh'],
        'peak_forecast_date': fc_sum['max_forecast_date']
    }

    bot = EnergyChatbotEngine(context=context)

    # Test Suite Queries
    test_queries = [
        ("Greetings", "Hello, who are you and what can you do?"),
        ("Forecast", "What is my predicted electricity consumption over the next 30 days?"),
        ("Peak Demand", "When will my peak electricity demand occur?"),
        ("Weekend vs Weekday", "Compare my weekend vs weekday consumption"),
        ("Historical Stats", "What is my historical average daily load?"),
        ("Model Accuracy", "How accurate is the Random Forest model and what is its R2 score?"),
        ("Feature Importance", "What are the most important features driving the forecast?"),
        ("Appliance Scheduling", "When is the cheapest/best time to run the washing machine?"),
        ("Bill Savings", "How much money can I save if I reduce energy by 15%?"),
        ("Weather Impact", "How does ambient temperature impact my electricity consumption?"),
        ("Sub-metering", "What is the breakdown of my sub-metering appliances?")
    ]

    for category, query in test_queries:
        print(f"\n[QUERY: {category}] -> '{query}'")
        resp = bot.generate_response(query)
        assert len(resp) > 50, f"Response too short for {category}"
        assert not ("error" in resp.lower() and "exception" in resp.lower()), f"Error in response for {category}"
        first_line = resp.strip().split('\n')[0]
        print(f"  ✓ Success: {first_line[:60]}... ({len(resp)} chars)")

    print("\n" + "=" * 70)
    print(" ⭐ CHATBOT ENGINE TESTS PASSED WITH 100% SUCCESS! ⭐ ")
    print("=" * 70)

if __name__ == "__main__":
    test_chatbot_engine()
