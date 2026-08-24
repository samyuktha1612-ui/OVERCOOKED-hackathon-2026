import pandas as pd
import numpy as np
import json
import os

from data_processing import load_raw_data

raw_path = 'data/household_power_consumption.txt'
if os.path.exists(raw_path):
    print("Computing 24-hour diurnal profile from UCI dataset...")
    df = pd.read_csv(
        raw_path,
        sep=';',
        na_values=['?'],
        usecols=['Time', 'Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'],
        low_memory=False
    )
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    df['Sub_metering_1'] = pd.to_numeric(df['Sub_metering_1'], errors='coerce')
    df['Sub_metering_2'] = pd.to_numeric(df['Sub_metering_2'], errors='coerce')
    df['Sub_metering_3'] = pd.to_numeric(df['Sub_metering_3'], errors='coerce')
    df = df.dropna(subset=['Global_active_power'])
    
    # Extract Hour
    df['Hour'] = df['Time'].str.split(':').str[0].astype(int)
    
    hourly = df.groupby('Hour').agg({
        'Global_active_power': ['mean', 'std', 'max', 'min'],
        'Sub_metering_1': 'mean',
        'Sub_metering_2': 'mean',
        'Sub_metering_3': 'mean'
    })
    
    hourly_dict = {
        'hours': list(range(24)),
        'hour_labels': [f"{h:02d}:00" for h in range(24)],
        'mean_power_kw': [round(float(x), 3) for x in hourly[('Global_active_power', 'mean')]],
        'std_power_kw': [round(float(x), 3) for x in hourly[('Global_active_power', 'std')]],
        'max_power_kw': [round(float(x), 3) for x in hourly[('Global_active_power', 'max')]],
        'min_power_kw': [round(float(x), 3) for x in hourly[('Global_active_power', 'min')]],
        'sub1_kitchen_wh': [round(float(x), 2) for x in hourly[('Sub_metering_1', 'mean')]],
        'sub2_laundry_wh': [round(float(x), 2) for x in hourly[('Sub_metering_2', 'mean')]],
        'sub3_climate_wh': [round(float(x), 2) for x in hourly[('Sub_metering_3', 'mean')]],
        'peak_hour': int(hourly[('Global_active_power', 'mean')].idxmax()),
        'peak_hour_label': f"{int(hourly[('Global_active_power', 'mean')].idxmax()):02d}:00",
        'peak_hour_power_kw': round(float(hourly[('Global_active_power', 'mean')].max()), 3),
        'min_hour': int(hourly[('Global_active_power', 'mean')].idxmin()),
        'min_hour_label': f"{int(hourly[('Global_active_power', 'mean')].idxmin()):02d}:00",
        'min_hour_power_kw': round(float(hourly[('Global_active_power', 'mean')].min()), 3)
    }
    
    os.makedirs('models', exist_ok=True)
    with open('models/hourly_profile.json', 'w') as f:
        json.dump(hourly_dict, f, indent=4)
    print("✓ Saved 24-hour hourly profile to models/hourly_profile.json:", hourly_dict['peak_hour_label'], "peak at", hourly_dict['peak_hour_power_kw'], "kW")
