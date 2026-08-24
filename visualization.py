"""
visualization.py
----------------
Modular Plotly visualization builders for Household Electricity Consumption
Analysis and Forecasting.
"""

import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Dict, Any


# Clean, modern styling palette
COLOR_PRIMARY = "#3B82F6"      # Bright blue
COLOR_SECONDARY = "#10B981"    # Emerald green
COLOR_ACCENT = "#F59E0B"       # Amber
COLOR_DANGER = "#EF4444"       # Rose red
COLOR_PURPLE = "#8B5CF6"       # Violet
COLOR_DARK = "#1E293B"         # Slate dark
COLOR_LIGHT_BG = "rgba(248, 250, 252, 0.8)"


def apply_theme(fig: go.Figure, title: str = "", height: int = 450) -> go.Figure:
    """Applies a clean, modern aesthetic to a Plotly figure."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=16, family="Inter, system-ui, sans-serif"),
            x=0.02,
            y=0.96
        ),
        template="plotly_white",
        height=height,
        margin=dict(l=40, r=30, t=50, b=40),
        hoverlabel=dict(
            bgcolor="rgba(15, 23, 42, 0.9)",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#FFFFFF"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.6)"
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(226, 232, 240, 0.7)",
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(226, 232, 240, 0.7)",
            zeroline=False
        )
    )
    return fig


def plot_overall_trend(daily_df: pd.DataFrame, target_col: str = "Daily_energy_kWh") -> go.Figure:
    """Plots overall daily electricity consumption over time with 7-day and 30-day moving averages."""
    df = daily_df.copy().sort_index()
    df['MA_7'] = df[target_col].rolling(window=7, min_periods=1).mean()
    df['MA_30'] = df[target_col].rolling(window=30, min_periods=1).mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[target_col],
        mode='lines',
        name='Daily Consumption (kWh)',
        line=dict(color='rgba(148, 163, 184, 0.45)', width=1),
        hovertemplate='%{x|%b %d, %Y}<br>Actual: <b>%{y:.2f} kWh</b><extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MA_7'],
        mode='lines',
        name='7-Day Moving Average',
        line=dict(color=COLOR_PRIMARY, width=2),
        hovertemplate='%{x|%b %d, %Y}<br>7-Day MA: <b>%{y:.2f} kWh</b><extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MA_30'],
        mode='lines',
        name='30-Day Moving Average',
        line=dict(color=COLOR_ACCENT, width=2.5),
        hovertemplate='%{x|%b %d, %Y}<br>30-Day MA: <b>%{y:.2f} kWh</b><extra></extra>'
    ))
    
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.06),
            type="date"
        )
    )
    return apply_theme(fig, title="Historical Household Electricity Consumption Trend", height=480)


def plot_daily_distribution(daily_df: pd.DataFrame, target_col: str = "Daily_energy_kWh") -> go.Figure:
    """Plots daily consumption histogram and box plot."""
    fig = px.histogram(
        daily_df,
        x=target_col,
        nbins=40,
        marginal="box",
        color_discrete_sequence=[COLOR_PRIMARY],
        opacity=0.8,
        labels={target_col: "Daily Consumption (kWh)"}
    )
    
    mean_val = daily_df[target_col].mean()
    median_val = daily_df[target_col].median()
    
    fig.add_vline(x=mean_val, line_dash="dash", line_color=COLOR_DANGER,
                  annotation_text=f"Mean: {mean_val:.2f} kWh", annotation_position="top right")
    fig.add_vline(x=median_val, line_dash="dot", line_color=COLOR_SECONDARY,
                  annotation_text=f"Median: {median_val:.2f} kWh", annotation_position="bottom right")
    
    return apply_theme(fig, title="Daily Energy Consumption Distribution & Spread", height=420)


def plot_monthly_consumption(daily_df: pd.DataFrame, target_col: str = "Daily_energy_kWh") -> go.Figure:
    """Plots monthly average and total consumption across months."""
    df = daily_df.copy()
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    available_months = [m for m in month_order if m in df['month_name'].unique()]
    
    fig = px.box(
        df,
        x='month_name',
        y=target_col,
        color='month_name',
        category_orders={'month_name': available_months},
        labels={'month_name': 'Month', target_col: 'Daily Energy (kWh)'},
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    fig.update_layout(showlegend=False)
    return apply_theme(fig, title="Monthly Seasonal Variations in Electricity Usage", height=430)


def plot_day_of_week_consumption(daily_df: pd.DataFrame, target_col: str = "Daily_energy_kWh") -> go.Figure:
    """Plots average consumption by day of week with error bars."""
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_stats = daily_df.groupby('day_name')[target_col].agg(['mean', 'std', 'count']).reindex(day_order).dropna().reset_index()
    
    colors = [COLOR_PRIMARY if d not in ['Saturday', 'Sunday'] else COLOR_ACCENT for d in dow_stats['day_name']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dow_stats['day_name'],
        y=dow_stats['mean'],
        error_y=dict(type='data', array=dow_stats['std'], visible=True, color="rgba(100, 116, 139, 0.6)"),
        marker_color=colors,
        text=[f"{v:.2f} kWh" for v in dow_stats['mean']],
        textposition="outside",
        hovertemplate='<b>%{x}</b><br>Average: %{y:.2f} kWh<br>Std Dev: ±%{error_y.array:.2f} kWh<extra></extra>'
    ))
    return apply_theme(fig, title="Average Electricity Consumption by Day of Week", height=420)


def plot_weekday_vs_weekend(daily_df: pd.DataFrame, target_col: str = "Daily_energy_kWh") -> go.Figure:
    """Plots violin and boxplot comparison of weekday vs weekend consumption."""
    df = daily_df.copy()
    df['Day_Type'] = df['is_weekend'].apply(lambda x: 'Weekend (Sat-Sun)' if x == 1 else 'Weekday (Mon-Fri)')
    
    fig = px.violin(
        df,
        x='Day_Type',
        y=target_col,
        color='Day_Type',
        box=True,
        points="outliers",
        color_discrete_map={
            'Weekday (Mon-Fri)': COLOR_PRIMARY,
            'Weekend (Sat-Sun)': COLOR_ACCENT
        },
        labels={'Day_Type': 'Day Type', target_col: 'Daily Energy (kWh)'}
    )
    fig.update_layout(showlegend=False)
    return apply_theme(fig, title="Weekday vs Weekend Electricity Usage Distribution", height=420)


def plot_hourly_consumption_pattern(hourly_dict: Optional[Dict[str, Any]] = None) -> go.Figure:
    """
    Plots the 24-Hour Diurnal Hourly Load Profile showing typical intraday electricity consumption.
    """
    if hourly_dict is None:
        # Load from models/hourly_profile.json if available
        if os.path.exists("models/hourly_profile.json"):
            with open("models/hourly_profile.json", "r") as f:
                hourly_dict = json.load(f)
        else:
            # Realistic synthetic fallback profile for display
            hours = list(range(24))
            base_kw = [0.55, 0.48, 0.45, 0.43, 0.44, 0.52, 0.85, 1.35, 1.42, 1.15,
                       1.05, 1.02, 1.10, 1.08, 1.05, 1.12, 1.35, 1.68, 1.88, 1.95,
                       1.85, 1.60, 1.15, 0.75]
            hourly_dict = {
                'hour_labels': [f"{h:02d}:00" for h in hours],
                'mean_power_kw': base_kw,
                'std_power_kw': [round(x * 0.25, 3) for x in base_kw],
                'peak_hour_label': '19:00',
                'peak_hour_power_kw': 1.95
            }
            
    hours = hourly_dict['hour_labels']
    mean_kw = hourly_dict['mean_power_kw']
    std_kw = hourly_dict.get('std_power_kw', [0.2 * x for x in mean_kw])
    
    upper = [m + s for m, s in zip(mean_kw, std_kw)]
    lower = [max(0.0, m - s) for m, s in zip(mean_kw, std_kw)]
    
    fig = go.Figure()
    
    # Uncertainty envelope
    fig.add_trace(go.Scatter(
        x=hours + hours[::-1],
        y=upper + lower[::-1],
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=True,
        name='Typical Intraday Spread (±1σ)'
    ))
    
    # Main profile line
    fig.add_trace(go.Scatter(
        x=hours,
        y=mean_kw,
        mode='lines+markers',
        name='Average Active Power (kW)',
        line=dict(color=COLOR_PRIMARY, width=3),
        marker=dict(size=6, color=COLOR_PRIMARY),
        hovertemplate='Time: <b>%{x}</b><br>Average Power: <b>%{y:.2f} kW</b><extra></extra>'
    ))
    
    # Highlight Peak Period (18:00 - 22:00)
    fig.add_vrect(
        x0="18:00", x1="22:00",
        fillcolor="rgba(239, 68, 68, 0.12)",
        layer="below", line_width=0,
        annotation_text="Evening Peak Window (18:00–22:00)",
        annotation_position="top left"
    )
    
    # Highlight Off-Peak Valley (01:00 - 05:00)
    fig.add_vrect(
        x0="01:00", x1="05:00",
        fillcolor="rgba(16, 185, 129, 0.10)",
        layer="below", line_width=0,
        annotation_text="Off-Peak Valley (01:00–05:00)",
        annotation_position="bottom left"
    )
    
    return apply_theme(fig, title="24-Hour Diurnal Electricity Consumption Pattern (Average Power in kW)", height=430)


def plot_submetering_breakdown(daily_df: pd.DataFrame) -> Optional[go.Figure]:
    """Plots sub-metering breakdown over time if sub-metering columns exist."""
    if 'Sub_metering_1' not in daily_df.columns:
        return None
        
    df = daily_df.copy().sort_index()
    df['Kitchen (Sub 1, kWh)'] = df['Sub_metering_1'] / 1000.0
    df['Laundry (Sub 2, kWh)'] = df['Sub_metering_2'] / 1000.0
    df['Climate/Heating (Sub 3, kWh)'] = df['Sub_metering_3'] / 1000.0
    df['Other Appliances (kWh)'] = df.get('Sub_metering_remainder', 0.0) / 1000.0
    
    smoothed = df[['Kitchen (Sub 1, kWh)', 'Laundry (Sub 2, kWh)', 'Climate/Heating (Sub 3, kWh)', 'Other Appliances (kWh)']].rolling(14, min_periods=1).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=smoothed.index, y=smoothed['Climate/Heating (Sub 3, kWh)'],
        mode='lines', stackgroup='one', name='Climate & Water Heater (Sub 3)',
        line=dict(width=0.5, color="#F59E0B")
    ))
    fig.add_trace(go.Scatter(
        x=smoothed.index, y=smoothed['Laundry (Sub 2, kWh)'],
        mode='lines', stackgroup='one', name='Laundry Room (Sub 2)',
        line=dict(width=0.5, color="#10B981")
    ))
    fig.add_trace(go.Scatter(
        x=smoothed.index, y=smoothed['Kitchen (Sub 1, kWh)'],
        mode='lines', stackgroup='one', name='Kitchen (Sub 1)',
        line=dict(width=0.5, color="#3B82F6")
    ))
    fig.add_trace(go.Scatter(
        x=smoothed.index, y=smoothed['Other Appliances (kWh)'],
        mode='lines', stackgroup='one', name='Other Base Load & Lighting',
        line=dict(width=0.5, color="#94A3B8")
    ))
    
    return apply_theme(fig, title="Sub-Metering Energy Breakdown Over Time", height=440)


def plot_weather_correlation(daily_df: pd.DataFrame, target_col: str = "Daily_energy_kWh") -> Optional[go.Figure]:
    """Plots Temperature vs Electricity Consumption if weather columns exist."""
    temp_col = next((c for c in daily_df.columns if 'temp' in c.lower()), None)
    if temp_col is None:
        return None
        
    fig = px.scatter(
        daily_df,
        x=temp_col,
        y=target_col,
        color=daily_df['Weather condition'] if 'Weather condition' in daily_df.columns else temp_col,
        trendline="ols",
        labels={temp_col: "Ambient Temperature (°C)", target_col: "Electricity Consumption (kWh)"},
        title="Impact of Ambient Temperature on Electricity Consumption"
    )
    return apply_theme(fig, title="Electricity Consumption vs Ambient Temperature (°C)", height=440)


def plot_occupancy_impact(daily_df: pd.DataFrame, target_col: str = "Daily_energy_kWh") -> Optional[go.Figure]:
    """Plots occupancy (number of people at home) vs consumption if available."""
    occ_col = next((c for c in daily_df.columns if 'people' in c.lower() or 'occupan' in c.lower()), None)
    if occ_col is None:
        return None
        
    fig = px.box(
        daily_df,
        x=occ_col,
        y=target_col,
        color=occ_col,
        labels={occ_col: "Number of People at Home", target_col: "Electricity Consumption (kWh)"}
    )
    fig.update_layout(showlegend=False)
    return apply_theme(fig, title="Household Occupancy vs Electricity Consumption", height=420)


def plot_peak_analysis(daily_df: pd.DataFrame, target_col: str = "Daily_energy_kWh", top_n: int = 10) -> go.Figure:
    """Plots top peak electricity consumption days."""
    top_days = daily_df.sort_values(target_col, ascending=False).head(top_n).copy()
    top_days['Date_Str'] = top_days.index.strftime('%Y-%m-%d (%a)')
    top_days = top_days.sort_values(target_col, ascending=True)
    
    fig = go.Figure(go.Bar(
        x=top_days[target_col],
        y=top_days['Date_Str'],
        orientation='h',
        marker=dict(
            color=top_days[target_col],
            colorscale='YlOrRd',
            showscale=False
        ),
        text=[f"{v:.2f} kWh" for v in top_days[target_col]],
        textposition="outside",
        hovertemplate='<b>%{y}</b><br>Peak Consumption: <b>%{x:.2f} kWh</b><extra></extra>'
    ))
    return apply_theme(fig, title=f"Top {top_n} Peak Electricity Consumption Days", height=440)


def plot_feature_importances(importances: Dict[str, float], top_n: int = 12) -> go.Figure:
    """Plots feature importances from the trained ML model."""
    top_items = list(importances.items())[:top_n]
    features = [k for k, v in top_items][::-1]
    scores = [v for k, v in top_items][::-1]
    
    fig = go.Figure(go.Bar(
        x=scores,
        y=features,
        orientation='h',
        marker=dict(color=COLOR_PRIMARY),
        text=[f"{s:.3f}" for s in scores],
        textposition="outside",
        hovertemplate='<b>%{y}</b><br>Importance Score: <b>%{x:.4f}</b><extra></extra>'
    ))
    return apply_theme(fig, title=f"Top {top_n} Key Forecasting Drivers (Feature Importance)", height=420)


def plot_actual_vs_predicted(
    test_dates: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted_primary: np.ndarray,
    predicted_secondary: Optional[np.ndarray] = None,
    primary_label: str = "Random Forest Forecast",
    secondary_label: str = "Baseline (Persistence)"
) -> go.Figure:
    """Plots Actual Test Data vs Primary ML Predictions and Secondary Predictions."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=actual,
        mode='lines',
        name='Actual Consumption',
        line=dict(color='#0F172A', width=2),
        hovertemplate='%{x|%b %d, %Y}<br>Actual: <b>%{y:.2f} kWh</b><extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=predicted_primary,
        mode='lines',
        name=primary_label,
        line=dict(color=COLOR_PRIMARY, width=2.5, dash='solid'),
        hovertemplate=f'%{{x|%b %d, %Y}}<br>{primary_label}: <b>%{{y:.2f}} kWh</b><extra></extra>'
    ))
    
    if predicted_secondary is not None:
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=predicted_secondary,
            mode='lines',
            name=secondary_label,
            line=dict(color=COLOR_ACCENT, width=1.5, dash='dot'),
            hovertemplate=f'%{{x|%b %d, %Y}}<br>{secondary_label}: <b>%{{y:.2f}} kWh</b><extra></extra>'
        ))
        
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.06),
            type="date"
        )
    )
    return apply_theme(fig, title="Out-of-Sample Test Evaluation: Actual vs ML Model Forecasts", height=480)


def plot_future_forecast(
    historical_tail_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    target_col: str = "Daily_energy_kWh"
) -> go.Figure:
    """Plots seamless transition from recent historical data into future forecast horizon."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=historical_tail_df.index,
        y=historical_tail_df[target_col],
        mode='lines+markers',
        name='Recent Historical',
        line=dict(color='#64748B', width=2),
        marker=dict(size=4),
        hovertemplate='%{x|%b %d, %Y}<br>Historical: <b>%{y:.2f} kWh</b><extra></extra>'
    ))
    
    last_hist_date = historical_tail_df.index[-1]
    last_hist_val = historical_tail_df[target_col].iloc[-1]
    first_fc_date = forecast_df.index[0]
    first_fc_val = forecast_df['Forecast_kWh'].iloc[0]
    
    fig.add_trace(go.Scatter(
        x=[last_hist_date, first_fc_date],
        y=[last_hist_val, first_fc_val],
        mode='lines',
        showlegend=False,
        line=dict(color=COLOR_PRIMARY, width=2.5, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df['Forecast_kWh'],
        mode='lines+markers',
        name='Future Forecast',
        line=dict(color=COLOR_PRIMARY, width=2.5),
        marker=dict(size=6, color=COLOR_PRIMARY),
        hovertemplate='%{x|%b %d, %Y (%a)}<br>Forecast: <b>%{y:.2f} kWh</b><extra></extra>'
    ))
    
    upper_bound = forecast_df['Forecast_kWh'] * 1.12
    lower_bound = np.maximum(0, forecast_df['Forecast_kWh'] * 0.88)
    
    fig.add_trace(go.Scatter(
        x=list(forecast_df.index) + list(forecast_df.index[::-1]),
        y=list(upper_bound) + list(lower_bound[::-1]),
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.12)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Expected Variation Band (±12%)'
    ))
    
    return apply_theme(fig, title="Future Multi-Day Electricity Consumption Forecast", height=460)


def plot_training_loss(history_dict: Dict[str, List[float]]) -> go.Figure:
    """Plots training and validation loss curves across epochs."""
    epochs = list(range(1, len(history_dict['loss']) + 1))
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history_dict['loss'],
        mode='lines+markers',
        name='Training Loss (MSE)',
        line=dict(color=COLOR_PRIMARY, width=2),
        marker=dict(size=4)
    ))
    
    if 'val_loss' in history_dict and history_dict['val_loss']:
        fig.add_trace(go.Scatter(
            x=epochs,
            y=history_dict['val_loss'],
            mode='lines+markers',
            name='Validation Loss (MSE)',
            line=dict(color=COLOR_ACCENT, width=2),
            marker=dict(size=4)
        ))
        
    return apply_theme(fig, title="Model Training & Validation Convergence Curve", height=380)
