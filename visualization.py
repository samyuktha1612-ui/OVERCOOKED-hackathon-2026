"""
visualization.py
----------------
Modular Plotly visualization builders for Household Electricity Consumption
Analysis and Forecasting.
Styled with a premium, futuristic dark energy-tech aesthetic (Electric Cyan & Blue).
"""

import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Dict, Any


# Futuristic dark-energy color palette
COLOR_PRIMARY = "#00F0FF"      # Electric Cyan
COLOR_SECONDARY = "#3B82F6"    # Electric Blue
COLOR_ACCENT = "#38BDF8"       # Sky Blue
COLOR_DANGER = "#F43F5E"       # Rose Red (Peak)
COLOR_SUCCESS = "#10B981"      # Emerald Green
COLOR_WARNING = "#F59E0B"      # Amber
COLOR_PURPLE = "#A855F7"       # Violet
COLOR_DARK_SURFACE = "#0E1626" # Dark Navy Card
COLOR_TEXT_MAIN = "#F8FAFC"    # Soft White
COLOR_TEXT_MUTED = "#94A3B8"   # Slate Grey


def apply_theme(fig: go.Figure, title: str = "", height: int = 450) -> go.Figure:
    """Applies a premium futuristic dark energy-tech aesthetic to a Plotly figure."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=15, family="Plus Jakarta Sans, Inter, system-ui, sans-serif", color=COLOR_TEXT_MAIN),
            x=0.02,
            y=0.96
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11, 17, 30, 0.55)",
        height=height,
        margin=dict(l=45, r=30, t=55, b=45),
        font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#E2E8F0"),
        hoverlabel=dict(
            bgcolor="#080C14",
            bordercolor=COLOR_PRIMARY,
            font_size=12,
            font_family="Plus Jakarta Sans, sans-serif",
            font_color=COLOR_PRIMARY
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(14, 22, 38, 0.75)",
            bordercolor="rgba(56, 189, 248, 0.25)",
            borderwidth=1,
            font=dict(size=11, color="#CBD5E1")
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.10)",
            linecolor="rgba(148, 163, 184, 0.20)",
            tickfont=dict(color="#94A3B8", size=10),
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.10)",
            linecolor="rgba(148, 163, 184, 0.20)",
            tickfont=dict(color="#94A3B8", size=10),
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
        line=dict(color='rgba(148, 163, 184, 0.35)', width=1),
        hovertemplate='%{x|%b %d, %Y}<br>Actual: <b>%{y:.2f} kWh</b><extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MA_7'],
        mode='lines',
        name='7-Day Moving Average',
        line=dict(color=COLOR_PRIMARY, width=2.2),
        hovertemplate='%{x|%b %d, %Y}<br>7-Day MA: <b>%{y:.2f} kWh</b><extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MA_30'],
        mode='lines',
        name='30-Day Moving Average',
        line=dict(color=COLOR_SECONDARY, width=2.5),
        hovertemplate='%{x|%b %d, %Y}<br>30-Day MA: <b>%{y:.2f} kWh</b><extra></extra>'
    ))
    
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                thickness=0.06,
                bgcolor="rgba(14, 22, 38, 0.85)",
                bordercolor="rgba(56, 189, 248, 0.3)"
            ),
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
        opacity=0.85,
        labels={target_col: "Daily Consumption (kWh)"}
    )
    
    mean_val = float(daily_df[target_col].mean())
    median_val = float(daily_df[target_col].median())
    
    fig.add_vline(x=mean_val, line_dash="dash", line_color=COLOR_DANGER,
                  annotation_text=f"Mean: {mean_val:.2f} kWh", annotation_position="top right",
                  annotation_font=dict(color="#FDA4AF", size=10))
    fig.add_vline(x=median_val, line_dash="dot", line_color=COLOR_ACCENT,
                  annotation_text=f"Median: {median_val:.2f} kWh", annotation_position="bottom right",
                  annotation_font=dict(color="#BAE6FD", size=10))
    
    return apply_theme(fig, title="Daily Energy Consumption Distribution & Spread", height=420)


def plot_monthly_consumption(daily_df: pd.DataFrame, target_col: str = "Daily_energy_kWh") -> go.Figure:
    """Plots monthly average and total consumption across months."""
    df = daily_df.copy()
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    available_months = [m for m in month_order if m in df['month_name'].unique()]
    
    palette = ['#00F0FF', '#38BDF8', '#3B82F6', '#60A5FA', '#93C5FD', '#818CF8', '#A855F7', '#C084FC', '#E879F9', '#F472B6', '#FB7185', '#FDA4AF']
    fig = px.box(
        df,
        x='month_name',
        y=target_col,
        color='month_name',
        category_orders={'month_name': available_months},
        labels={'month_name': 'Month', target_col: 'Daily Energy (kWh)'},
        color_discrete_sequence=palette
    )
    fig.update_layout(showlegend=False)
    return apply_theme(fig, title="Monthly Seasonal Variations in Electricity Usage", height=430)


def plot_day_of_week_consumption(daily_df: pd.DataFrame, target_col: str = "Daily_energy_kWh") -> go.Figure:
    """Plots average consumption by day of week with error bars."""
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_stats = daily_df.groupby('day_name')[target_col].agg(['mean', 'std', 'count']).reindex(day_order).dropna().reset_index()
    
    colors = [COLOR_SECONDARY if d not in ['Saturday', 'Sunday'] else COLOR_PRIMARY for d in dow_stats['day_name']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dow_stats['day_name'],
        y=dow_stats['mean'],
        error_y=dict(type='data', array=dow_stats['std'], visible=True, color="rgba(148, 163, 184, 0.4)"),
        marker_color=colors,
        text=[f"{v:.2f} kWh" for v in dow_stats['mean']],
        textposition="outside",
        textfont=dict(color="#F8FAFC", size=10),
        hovertemplate='<b>%{x}</b><br>Average: <b>%{y:.2f} kWh</b><br>Std Dev: ±%{error_y.array:.2f} kWh<extra></extra>'
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
            'Weekday (Mon-Fri)': COLOR_SECONDARY,
            'Weekend (Sat-Sun)': COLOR_PRIMARY
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
        if os.path.exists("models/hourly_profile.json"):
            with open("models/hourly_profile.json", "r") as f:
                hourly_dict = json.load(f)
        else:
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
        fillcolor='rgba(0, 240, 255, 0.12)',
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
        fillcolor="rgba(244, 63, 94, 0.14)",
        layer="below", line_width=0,
        annotation_text="Evening Peak (18:00–22:00)",
        annotation_position="top left",
        annotation_font=dict(color="#FDA4AF", size=10)
    )
    
    # Highlight Off-Peak Valley (01:00 - 05:00)
    fig.add_vrect(
        x0="01:00", x1="05:00",
        fillcolor="rgba(16, 185, 129, 0.12)",
        layer="below", line_width=0,
        annotation_text="Off-Peak Valley (01:00–05:00)",
        annotation_position="bottom left",
        annotation_font=dict(color="#6EE7B7", size=10)
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
        line=dict(width=0.5, color=COLOR_PRIMARY)
    ))
    fig.add_trace(go.Scatter(
        x=smoothed.index, y=smoothed['Laundry (Sub 2, kWh)'],
        mode='lines', stackgroup='one', name='Laundry Room (Sub 2)',
        line=dict(width=0.5, color=COLOR_SECONDARY)
    ))
    fig.add_trace(go.Scatter(
        x=smoothed.index, y=smoothed['Kitchen (Sub 1, kWh)'],
        mode='lines', stackgroup='one', name='Kitchen (Sub 1)',
        line=dict(width=0.5, color=COLOR_PURPLE)
    ))
    fig.add_trace(go.Scatter(
        x=smoothed.index, y=smoothed['Other Appliances (kWh)'],
        mode='lines', stackgroup='one', name='Other Base Load & Lighting',
        line=dict(width=0.5, color="#64748B")
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
        trendline_color_override=COLOR_DANGER,
        color_discrete_sequence=[COLOR_PRIMARY, COLOR_SECONDARY, COLOR_PURPLE, COLOR_WARNING],
        labels={temp_col: "Ambient Temperature (°C)", target_col: "Electricity Consumption (kWh)"}
    )
    return apply_theme(fig, title="Impact of Ambient Temperature on Electricity Consumption", height=440)


def plot_occupancy_impact(daily_df: pd.DataFrame, target_col: str = "Daily_energy_kWh") -> Optional[go.Figure]:
    """Plots occupancy (number of people at home) vs consumption if available."""
    occ_col = next((c for c in daily_df.columns if 'people' in c.lower() or 'occupan' in c.lower()), None)
    if occ_col is None:
        return None
        
    palette_occ = ['#00F0FF', '#38BDF8', '#3B82F6', '#818CF8', '#A855F7']
    fig = px.box(
        daily_df,
        x=occ_col,
        y=target_col,
        color=occ_col,
        color_discrete_sequence=palette_occ,
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
            colorscale=[[0, COLOR_SECONDARY], [0.5, COLOR_PRIMARY], [1, COLOR_DANGER]],
            showscale=False
        ),
        text=[f"{v:.2f} kWh" for v in top_days[target_col]],
        textposition="outside",
        textfont=dict(color="#F8FAFC", size=10),
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
        marker=dict(
            color=scores,
            colorscale=[[0, COLOR_SECONDARY], [1, COLOR_PRIMARY]],
            showscale=False
        ),
        text=[f"{s:.3f}" for s in scores],
        textposition="outside",
        textfont=dict(color="#F8FAFC", size=10),
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
        name='Actual Ground Truth',
        line=dict(color='#94A3B8', width=1.8),
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
            rangeslider=dict(
                visible=True,
                thickness=0.06,
                bgcolor="rgba(14, 22, 38, 0.85)",
                bordercolor="rgba(56, 189, 248, 0.3)"
            ),
            type="date"
        )
    )
    return apply_theme(fig, title="Out-of-Sample Test Evaluation: Actual vs Model Forecasts", height=480)


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
        name='Recent Historical Actuals',
        line=dict(color='#94A3B8', width=2),
        marker=dict(size=4, color='#94A3B8'),
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
        line=dict(color=COLOR_PRIMARY, width=2.2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df['Forecast_kWh'],
        mode='lines+markers',
        name='Future AI Forecast',
        line=dict(color=COLOR_PRIMARY, width=2.8),
        marker=dict(size=6, color=COLOR_PRIMARY),
        hovertemplate='%{x|%b %d, %Y (%a)}<br>Forecast: <b>%{y:.2f} kWh</b><extra></extra>'
    ))
    
    upper_bound = forecast_df['Forecast_kWh'] * 1.12
    lower_bound = np.maximum(0, forecast_df['Forecast_kWh'] * 0.88)
    
    fig.add_trace(go.Scatter(
        x=list(forecast_df.index) + list(forecast_df.index[::-1]),
        y=list(upper_bound) + list(lower_bound[::-1]),
        fill='toself',
        fillcolor='rgba(0, 240, 255, 0.14)',
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
        line=dict(color=COLOR_PRIMARY, width=2.2),
        marker=dict(size=4, color=COLOR_PRIMARY),
        hovertemplate='Epoch %{x}<br>Train Loss: <b>%{y:.5f}</b><extra></extra>'
    ))
    
    if 'val_loss' in history_dict and history_dict['val_loss']:
        fig.add_trace(go.Scatter(
            x=epochs,
            y=history_dict['val_loss'],
            mode='lines+markers',
            name='Validation Loss (MSE)',
            line=dict(color=COLOR_SECONDARY, width=2.2),
            marker=dict(size=4, color=COLOR_SECONDARY),
            hovertemplate='Epoch %{x}<br>Val Loss: <b>%{y:.5f}</b><extra></extra>'
        ))
        
    return apply_theme(fig, title="LSTM Neural Network: Training vs Validation Loss Convergence Curve", height=390)
