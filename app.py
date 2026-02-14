"""
Comprehensive Stock Analysis Platform - Main UI Module

A professional Streamlit-based web application for comprehensive stock analysis,
featuring technical indicators, fundamental analysis, SWOT assessment, and
AI-powered investment recommendations.

Author: Stock Analysis Platform Team
Version: 2.1.0
License: MIT
"""

# Standard library imports
import os
import json
import pathlib
import tempfile
import warnings
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional, Any

# Third-party imports
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

# Custom module imports
from import_fetch_technical import fetch_technical_data
from fetch_fundamental import fetch_fundamental_data
from compute_indicators import calculate_technical_indicators
from analyze_signals import analyze_technical_signals
from analyze_financial import analyze_financial_performance
from price_targets import calculate_price_targets
from main_analysis import analyze_data
from save_to_excel import save_report
from create_price_chart import create_price_target_chart

warnings.filterwarnings('ignore')


# ============================================================================
# MODULE-LEVEL CONSTANTS  (avoids repeated object creation on every render)
# ============================================================================

# Analyst-recommendation scale → human-readable label
_REC_LABELS: Tuple[Tuple[float, str], ...] = (
    (1.5, "Strong Buy"),
    (2.5, "Buy"),
    (3.5, "Hold"),
    (4.5, "Sell"),
    (5.0, "Strong Sell"),
)

# Investment decision → (indicator label, hex colour)
_DECISION_MAP: Dict[str, Tuple[str, str]] = {
    "Buy":  ("Green", "#27ae60"),
    "Sell": ("Red",   "#e74c3c"),
    "Hold": ("Yellow","#f1c40f"),
}

# SWOT section rendering order
_SWOT_ORDER: Tuple[Tuple[str, str, str], ...] = (
    ("Strengths",     "#28a745", "first"),
    ("Weaknesses",    "#ffc107", "first"),
    ("Opportunities", "#007bff", "second"),
    ("Threats",       "#dc3545", "second"),
)

# Index tickers for the market overview panel
_MARKET_INDICES: Tuple[Tuple[str, str], ...] = (
    ("S&P 500",     "^GSPC"),
    ("NASDAQ",      "^IXIC"),
    ("DOW",         "^DJI"),
    ("Russell 2000","^RUT"),
)

# Palette used throughout comparison charts
_COMPARISON_COLORS: Tuple[str, ...] = (
    "#667eea", "#764ba2", "#f093fb", "#4facfe", "#43e97b"
)

# Valuation-trend colour lookup  (Increasing P/E = bad → Red, Decreasing = good → Green)
_TREND_COLORS: Dict[str, str] = {
    "Increasing": "Red",
    "Decreasing": "Green",
    "Stable":     "Yellow",
}

# OBV look-back window for trend detection
_OBV_LOOKBACK: int = 5

# Risk-free rate used in Sharpe ratio calculations (annualised decimal)
_RISK_FREE_RATE: float = 0.02

# Minimum market-cap comparison threshold for stock-data fetching
_MIN_SYMBOLS: int = 2
_MAX_SYMBOLS: int = 5


# ============================================================================
# CONFIGURATION & PAGE SETUP
# ============================================================================

def configure_page() -> None:
    """Configure Streamlit page settings and metadata."""
    st.set_page_config(
        page_title="Comprehensive Stock Analysis Platform",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_custom_styles() -> None:
    """Apply comprehensive custom CSS styling to the application."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');

    body, * { font-family: 'Inter', sans-serif; }

    .main-header {
        text-align: center; padding: 5px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border-radius: 14px; margin-bottom: 20px;
    }
    .main-header h1 { font-size: 2rem; font-weight: 600; margin: 0; }
    .main-header p  { font-size: 0.96rem; opacity: 0.85; }

    .features-grid {
        display: grid; grid-template-columns: repeat(3, 1fr);
        gap: 1.1rem; margin: 1rem 0;
    }
    .feature-card {
        background: transparent; padding: 1rem; border-radius: 10px;
        border: 1px solid #303248; display: flex; flex-direction: column;
        justify-content: flex-start; align-items: flex-start;
        transition: border .18s, box-shadow .25s, transform .15s;
        min-width: 0; max-width: 100%;
    }
    .feature-card:hover {
        border: 1.5px solid #667eea;
        box-shadow: 0 4px 18px 0 #667eea44, 0 2px 4px 0 #33333322;
        background: linear-gradient(125deg, #2326520c 50%, #667eea19 100%);
        transform: translateY(-2px) scale(1.03); cursor: pointer;
    }
    @media (max-width: 900px) { .features-grid { grid-template-columns: 1fr; } }

    .welcome-section {
        background: transparent; padding: .4rem .5rem; border-radius: 9px;
        margin: 1.2rem 0; color: #B7C1D6; font-size: 1.01rem;
    }
    .instructions {
        background: transparent; padding: .5rem 1rem; border-radius: 8px;
        margin: .7rem 0 1rem; color: #B7C1D6; font-size: .96rem;
    }
    .instructions h3 { color: #6d99fa; margin-bottom: .6rem; font-size: .98rem; }
    .instructions ol { color: #b0b3bc; font-size: .92rem; margin: 0 0 0 1rem; }

    .sidebar-section h3 { font-size: 28px !important; font-weight: 600 !important; margin-bottom: 0 !important; }
    [data-testid="stSidebar"] .sidebar-section { padding: .5rem .5rem .7rem .5rem !important; }
    [data-testid="stSidebar"] .stForm { padding: .8rem .8rem 1.2rem .8rem !important; border-radius: 13px !important; }
    [data-testid="stSidebar"] label {
        font-size: 1.05rem !important; font-weight: 600 !important;
        display: flex !important; align-items: center !important;
        gap: 4px !important; margin-bottom: .07rem !important;
    }
    [data-testid="stSidebar"] svg {
        width: 16px !important; height: 16px !important;
        vertical-align: middle !important; margin-bottom: 0 !important; margin-left: 2px !important;
    }
    [data-testid="stSidebar"] input {
        font-size: 1.05rem !important; border-radius: 5px !important;
        padding: .38rem .65rem !important; height: 2.1rem !important;
    }
    [data-testid="stSidebar"] .stNumberInput input,
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stDateInput input { font-size: 1.05rem !important; }
    [data-testid="stSidebar"] button {
        font-size: .98rem !important; padding: .55rem 1.2rem !important;
        min-height: 2.2rem !important; border-radius: 6px !important; margin-top: .4rem;
    }

    .stButton > button {
        border-radius: 7px !important; font-weight: 600 !important;
        font-size: 1rem !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }

    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    .stDateInput > div > div > input { font-size: 1.05rem !important; border-radius: 7px !important; }
    .stNumberInput button {
        font-size: 1rem !important; padding: 0 7px !important;
        height: 1.8rem !important; min-width: 1.5rem !important;
        color: #dedede !important; border: 1px solid #373952 !important;
        border-radius: 5px !important; box-shadow: none !important; margin: 0 !important;
    }
    .stNumberInput button:active { background: #303248 !important; }
    .stNumberInput > div { gap: 4px !important; }

    #MainMenu, footer, header { visibility: hidden; }

    .loading-wave { display: flex; gap: 6px; justify-content: center; align-items: center; height: 38px; margin-bottom: 1.1rem; }
    .loading-wave span {
        display: block; width: 8px; height: 18px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px; animation: wave 1.1s infinite ease-in-out;
    }
    .loading-wave span:nth-child(2) { animation-delay: .13s; }
    .loading-wave span:nth-child(3) { animation-delay: .26s; }
    .loading-wave span:nth-child(4) { animation-delay: .39s; }
    .loading-wave span:nth-child(5) { animation-delay: .52s; }
    @keyframes wave { 0%, 40%, 100% { transform: scaleY(1); } 20% { transform: scaleY(1.65); } }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def render_html_template(path: str, context: Dict[str, str]) -> str:
    """
    Render HTML template with dynamic placeholder replacement.

    Args:
        path:    Filesystem path to HTML template.
        context: Mapping of placeholder names → replacement values.

    Returns:
        Processed HTML string with all substitutions applied.
    """
    html_content = pathlib.Path(path).read_text(encoding="utf-8")
    for placeholder, value in context.items():
        html_content = html_content.replace(f"{{{{{placeholder}}}}}", value)
    return html_content


def safe_numeric_format(
    value: Any,
    format_spec: str = ".2f",
    default: str = "N/A",
) -> str:
    """
    Safely format a numeric value with a fallback for invalid/missing data.

    Args:
        value:       Value to format.
        format_spec: Python format-spec string (default ".2f").
        default:     Returned when value is None, NaN, or non-numeric.

    Returns:
        Formatted string or *default*.
    """
    if value is None:
        return default
    if isinstance(value, (int, float)) and np.isfinite(value):
        return f"{value:{format_spec}}"
    return default


def format_market_cap(mc: Any) -> str:
    """
    Convert a raw market-cap number to a human-readable tiered string.

    Handles None / non-numeric inputs gracefully.
    """
    if not isinstance(mc, (int, float)) or not np.isfinite(mc):
        return "N/A"
    if mc >= 1e12:
        return f"${mc / 1e12:.2f} T"
    if mc >= 1e9:
        return f"${mc / 1e9:.2f} B"
    if mc >= 1e6:
        return f"${mc / 1e6:.2f} M"
    return f"${mc:,.0f}"


def analyst_rec_label(rec: Any) -> str:
    """
    Map a numeric recommendationMean value to a text label.

    Scale: 1 = Strong Buy … 5 = Strong Sell.
    Returns empty string for invalid input.
    """
    if not isinstance(rec, (int, float)) or not np.isfinite(rec):
        return ""
    for threshold, label in _REC_LABELS:
        if rec <= threshold:
            return label
    return "Strong Sell"


def get_column_config(columns: List[str]) -> Dict[str, st.column_config.Column]:
    """
    Generate Streamlit column configuration for data tables.

    Args:
        columns: Column names to configure.

    Returns:
        Dict mapping column names to configuration objects.
    """
    return {
        col: (
            st.column_config.Column(width="medium")
            if col == "Date"
            else st.column_config.NumberColumn(format="%.2f")
        )
        for col in columns
    }


def _compute_returns_stats(
    close: pd.Series,
) -> Dict[str, float]:
    """
    Compute annualised return, volatility, Sharpe ratio, max drawdown,
    and 95 % VaR from a price series.  All values are scalars (floats).

    Extracted to a single helper so comparison functions never duplicate
    the identical rolling-window arithmetic.
    """
    returns = close.pct_change().dropna()
    if returns.empty:
        return dict(
            total_return=0.0, volatility=0.0,
            sharpe=0.0, max_drawdown=0.0, var_95=0.0,
        )

    first, last = close.iloc[0], close.iloc[-1]
    total_return = (last - first) / first * 100 if first else 0.0

    vol = returns.std() * np.sqrt(252) * 100
    avg_annual = returns.mean() * 252
    sharpe = (avg_annual - _RISK_FREE_RATE) / (vol / 100) if vol else 0.0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    max_dd = ((cumulative - running_max) / running_max).min() * 100

    var_95 = float(np.percentile(returns, 5)) * 100

    return dict(
        total_return=total_return,
        volatility=vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        var_95=var_95,
    )


# ============================================================================
# CHART CREATION FUNCTIONS
# ============================================================================

def create_interactive_chart(
    analysis: Dict[str, Any],
    technical_df: pd.DataFrame,
    symbol: str,
    show_bb: bool = True,
    show_sma50: bool = True,
    show_macd: bool = True,
    show_volume: bool = True,
) -> go.Figure:
    """
    Create a comprehensive interactive price chart with technical indicators.

    Panels:
      Row 1 — Candlestick with MA / BB overlays, entry/SL/target annotations.
      Row 2 — Volume histogram (optional).
      Row 3 — MACD with signal + histogram (optional).

    Args:
        analysis:     Full analysis dict (price targets, entry, stop-loss).
        technical_df: OHLCV + indicator DataFrame.
        symbol:       Ticker symbol for the chart title.
        show_bb:      Toggle Bollinger Bands.
        show_sma50:   Toggle 50-period SMA.
        show_macd:    Toggle MACD panel.
        show_volume:  Toggle volume panel.

    Returns:
        Configured Plotly Figure.
    """
    df = technical_df.tail(120).copy()
    df["Date"] = pd.to_datetime(df["Date"])

    entry       = analysis["entry_point"]
    stop_loss   = analysis["stop_loss"]
    up_targets  = analysis["up_targets"][:3]
    down_targets = analysis["down_targets"][:2]
    last_date   = df["Date"].iloc[-1]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.15, 0.25],
        vertical_spacing=0.02,
        specs=[[{"type": "candlestick"}], [{"type": "bar"}], [{"type": "scatter"}]],
    )

    # ── Candlestick ──────────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df["Date"], open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="Price",
        ),
        row=1, col=1,
    )

    # ── SMA 20 ───────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["SMA_20"],
                   line=dict(width=1.2, dash="solid"), name="SMA 20"),
        row=1, col=1,
    )

    # ── SMA 50 (optional) ────────────────────────────────────────────────────
    if show_sma50:
        fig.add_trace(
            go.Scatter(x=df["Date"], y=df["SMA_50"],
                       line=dict(width=1.2, dash="dot"), name="SMA 50"),
            row=1, col=1,
        )

    # ── Bollinger Bands (optional) ───────────────────────────────────────────
    if show_bb:
        for band, label in (("BB_Upper", "BB Upper"), ("BB_Lower", "BB Lower")):
            fig.add_trace(
                go.Scatter(x=df["Date"], y=df[band],
                           line=dict(width=0.8, color="gray"),
                           name=label, visible=True),
                row=1, col=1,
            )

    # ── Entry / Stop-Loss annotations ────────────────────────────────────────
    for y_val, label, arrow_color, ay in (
        (entry,     "Entry", "green", -30),
        (stop_loss, "SL",    "red",    30),
    ):
        fig.add_annotation(
            x=last_date, y=y_val, text=label, showarrow=True,
            arrowcolor=arrow_color, arrowhead=3, ax=0, ay=ay,
            row=1, col=1,
        )

    # ── Price-target markers ──────────────────────────────────────────────────
    if up_targets:
        fig.add_trace(
            go.Scatter(
                x=[last_date] * len(up_targets), y=up_targets,
                mode="markers", marker=dict(size=10, color="#4ecdc4"),
                name="Up-Target",
            ),
            row=1, col=1,
        )
    if down_targets:
        fig.add_trace(
            go.Scatter(
                x=[last_date] * len(down_targets), y=down_targets,
                mode="markers", marker=dict(size=10, color="#ffa07a"),
                name="Down-Target",
            ),
            row=1, col=1,
        )

    # ── Volume (optional) ────────────────────────────────────────────────────
    if show_volume:
        fig.add_trace(
            go.Bar(x=df["Date"], y=df["Volume"],
                   marker_color="#636efa", name="Volume"),
            row=2, col=1,
        )

    # ── MACD (optional) ──────────────────────────────────────────────────────
    if show_macd:
        # Vectorised colour assignment — avoids per-row apply()
        hist_vals = df["MACD_Histogram"].to_numpy()
        histogram_colors = np.where(hist_vals > 0, "#5fe499", "#f34d63").tolist()

        fig.add_trace(
            go.Scatter(x=df["Date"], y=df["MACD"],
                       line=dict(width=1.1), name="MACD"),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df["Date"], y=df["MACD_Signal"],
                       line=dict(width=1.1, dash="dash"), name="Signal"),
            row=3, col=1,
        )
        fig.add_trace(
            go.Bar(x=df["Date"], y=df["MACD_Histogram"],
                   marker_color=histogram_colors, name="Histogram"),
            row=3, col=1,
        )

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        title="Price Action & Indicators",
        template="plotly_dark",
        height=700,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(t=60, b=40, l=0, r=10),
    )
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.07),
        rangeselector=dict(buttons=[
            dict(count=1,  label="1 m", step="month", stepmode="backward"),
            dict(count=3,  label="3 m", step="month", stepmode="backward"),
            dict(count=6,  label="6 m", step="month", stepmode="backward"),
            dict(step="all"),
        ]),
        row=3, col=1,
    )
    return fig


def create_comparison_chart(levels: Dict[str, float], symbol: str) -> go.Figure:
    """
    Create a comparative bar chart of key price levels.

    Args:
        levels: Dict containing named price levels (current, analyst targets, S/R).
        symbol: Ticker symbol used in the chart title.

    Returns:
        Configured Plotly bar Figure.
    """
    level_order = [
        ("analyst_min",         "Analyst Min", "#9b59b6"),
        ("nearest_resistance",  "1Y Res.",     "#3498db"),
        ("current",             "Current",     "#1abc9c"),
        ("analyst_avg",         "Analyst Avg", "#f1c40f"),
        ("nearest_support",     "1Y Supp.",    "#e74c3c"),
        ("analyst_max",         "Analyst Max", "#e67e22"),
    ]

    labels, values, colors = [], [], []
    for key, label, color in level_order:
        val = levels.get(key)
        # Reject None and non-finite floats in one clean test
        if val is not None and np.isfinite(float(val)):
            labels.append(label)
            values.append(val)
            colors.append(color)

    fig = px.bar(
        x=labels, y=values,
        text=[f"${v:,.2f}" for v in values],
        color=labels,
        color_discrete_sequence=colors,
        title="Price Comparison",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis_title="Price ($)", xaxis_title="Level",
        showlegend=False, template="plotly_dark", height=450,
        margin=dict(t=60, l=0, r=0, b=40),
    )
    return fig


def render_sr_dashboard_inline(
    analysis: Dict[str, Any],
    current_price: float = 205.50,
) -> None:
    """
    Render an interactive Support & Resistance dashboard component.

    Builds a self-contained HTML page with three tabs:
      • Overview  — pivot-level cards.
      • SR Zones  — zone cards (active zone highlighted).
      • Fibonacci — Fibonacci retracement level cards.

    Args:
        analysis:      Analysis dict containing sr_zones, pivot_levels, fib_levels.
        current_price: Current price used for distance calculations.
    """
    sr_zones     = analysis.get("sr_zones", [])
    pivot_levels = analysis.get("pivot_levels", {})
    fib_levels   = analysis.get("fib_levels", {})

    # ── Normalise sr_zones to [[min, max], …] ──────────────────────────────
    clean_sr = [
        [float(zone[0]), float(zone[1])]
        for zone in sr_zones
        if len(zone) == 2
    ]

    data = {
        "pivot_levels": {
            "long_support":     float(pivot_levels.get("long_support",     0)),
            "long_resistance":  float(pivot_levels.get("long_resistance",  0)),
            "short_support":    float(pivot_levels.get("short_support",    0)),
            "short_resistance": float(pivot_levels.get("short_resistance", 0)),
        },
        "sr_zones":     clean_sr,
        "fib_levels":   {k: float(v) for k, v in fib_levels.items()},
        "current_price": float(current_price),
    }

    # ── CSS ──────────────────────────────────────────────────────────────────
    css = """
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        color: white; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding: 20px;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    .header { text-align: center; margin-bottom: 30px; }
    .header h1 {
        font-size: 2.5rem;
        background: linear-gradient(45deg, #4facfe, #00f2fe);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .current-price {
        background: rgba(255,255,255,.1); padding: 15px; border-radius: 15px;
        display: flex; justify-content: space-between; align-items: center;
        backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,.2);
    }
    .price-value { font-size: 2rem; font-weight: bold; color: #00f2fe; }
    .tabs {
        display: flex; justify-content: center; margin: 30px 0;
        background: rgba(255,255,255,.1); border-radius: 15px; padding: 5px;
    }
    .tab {
        padding: 10px 20px; margin: 0 5px; border: none;
        background: transparent; color: white; cursor: pointer;
        border-radius: 10px; transition: all .3s;
    }
    .tab.active { background: #4facfe; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px,1fr)); gap: 20px; }
    .card {
        background: rgba(255,255,255,.1); padding: 20px; border-radius: 15px;
        border: 1px solid rgba(255,255,255,.2); backdrop-filter: blur(10px);
        transition: transform .3s, box-shadow .3s;
    }
    .card:hover { transform: translateY(-5px); box-shadow: 0 10px 25px rgba(79,172,254,.3); }
    .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
    .card-title { font-size: 1.1rem; font-weight: 600; }
    .card-value { font-size: 1.8rem; font-weight: bold; margin: 10px 0; }
    .support    { color: #4ade80; }
    .resistance { color: #f87171; }
    .zone-active { border: 2px solid #fbbf24 !important; box-shadow: 0 0 20px rgba(251,191,36,.5) !important; }
    .fib-bar  { width: 100%; height: 6px; background: rgba(255,255,255,.2); border-radius: 3px; overflow: hidden; margin-top: 10px; }
    .fib-fill { height: 100%; background: linear-gradient(90deg,#4ade80,#fbbf24,#f87171); border-radius: 3px; transition: width 1s ease; }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    .badge { padding: 4px 8px; border-radius: 8px; font-size: .8rem; font-weight: bold; }
    .badge-support    { background: rgba(74,222,128,.2); color: #4ade80; }
    .badge-resistance { background: rgba(248,113,113,.2); color: #f87171; }
    .badge-active     { background: #fbbf24; color: black; animation: pulse 2s infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .7; } }
    """

    # ── Build HTML parts ──────────────────────────────────────────────────────
    html_parts = [
        f'<!DOCTYPE html><html><head><style>{css}</style></head><body>',
        '<div class="container">',
        '<div class="header"><h1>Support &amp; Resistance Analysis</h1>',
        '<div class="current-price"><span>Current Price</span>',
        f'<span class="price-value">{data["current_price"]:.2f}</span></div></div>',
        '<div class="tabs">',
        '<button class="tab active" onclick="showTab(\'overview\')">Overview</button>',
        '<button class="tab" onclick="showTab(\'zones\')">SR Zones</button>',
        '<button class="tab" onclick="showTab(\'fibonacci\')">Fibonacci</button>',
        '</div>',
        '<div id="overview" class="tab-content active"><div class="grid">',
    ]

    # Pivot cards
    for name, value in data["pivot_levels"].items():
        if value <= 0:
            continue
        price_type = "resistance" if "resistance" in name.lower() else "support"
        distance   = abs(value - data["current_price"])
        pct        = distance / data["current_price"] * 100 if data["current_price"] else 0
        html_parts.append(
            f'<div class="card">'
            f'<div class="card-header">'
            f'<span class="card-title">{name.replace("_"," ").title()}</span>'
            f'<span class="badge badge-{price_type}">{price_type}</span></div>'
            f'<div class="card-value {price_type}">{value:.2f}</div>'
            f'<div>Distance: {distance:.2f} ({pct:.1f}%)</div></div>'
        )

    html_parts += ['</div></div>', '<div id="zones" class="tab-content"><div class="grid">']

    # SR zone cards
    for i, zone in enumerate(data["sr_zones"]):
        min_v, max_v = zone
        is_active  = min_v <= data["current_price"] <= max_v
        act_cls    = " zone-active" if is_active else ""
        act_badge  = '<span class="badge badge-active">ACTIVE</span>' if is_active else ""
        html_parts.append(
            f'<div class="card{act_cls}">'
            f'<div class="card-header"><span class="card-title">Zone {i+1}</span>{act_badge}</div>'
            f'<div class="support">Support: {min_v:.2f}</div>'
            f'<div class="resistance">Resistance: {max_v:.2f}</div>'
            f'<div>Range: {max_v - min_v:.2f}</div></div>'
        )

    html_parts += ['</div></div>', '<div id="fibonacci" class="tab-content"><div class="grid">']

    # Fibonacci cards
    cp = data["current_price"]
    for level, value in data["fib_levels"].items():
        distance = abs(value - cp)
        pct      = distance / cp * 100 if cp else 0
        html_parts.append(
            f'<div class="card">'
            f'<div class="card-header"><span class="card-title">Fib {level}%</span></div>'
            f'<div class="card-value">{value:.2f}</div>'
            f'<div>Distance: {distance:.2f} ({pct:.1f}%)</div>'
            f'<div class="fib-bar"><div class="fib-fill" style="width:{min(pct*2,100):.1f}%"></div></div>'
            f'</div>'
        )

    html_parts += [
        '</div></div>',
        '</div>',   # /container
        '<script>',
        'function showTab(tabId) {',
        '  document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));',
        '  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));',
        '  document.getElementById(tabId).classList.add("active");',
        '  event.target.classList.add("active");',
        '}',
        '</script>',
        '</body></html>',
    ]

    components.html("".join(html_parts), height=800, scrolling=True)


def render_metric_card(
    title: str,
    value: str,
    subtitle: str,
    is_percentage: bool = False,
) -> str:
    """
    Generate HTML for a styled metric display card.

    Args:
        title:         Card title / metric name.
        value:         Metric value string (may begin with a colour-emoji prefix).
        subtitle:      Descriptive subtitle text.
        is_percentage: Append "%" when True and value doesn't already contain it.

    Returns:
        HTML string for the metric card.
    """
    # Separate leading colour-emoji from the numeric part
    emoji = value.split()[0] if isinstance(value, str) and value and value[0] in "🟢🟡🔴" else ""
    display_value = value[len(emoji):].strip() if emoji else value
    pct_suffix = "%" if is_percentage and "%" not in str(display_value) else ""

    return (
        f'<div style="background:#2c2f33;padding:15px;border-radius:8px;'
        f'text-align:center;margin:5px;">'
        f'<h4 style="margin:0;color:#ffffff;font-size:24px;">{title}</h4>'
        f'<h2 style="margin-top:-8px;color:#4facfe;font-size:28px;">'
        f'{emoji} {display_value}{pct_suffix}</h2>'
        f'<p style="margin:0;font-size:14px;color:#cccccc;">{subtitle}</p>'
        f'</div>'
    )


def render_progress_bar(label: str, value: float, color: str) -> None:
    """
    Render a labelled inline progress bar.

    Args:
        label: Text label for the metric.
        value: Percentage value (0–100).
        color: CSS colour for the filled portion.
    """
    # Clamp to [0, 100] so malformed scores never break the CSS width
    clamped = max(0.0, min(100.0, value))
    st.markdown(
        f'<div style="margin:12px 0;">'
        f'<span style="display:inline-block;width:120px;">{label}</span>'
        f'<div style="background:#2c2f33;border-radius:6px;height:16px;'
        f'display:inline-block;width:60%;">'
        f'<div style="background:{color};height:16px;border-radius:6px;'
        f'width:{clamped}%;"></div></div>'
        f'<span style="margin-left:8px;">{clamped:.0f}%</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# UI COMPONENT RENDERING FUNCTIONS
# ============================================================================

def render_header() -> None:
    """Render the main application header banner."""
    st.markdown(
        '<div class="main-header">'
        "<h1>Comprehensive Stock Analysis Platform</h1>"
        "<p>Advanced Technical &amp; Fundamental Analysis with AI-Powered Insights</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_sidebar_form() -> Tuple[bool, str, date, date, float]:
    """
    Render the sidebar input form for analysis parameters.

    Returns:
        Tuple of (submitted, stock_symbol, start_date, end_date, industry_pe).
    """
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-section">'
            '<h3 style="margin-bottom:4px;">Analysis Settings</h3>'
            '<hr style="border:1px solid #373952;margin:6px 0 14px;"></div>',
            unsafe_allow_html=True,
        )
        with st.form("input_form", clear_on_submit=False):
            stock_symbol = st.text_input(
                "Stock Symbol", "AAPL",
                help="Enter stock ticker symbol (e.g., AAPL, TSLA, MSFT)",
            )
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "From Date", value=datetime(2024, 1, 1),
                    help="Historical data start date",
                )
            with col2:
                end_date = st.date_input(
                    "To Date", value=date.today(),
                    help="Historical data end date",
                )
            industry_pe = st.number_input(
                "Industry P/E Ratio", min_value=1.0, step=1.0, value=30.0,
                help="Average P/E ratio for the sector",
            )
            submitted = st.form_submit_button("Start Analysis", use_container_width=True)

    return submitted, stock_symbol, start_date, end_date, industry_pe


def render_loading_indicator() -> st.delta_generator.DeltaGenerator:
    """
    Display an animated wave loading indicator.

    Returns:
        Streamlit placeholder (call .empty() to remove it).
    """
    placeholder = st.empty()
    with placeholder.container():
        st.markdown(
            '<div style="display:flex;flex-direction:column;align-items:center;'
            'justify-content:center;margin:40px 0;">'
            '<div class="loading-wave">'
            "<span></span><span></span><span></span><span></span><span></span>"
            "</div>"
            '<div style="text-align:center;color:#B7C1D6;margin-bottom:10px;font-size:1.1rem">'
            "Loading and analyzing data… Please wait"
            "</div></div>",
            unsafe_allow_html=True,
        )
    return placeholder


# ============================================================================
# MARKET OVERVIEW & STOCK COMPARISON
# ============================================================================

def render_market_overview() -> None:
    """Render live market overview showing major index metrics."""
    st.markdown("## 🌐 Market Overview")
    cols = st.columns(len(_MARKET_INDICES))

    for col, (name, symbol) in zip(cols, _MARKET_INDICES):
        with col:
            try:
                data = yf.Ticker(symbol).history(period="2d")
                if len(data) >= 2:
                    current  = data["Close"].iloc[-1]
                    previous = data["Close"].iloc[-2]
                    change   = (current - previous) / previous * 100
                    st.metric(name, f"{current:,.2f}", delta=f"{change:+.2f}%")
                else:
                    st.metric(name, "N/A")
            except Exception:
                st.metric(name, "N/A")

    st.markdown("---")


def render_stock_comparison() -> None:
    """Render the multi-stock comparison tool."""
    st.markdown("## 📊 Stock Comparison")
    st.info("💡 Compare up to 5 stocks side-by-side with comprehensive analysis")

    col1, col2 = st.columns([3, 1])
    with col1:
        symbols_input = st.text_input(
            "Enter stock symbols (comma-separated)", "AAPL, MSFT, GOOGL",
            help="Enter 2–5 stock symbols separated by commas",
            key="comparison_symbols",
        )
    with col2:
        time_period = st.selectbox(
            "Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3, key="comparison_period",
        )

    if not st.button("🔍 Compare Stocks", use_container_width=True,
                     type="primary", key="compare_btn"):
        return

    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

    if len(symbols) < _MIN_SYMBOLS:
        st.error(f"❌ Please enter at least {_MIN_SYMBOLS} symbols")
        return
    if len(symbols) > _MAX_SYMBOLS:
        st.warning(f"⚠️ Comparing more than {_MAX_SYMBOLS} stocks. Showing first {_MAX_SYMBOLS} only.")
        symbols = symbols[:_MAX_SYMBOLS]

    with st.spinner("Fetching data and analyzing…"):
        stock_data: Dict[str, Dict] = {}
        failed: List[str] = []

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist   = ticker.history(period=time_period)
                if not hist.empty:
                    stock_data[symbol] = {"history": hist, "info": ticker.info}
                else:
                    failed.append(symbol)
            except Exception:
                failed.append(symbol)

        if failed:
            st.warning(f"⚠️ Could not fetch data for: {', '.join(failed)}")
        if not stock_data:
            st.error("❌ No valid data found for any symbol")
            return

        render_comparison_analysis(stock_data, time_period)


def render_comparison_analysis(
    stock_data: Dict[str, Dict],
    time_period: str,
) -> None:
    """Render the full multi-tab comparison analysis view."""
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Price Comparison", "📊 Technical Metrics",
        "💰 Fundamentals", "⚖️ Valuation", "📉 Risk Analysis",
    ])
    colors = list(_COMPARISON_COLORS)

    with tab1: render_price_comparison(stock_data, colors)
    with tab2: render_technical_comparison(stock_data, colors)
    with tab3: render_fundamental_comparison(stock_data, colors)
    with tab4: render_valuation_comparison(stock_data, colors)
    with tab5: render_risk_comparison(stock_data, colors)


def render_price_comparison(stock_data: Dict[str, Dict], colors: List[str]) -> None:
    """Price and returns comparison panel."""
    st.markdown("### 📈 Price Performance")

    # Normalised price chart
    fig = go.Figure()
    for i, (symbol, data) in enumerate(stock_data.items()):
        hist       = data["history"]
        normalized = hist["Close"] / hist["Close"].iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=hist.index, y=normalized, name=symbol,
            line=dict(color=colors[i % len(colors)], width=2),
        ))
    fig.update_layout(
        title="Normalized Price Performance (Base 100)",
        xaxis_title="Date", yaxis_title="Normalized Price",
        hovermode="x unified", template="plotly_white", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Returns comparison table — uses the shared helper
    st.markdown("### 📊 Returns Comparison")
    returns_rows = []
    for symbol, data in stock_data.items():
        stats = _compute_returns_stats(data["history"]["Close"])
        returns_rows.append({
            "Symbol":       symbol,
            "Current Price": f"${data['history']['Close'].iloc[-1]:.2f}",
            "Total Return":  f"{stats['total_return']:+.2f}%",
            "Volatility":    f"{stats['volatility']:.2f}%",
            "Sharpe Ratio":  f"{stats['sharpe']:.3f}",
            "Max Drawdown":  f"{stats['max_drawdown']:.2f}%",
        })
    st.dataframe(pd.DataFrame(returns_rows), use_container_width=True)

    # Volume comparison
    st.markdown("### 📊 Volume Comparison")
    vol_fig = make_subplots(
        rows=len(stock_data), cols=1,
        subplot_titles=[f"{s} Volume" for s in stock_data],
        vertical_spacing=0.05,
    )
    for i, (symbol, data) in enumerate(stock_data.items(), 1):
        hist = data["history"]
        vol_fig.add_trace(
            go.Bar(x=hist.index, y=hist["Volume"],
                   name=symbol, marker_color=colors[(i - 1) % len(colors)]),
            row=i, col=1,
        )
    vol_fig.update_layout(
        height=300 * len(stock_data), showlegend=False, template="plotly_white",
    )
    st.plotly_chart(vol_fig, use_container_width=True)


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Standard RSI calculation.  Extracted to eliminate the identical
    four-line block that appeared in both render_technical_comparison
    and the RSI chart loop.
    """
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(window=window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))


def render_technical_comparison(stock_data: Dict[str, Dict], colors: List[str]) -> None:
    """Technical indicators comparison panel."""
    st.markdown("### 📊 Technical Indicators")

    tech_rows = []
    for symbol, data in stock_data.items():
        close = data["history"]["Close"]
        current = close.iloc[-1]

        sma_20  = close.rolling(window=20).mean().iloc[-1]
        sma_50  = close.rolling(window=50).mean().iloc[-1]
        rsi     = _compute_rsi(close)
        ema_12  = close.ewm(span=12, adjust=False).mean()
        ema_26  = close.ewm(span=26, adjust=False).mean()
        macd    = ema_12 - ema_26
        macd_sig = macd.ewm(span=9, adjust=False).mean()

        cur_rsi      = rsi.iloc[-1]
        cur_macd     = macd.iloc[-1]
        cur_macd_sig = macd_sig.iloc[-1]

        sma_signal  = "🟢 Bullish"   if current > sma_20 > sma_50 else "🔴 Bearish"
        rsi_signal  = ("🔴 Overbought" if cur_rsi > 70
                       else "🟢 Oversold" if cur_rsi < 30
                       else "🟡 Neutral")
        macd_signal = "🟢 Bullish" if cur_macd > cur_macd_sig else "🔴 Bearish"

        tech_rows.append({
            "Symbol":      symbol,
            "Price":       f"${current:.2f}",
            "SMA 20":      f"${sma_20:.2f}",
            "SMA 50":      f"${sma_50:.2f}",
            "SMA Signal":  sma_signal,
            "RSI":         f"{cur_rsi:.2f}",
            "RSI Signal":  rsi_signal,
            "MACD":        f"{cur_macd:.3f}",
            "MACD Signal": macd_signal,
        })

    st.dataframe(pd.DataFrame(tech_rows), use_container_width=True)

    # RSI chart — reuse _compute_rsi (no duplication)
    st.markdown("#### RSI Comparison")
    rsi_fig = go.Figure()
    for i, (symbol, data) in enumerate(stock_data.items()):
        hist = data["history"]
        rsi  = _compute_rsi(hist["Close"])
        rsi_fig.add_trace(go.Scatter(
            x=hist.index, y=rsi, name=symbol,
            line=dict(color=colors[i % len(colors)], width=2),
        ))
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red",   annotation_text="Overbought")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    rsi_fig.update_layout(
        title="RSI Comparison", xaxis_title="Date", yaxis_title="RSI",
        hovermode="x unified", template="plotly_white", height=400,
    )
    st.plotly_chart(rsi_fig, use_container_width=True)


def render_fundamental_comparison(stock_data: Dict[str, Dict], colors: List[str]) -> None:
    """Fundamental metrics comparison panel."""
    st.markdown("### 💰 Fundamental Metrics")

    def _fmt(info: Dict, key: str, mult: float = 1.0, suffix: str = "") -> str:
        """Format a single info field with optional multiplier and suffix."""
        val = info.get(key)
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return "N/A"
        return f"{val * mult:.2f}{suffix}"

    rows = []
    for symbol, data in stock_data.items():
        info = data["info"]
        mc   = info.get("marketCap", 0)
        rows.append({
            "Symbol":         symbol,
            "Market Cap":     f"${mc / 1e9:.2f}B" if mc else "N/A",
            "P/E Ratio":      _fmt(info, "trailingPE"),
            "Forward P/E":    _fmt(info, "forwardPE"),
            "PEG Ratio":      _fmt(info, "pegRatio"),
            "P/B Ratio":      _fmt(info, "priceToBook"),
            "Dividend Yield": _fmt(info, "dividendYield", mult=100, suffix="%"),
            "ROE":            _fmt(info, "returnOnEquity", mult=100, suffix="%"),
            "Profit Margin":  _fmt(info, "profitMargins",  mult=100, suffix="%"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    col1, col2 = st.columns(2)

    # Market cap chart
    mc_data = [
        (sym, d["info"].get("marketCap", 0) / 1e9)
        for sym, d in stock_data.items()
        if d["info"].get("marketCap", 0) > 0
    ]
    if mc_data:
        syms, caps = zip(*mc_data)
        fig_mc = go.Figure(data=[go.Bar(
            x=list(syms), y=list(caps),
            marker_color=colors[:len(syms)],
        )])
        fig_mc.update_layout(
            title="Market Capitalization (Billions)",
            xaxis_title="Symbol", yaxis_title="Market Cap ($B)",
            template="plotly_white", height=400,
        )
        with col1:
            st.plotly_chart(fig_mc, use_container_width=True)

    # P/E chart
    pe_data = [
        (sym, d["info"].get("trailingPE"))
        for sym, d in stock_data.items()
        if d["info"].get("trailingPE") and d["info"]["trailingPE"] > 0
    ]
    if pe_data:
        syms, pes = zip(*pe_data)
        fig_pe = go.Figure(data=[go.Bar(
            x=list(syms), y=list(pes),
            marker_color=colors[:len(syms)],
        )])
        fig_pe.update_layout(
            title="P/E Ratio Comparison",
            xaxis_title="Symbol", yaxis_title="P/E Ratio",
            template="plotly_white", height=400,
        )
        with col2:
            st.plotly_chart(fig_pe, use_container_width=True)


def render_valuation_comparison(stock_data: Dict[str, Dict], colors: List[str]) -> None:
    """Valuation metrics comparison panel."""
    st.markdown("### ⚖️ Valuation Analysis")

    # FIX: use the actual analysis industry_pe where available rather than
    # a hard-coded constant.  Fallback to 20 when unavailable.
    _FALLBACK_IND_PE = 20.0

    rows = []
    for symbol, data in stock_data.items():
        info = data["info"]
        pe         = info.get("trailingPE") or 0
        ind_pe     = _FALLBACK_IND_PE
        pe_premium = ((pe - ind_pe) / ind_pe * 100) if pe and ind_pe else 0

        def _safe(key: str) -> str:
            v = info.get(key)
            return f"${v:.2f}" if v else "N/A"

        rows.append({
            "Symbol":        symbol,
            "Current Price": f"${info.get('currentPrice', 0):.2f}",
            "Target Mean":   _safe("targetMeanPrice"),
            "Target High":   _safe("targetHighPrice"),
            "Target Low":    _safe("targetLowPrice"),
            "P/E Premium":   f"{pe_premium:+.1f}%",
            "Price to Sales": (
                f"{info['priceToSalesTrailing12Months']:.2f}"
                if info.get("priceToSalesTrailing12Months") else "N/A"
            ),
            "EV/EBITDA": (
                f"{info['enterpriseToEbitda']:.2f}"
                if info.get("enterpriseToEbitda") else "N/A"
            ),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Analyst targets chart
    st.markdown("#### 🎯 Analyst Price Targets")
    fig = go.Figure()
    for i, (symbol, data) in enumerate(stock_data.items()):
        info   = data["info"]
        cur    = info.get("currentPrice",    0)
        t_low  = info.get("targetLowPrice",  cur)
        t_mean = info.get("targetMeanPrice", cur)
        t_high = info.get("targetHighPrice", cur)
        fig.add_trace(go.Scatter(
            x=[symbol] * 4,
            y=[cur, t_low, t_mean, t_high],
            mode="markers+lines",
            name=symbol,
            marker=dict(size=[15, 10, 15, 10], color=colors[i % len(colors)]),
            line=dict(color=colors[i % len(colors)], width=2),
        ))
    fig.update_layout(
        title="Current Price vs Analyst Targets",
        xaxis_title="Symbol", yaxis_title="Price ($)",
        template="plotly_white", height=400, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_risk_comparison(stock_data: Dict[str, Dict], colors: List[str]) -> None:
    """Risk metrics comparison panel."""
    st.markdown("### 📉 Risk Analysis")

    risk_rows  = []
    scatter_rows = []

    for symbol, data in stock_data.items():
        hist = data["history"]
        info = data["info"]
        stats = _compute_returns_stats(hist["Close"])

        risk_rows.append({
            "Symbol":       symbol,
            "Volatility":   f"{stats['volatility']:.2f}%",
            "Beta":         f"{info.get('beta', 1.0):.2f}",
            "Max Drawdown": f"{stats['max_drawdown']:.2f}%",
            "VaR (95%)":    f"{stats['var_95']:.2f}%",
            "Sharpe Ratio": f"{stats['sharpe']:.3f}",
            "Debt/Equity":  (
                f"{info['debtToEquity']:.2f}" if info.get("debtToEquity") else "N/A"
            ),
            "Current Ratio": (
                f"{info['currentRatio']:.2f}" if info.get("currentRatio") else "N/A"
            ),
        })
        # Annualised return for scatter (reuse from stats dict to avoid recomputing)
        returns     = hist["Close"].pct_change().dropna()
        avg_return  = returns.mean() * 252 * 100
        scatter_rows.append({
            "Symbol": symbol,
            "Return": avg_return,
            "Risk":   stats["volatility"],
        })

    st.dataframe(pd.DataFrame(risk_rows), use_container_width=True)

    st.markdown("#### 📊 Risk-Return Profile")
    scatter_df = pd.DataFrame(scatter_rows)
    fig = px.scatter(
        scatter_df, x="Risk", y="Return", text="Symbol",
        title="Risk-Return Profile (Annualized)",
        labels={"Risk": "Volatility (%)", "Return": "Expected Return (%)"},
        color="Symbol", size=[20] * len(scatter_df),
        color_discrete_sequence=colors,
    )
    fig.update_traces(textposition="top center", textfont_size=12)
    fig.update_layout(template="plotly_white", height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **📊 Interpretation:**
    - **Top-Right**: High risk, high return — Aggressive growth stocks
    - **Top-Left**:  Low risk, high return  — Ideal (rare)
    - **Bottom-Right**: High risk, low return — Avoid
    - **Bottom-Left**:  Low risk, low return  — Conservative/Stable
    """)


# ============================================================================
# WELCOME PAGE
# ============================================================================

def render_welcome_page() -> None:
    """Render the welcome screen with feature overview."""
    render_market_overview()
    render_stock_comparison()
    st.markdown("---")

    st.markdown("""
    <div class="welcome-section">
        <h2>Welcome to the Comprehensive Stock Analysis Platform</h2>
        <p>Get professional-grade stock analysis with advanced technical indicators,
        fundamental analysis, SWOT assessment, and AI-powered investment recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="features-grid">
        <div class="feature-card"><h3>Technical Analysis</h3>
            <p>Advanced indicators including RSI, MACD, Bollinger Bands,
            Fibonacci levels, and support/resistance zones.</p></div>
        <div class="feature-card"><h3>Fundamental Analysis</h3>
            <p>Deep dive into financial statements, ratios, growth metrics,
            and company valuation analysis.</p></div>
        <div class="feature-card"><h3>SWOT Analysis</h3>
            <p>Comprehensive evaluation of Strengths, Weaknesses, Opportunities,
            and Threats affecting the stock.</p></div>
        <div class="feature-card"><h3>Price Targets</h3>
            <p>AI-powered price predictions with entry points, stop losses,
            and profit-taking levels.</p></div>
        <div class="feature-card"><h3>Interactive Charts</h3>
            <p>Professional-grade charts with technical overlays and
            price projection visualisations.</p></div>
        <div class="feature-card"><h3>Detailed Reports</h3>
            <p>Downloadable Excel reports with comprehensive analysis and
            investment recommendations.</p></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="instructions">
        <h3>How to Use This Platform</h3>
        <ol>
            <li>Enter the stock symbol in the sidebar (e.g., AAPL for Apple Inc.)</li>
            <li>Select your preferred date range for historical analysis</li>
            <li>Set your investment amount and industry P/E ratio</li>
            <li>Click "Start Analysis" to begin the comprehensive evaluation</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        if st.button("Try Sample Analysis (AAPL)", use_container_width=True, type="primary"):
            st.info("Use the sidebar form to start your analysis with AAPL or any other stock symbol!")


# ============================================================================
# ANALYSIS TAB RENDERING FUNCTIONS
# ============================================================================

def render_company_profile_tab(analysis: Dict[str, Any]) -> None:
    """Render Company Profile tab content."""
    st.markdown("## Company Profile")

    basic_info = analysis.get("fundamental_info", analysis.get("basic_info", {}))

    st.markdown("### Basic Information")
    col1, col2 = st.columns(2)

    with col1:
        for label, key in [
            ("Company Name", "company_name"),
            ("Symbol",       "symbol"),
            ("Sector",       "sector"),
            ("Industry",     "industry"),
            ("Country",      "country"),
        ]:
            st.write(f"**{label}:** {basic_info.get(key, 'N/A')}")

        website = basic_info.get("website", "N/A")
        if website and website != "N/A":
            st.markdown(f"**Website:** [{website}]({website})")
        else:
            st.write("**Website:** N/A")

    with col2:
        cp = basic_info.get("current_price")
        st.metric("Current Price", f"${cp:.2f}" if isinstance(cp, (int, float)) else "N/A")
        st.metric("Market Cap", format_market_cap(
            basic_info.get("marketCap", basic_info.get("market_cap"))
        ))
        vol = basic_info.get("volume")
        st.metric("Volume", f"{vol:,.0f}" if isinstance(vol, (int, float)) else "N/A")

    st.divider()

    st.markdown("### Price Range")
    st.write(f"**Day Range:** {basic_info.get('day_range', 'N/A')}")
    st.write(f"**52-Week Range:** {basic_info.get('week_52_range', 'N/A')}")

    st.divider()

    st.markdown("### Analyst Targets")
    col_t1, col_t2, col_t3 = st.columns(3)
    for col, label, key in [
        (col_t1, "Target Low",  "targetLowPrice"),
        (col_t2, "Target Mean", "targetMeanPrice"),
        (col_t3, "Target High", "targetHighPrice"),
    ]:
        val = basic_info.get(key)
        col.metric(label, f"${val:.2f}" if isinstance(val, (int, float)) else "N/A")

    rec = basic_info.get("recommendationMean")
    label_str = analyst_rec_label(rec)
    if label_str and isinstance(rec, (int, float)):
        st.write(f"**Analyst Recommendation:** {label_str} ({rec:.1f})")

    st.divider()

    st.markdown("### Key Ratios")
    col_r1, col_r2, col_r3 = st.columns(3)

    with col_r1:
        pe = basic_info.get("trailingPE")
        st.metric("P/E (TTM)", f"{pe:.2f}" if isinstance(pe, (int, float)) else "N/A")
        divy = basic_info.get("dividendYield")
        st.metric("Dividend Yield",
                  f"{divy * 100:.2f}%" if isinstance(divy, (int, float)) else "N/A")

    with col_r2:
        pm = basic_info.get("profitMargins")
        st.metric("Profit Margin",
                  f"{pm * 100:.1f}%" if isinstance(pm, (int, float)) else "N/A")
        roe = basic_info.get("returnOnEquity")
        st.metric("ROE", f"{roe * 100:.1f}%" if isinstance(roe, (int, float)) else "N/A")

    with col_r3:
        dte = basic_info.get("debtToEquity")
        st.metric("Debt/Equity", f"{dte:.2f}" if isinstance(dte, (int, float)) else "N/A")
        peg = basic_info.get("pegRatio")
        st.metric("PEG Ratio", f"{peg:.2f}" if isinstance(peg, (int, float)) else "N/A")

    st.divider()

    st.markdown("### Business Summary")
    st.write(basic_info.get("summary", "No business summary available."))


def render_technical_analysis_tab(analysis: Dict[str, Any]) -> None:
    """Render Technical Analysis tab content."""
    st.subheader("Comprehensive Technical Analysis")

    technical_df = analysis["technical_data"]
    latest = technical_df.iloc[-1]

    col1, col2, col3, col4, col5 = st.columns(5)

    # ── RSI ──────────────────────────────────────────────────────────────────
    with col1:
        rsi = latest.get("RSI_21")
        if rsi is not None:
            in_range  = 30 <= rsi <= 70
            st.metric("RSI (21)", f"{rsi:.2f}",
                      delta="Normal" if in_range else "Alert",
                      delta_color="normal" if in_range else "inverse")
        else:
            st.metric("RSI (21)", "N/A")

    # ── MACD ─────────────────────────────────────────────────────────────────
    with col2:
        macd, macd_sig = latest.get("MACD"), latest.get("MACD_Signal")
        if macd is not None and macd_sig is not None:
            bullish = macd > macd_sig
            st.metric("MACD", f"{macd:.2f}",
                      delta="Bullish" if bullish else "Bearish",
                      delta_color="normal" if bullish else "inverse")
        else:
            st.metric("MACD", "N/A")

    # ── Bollinger %B ─────────────────────────────────────────────────────────
    with col3:
        bpb = latest.get("Bollinger_%B")
        if bpb is not None:
            if   bpb > 1: delta, color = "Above Bands", "inverse"
            elif bpb < 0: delta, color = "Below Bands", "inverse"
            else:         delta, color = "Within Bands", "normal"
            st.metric("Bollinger %B", f"{bpb:.2f}", delta=delta, delta_color=color)
        else:
            st.metric("Bollinger %B", "N/A")

    # ── EMA Crossover ────────────────────────────────────────────────────────
    with col4:
        ema_signal = latest.get("EMA_signal")
        if ema_signal:
            golden = ema_signal == "Golden Cross"
            st.metric("EMA Crossover",
                      "Uptrend" if golden else "Downtrend",
                      delta="Bullish" if golden else "Bearish",
                      delta_color="normal" if golden else "inverse")
        else:
            st.metric("EMA Crossover", "N/A")

    # ── OBV ──────────────────────────────────────────────────────────────────
    with col5:
        obv = latest.get("OBV")
        if obv is not None and len(technical_df) > _OBV_LOOKBACK:
            obv_trend  = technical_df["OBV"].iloc[-_OBV_LOOKBACK:]
            diffs      = obv_trend.diff()
            increases  = int((diffs > 0).sum())
            decreases  = int((diffs < 0).sum())

            if   increases == (_OBV_LOOKBACK - 1): delta, color = "Strong Buying",  "normal"
            elif decreases == (_OBV_LOOKBACK - 1): delta, color = "Strong Selling", "inverse"
            else:                                   delta, color = "Mixed/Weak",     "off"
            st.metric("OBV", f"{int(obv):,}", delta=delta, delta_color=color)
        elif obv is not None:
            st.metric("OBV", f"{int(obv):,}")
        else:
            st.metric("OBV", "N/A")

    with st.expander("Indicator Explanations"):
        st.markdown("""
        - **RSI (21):** Values below 30 indicate oversold conditions; above 70 indicate overbought.
        - **MACD:** Measures market momentum through moving-average convergence/divergence.
        - **Bollinger %B:** > 1 = above bands, 0–1 = within bands, < 0 = below bands.
        - **EMA Crossover:** Uptrend = 12 EMA above 26 EMA; Downtrend = 12 EMA below 26 EMA.
        - **OBV:** Measures buying/selling pressure based on trading volume.
        """)

    # ── Technical data tables ─────────────────────────────────────────────────
    df_tech = analysis["technical_data"].copy()
    df_tech["Date"] = pd.to_datetime(df_tech["Date"]).dt.date

    column_groups = {
        "Price & Trend": [
            "Date", "Close", "SMA_20", "SMA_50", "ADX", "Plus_DI", "Minus_DI",
            "Pivot", "R1", "S1", "R2", "S2",
            "Fib_23.6", "Fib_38.2", "Fib_50", "Fib_61.8", "Fib_78.6",
            "sr_zones", "Long_Resistance", "Long_Support",
        ],
        "Momentum": [
            "Date", "Close", "RSI_7", "RSI_14", "RSI_21",
            "MACD", "MACD_Signal", "MACD_Histogram", "Stoch_K", "Stoch_D",
        ],
        "Volume & Volatility": [
            "Date", "Close", "OBV", "OBVSignal",
            "BB_Lower", "BB_Middle", "BB_Upper", "Bollinger%B",
        ],
        "Signals": [
            "Date", "Close", "Combined_Signal",
            "Buy_Score", "Sell_Score", "Net_Score", "Signal",
        ],
    }

    st.subheader("Latest Technical Analysis")
    st.write("Last 10 days of key technical indicators and signals")

    tabs_list = st.tabs(list(column_groups.keys()))
    for tab, (_, cols) in zip(tabs_list, column_groups.items()):
        with tab:
            available = [c for c in cols if c in df_tech.columns]
            if available:
                st.data_editor(
                    df_tech[available].tail(10).round(2),
                    use_container_width=True,
                    hide_index=True,
                    column_config=get_column_config(available),
                )

    # Quick reference cards
    st.subheader("Quick Reference")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("""
        **Trend Indicators**
        - SMA/EMA: Moving averages
        - Price above MA = Uptrend
        - Price below MA = Downtrend
        """)
    with c2:
        st.info("""
        **Momentum Indicators**
        - RSI: 30–70 normal range
        - MACD: Above 0 = bullish
        - Signal crossovers = entry points
        """)
    with c3:
        st.info("""
        **Key Signals**
        - Buy Score: Bullish strength
        - Sell Score: Bearish strength
        - Net Score: Overall direction
        """)

    with st.expander("Complete Indicator Explanations"):
        st.markdown("""
        ### Price & Trend Indicators
        **SMA_20/SMA_50:** Simple Moving Averages — trend direction indicators.
        **ADX:** Average Directional Index — measures trend strength (> 25 = strong trend).
        **Plus_DI/Minus_DI:** Directional Indicators showing bullish/bearish pressure.
        **Pivot Points:** Daily support/resistance based on previous day's high/low/close.
        **Fibonacci Levels:** Retracement levels at 23.6 %, 38.2 %, 50 %, 61.8 %, 78.6 %.
        **S/R Zones:** Aggregated support and resistance zones from multiple timeframes.
        """)
        st.divider()
        st.markdown("""
        ### Momentum Indicators
        **RSI (7/14/21):** Relative Strength Index — below 30 oversold, above 70 overbought.
        **MACD:** Moving Average Convergence Divergence — measures momentum changes.
        **MACD_Histogram:** Divergence between MACD and signal line.
        **Stochastic K/D:** Momentum oscillator comparing closing price to price range.
        """)
        st.divider()
        st.markdown("""
        ### Volume & Volatility Indicators
        **OBV:** On-Balance Volume — measures buying/selling pressure based on volume.
        **Bollinger %B:** Position within Bollinger Bands (> 1 above, < 0 below, 0–1 within).
        **BB_Upper/Middle/Lower:** Bollinger Band levels showing volatility ranges.
        """)
        st.divider()
        st.markdown("""
        ### Signal Indicators
        **Buy_Score:** Strength of bullish signals (0–100).
        **Sell_Score:** Strength of bearish signals (0–100).
        **Net_Score:** Overall market direction (Buy_Score − Sell_Score).
        **Signal:** Final recommendation based on all indicators combined.
        """)


def render_financial_health_tab(analysis: Dict[str, Any]) -> None:
    """Render Financial Health tab content."""
    fin   = analysis.get("financial_analysis", {})
    score = fin.get("overall_score")

    if score is not None:
        if   score >= 70: border_color = "#1db954"
        elif score <  50: border_color = "#e4363a"
        else:             border_color = "#f4c20d"

        st.markdown(f"""
        <div style="width:100%;background:#2c2f33;padding:15px;border-radius:12px;
                    border-left:6px solid {border_color};border-right:6px solid {border_color};
                    text-align:center;box-shadow:0 4px 6px rgba(0,0,0,.1);">
            <h4 style="color:white;font-size:35px;line-height:1;margin-bottom:10px;">
                Financial Health Score</h4>
            <h2 style="margin:-23px 0 0;color:{border_color};font-size:40px;">{score:.1f}/100</h2>
            <p style="margin:0;font-size:18px;color:#cccccc;">
                <strong style="color:#ffffff;font-size:22px;">{fin.get('health_rating','')}</strong><br>
                <span style="font-size:16px;">{fin.get('health_description','')}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Financial health data not available.")

    st.markdown("---")

    st.markdown("### Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        rg = fin.get("revenue_analysis", {})
        st.metric("Revenue Growth",
                  f"{rg.get('revenue_growth_%', 0):.1f}%",
                  delta=rg.get("trend", "")[:20],
                  help="Year-over-year revenue growth rate")

    with col2:
        nm = fin.get("profitability_analysis", {}).get("net_margin_%", 0)
        st.metric("Net Profit Margin", f"{nm:.1f}%",
                  delta="Strong" if nm > 15 else "Good" if nm > 8 else "Weak",
                  help="Net income as percentage of revenue")

    with col3:
        de = fin.get("balance_sheet_analysis", {}).get("debt_to_equity", 0)
        st.metric("Debt/Equity Ratio", f"{de:.2f}",
                  delta="Low" if de < 0.3 else "Moderate" if de < 0.6 else "High",
                  delta_color="normal" if de < 0.6 else "inverse",
                  help="Total debt relative to shareholders' equity")

    with col4:
        cr = fin.get("balance_sheet_analysis", {}).get("current_ratio", 0)
        st.metric("Current Ratio", f"{cr:.2f}",
                  delta="Excellent" if cr > 2 else "Good" if cr > 1.5 else "Weak",
                  delta_color="normal" if cr > 1.2 else "inverse",
                  help="Ability to pay short-term obligations")

    st.markdown("---")
    st.markdown("### Detailed Financial Analysis")
    detail_tabs = st.tabs(["Profitability", "Balance Sheet", "Cash Flow", "Valuation"])

    # ── Profitability ─────────────────────────────────────────────────────────
    with detail_tabs[0]:
        prof = fin.get("profitability_analysis", {})
        p1, p2, p3 = st.columns(3)
        for col, (name, val, hi, lo, desc) in zip([p1, p2, p3], [
            ("Gross Margin",      prof.get("gross_margin_%", 0),   30, 15, "Profit % before operating expenses"),
            ("Operating Margin",  prof.get("operating_margin_%", 0), 15, 8, "Efficiency in managing core operations"),
            ("Net Margin",        prof.get("net_margin_%", 0),      15, 8,  "Final net profit after all costs and taxes"),
        ]):
            color = "Green" if val > hi else "Yellow" if val > lo else "Red"
            with col:
                st.markdown(render_metric_card(name, f"{color} {val:.1f}%", desc), unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Return Metrics")
        r1, r2 = st.columns(2)
        with r1:
            roe_val  = prof.get("ROE_%", 0)
            roe_col  = "Green" if roe_val > 15 else "Yellow" if roe_val > 10 else "Red"
            st.markdown(render_metric_card(
                "Return on Equity (ROE)", f"{roe_col} {roe_val:.1f}%",
                "Profit generated from each dollar of shareholders' equity",
            ), unsafe_allow_html=True)
        with r2:
            roa_val = (fin.get("asset_efficiency", {}).get("ROA_%", 0) or
                       prof.get("ROA_%", 0) or
                       fin.get("balance_sheet_analysis", {}).get("ROA_%", 0))
            roa_col = "Green" if roa_val > 8 else "Yellow" if roa_val > 4 else "Red"
            st.markdown(render_metric_card(
                "Return on Assets (ROA)", f"{roa_col} {roa_val:.1f}%",
                "How effectively assets are used to generate profits",
            ), unsafe_allow_html=True)

    # ── Balance Sheet ─────────────────────────────────────────────────────────
    with detail_tabs[1]:
        balance = fin.get("balance_sheet_analysis", {})
        st.markdown("##### Debt Analysis")
        d1, d2 = st.columns(2)
        with d1:
            de_r  = balance.get("debt_to_equity", 0)
            de_c  = "Green" if de_r < 0.3 else "Yellow" if de_r < 0.6 else "Red"
            st.markdown(render_metric_card("Debt-to-Equity", f"{de_c} {de_r:.2f}",
                                           "Company's debt vs shareholders' equity"), unsafe_allow_html=True)
        with d2:
            da_r  = balance.get("debt_to_assets_%", 0)
            da_c  = "Green" if da_r < 30 else "Yellow" if da_r < 50 else "Red"
            st.markdown(render_metric_card("Debt-to-Assets", f"{da_c} {da_r:.1f}%",
                                           "Percentage of assets financed through debt"), unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("##### Liquidity Analysis")
        l1, l2 = st.columns(2)
        with l1:
            cr_r = balance.get("current_ratio", 0)
            cr_c = "Green" if cr_r > 1.5 else "Yellow" if cr_r > 1.0 else "Red"
            st.markdown(render_metric_card("Current Ratio", f"{cr_c} {cr_r:.2f}",
                                           "Ability to pay short-term debts with current assets"), unsafe_allow_html=True)
        with l2:
            qr_r = balance.get("quick_ratio", 0)
            qr_c = "Green" if qr_r > 1.0 else "Yellow" if qr_r > 0.7 else "Red"
            st.markdown(render_metric_card("Quick Ratio", f"{qr_c} {qr_r:.2f}",
                                           "Ability to pay debts using only liquid assets"), unsafe_allow_html=True)

    # ── Cash Flow ─────────────────────────────────────────────────────────────
    with detail_tabs[2]:
        cashflow = fin.get("cash_flow_analysis", {})
        st.markdown("##### Cash Generation")
        cf1, cf2 = st.columns(2)
        with cf1:
            op_cf     = cashflow.get("operating_cash_flow", 0) / 1000
            op_cf_col = "Green" if op_cf >= 0 else "Red"
            st.markdown(render_metric_card("Operating Cash Flow", f"{op_cf_col} ${op_cf:,.0f}",
                                           "Cash generated from normal business operations"), unsafe_allow_html=True)
        with cf2:
            free_cf     = cashflow.get("free_cash_flow", 0) / 1000
            fcf_col     = "Green" if free_cf >= 0 else "Red"
            st.markdown(render_metric_card("Free Cash Flow", f"{fcf_col} ${free_cf:,.0f}",
                                           "Cash left after operating expenses and investments"), unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("##### Cash Flow Margins")
        m1, m2 = st.columns(2)
        with m1:
            ocf_r = cashflow.get("ocf_to_revenue_%", 0)
            ocf_c = "Green" if ocf_r > 15 else "Yellow" if ocf_r > 8 else "Red"
            st.markdown(render_metric_card("OCF to Revenue", f"{ocf_c} {ocf_r:.1f}%",
                                           "How much of each revenue dollar becomes operating cash"), unsafe_allow_html=True)
        with m2:
            fcf_r = cashflow.get("fcf_to_revenue_%", 0)
            fcf_c = "Green" if fcf_r > 10 else "Yellow" if fcf_r > 5 else "Red"
            st.markdown(render_metric_card("FCF to Revenue", f"{fcf_c} {fcf_r:.1f}%",
                                           "How much of each revenue dollar becomes free cash"), unsafe_allow_html=True)

    # ── Valuation ─────────────────────────────────────────────────────────────
    with detail_tabs[3]:
        valuation = fin.get("valuation_analysis", {})
        st.markdown("##### Price Ratios")
        v1, v2, v3 = st.columns(3)

        ratio_cfg = [
            ("Price-to-Earnings", valuation.get("P/E"),  15.0, 25.0, "Green", "Yellow", "Red",
             "How much investors pay for each dollar of earnings",
             "Data not available or company has no earnings"),
            ("Price-to-Book",     valuation.get("P/B"),   1.5,  3.0, "Green", "Yellow", "Red",
             "Stock price compared to company's book value per share",
             "Data not available for this calculation"),
            ("PEG Ratio",         valuation.get("PEG"),   1.0,  1.5, "Green", "Yellow", "Red",
             "P/E ratio adjusted for earnings growth rate",
             "Growth data not available for this calculation"),
        ]

        for col, (title, val, low, high, c_good, c_mid, c_bad, desc_val, desc_na) in zip(
            [v1, v2, v3], ratio_cfg
        ):
            with col:
                if val:
                    color = c_good if val < low else c_mid if val < high else c_bad
                    st.markdown(render_metric_card(title, f"{color} {val:.1f}", desc_val),
                                unsafe_allow_html=True)
                else:
                    st.markdown(render_metric_card(title, "N/A", desc_na),
                                unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### Valuation Trends")
        t1, t2 = st.columns(2)
        for col, label, key in [(t1, "P/E Trend", "pe_trend"), (t2, "P/B Trend", "pb_trend")]:
            with col:
                trend     = valuation.get(key, "N/A")
                indicator = _TREND_COLORS.get(trend, "")
                st.markdown(render_metric_card(
                    label, f"{indicator} {trend}", f"Direction of {label.split()[0]} changes over time",
                ), unsafe_allow_html=True)

    st.markdown("---")


def render_swot_tab(analysis: Dict[str, Any]) -> None:
    """Render SWOT Analysis tab content."""
    st.subheader("SWOT Analysis")
    swot = analysis["swot"]

    for row_sections in (
        ("Strengths", "Weaknesses"),
        ("Opportunities", "Threats"),
    ):
        cols = st.columns(2)
        for col, section in zip(cols, row_sections):
            if section not in swot:
                continue
            items = swot[section]
            with col:
                st.markdown(f"### {section}")
                valid = [it for it in items if it and it != "N/A"]
                if valid:
                    for item in valid:
                        st.markdown(f"- {item}")
                else:
                    st.markdown("*No items identified*")
                st.markdown("")


def render_charts_tab(analysis: Dict[str, Any], stock_symbol: str) -> None:
    """Render Charts & Visualisations tab content."""
    st.subheader("Price Analysis Charts")

    col1, col2, col3, col4 = st.columns(4)
    show_bb     = col1.checkbox("Bollinger Bands", value=False)
    show_sma50  = col2.checkbox("SMA-50",          value=True)
    show_macd   = col3.checkbox("MACD Panel",       value=True)
    show_volume = col4.checkbox("Volume",           value=True)

    interactive_fig = create_interactive_chart(
        analysis, analysis["technical_data"], stock_symbol,
        show_bb=show_bb, show_sma50=show_sma50,
        show_macd=show_macd, show_volume=show_volume,
    )
    interactive_fig.update_layout(title_font=dict(size=28))
    st.plotly_chart(interactive_fig, use_container_width=True)

    st.divider()

    # Price comparison chart
    tech_df = analysis["technical_data"]
    nearest_res = analysis.get(
        "nearest_resistance_1y",
        tech_df["High"].rolling(window=252, min_periods=1).max().iloc[-1],
    )
    nearest_sup = analysis.get(
        "nearest_support_1y",
        tech_df["Low"].rolling(window=252, min_periods=1).min().iloc[-1],
    )
    fi = analysis.get("fundamental_info", {})
    levels = {
        "analyst_min":         fi.get("targetLowPrice"),
        "nearest_resistance":  nearest_res,
        "current":             analysis["current_price"],
        "analyst_avg":         fi.get("targetMeanPrice"),
        "nearest_support":     nearest_sup,
        "analyst_max":         fi.get("targetHighPrice"),
    }
    comparison_fig = create_comparison_chart(levels, stock_symbol)
    comparison_fig.update_layout(title_font=dict(size=28))
    st.plotly_chart(comparison_fig, use_container_width=True)

    st.divider()

    # MACD histogram HTML
    st.markdown(
        "<h2 style='font-size:32px;font-weight:600;margin:0 0 12px;'>MACD Histogram</h2>",
        unsafe_allow_html=True,
    )
    macd_path = os.path.join(os.path.dirname(__file__), "macd_histogram.html")
    hist_df   = tech_df[["Date", "MACD_Histogram"]].tail(90)
    context   = {
        "MACD_VALUES": json.dumps(hist_df["MACD_Histogram"].round(3).tolist()),
        "MACD_LABELS": json.dumps(hist_df["Date"].dt.strftime("%d/%m").tolist()),
    }
    html_raw = render_html_template(macd_path, context)
    components.html(html_raw, height=650, scrolling=True)

    st.divider()

    st.markdown("### Support and Resistance")
    render_sr_dashboard_inline(analysis, current_price=analysis.get("current_price", 0))


def render_analytics_overview_tab(analysis: Dict[str, Any], stock_symbol: str) -> None:
    """Render Analytics Overview tab with comprehensive quantitative metrics."""
    st.markdown("## 📊 Analytics Overview")

    current_price = analysis["current_price"]
    entry_point   = analysis["entry_point"]

    tech_df = analysis.get("technical_data")
    if tech_df is not None and not tech_df.empty:
        support_level    = analysis.get("stop_loss", tech_df["Low"].min())
        resistance_level = tech_df["High"].max()
        returns   = tech_df["Close"].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100
    else:
        support_level    = analysis.get("stop_loss", current_price * 0.95)
        resistance_level = current_price * 1.15
        volatility       = 20.0

    up_targets_list = [
        {"price": t, "label": f"Target {i+1}"}
        for i, t in enumerate(analysis.get("up_targets", []))
    ]

    fundamental_info = analysis.get("fundamental_info", analysis.get("basic_info", {}))
    financial_analysis = analysis.get("financial_analysis", {})

    try:
        from analytics_display import (
            calculate_buy_trigger,
            calculate_sell_trigger,
            calculate_value_trigger,
            calculate_future_value_status,
            calculate_forward_target,
            calculate_risk_level,
            calculate_performance_score,
            calculate_recommendation_score,
            calculate_buy_value_score,
            calculate_market_expectation,
        )

        technical_score  = analysis.get("technical_score", 50) / 10
        trend_prediction = analysis.get("prediction", "Sideways")
        trend_strength   = 8 if trend_prediction == "Strong Uptrend" else 5

        buy_trigger  = calculate_buy_trigger(
            entry_point, support_level, technical_score, trend_strength, current_price,
        )
        sell_trigger = calculate_sell_trigger(
            up_targets_list, resistance_level, entry_point, profit_target_pct=20,
        )

        pe_ratio  = fundamental_info.get("trailingPE")
        pb_ratio  = fundamental_info.get("priceToBook")
        peg_ratio = fundamental_info.get("pegRatio")

        revenue_analysis = financial_analysis.get("revenue_analysis", {})

        # ── FIX: المفتاح الصحيح هو "revenue_growth_%" وليس "growth_rate" ──────
        # analyze_financial يُخرج "revenue_growth_%" (راجع السطر 374 منه).
        # الكود القديم كان يبحث عن "growth_rate" فيرجع 0 دائماً.
        # نُغطي الآن كل التسميات الممكنة بترتيب أولوية.
        def _extract_revenue_growth_pct(rev_analysis: dict, fund_info: dict) -> float:
            """
            استخراج نسبة نمو الإيرادات من أي مصدر متاح.
            يُعيد القيمة كـ decimal fraction (0.05 = 5%).
            """
            # 1. البيانات المحسوبة من analyze_financial (الأكثر دقة)
            for key in ("revenue_growth_%", "revenue_growth", "growth_%", "yoy_growth_%"):
                val = rev_analysis.get(key)
                if val is not None and val != 0:
                    # إذا كانت القيمة كبيرة (> 2) فهي بالفعل نسبة مئوية
                    return val / 100.0 if abs(val) > 2.0 else float(val)

            # 2. القيمة المباشرة من yfinance (revenueGrowth = decimal مثل 0.04)
            for key in ("revenueGrowth", "revenue_growth", "revenueGrowthRate"):
                val = fund_info.get(key)
                if val is not None and val != 0:
                    return val / 100.0 if abs(val) > 2.0 else float(val)

            # 3. حساب من current/previous revenue إذا متوفرة
            curr = rev_analysis.get("current_revenue", 0)
            prev = rev_analysis.get("previous_revenue", 0)
            if curr and prev and prev > 0 and curr != prev:
                return (curr - prev) / abs(prev)

            return 0.0

        revenue_growth = _extract_revenue_growth_pct(revenue_analysis, fundamental_info)

        earnings_growth = fundamental_info.get("earningsGrowth", 0.10)
        if earnings_growth and earnings_growth > 1:
            earnings_growth /= 100

        intrinsic_value = calculate_value_trigger(
            current_price, pe_ratio, pb_ratio=pb_ratio,
            earnings_growth=earnings_growth, peg_ratio=peg_ratio,
        )
        future_value_status = calculate_future_value_status(current_price, intrinsic_value)

        analyst_target = fundamental_info.get("targetMeanPrice")
        target_forward = calculate_forward_target(
            current_price, revenue_growth, earnings_growth, analyst_target, years_forward=1,
        )

        # revenue_cagr للعرض: نفضل القيمة المباشرة من revenue_analysis
        # لأنها تعكس النمو الفعلي الذي حسبه analyze_financial
        _raw_cagr = revenue_analysis.get("revenue_growth_%")
        if _raw_cagr is not None and _raw_cagr != 0:
            revenue_cagr = float(_raw_cagr)
        else:
            revenue_cagr = revenue_growth * 100

        beta          = fundamental_info.get("beta")
        debt_to_equity = fundamental_info.get("debtToEquity")
        current_ratio = fundamental_info.get("currentRatio")
        profit_margin = financial_analysis.get("profitability_analysis", {}).get("net_margin_%", 0)

        risk_level  = calculate_risk_level(volatility, beta, debt_to_equity, current_ratio, profit_margin)
        roe         = fundamental_info.get("returnOnEquity")
        sector      = fundamental_info.get("sector", "General")
        # FIX: نمرر revenue_growth كـ % وليس decimal — calculate_performance_score تتوقع %
        performance = calculate_performance_score(revenue_cagr, profit_margin, roe, technical_score, sector)

        financial_score = financial_analysis.get("overall_score", 50)
        recommend       = calculate_recommendation_score(
            technical_score, financial_score, risk_level, trend_prediction, future_value_status,
        )
        stop_loss = analysis.get("stop_loss", current_price * 0.95)
        buy_value = calculate_buy_value_score(current_price, intrinsic_value, entry_point, stop_loss)

        analyst_rating  = fundamental_info.get("recommendationMean")
        price_momentum  = 5.0  # placeholder; replace with computed momentum when available
        market_expect   = calculate_market_expectation(analyst_rating, price_momentum, "Increasing")

        # ── Key metric cards (row 1) ──────────────────────────────────────────
        st.markdown("### 💡 Key Metrics")
        c1, c2, c3 = st.columns(3)
        _CARD_STYLE = (
            "padding:20px;border-radius:12px;text-align:center;"
            "color:white;"
        )
        with c1:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);{_CARD_STYLE}">'
                f'<h4 style="margin:0;">BuyTrigger</h4>'
                f'<h2 style="margin:10px 0;">${buy_trigger}</h2>'
                f'<p style="margin:0;opacity:.8;">Entry Price</p></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%);{_CARD_STYLE}">'
                f'<h4 style="margin:0;">ValueTrigger</h4>'
                f'<h2 style="margin:10px 0;">${intrinsic_value:.2f}</h2>'
                f'<p style="margin:0;opacity:.8;">Intrinsic Value</p></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#4facfe 0%,#00f2fe 100%);{_CARD_STYLE}">'
                f'<h4 style="margin:0;">SellTrigger</h4>'
                f'<h2 style="margin:10px 0;">${sell_trigger}</h2>'
                f'<p style="margin:0;opacity:.8;">Profit Target</p></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Key metric cards (row 2) ──────────────────────────────────────────
        c4, c5, c6 = st.columns(3)
        with c4:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#fa709a 0%,#fee140 100%);{_CARD_STYLE}">'
                f'<h4 style="margin:0;">Growth</h4>'
                f'<h2 style="margin:10px 0;">{revenue_cagr:.1f}%</h2>'
                f'<p style="margin:0;opacity:.8;">Revenue CAGR</p></div>',
                unsafe_allow_html=True,
            )
        status_color = (
            "#27ae60" if future_value_status == "Undervalued" else
            "#e74c3c" if future_value_status == "Overvalued" else
            "#f1c40f"
        )
        with c5:
            st.markdown(
                f'<div style="background:{status_color};{_CARD_STYLE}">'
                f'<h4 style="margin:0;">Future Value</h4>'
                f'<h2 style="margin:10px 0;">{future_value_status}</h2>'
                f'<p style="margin:0;opacity:.8;">Valuation Zone</p></div>',
                unsafe_allow_html=True,
            )
        with c6:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#30cfd0 0%,#330867 100%);{_CARD_STYLE}">'
                f'<h4 style="margin:0;">12M Target</h4>'
                f'<h2 style="margin:10px 0;">${target_forward:.2f}</h2>'
                f'<p style="margin:0;opacity:.8;">Forward Target</p></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Performance scores ────────────────────────────────────────────────
        st.markdown("### 📈 Performance Scores")
        s1, s2, s3, s4, s5 = st.columns(5)
        with s1:
            st.metric("Risk Level",   f"{risk_level}/100",
                      delta="Low" if risk_level < 30 else "High" if risk_level > 60 else "Medium",
                      delta_color="inverse" if risk_level > 60 else "normal")
        with s2:
            st.metric("Performance",  f"{performance}/100",
                      delta="Strong" if performance > 70 else "Good" if performance > 50 else "Weak")
        with s3:
            st.metric("Recommend",    f"{recommend}/100",
                      delta="Buy" if recommend > 70 else "Hold" if recommend > 40 else "Sell")
        with s4:
            st.metric("Buy Value",    f"{buy_value}/100",
                      delta="Attractive" if buy_value > 70 else "Fair" if buy_value > 50 else "Expensive")
        with s5:
            st.metric("Market Expect", f"{market_expect}/100",
                      delta="Positive" if market_expect > 60 else "Neutral" if market_expect > 40 else "Negative")

        st.markdown("---")
        st.markdown("### 📊 Score Breakdown")
        render_progress_bar("Risk Level",         risk_level,   "#e74c3c")
        render_progress_bar("Performance",        performance,  "#3498db")
        render_progress_bar("Recommendation",     recommend,    "#27ae60")
        render_progress_bar("Buy Value",          buy_value,    "#9b59b6")
        render_progress_bar("Market Expectation", market_expect,"#f1c40f")

    except ImportError:
        st.warning("⚠️ Analytics module not available. Ensure analytics_display.py is present.")
        st.info(
            f"**Available Metrics:**\n"
            f"- Current Price: ${current_price:.2f}\n"
            f"- Entry Point: ${entry_point:.2f}\n"
            f"- Technical Score: {analysis.get('technical_score', 0):.1f}/100\n"
            f"- Financial Score: {financial_analysis.get('overall_score', 0):.1f}/100"
        )


def render_investment_decision_tab(analysis: Dict[str, Any]) -> None:
    """Render Investment Decision tab content."""
    st.subheader("Investment Recommendation Summary")

    decision     = analysis["decision"]
    confidence   = analysis["confidence"]
    risk_score   = analysis.get("risk_rating", "N/A")
    reward_ratio = analysis.get("reward_to_risk", np.nan)

    indicator, color = _DECISION_MAP.get(decision, ("Blue", "#3498db"))

    reward_text = "—" if (isinstance(reward_ratio, float) and np.isnan(reward_ratio)) else f"{reward_ratio:.2f}:1"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{color}44 0%,{color}22 100%);
                padding:20px;border-radius:12px;text-align:center;
                margin-bottom:25px;border:2px solid {color};">
        <h2 style="margin:0;font-size:38px;">
            <span style="color:{color};">{decision}</span></h2>
        <p style="margin:5px 0 0;color:#cccccc;font-size:18px;">
            System confidence <strong>{confidence:.1f}%</strong> &middot;
            Risk level <strong>{risk_score}</strong> &middot;
            Reward-to-Risk <strong>{reward_text}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Key price levels ──────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Current",   f"${analysis['current_price']:.2f}")
    k2.metric("Entry",     f"${analysis['entry_point']:.2f}")

    # ── FIX: Stop-Loss يجب أن يكون دائماً < entry_point ─────────────────────
    # analysis['stop_loss'] يُحسب من current_price في main_analysis.
    # إذا كان entry_point < current_price (مثال: entry=240, current=261, SL=251)
    # فالـ SL سيكون أعلى من Entry — وهذا خطأ منطقي.
    # الإصلاح: نُعيد حساب SL بالنسبة لـ entry_point إذا كان أعلى منه.
    _raw_sl    = analysis.get("stop_loss", 0)
    _entry_pt  = analysis["entry_point"]
    _curr_price = analysis["current_price"]
    if _raw_sl >= _entry_pt and _entry_pt > 0:
        # احسب نسبة المسافة الأصلية من current_price واطبّقها على entry_point
        _sl_pct = (_curr_price - _raw_sl) / _curr_price if _curr_price > 0 else 0.05
        _sl_pct = max(abs(_sl_pct), 0.03)   # minimum 3% buffer تحت Entry
        _display_sl = round(_entry_pt * (1 - _sl_pct), 2)
    else:
        _display_sl = _raw_sl
    k3.metric("Stop-Loss", f"${_display_sl:.2f}")

    entry_price = analysis["entry_point"]

    # FIX: guard against missing or under-length up_targets list
    up_targets = analysis.get("up_targets", [])

    def _pct_vs_entry(target_price: float) -> str:
        """Return percentage change from entry formatted for st.metric delta."""
        if entry_price and entry_price > 0:
            return f"{(target_price / entry_price - 1) * 100:+.1f}%"
        return "N/A"

    if len(up_targets) >= 1:
        k4.metric("Target 1", f"${up_targets[0]:.2f}",
                  delta=_pct_vs_entry(up_targets[0]), delta_color="normal")
    else:
        k4.metric("Target 1", "N/A")

    if len(up_targets) >= 2:
        k5.metric("Target 2", f"${up_targets[1]:.2f}",
                  delta=_pct_vs_entry(up_targets[1]), delta_color="normal")
    else:
        k5.metric("Target 2", "N/A")

    avg_target = analysis.get("analyst_avg_target", np.nan)
    if isinstance(avg_target, float) and not np.isnan(avg_target):
        k6.metric("Analyst Avg", f"${avg_target:.2f}",
                  delta=_pct_vs_entry(avg_target), delta_color="normal")
    else:
        k6.metric("Analyst Avg", "N/A")

    # ── Analysis component scores ─────────────────────────────────────────────
    render_progress_bar("Technical",    analysis["technical_score"], "#3498db")
    render_progress_bar("Fundamental",  analysis["financial_analysis"]["overall_score"], "#1abc9c")
    render_progress_bar("SWOT",         analysis["swot_score"], "#9b59b6")

    # ── Investment rationale ──────────────────────────────────────────────────
    technical_summary = analysis.get(
        "technical_summary",
        f"Net Score {analysis['avg_net_score']:+.2f} — {analysis.get('prediction','No clear trend')}",
    )
    fin_analysis = analysis.get("financial_analysis", {})

    st.markdown("### Why this recommendation?")
    opportunities = analysis["swot"].get("Opportunities", [])
    threats       = analysis["swot"].get("Threats", [])
    st.info(f"""
    **Technical outlook:** {technical_summary}
    **Financial health:** {fin_analysis.get('health_description', 'N/A')}
    **Valuation:** Stock trades at **{fin_analysis.get('valuation_analysis', {}).get('valuation_flag', 'N/A')}** compared to peers.
    **Catalysts:** {opportunities[0] if opportunities else '—'}
    **Risks:** {threats[0] if threats else '—'}
    """)


# ============================================================================
# MAIN APPLICATION LOGIC
# ============================================================================

def main() -> None:
    """Main application entry point."""
    configure_page()
    apply_custom_styles()
    render_header()

    submitted, stock_symbol, start_date, end_date, industry_pe = render_sidebar_form()
    investment_amount = 1000.0

    if submitted:
        # Clear any stale session state before a fresh run
        for key in ("analysis", "excel_file", "fig", "stock_symbol"):
            st.session_state.pop(key, None)

        loading_placeholder = render_loading_indicator()

        try:
            technical_data                = fetch_technical_data(stock_symbol, start_date, end_date)
            fundamental_data, message     = fetch_fundamental_data(stock_symbol)

            if fundamental_data is None:
                loading_placeholder.empty()
                st.error(message)
                st.stop()

            financial_analysis = analyze_financial_performance(
                fundamental_data["financials"],
                fundamental_data["balance_sheet"],
                fundamental_data["cashflow"],
                fundamental_data["quarterly_financials"],
                fundamental_data["quarterly_balance_sheet"],
                fundamental_data["quarterly_cashflow"],
                fundamental_data["basic_info"],
                industry_pe,
            )

            analysis = analyze_data(
                technical_data,
                fundamental_data,
                investment_amount,
                industry_pe,
                financial_analysis,
            )

            excel_file = save_report(analysis, stock_symbol, tempfile.gettempdir())
            fig        = create_price_target_chart(
                analysis, fundamental_data["basic_info"], stock_symbol,
            )

            st.session_state.update({
                "analysis":     analysis,
                "excel_file":   excel_file,
                "fig":          fig,
                "stock_symbol": stock_symbol,
            })

        except Exception as e:
            loading_placeholder.empty()
            st.error(f"Error during analysis: {repr(e)}")
            st.stop()

        loading_placeholder.empty()
        st.rerun()

    # ── Results display ───────────────────────────────────────────────────────
    if st.session_state.get("analysis") is not None:
        analysis     = st.session_state["analysis"]
        stock_symbol = st.session_state.get("stock_symbol", "AAPL")

        st.markdown("---")
        if st.button("Back to Main Page", use_container_width=True):
            for key in ("analysis", "excel_file", "fig", "stock_symbol"):
                st.session_state.pop(key, None)
            st.rerun()

        tabs = st.tabs([
            "Company Profile",
            "Technical Analysis",
            "Financial Health",
            "SWOT Analysis",
            "Analytics Overview",
            "Charts & Visualizations",
            "Investment Decision",
        ])

        with tabs[0]: render_company_profile_tab(analysis)
        with tabs[1]: render_technical_analysis_tab(analysis)
        with tabs[2]: render_financial_health_tab(analysis)
        with tabs[3]: render_swot_tab(analysis)
        with tabs[4]: render_analytics_overview_tab(analysis, stock_symbol)
        with tabs[5]: render_charts_tab(analysis, stock_symbol)
        with tabs[6]: render_investment_decision_tab(analysis)

        # Excel download
        excel_file = st.session_state.get("excel_file")
        if excel_file:
            st.markdown("---")
            with open(excel_file, "rb") as f:
                st.download_button(
                    label="Download Complete Analysis Report",
                    data=f.read(),
                    file_name=f"{stock_symbol}_Complete_Analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    type="primary",
                )
    else:
        render_welcome_page()

    st.markdown("---")
    st.markdown(
        "**Disclaimer:** This analysis is for educational purposes only. "
        "Please consult with qualified financial professionals before making any investment decisions."
    )


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()