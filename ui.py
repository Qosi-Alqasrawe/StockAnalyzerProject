import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime
import warnings
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit.components.v1 as components
import json
import pathlib	
from plotly.subplots import make_subplots
from datetime import date   # ضعه أعلى الملف إذا لم يكن موجوداً



# Import custom modules
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

# --------------------- HTML Template Utilities ---------------------
def render_html_template(path: str, context: dict) -> str:
    """Simple placeholder replacement for small HTML snippets."""
    html = pathlib.Path(path).read_text(encoding="utf8")
    for key, value in context.items():
        html = html.replace(f"{{{{{key}}}}}", value)
    return html

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="📊 Comprehensive Stock Analysis Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM STYLES ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');

/* ========== Base Styles ========== */
body, * { font-family: 'Inter', sans-serif; }

/* ========== Main Header ========== */
.main-header {
    text-align: center;
    padding: 5px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 14px;
    margin-bottom: 20px;
}
.main-header h1 { font-size: 2rem; font-weight: 600; margin: 0; }
.main-header p  { font-size: 0.96rem; opacity: 0.85; }

/* ========== Feature Cards ========== */
.features-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.1rem; margin: 1rem 0; }
.feature-card {
    background: transparent;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #303248;
    display: flex; flex-direction: column; justify-content: flex-start; align-items: flex-start;
    transition: border 0.18s, box-shadow 0.25s, transform 0.15s;
    min-width: 0; max-width: 100%;
}
.feature-card:hover {
    border: 1.5px solid #667eea;
    box-shadow: 0 4px 18px 0 #667eea44, 0 2px 4px 0 #33333322;
    background: linear-gradient(125deg, #2326520c 50%, #667eea19 100%);
    transform: translateY(-2px) scale(1.03);
    cursor: pointer;
}
@media (max-width: 900px) { .features-grid { grid-template-columns: 1fr; } }

/* ========== Welcome & Instructions ========== */
.welcome-section {
    background: transparent;
    padding: 0.4rem 0.5rem;
    border-radius: 9px;
    margin: 1.2rem 0;
    color: #B7C1D6;
    font-size: 1.01rem;
}
.instructions {
    background: transparent;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    margin: 0.7rem 0 1rem 0;
    color: #B7C1D6;
    font-size: 0.96rem;
}
.instructions h3 { color: #6d99fa; margin-bottom: 0.6rem; font-size: 0.98rem; }
.instructions ol { color: #b0b3bc; font-size: 0.92rem; margin: 0 0 0 1rem; }

/* ========== Sidebar Styles ========== */
.sidebar-section h3 {
    font-size: 28px !important;          /* ← أكبر */
    font-weight: 600 !important;
    margin-bottom: 0 !important;
}
[data-testid="stSidebar"] .sidebar-section {
    padding: 0.5rem 0.5rem 0.7rem 0.5rem !important;
}

/* صندوق الفورم أطول + حاشية أكبر */
[data-testid="stSidebar"] .stForm {
    padding: 0.8rem 0.8rem 1.2rem 0.8rem !important; /* حاشية موسّعة */
    border-radius: 13px !important;
}

/* Label بحجم خطّ أكبر */
[data-testid="stSidebar"] label {
    font-size: 1.05rem !important;       /* ← كان 0.93rem */
    font-weight: 600 !important;
    display: flex !important; align-items: center !important;
    gap: 4px !important; margin-bottom: 0.07rem !important;
}

/* أيقونات مساعدة */
[data-testid="stSidebar"] svg {
    width: 16px !important; height: 16px !important;
    vertical-align: middle !important; margin-bottom: 0px !important; margin-left: 2px !important;
}

/* خانات الإدخال بحجم خطّ أكبر */
[data-testid="stSidebar"] input {
    font-size: 1.05rem !important;       /* ← كان 0.92rem */
    border-radius: 5px !important;
    padding: 0.38rem 0.65rem !important; /* حاشية داخلية أكبر */
    height: 2.1rem !important;
}
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stTextInput  input,
[data-testid="stSidebar"] .stDateInput  input {
    font-size: 1.05rem !important;       /* موحّد */
}

/* زرّ داخل الـ Sidebar */
[data-testid="stSidebar"] button {
    font-size: 0.98rem !important;       /* أكبر قليلاً */
    padding: 0.55rem 1.2rem !important;
    min-height: 2.2rem !important;
    border-radius: 6px !important;
    margin-top: 0.4rem;
}

/* ========== Button Styles (خارج الـ Sidebar أيضاً) ========== */
.stButton > button {
    border-radius: 7px !important;
    font-weight: 600 !important;
    font-size: 1.0rem !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}

/* داخل عناصر الإدخال المركبة */
.stNumberInput > div > div > input,
.stTextInput  > div > div > input,
.stDateInput  > div > div > input {
    font-size: 1.05rem !important;
    border-radius: 7px !important;
}
.stNumberInput button {
    font-size: 1rem !important;
    padding: 0 7px !important;
    height: 1.8rem !important;
    min-width: 1.5rem !important;
    color: #dedede !important;
    border: 1px solid #373952 !important;
    border-radius: 5px !important;
    box-shadow: none !important;
    margin: 0px !important;
}
.stNumberInput button:active { background: #303248 !important; }
.stNumberInput > div { gap: 4px !important; }

/* ========== Hide Streamlit Default Elements ========== */
#MainMenu, footer, header { visibility: hidden; }

/* ========== Loading Animation ========== */
.loading-wave {
    display: flex; gap: 6px; justify-content: center; align-items: center;
    height: 38px; margin-bottom: 1.1rem;
}
.loading-wave span {
    display: block;
    width: 8px; height: 18px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 4px;
    animation: wave 1.1s infinite ease-in-out;
}
.loading-wave span:nth-child(2) { animation-delay: 0.13s; }
.loading-wave span:nth-child(3) { animation-delay: 0.26s; }
.loading-wave span:nth-child(4) { animation-delay: 0.39s; }
.loading-wave span:nth-child(5) { animation-delay: 0.52s; }

@keyframes wave {
    0%, 40%, 100% { transform: scaleY(1); }
    20% { transform: scaleY(1.65); }
}
</style>
""", unsafe_allow_html=True)
# ==================== UNIFIED CHART FUNCTION ====================
def create_interactive_chart(
        analysis, technical_df, symbol,
        show_bb: bool = True,
        show_sma50: bool = True,
        show_macd: bool = True,
        show_volume: bool = True):
    """
    بناء مخطّط تفاعلي يضمّ السعر، المتوسطات، حجم التداول، MACD،
    إضافة إلى نقاط الدخول، وقف الخسارة، والأهداف.
    """

    # 1) تجهيز آخر 120 شمعة
    df = technical_df.tail(120).copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # 2) إنشاء الشكل بثلاث لوحات
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.15, 0.25],
        vertical_spacing=0.02,
        specs=[[{"type": "candlestick"}],
               [{"type": "bar"}],
               [{"type": "scatter"}]]
    )

    # --- (أ) لوحة السعر و المؤشّرات ---
    fig.add_trace(
        go.Candlestick(
            x=df["Date"], open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="Price"),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=df["SMA_20"],
            line=dict(width=1.2, dash="solid"),
            name="SMA 20"),
        row=1, col=1
    )

    if show_sma50:
        fig.add_trace(
            go.Scatter(
                x=df["Date"], y=df["SMA_50"],
                line=dict(width=1.2, dash="dot"),
                name="SMA 50"),
            row=1, col=1
        )

    # --- Bollinger Bands ---
    bb_visibility = True if show_bb else "legendonly"

    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=df["BB_Upper"],
            line=dict(width=0.8, color="gray"),
            name="BB Upper",
            visible=bb_visibility),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=df["BB_Lower"],
            line=dict(width=0.8, color="gray"),
            name="BB Lower",
            visible=bb_visibility),
        row=1, col=1
    )

    # --- (ب) نقاط الدخول، وقف الخسارة، الأهداف ---
    entry        = analysis["entry_point"]
    stop_loss    = analysis["stop_loss"]
    up_targets   = analysis["up_targets"][:3]
    down_targets = analysis["down_targets"][:2]
    last_date    = df["Date"].iloc[-1]

    # Entry و SL
    fig.add_annotation(
        x=last_date, y=entry, text="Entry",
        showarrow=True, arrowcolor="green",
        arrowhead=3, ax=0, ay=-30,
        row=1, col=1)

    fig.add_annotation(
        x=last_date, y=stop_loss, text="SL",
        showarrow=True, arrowcolor="red",
        arrowhead=3, ax=0, ay=30,
        row=1, col=1)

    # ---------- الأهداف ----------

    # Up-Targets (trace واحد)
    fig.add_trace(
        go.Scatter(
            x=[last_date] * len(up_targets),     # نفس التاريخ لكل هدف
            y=up_targets,
            mode="markers",
            marker=dict(size=10, color="#4ecdc4"),
            name="Up-Target"                     # يظهر مرّة وحدة في الـ Legend
        ),
        row=1, col=1
    )

    # Down-Targets (trace واحد)
    fig.add_trace(
        go.Scatter(
            x=[last_date] * len(down_targets),
            y=down_targets,
            mode="markers",
            marker=dict(size=10, color="#ffa07a"),
            name="Down-Target"
        ),
        row=1, col=1
    )

    # --- (ج) حجم التداول ---
    if show_volume:
        fig.add_trace(
            go.Bar(
                x=df["Date"], y=df["Volume"],
                marker_color="#636efa", name="Volume"),
            row=2, col=1
        )

    # --- (د) لوحة الـ MACD ---
    if show_macd:
        fig.add_trace(
            go.Scatter(
                x=df["Date"], y=df["MACD"],
                line=dict(width=1.1),
                name="MACD"),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df["Date"], y=df["MACD_Signal"],
                line=dict(width=1.1, dash="dash"),
                name="Signal"),
            row=3, col=1
        )
        fig.add_trace(
            go.Bar(
                x=df["Date"], y=df["MACD_Histogram"],
                marker_color=df["MACD_Histogram"].apply(
                    lambda v: "#5fe499" if v > 0 else "#f34d63"),
                name="Histogram"),
            row=3, col=1
        )

    # 3) ضبط المظهر العام
    fig.update_layout(
        title="📈 Price Action & Indicators",
        template="plotly_dark",
        height=700,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(t=60, b=40, l=0, r=10)
    )

    # Range-slider و Range-selector
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.07),
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1 m", step="month",
                     stepmode="backward"),
                dict(count=3, label="3 m", step="month",
                     stepmode="backward"),
                dict(count=6, label="6 m", step="month",
                     stepmode="backward"),
                dict(step="all")
            ]),
        row=3, col=1
    )

    return fig

# =============== SIMPLE COMPARISON CHART ===============
def create_comparison_chart(levels: dict, symbol: str):
    """
    يرسم عمودًا مقارنًا يوضّح السعر الحالي، أقرب مقاومة/دعم سنوية،
    وأهداف المحلّلين (Min / Avg / Max) بالترتيب:
    Analyst Min → 1Y Res. → Current → Analyst Avg → 1Y Supp. → Analyst Max
    """
    import plotly.express as px
    import numpy as np

    # الترتيب المطلوب للمفاتيح
    order = [
        ("analyst_min",        "Analyst Min"),
        ("nearest_resistance", "1Y Res."),
        ("current",            "Current"),
        ("analyst_avg",        "Analyst Avg"),
        ("nearest_support",    "1Y Supp."),
        ("analyst_max",        "Analyst Max"),
    ]

    labels, values = [], []
    for key, label in order:
        val = levels.get(key)
        # أضف العنصر فقط إذا كانت القيمة عددية
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            labels.append(label)
            values.append(val)

    fig = px.bar(
        x=labels,
        y=values,
        text=[f"${v:,.2f}" for v in values],
        color=labels,
        color_discrete_sequence=[
            "#9b59b6",   # Analyst Min
            "#3498db",   # 1Y Resistance
            "#1abc9c",   # Current
            "#f1c40f",   # Analyst Avg
            "#e74c3c",   # 1Y Support
            "#e67e22",   # Analyst Max
        ],
    	title="💰 Price Comparison"          # ← بدل ما كان  f"Price Comparison – {symbol}"
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis_title="Price ($)",
        xaxis_title="Level",
        showlegend=False,
        margin=dict(t=60, l=0, r=0, b=40),
        template="plotly_dark",
        height=450
    )
    return fig


# ==================== SUPPORT & RESISTANCE DASHBOARD ====================
def render_sr_dashboard_inline(analysis, current_price=205.50):
    """Render interactive Support & Resistance dashboard"""
    
    # Extract data
    sr_zones = analysis.get('sr_zones', [])
    pivot_levels = analysis.get('pivot_levels', {})
    fib_levels = analysis.get('fib_levels', {})
    
    # Prepare data for component
    data = {
        "pivot_levels": {
            "long_support": pivot_levels.get('long_support', 0),
            "long_resistance": pivot_levels.get('long_resistance', 0), 
            "short_support": pivot_levels.get('short_support', 0),
            "short_resistance": pivot_levels.get('short_resistance', 0)
        },
        "sr_zones": [[float(zone[0]), float(zone[1])] for zone in sr_zones if len(zone) == 2],
        "fib_levels": {k: float(v) for k, v in fib_levels.items()},
        "current_price": float(current_price)
    }
    
    # HTML Template
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
                color: white; 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                padding: 20px;
            }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .header h1 {{ 
                font-size: 2.5rem; 
                background: linear-gradient(45deg, #4facfe, #00f2fe);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 10px;
            }}
            .current-price {{ 
                background: rgba(255,255,255,0.1); 
                padding: 15px; 
                border-radius: 15px; 
                display: flex; 
                justify-content: space-between; 
                align-items: center;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.2);
            }}
            .price-value {{ font-size: 2rem; font-weight: bold; color: #00f2fe; }}
            .tabs {{ 
                display: flex; 
                justify-content: center; 
                margin: 30px 0; 
                background: rgba(255,255,255,0.1);
                border-radius: 15px;
                padding: 5px;
            }}
            .tab {{ 
                padding: 10px 20px; 
                margin: 0 5px; 
                border: none; 
                background: transparent; 
                color: white; 
                cursor: pointer; 
                border-radius: 10px;
                transition: all 0.3s;
            }}
            .tab.active {{ background: #4facfe; }}
            .grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); 
                gap: 20px; 
            }}
            .card {{ 
                background: rgba(255,255,255,0.1); 
                padding: 20px; 
                border-radius: 15px; 
                border: 1px solid rgba(255,255,255,0.2);
                backdrop-filter: blur(10px);
                transition: transform 0.3s, box-shadow 0.3s;
            }}
            .card:hover {{ 
                transform: translateY(-5px); 
                box-shadow: 0 10px 25px rgba(79, 172, 254, 0.3);
            }}
            .card-header {{ 
                display: flex; 
                justify-content: space-between; 
                align-items: center; 
                margin-bottom: 15px; 
            }}
            .card-title {{ font-size: 1.1rem; font-weight: 600; }}
            .card-value {{ font-size: 1.8rem; font-weight: bold; margin: 10px 0; }}
            .support {{ color: #4ade80; }}
            .resistance {{ color: #f87171; }}
            .zone-active {{ 
                border: 2px solid #fbbf24 !important; 
                box-shadow: 0 0 20px rgba(251, 191, 36, 0.5) !important;
            }}
            .fib-bar {{ 
                width: 100%; 
                height: 6px; 
                background: rgba(255,255,255,0.2); 
                border-radius: 3px; 
                overflow: hidden; 
                margin-top: 10px;
            }}
            .fib-fill {{ 
                height: 100%; 
                background: linear-gradient(90deg, #4ade80, #fbbf24, #f87171); 
                border-radius: 3px;
                transition: width 1s ease;
            }}
            .tab-content {{ display: none; }}
            .tab-content.active {{ display: block; }}
            .badge {{ 
                padding: 4px 8px; 
                border-radius: 8px; 
                font-size: 0.8rem; 
                font-weight: bold;
            }}
            .badge-support {{ background: rgba(74, 222, 128, 0.2); color: #4ade80; }}
            .badge-resistance {{ background: rgba(248, 113, 113, 0.2); color: #f87171; }}
            .badge-active {{ background: #fbbf24; color: black; animation: pulse 2s infinite; }}
            @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.7; }} }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Support & Resistance Analysis</h1>
                <div class="current-price">
                    <span>💹 Current Price</span>
                    <span class="price-value">{data['current_price']:.2f}</span>
                </div>
            </div>
            
            <div class="tabs">
                <button class="tab active" onclick="showTab('overview')">📈 Overview</button>
                <button class="tab" onclick="showTab('zones')">🎯 SR Zones</button>
                <button class="tab" onclick="showTab('fibonacci')">⚡ Fibonacci</button>
            </div>
            
            <div id="overview" class="tab-content active">
                <div class="grid">
    """
    
    # Add pivot level cards
    for name, value in data["pivot_levels"].items():
        if value > 0:
            price_type = "resistance" if "resistance" in name.lower() else "support"
            distance = abs(value - data["current_price"])
            percentage = (distance / data["current_price"]) * 100
            
            html_template += f"""
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">{name.replace('_', ' ').title()}</span>
                            <span class="badge badge-{price_type}">{price_type}</span>
                        </div>
                        <div class="card-value {price_type}">{value:.2f}</div>
                        <div>Distance: {distance:.2f} ({percentage:.1f}%)</div>
                    </div>
            """
    
    html_template += """
                </div>
            </div>
            
            <div id="zones" class="tab-content">
                <div class="grid">
    """
    
    # Add S&R zones
    for i, zone in enumerate(data["sr_zones"]):
        min_val, max_val = zone
        is_active = min_val <= data["current_price"] <= max_val
        active_class = "zone-active" if is_active else ""
        
        html_template += f"""
                    <div class="card {active_class}">
                        <div class="card-header">
                            <span class="card-title">Zone {i+1}</span>
                            {'<span class="badge badge-active">ACTIVE</span>' if is_active else ''}
                        </div>
                        <div class="support">Support: {min_val:.2f}</div>
                        <div class="resistance">Resistance: {max_val:.2f}</div>
                        <div>Range: {(max_val - min_val):.2f}</div>
                    </div>
        """
    
    html_template += """
                </div>
            </div>
            
            <div id="fibonacci" class="tab-content">
                <div class="grid">
    """
    
    # Add Fibonacci levels
    for level, value in data["fib_levels"].items():
        distance = abs(value - data["current_price"])
        percentage = (distance / data["current_price"]) * 100
        
        html_template += f"""
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Fib {level}%</span>
                        </div>
                        <div class="card-value">{value:.2f}</div>
                        <div>Distance: {distance:.2f} ({percentage:.1f}%)</div>
                        <div class="fib-bar">
                            <div class="fib-fill" style="width: {min(percentage*2, 100)}%"></div>
                        </div>
                    </div>
        """
    
    html_template += """
                </div>
            </div>
        </div>
        
        <script>
            function showTab(tabId) {
                const contents = document.querySelectorAll('.tab-content');
                contents.forEach(content => content.classList.remove('active'));
                
                const tabs = document.querySelectorAll('.tab');
                tabs.forEach(tab => tab.classList.remove('active'));
                
                document.getElementById(tabId).classList.add('active');
                event.target.classList.add('active');
            }
        </script>
    </body>
    </html>
    """
    
    components.html(html_template, height=800, scrolling=True)

# ==================== MAIN APP ====================
# Header
st.markdown("""
<div class="main-header">
    <h1>📊 Comprehensive Stock Analysis Platform</h1>
    <p>Advanced Technical & Fundamental Analysis with AI-Powered Insights</p>
</div>
""", unsafe_allow_html=True)

# ───────── Sidebar ─────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-section">
            <h3 style="margin-bottom:4px;">📈 Analysis Settings</h3>
            <hr style="border:1px solid #373952; margin:6px 0 14px;">
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("input_form", clear_on_submit=False):
        # 🔹 الرمز والتواريخ في صفّين متتاليين
        stock_symbol = st.text_input(
            "Stock Symbol", "AAPL",
            help="Enter stock ticker symbol (e.g., AAPL, TSLA, MSFT)"
        )

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "From Date", value=datetime(2024, 1, 1),
                help="Historical data start date"
            )
        with col2:
            end_date = st.date_input(
                "To Date", value=date.today(),
                help="Historical data end date"
            )

        # 🔹 حقل P/E فقط
        industry_pe = st.number_input(
            "Industry P/E Ratio", min_value=1.0, step=1.0, value=30.0,
            help="Average P/E ratio for the sector"
        )

        # زرّ التحليل
        submitted = st.form_submit_button(
            "🔍 Start Analysis", use_container_width=True
        )

# قيمة افتراضيّة للاستثمار بعد حذف الحقل من الواجهة
investment_amount = 1000.0
# ========= Main Analysis Process =========
if submitted:
    # 1) نظّف أي نتائج سابقة في session_state
    for key in ["analysis", "excel_file", "fig", "stock_symbol"]:
        st.session_state.pop(key, None)

    # 2) أظهر مؤشر التحميل
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown(
            """
            <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;margin:40px 0;">
                <div class="loading-wave">
                    <span></span><span></span><span></span><span></span><span></span>
                </div>
                <div style="text-align:center;color:#B7C1D6;margin-bottom:10px;font-size:1.1rem">
                    Loading and analyzing data… Please wait
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    try:
        # 3) جلب البيانات
        technical_data = fetch_technical_data(stock_symbol, start_date, end_date)
        fundamental_data, message = fetch_fundamental_data(stock_symbol)

        if fundamental_data is None:
            loading_placeholder.empty()
            st.error(message)
            st.stop()

        # 4) تحليل الأداء المالي
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

        # 5) التحليل الرئيسي
        analysis = analyze_data(
            technical_data,
            fundamental_data,
            investment_amount,
            industry_pe,
            financial_analysis,
        )

        # 6) إنشاء التقارير
        excel_file = save_report(analysis, stock_symbol, tempfile.gettempdir())
        fig = create_price_target_chart(
            analysis, fundamental_data["basic_info"], stock_symbol
        )

        # 7) حفظ النتائج في session_state
        st.session_state.update(
            {
                "analysis": analysis,
                "excel_file": excel_file,
                "fig": fig,
                "stock_symbol": stock_symbol,
            }
        )

    except Exception as e:
        loading_placeholder.empty()
        st.error(f"❌ Error: {repr(e)}")
        st.stop()

    # 8) أخفِ المؤشر وأعد تشغيل الصفحة لعرض النتائج
    loading_placeholder.empty()
    st.rerun()

# Display Analysis Results
if 'analysis' in st.session_state and st.session_state['analysis'] is not None:
    analysis = st.session_state['analysis']
    stock_symbol = st.session_state.get('stock_symbol', 'AAPL')

    st.markdown("---")
    if st.button("⬅️ Back to Main Page", use_container_width=True):
        for key in ['analysis', 'excel_file', 'fig', 'stock_symbol']:
            st.session_state.pop(key, None)
        st.rerun()


    # Analysis Tabs
    tabs = st.tabs([
        "🏢 Company Profile",
        "📊 Technical Analysis",
        "📈 Financial Health",
        "🧠 SWOT Analysis",
        "📉 Charts & Visualizations",
        "✅ Investment Decision"
    ])

    # ===== Tab 0: Company Profile =====
    with tabs[0]:
        st.markdown("## 🏢 Company Profile")

        # جرّب أولاً fundamental_info وإلّا basic_info
        basic_info = analysis.get("fundamental_info",
                                  analysis.get("basic_info", {}))

        # ---------- 🏷️ Basic Information ----------
        st.markdown("### 🏷️ Basic Information")
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Company Name:** {basic_info.get('company_name', 'N/A')}")
            st.write(f"**Symbol:** {basic_info.get('symbol', 'N/A')}")
            st.write(f"**Sector:** {basic_info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {basic_info.get('industry', 'N/A')}")
            st.write(f"**Country:** {basic_info.get('country', 'N/A')}")

            website = basic_info.get('website', 'N/A')
            if website and website != 'N/A':
                st.markdown(f"**Website:** [{website}]({website})")
            else:
                st.write("**Website:** N/A")

        with col2:
            # Current price
            cp = basic_info.get('current_price')
            cp_txt = f"${cp:.2f}" if isinstance(cp, (int, float)) else "N/A"
            st.metric("Current Price", cp_txt)

            # Market-cap (مبسّط)
            mc = basic_info.get('market_cap')
            if isinstance(mc, (int, float)):
                if mc >= 1e12: mc_txt = f"${mc/1e12:.2f} T"
                elif mc >= 1e9: mc_txt = f"${mc/1e9:.2f} B"
                elif mc >= 1e6: mc_txt = f"${mc/1e6:.2f} M"
                else: mc_txt = f"${mc:,.0f}"
            else:
                mc_txt = "N/A"
            st.metric("Market Cap", mc_txt)

            # Volume
            vol = basic_info.get('volume')
            vol_txt = f"{vol:,.0f}" if isinstance(vol, (int, float)) else "N/A"
            st.metric("Volume", vol_txt)

        st.divider()

        # ---------- 📈 Price Range ----------
        st.markdown("### 📈 Price Range")
        day_range = basic_info.get('day_range', 'N/A')
        week_range = basic_info.get('week_52_range', 'N/A')
        st.write(f"**Day Range:** {day_range}")
        st.write(f"**52-Week Range:** {week_range}")

        st.divider()

        # ---------- 🎯 Analyst Targets ----------
        st.markdown("### 🎯 Analyst Targets")
        col_t1, col_t2, col_t3 = st.columns(3)

        with col_t1:
            low = basic_info.get('targetLowPrice')
            st.metric("Target Low", f"${low:.2f}" if isinstance(low, (int, float)) else "N/A")

        with col_t2:
            mean = basic_info.get('targetMeanPrice')
            st.metric("Target Mean", f"${mean:.2f}" if isinstance(mean, (int, float)) else "N/A")

        with col_t3:
            high = basic_info.get('targetHighPrice')
            st.metric("Target High", f"${high:.2f}" if isinstance(high, (int, float)) else "N/A")

        # توصية المحللين (اختياري)
        rec = basic_info.get('recommendationMean')
        if isinstance(rec, (int, float)):
            txt = ("🟢 Strong Buy" if rec <= 1.5 else
                   "🔵 Buy"        if rec <= 2.5 else
                   "🟡 Hold"       if rec <= 3.5 else
                   "🟠 Sell"       if rec <= 4.5 else
                   "🔴 Strong Sell")
            st.write(f"**Analyst Recommendation:** {txt} ({rec:.1f})")

        st.divider()

        # ---------- 🔢 Key Ratios ----------
        st.markdown("### 🔢 Key Ratios")
        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            pe = basic_info.get('trailingPE')
            st.metric("P/E (TTM)", f"{pe:.2f}" if isinstance(pe, (int, float)) else "N/A")

            divy = basic_info.get('dividendYield')
            st.metric("Dividend Yield", f"{divy*100:.2f}%" if isinstance(divy, (int, float)) else "N/A")

        with col_r2:
            pm = basic_info.get('profitMargins')
            st.metric("Profit Margin", f"{pm*100:.1f}%" if isinstance(pm, (int, float)) else "N/A")

            roe = basic_info.get('returnOnEquity')
            st.metric("ROE", f"{roe*100:.1f}%" if isinstance(roe, (int, float)) else "N/A")

        with col_r3:
            dte = basic_info.get('debtToEquity')
            st.metric("Debt/Equity", f"{dte:.2f}" if isinstance(dte, (int, float)) else "N/A")

            peg = basic_info.get('pegRatio')
            st.metric("PEG Ratio", f"{peg:.2f}" if isinstance(peg, (int, float)) else "N/A")

        st.divider()

        # ---------- 📄 Business Summary ----------
        st.markdown("### 📄 Business Summary")
        summary = basic_info.get('summary', 'No business summary available.')
        st.write(summary)



    # Technical Analysis Tab
    with tabs[1]:
        st.subheader("📊 Comprehensive Technical Analysis")
        
        # Get latest technical data
        technical_df = st.session_state['analysis']['technical_data']
        latest = technical_df.iloc[-1]
        prev = technical_df.iloc[-2] if len(technical_df) > 1 else latest
        
        # Display key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # RSI (21)
        with col1:
            rsi = latest.get('RSI_21', None)
            if rsi is not None:
                if 30 <= rsi <= 70:
                    delta, color = "🟢 Normal", "normal"
                else:
                    delta, color = "🔴 Alert", "inverse"
                st.metric("RSI (21)", f"{rsi:.2f}", delta=delta, delta_color=color)
            else:
                st.metric("RSI (21)", "N/A")
        
        # MACD
        with col2:
            macd = latest.get('MACD', None)
            macd_sig = latest.get('MACD_Signal', None)
            if macd is not None and macd_sig is not None:
                if macd > macd_sig:
                    delta, color = "🟢 Bullish", "normal"
                else:
                    delta, color = "🔴 Bearish", "inverse"
                st.metric("MACD", f"{macd:.2f}", delta=delta, delta_color=color)
            else:
                st.metric("MACD", "N/A")
        
        # Bollinger %B
        with col3:
            bpb = latest.get('Bollinger_%B', None)
            if bpb is not None:
                if bpb > 1:
                    delta, color = "🔴 Above Bands", "inverse"
                elif bpb < 0:
                    delta, color = "🔴 Below Bands", "inverse"
                else:
                    delta, color = "🟢 Within Bands", "normal"
                st.metric("Bollinger %B", f"{bpb:.2f}", delta=delta, delta_color=color)
            else:
                st.metric("Bollinger %B", "N/A")
        
        # EMA Crossover
        with col4:
            ema_signal = latest.get('EMA_signal', None)
            if ema_signal:
                if ema_signal == "Golden Cross":
                    display_signal = "Uptrend"
                    delta, color = "🟢 Bullish", "normal"
                else:
                    display_signal = "Downtrend"
                    delta, color = "🔴 Bearish", "inverse"
                st.metric("EMA Crossover", display_signal, delta=delta, delta_color=color)
            else:
                st.metric("EMA Crossover", "N/A")
        
        # OBV
        with col5:
            obv = latest.get('OBV', None)
            obv_window = 5
            if obv is not None and len(technical_df) > obv_window:
                obv_trend = technical_df['OBV'].iloc[-obv_window:]
                increases = (obv_trend.diff() > 0).sum()
                decreases = (obv_trend.diff() < 0).sum()
                if increases == obv_window - 1:
                    delta, color = "🟢 Strong Buying", "normal"
                elif decreases == obv_window - 1:
                    delta, color = "🔴 Strong Selling", "inverse"
                else:
                    delta, color = "🟡 Mixed/Weak", "off"
                st.metric("OBV", f"{int(obv):,}", delta=delta, delta_color=color)
            elif obv is not None:
                st.metric("OBV", f"{int(obv):,}")
            else:
                st.metric("OBV", "N/A")

        # Indicator Explanations
        with st.expander("📋 Indicator Explanations"):
            st.markdown("""
            - **RSI (21):** Values below 30 indicate oversold conditions, above 70 indicate overbought conditions.
            - **MACD:** Measures market momentum through moving average convergence/divergence.
            - **Bollinger %B:** Above 1 → expansion, between 0 and 1 → within bands, below 0 → contraction.
            - **EMA Crossover:** Uptrend means 12 EMA is above 26 EMA, Downtrend means 12 EMA is below 26 EMA.
            - **OBV:** Measures buying/selling pressure based on trading volume.
            """)

        # Technical Data Display
        df_tech = st.session_state.analysis['technical_data'].copy()
        df_tech['Date'] = pd.to_datetime(df_tech['Date']).dt.date

        # Column definitions
        price_trend_cols = [
            'Date', 'Close', 'SMA_20', 'SMA_50', 'ADX', 'Plus_DI', 'Minus_DI',
            'Pivot', 'R1', 'S1', 'R2', 'S2', 'Fib_23.6', 'Fib_38.2', 'Fib_50',
            'Fib_61.8', 'Fib_78.6', 'sr_zones', 'Long_Resistance', 'Long_Support'
        ]

        momentum_cols = [
            'Date', 'Close', 'RSI_7', 'RSI_14', 'RSI_21',
            'MACD', 'MACD_Signal', 'MACD_Histogram', 'Stoch_K', 'Stoch_D'
        ]

        volume_volatility_cols = [
            'Date', 'Close', 'OBV', 'OBVSignal',
            'BB_Lower', 'BB_Middle', 'BB_Upper', 'Bollinger%B'
        ]

        signal_cols = [
            'Date', 'Close', 'Combined_Signal',
            'Buy_Score', 'Sell_Score', 'Net_Score', 'Signal'
        ]

        st.subheader("📊 Latest Technical Analysis")
        st.write("Last 10 days of key technical indicators and signals")

        # Create tabs for indicator categories
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Price & Trend",
            "⚡ Momentum", 
            "📊 Volume & Volatility",
            "🎯 Signals"
        ])

        # Helper function for column configuration
        def get_col_config(cols):
            return {
                **{
                    col: st.column_config.NumberColumn(format="%.2f")
                    for col in cols if col != 'Date'
                },
                "Date": st.column_config.Column(width="medium")
            }

#________________________________________________________

        with tab1:
            available_cols = [c for c in price_trend_cols if c in df_tech.columns]
            if available_cols:
                st.data_editor(
                    df_tech[available_cols].tail(10).round(2),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        **{
                            col: st.column_config.NumberColumn(
                                format="%.2f"
                            )
                            for col in available_cols if col != 'Date'
                        },
                        "Date": st.column_config.Column(
                            width="medium"
                        )
                    }
                )

        with tab2:
            available_cols = [c for c in momentum_cols if c in df_tech.columns]
            if available_cols:
                st.data_editor(
                    df_tech[available_cols].tail(10).round(2),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        **{
                            col: st.column_config.NumberColumn(
                                format="%.2f"
                            )
                            for col in available_cols if col != 'Date'
                        },
                        "Date": st.column_config.Column(
                            width="medium"
                        )
                    }
                )

        with tab3:
            available_cols = [c for c in volume_volatility_cols if c in df_tech.columns]
            if available_cols:
                st.data_editor(
                    df_tech[available_cols].tail(10).round(2),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        **{
                            col: st.column_config.NumberColumn(
                                format="%.2f"
                            )
                            for col in available_cols if col != 'Date'
                        },
                        "Date": st.column_config.Column(
                            width="medium"
                        )
                    }
                )

        with tab4:
            available_cols = [c for c in signal_cols if c in df_tech.columns]
            if available_cols:
                styled_df = df_tech[available_cols].tail(10).round(2)
                st.data_editor(
                    styled_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        **{
                            col: st.column_config.NumberColumn(
                                format="%.2f"
                            )
                            for col in available_cols if col != 'Date'
                        },
                        "Date": st.column_config.Column(
                            width="medium"
                        )
                    }
                )

        # Quick indicators summary
        st.subheader("📋 Quick Reference")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.info("""
            **Trend Indicators**
            - SMA/EMA: Moving averages
            - Price above MA = Uptrend
            - Price below MA = Downtrend
            """)

        with col2:
            st.info("""
            **Momentum Indicators**
            - RSI: 30-70 normal range
            - MACD: Above 0 = bullish
            - Signal crossovers = entry points
            """)

        with col3:
            st.info("""
            **Key Signals**
            - Buy Score: Bullish strength
            - Sell Score: Bearish strength
            - Net Score: Overall direction
            """)

        # Detailed Indicator Explanations - Better formatted
        with st.expander("📊 Complete Indicator Explanations"):

            # Price & Trend Indicators Section
            st.markdown("### 📈 **Price & Trend Indicators**")
            st.markdown("""
            **🔹 SMA_20/SMA_50:** Simple Moving Averages for 20 and 50 periods - trend direction indicators

            **🔹 ADX:** Average Directional Index - measures trend strength (>25 = strong trend)

            **🔹 Plus_DI/Minus_DI:** Directional Indicators showing bullish/bearish pressure

            **🔹 Pivot Points:** Daily support/resistance levels based on previous day's high/low/close

            **🔹 Fibonacci Levels:** Retracement levels at 23.6%, 38.2%, 50%, 61.8%, 78.6%

            **🔹 S/R Zones:** Aggregated support and resistance zones from multiple timeframes
            """)
            st.divider()

            # Momentum Indicators Section  
            st.markdown("### ⚡ **Momentum Indicators**")
            st.markdown("""
            **🔹 RSI (7/14/21):** Relative Strength Index - below 30 oversold, above 70 overbought

            **🔹 MACD:** Moving Average Convergence Divergence - measures momentum changes

            **🔹 MACD_Histogram:** Shows divergence between MACD and signal line

            **🔹 Stochastic K/D:** Momentum oscillator comparing closing price to price range
            """)
            st.divider()

            # Volume & Volatility Indicators Section
            st.markdown("### 📊 **Volume & Volatility Indicators**")
            st.markdown("""
            **🔹 OBV:** On-Balance Volume - measures buying/selling pressure based on volume

            **🔹 Bollinger %B:** Position within Bollinger Bands (>1 above, <0 below, 0-1 within)

            **🔹 BB_Upper/Middle/Lower:** Bollinger Band levels showing volatility ranges
            """)
            st.divider()

            # Signal Indicators Section
            st.markdown("### 🎯 **Signal Indicators**")
            st.markdown("""
            **🔹 Buy_Score:** Strength of bullish signals (0-100)

            **🔹 Sell_Score:** Strength of bearish signals (0-100)

            **🔹 Net_Score:** Overall market direction (Buy_Score - Sell_Score)

            **🔹 Signal:** Final recommendation based on all indicators combined
            """)
#_________________________________________________________

    with tabs[2]:
        fin = analysis.get('financial_analysis', {})
        score = fin.get('overall_score', None)
        if score is not None:
            score_color = ("🟢" if score >= 70 else "🔴" if score < 50 else "🟡")
            border_color = "#1db954" if score >= 70 else "#e4363a" if score < 50 else "#f4c20d"
            st.markdown(f"""
            <div style="
                width: 100%;
                background-color: #2c2f33;
                padding: 15px;
                border-radius: 12px;
                border-left: 6px solid {border_color};
                border-right: 6px solid {border_color}; 
                text-align: center;
                max-width: 100%;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            ">
                <h4 style="
                    text-align: center;
                    color: white;
                    font-size: 35px;
                    line-height: 1;
                    margin-bottom: 10px;
                ">
                    {score_color} Financial Health Score
                </h4>
                <h2 style="margin: -23px 0 0px; color: {border_color}; font-size: 40px;">
                    {score:.1f}/100
                </h2>
                <p style="
                    margin: 0; 
                    text-align: center; 
                    font-size: 18px; 
                    color: #cccccc;
                "> 
                    <strong style="color: #ffffff; font-size: 22px;">{fin.get('health_rating','')}</strong><br>
                    <span style="font-size: 16px;">{fin.get('health_description','')}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Financial health data not available.")
        st.markdown("---")

        # === 2. المؤشرات الرئيسية ===
        st.markdown("### 🎯 Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            rev_growth = fin.get('revenue_analysis', {}).get('revenue_growth_%', 0)
            st.metric(
                label="📈 Revenue Growth",
                value=f"{rev_growth:.1f}%",
                delta=fin.get('revenue_analysis', {}).get('trend','')[:20],
                help="Year-over-year revenue growth rate"
            )

        
        with col2:
            net_margin = fin['profitability_analysis'].get('net_margin_%', 0)
            st.metric(
                label="💰 Net Profit Margin",
                value=f"{net_margin:.1f}%",
                delta="Strong" if net_margin > 15 else "Good" if net_margin > 8 else "Weak",
                help="Net income as percentage of revenue"
            )
        
        with col3:
            debt_eq = fin['balance_sheet_analysis'].get('debt_to_equity', 0)
            debt_status = "Low" if debt_eq < 0.3 else "Moderate" if debt_eq < 0.6 else "High"
            st.metric(
                label="⚖️ Debt/Equity Ratio",
                value=f"{debt_eq:.2f}",
                delta=debt_status,
                delta_color="normal" if debt_eq < 0.6 else "inverse",
                help="Total debt relative to shareholders' equity"
            )
        
        with col4:
            current_ratio = fin['balance_sheet_analysis'].get('current_ratio', 0)
            liquidity_status = "Excellent" if current_ratio > 2 else "Good" if current_ratio > 1.5 else "Weak"
            st.metric(
                label="💧 Current Ratio",
                value=f"{current_ratio:.2f}",
                delta=liquidity_status,
                delta_color="normal" if current_ratio > 1.2 else "inverse",
                help="Ability to pay short-term obligations"
            )
        
        st.markdown("---")
        
# === 3. تحليل تفصيلي منظم ===
        st.markdown("### 📊 Detailed Financial Analysis")
        detail_tabs = st.tabs(["💰 Profitability", "🏦 Balance Sheet", "💸 Cash Flow", "🏷️ Valuation"])

        with detail_tabs[0]:
            prof = fin['profitability_analysis']
            # عرض المقاييس على 3 أعمدة بداخلها بطاقات منسقة
            p1, p2, p3 = st.columns(3)

            with p1:
                gross_margin = prof.get('gross_margin_%', 0)
                gross_color = "🟢" if gross_margin > 30 else "🟡" if gross_margin > 15 else "🔴"
                st.markdown(f"""
                <div style="background: #2c2f33; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;">
                    <h4 style="margin: 0; color: #ffffff; font-size: 24px;">Gross Margin</h4>
                    <h2 style="margin-top: -8px; color: #4facfe; font-size: 28px;">{gross_color} {gross_margin:.1f}%</h2>
                    <p style="margin: 0; font-size: 14px; color: #cccccc;">Profit percentage before operating expenses</p>
                </div>
                """, unsafe_allow_html=True)

            with p2:
                op_margin = prof.get('operating_margin_%', 0)
                op_color = "🟢" if op_margin > 15 else "🟡" if op_margin > 8 else "🔴"
                st.markdown(f"""
                <div style="background: #2c2f33; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;">
                    <h4 style="margin: 0; color: #ffffff; font-size: 24px;">Operating Margin</h4>
                    <h2 style="margin-top: -8px; color: #4facfe; font-size: 28px;">{op_color} {op_margin:.1f}%</h2>
                    <p style="margin: 0; font-size: 14px; color: #cccccc;">Efficiency in managing core business operations</p>
                </div>
                """, unsafe_allow_html=True)

            with p3:
                net_margin = prof.get('net_margin_%', 0)
                net_color = "🟢" if net_margin > 15 else "🟡" if net_margin > 8 else "🔴"
                st.markdown(f"""
                <div style="background: #2c2f33; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;">
                    <h4 style="margin: 0; color: #ffffff; font-size: 24px;">Net Margin</h4>
                    <h2 style="margin-top: -8px; color: #4facfe; font-size: 28px;">{net_color} {net_margin:.1f}%</h2>
                    <p style="margin: 0; font-size: 14px; color: #cccccc;">Final net profit after all costs and taxes</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # عرض العائدات
            st.markdown("#### 📈 Return Metrics")
            r1, r2 = st.columns(2)

            with r1:
                roe = prof.get('ROE_%', 0)
                roe_color = "🟢" if roe > 15 else "🟡" if roe > 10 else "🔴"
                st.markdown(f"""
                <div style="background: #2c2f33; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;">
                    <h4 style="margin: 0; color: #ffffff; font-size: 24px;">Return on Equity (ROE)</h4>
                    <h2 style="margin-top: -8px; color: #4facfe; font-size: 28px;">{roe_color} {roe:.1f}%</h2>
                    <p style="margin: 0; font-size: 14px; color: #cccccc;">Profit generated from each dollar of shareholders' equity</p>
                </div>
                """, unsafe_allow_html=True)

            with r2:
                # استخراج ROA من الأماكن المحتملة
                roa = 0
                if 'asset_efficiency' in fin:
                    roa = fin['asset_efficiency'].get('ROA_%', 0)
                elif 'ROA_%' in prof:
                    roa = prof.get('ROA_%', 0)
                elif 'balance_sheet_analysis' in fin:
                    roa = fin['balance_sheet_analysis'].get('ROA_%', 0)
                else:
                    for section in fin.values():
                        if isinstance(section, dict) and 'ROA_%' in section:
                            roa = section['ROA_%']
                            break
                roa_color = "🟢" if roa > 8 else "🟡" if roa > 4 else "🔴"
                st.markdown(f"""
                <div style="background: #2c2f33; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;">
                    <h4 style="margin: 0; color: #ffffff; font-size: 24px;">Return on Assets (ROA)</h4>
                    <h2 style="margin-top: -8px; color: #4facfe; font-size: 28px;">{roa_color} {roa:.1f}%</h2>
                    <p style="margin: 0; font-size: 14px; color: #cccccc;">How effectively assets are used to generate profits</p>
                </div>
                """, unsafe_allow_html=True)

        with detail_tabs[1]:
            balance = fin['balance_sheet_analysis']
            # Debt و Liquidity على عمودين
            st.markdown("##### 💳 Debt Analysis")
            d1, d2 = st.columns(2)
            with d1:
                de_ratio = balance.get('debt_to_equity', 0)
                de_color = "🟢" if de_ratio < 0.3 else "🟡" if de_ratio < 0.6 else "🔴"
                st.markdown(f"""
                <div style="background: #2c2f33; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;">
                    <h4 style="margin: 0; color: #ffffff; font-size: 24px;">Debt-to-Equity</h4>
                    <h2 style="margin-top: -8px; color: #4facfe; font-size: 28px;">{de_color} {de_ratio:.2f}</h2>
                    <p style="margin: 0; font-size: 14px; color: #cccccc;">Company's debt level compared to shareholders' equity</p>
                </div>
                """, unsafe_allow_html=True)
            with d2:
                da_ratio = balance.get('debt_to_assets_%', 0)
                da_color = "🟢" if da_ratio < 30 else "🟡" if da_ratio < 50 else "🔴"
                st.markdown(f"""
                <div style="background: #2c2f33; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;">
                    <h4 style="margin: 0; color: #ffffff; font-size: 24px;">Debt-to-Assets</h4>
                    <h2 style="margin-top: -8px; color: #4facfe; font-size: 28px;">{da_color} {da_ratio:.1f}%</h2>
                    <p style="margin: 0; font-size: 14px; color: #cccccc;">Percentage of assets financed through debt</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("##### 💧 Liquidity Analysis")
            l1, l2 = st.columns(2)
            with l1:
                current_ratio = balance.get('current_ratio', 0)
                cr_color = "🟢" if current_ratio > 1.5 else "🟡" if current_ratio > 1.0 else "🔴"
                st.markdown(f"""
                <div style="background: #2c2f33; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;">
                    <h4 style="margin: 0; color: #ffffff; font-size: 24px;">Current Ratio</h4>
                    <h2 style="margin-top: -8px; color: #4facfe; font-size: 28px;">{cr_color} {current_ratio:.2f}</h2>
                    <p style="margin: 0; font-size: 14px; color: #cccccc;">Ability to pay short-term debts with current assets</p>
                </div>
                """, unsafe_allow_html=True)
            with l2:
                quick_ratio = balance.get('quick_ratio', 0)
                qr_color = "🟢" if quick_ratio > 1.0 else "🟡" if quick_ratio > 0.7 else "🔴"
                st.markdown(f"""
                <div style="background: #2c2f33; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;">
                    <h4 style="margin: 0; color: #ffffff; font-size: 24px;">Quick Ratio</h4>
                    <h2 style="margin-top: -8px; color: #4facfe; font-size: 28px;">{qr_color} {quick_ratio:.2f}</h2>
                    <p style="margin: 0; font-size: 14px; color: #cccccc;">Ability to pay debts using only liquid assets</p>
                </div>
                """, unsafe_allow_html=True)

        with detail_tabs[2]:
            cashflow = fin['cash_flow_analysis']
            st.markdown("##### 💰 Cash Generation")
            cf1, cf2 = st.columns(2)
            with cf1:
                op_cf = cashflow.get('operating_cash_flow', 0) / 1000
                # عرض كامل بالدولار مع تنسيق الأرقام، بدون تحويل للملايين حفاظاً على المنطق الأصلي
                op_cf_color = "🟢" if op_cf >= 0 else "🔴"
                st.markdown(f"""
                <div style="background: #2c2f33; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;">
                    <h4 style="margin: 0; color: #ffffff; font-size: 24px;">Operating Cash Flow</h4>
                    <h2 style="margin-top: -8px; color: #4facfe; font-size: 28px;">{op_cf_color} ${op_cf:,.0f}</h2>
                    <p style="margin: 0; font-size: 14px; color: #cccccc;">Cash generated from normal business operations</p>
                </div>
                """, unsafe_allow_html=True)
            with cf2:
                free_cf = cashflow.get('free_cash_flow', 0) / 1000
                fcf_color = "🟢" if free_cf >= 0 else "🔴"
                st.markdown(f"""
                <div style="background: #2c2f33; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;">
                    <h4 style="margin: 0; color: #ffffff; font-size: 24px;">Free Cash Flow</h4>
                    <h2 style="margin-top: -8px; color: #4facfe; font-size: 28px;">{fcf_color} ${free_cf:,.0f}</h2>
                    <p style="margin: 0; font-size: 14px; color: #cccccc;">Cash left after covering operating expenses and investments</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("##### 📊 Cash Flow Margins")
            m1, m2 = st.columns(2)
            with m1:
                ocf_to_rev = cashflow.get('ocf_to_revenue_%', 0)
                ocf_color = "🟢" if ocf_to_rev > 15 else "🟡" if ocf_to_rev > 8 else "🔴"
                st.markdown(f"""
                <div style="background: #2c2f33; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;">
                    <h4 style="margin: 0; color: #ffffff; font-size: 24px;">OCF to Revenue</h4>
                    <h2 style="margin-top: -8px; color: #4facfe; font-size: 28px;">{ocf_color} {ocf_to_rev:.1f}%</h2>
                    <p style="margin: 0; font-size: 14px; color: #cccccc;">How much of each revenue dollar becomes operating cash</p>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                fcf_to_rev = cashflow.get('fcf_to_revenue_%', 0)
                fcf_margin_color = "🟢" if fcf_to_rev > 10 else "🟡" if fcf_to_rev > 5 else "🔴"
                st.markdown(f"""
                <div style="background: #2c2f33; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;">
                    <h4 style="margin: 0; color: #ffffff; font-size: 24px;">FCF to Revenue</h4>
                    <h2 style="margin-top: -8px; color: #4facfe; font-size: 28px;">{fcf_margin_color} {fcf_to_rev:.1f}%</h2>
                    <p style="margin: 0; font-size: 14px; color: #cccccc;">How much of each revenue dollar becomes free cash</p>
                </div>
                """, unsafe_allow_html=True)

        with detail_tabs[3]:
            valuation = fin['valuation_analysis']
            st.markdown("##### 📈 Price Ratios")
            v1, v2, v3 = st.columns(3)
            with v1:
                pe = valuation.get('P/E')
                if pe:
                    pe_color = "🟢" if pe < 15 else "🟡" if pe < 25 else "🔴"
                    st.markdown(f"""
                    <div style="background: #2c2f33; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;">
                        <h4 style="margin: 0; color: #ffffff; font-size: 24px;">Price-to-Earnings</h4>
                        <h2 style="margin-top: -8px; color: #4facfe; font-size: 28px;">{pe_color} {pe:.1f}</h2>
                        <p style="margin: 0; font-size: 14px; color: #cccccc;">How much investors pay for each dollar of company earnings</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: #2c2f33; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;">
                        <h4 style="margin: 0; color: #ffffff; font-size: 24px;">Price-to-Earnings</h4>
                        <h2 style="margin-top: -8px; color: #999999; font-size: 28px;">N/A</h2>
                        <p style="margin: 0; font-size: 14px; color: #cccccc;">Data not available or company has no earnings</p>
                    </div>
                    """, unsafe_allow_html=True)
            with v2:
                pb = valuation.get('P/B')
                if pb:
                    pb_color = "🟢" if pb < 1.5 else "🟡" if pb < 3.0 else "🔴"
                    st.markdown(f"""
                    <div style="background: #2c2f33; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;">
                        <h4 style="margin: 0; color: #ffffff; font-size: 24px;">Price-to-Book</h4>
                        <h2 style="margin-top: -8px; color: #4facfe; font-size: 28px;">{pb_color} {pb:.1f}</h2>
                        <p style="margin: 0; font-size: 14px; color: #cccccc;">Stock price compared to company's book value per share</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: #2c2f33; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;">
                        <h4 style="margin: 0; color: #ffffff; font-size: 24px;">Price-to-Book</h4>
                        <h2 style="margin-top: -8px; color: #999999; font-size: 28px;">N/A</h2>
                        <p style="margin: 0; font-size: 14px; color: #cccccc;">Data not available for this calculation</p>
                    </div>
                    """, unsafe_allow_html=True)
            with v3:
                peg = valuation.get('PEG')
                if peg:
                    peg_color = "🟢" if peg < 1.0 else "🟡" if peg < 1.5 else "🔴"
                    st.markdown(f"""
                    <div style="background: #2c2f33; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;">
                        <h4 style="margin: 0; color: #ffffff; font-size: 24px;">PEG Ratio</h4>
                        <h2 style="margin-top: -8px; color: #4facfe; font-size: 28px;">{peg_color} {peg:.1f}</h2>
                        <p style="margin: 0; font-size: 14px; color: #cccccc;">P/E ratio adjusted for the company's earnings growth rate</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: #2c2f33; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;">
                        <h4 style="margin: 0; color: #ffffff; font-size: 24px;">PEG Ratio</h4>
                        <h2 style="margin-top: -8px; color: #999999; font-size: 28px;">N/A</h2>
                        <p style="margin: 0; font-size: 14px; color: #cccccc;">Growth data not available for this calculation</p>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("##### 📊 Valuation Trends")
            t1, t2 = st.columns(2)
            with t1:
                pe_trend = valuation.get('pe_trend', 'N/A')
                trend_color = {"Increasing": "🔴", "Decreasing": "🟢", "Stable": "🟡"}.get(pe_trend, "")
                st.markdown(f"""
                <div style="background: #2c2f33; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;">
                    <h4 style="margin: 0; color: #ffffff; font-size: 24px;">P/E Trend</h4>
                    <h2 style="margin-top: -8px; color: #4facfe; font-size: 28px;">{trend_color} {pe_trend}</h2>
                    <p style="margin: 0; font-size: 14px; color: #cccccc;">Direction of P/E ratio changes over time</p>
                </div>
                """, unsafe_allow_html=True)
            with t2:
                pb_trend = valuation.get('pb_trend', 'N/A')
                pb_trend_color = {"Increasing": "🔴", "Decreasing": "🟢", "Stable": "🟡"}.get(pb_trend, "")
                st.markdown(f"""
                <div style="background: #2c2f33; padding: 15px; border-radius: 8px; text-align: center; margin: 5px;">
                    <h4 style="margin: 0; color: #ffffff; font-size: 24px;">P/B Trend</h4>
                    <h2 style="margin-top: -8px; color: #4facfe; font-size: 28px;">{pb_trend_color} {pb_trend}</h2>
                    <p style="margin: 0; font-size: 14px; color: #cccccc;">Direction of P/B ratio changes over time</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

# Tab 3: SWOT Analysis
    with tabs[3]:
        st.subheader("🧠 SWOT Analysis")
        
        swot_sections = analysis['swot']
        
        # تقسيم SWOT إلى صفين (2x2)
        swot_config = {
            "Strengths": {"emoji": "💪", "color": "#28a745"},
            "Weaknesses": {"emoji": "⚠️", "color": "#ffc107"},
            "Opportunities": {"emoji": "🚀", "color": "#007bff"},
            "Threats": {"emoji": "⚡", "color": "#dc3545"}
        }
        
        # الصف الأول: Strengths & Weaknesses
        first_row = st.columns(2)
        sections_first = ["Strengths", "Weaknesses"]
        
        for i, section in enumerate(sections_first):
            if section in swot_sections:
                items = swot_sections[section]
                config = swot_config[section]
                
                with first_row[i]:
                    st.markdown(f"### {config['emoji']} {section}")
                    
                    if items and any(item and item != "N/A" for item in items):
                        for item in items:
                            if item and item != "N/A":
                                st.markdown(f"• {item}")
                    else:
                        st.markdown("*No items identified*")
                    
                    st.markdown("")  # مسافة فارغة
        
        # الصف الثاني: Opportunities & Threats
        second_row = st.columns(2)
        sections_second = ["Opportunities", "Threats"]
        
        for i, section in enumerate(sections_second):
            if section in swot_sections:
                items = swot_sections[section]
                config = swot_config[section]
                
                with second_row[i]:
                    st.markdown(f"### {config['emoji']} {section}")
                    
                    if items and any(item and item != "N/A" for item in items):
                        for item in items:
                            if item and item != "N/A":
                                st.markdown(f"• {item}")
                    else:
                        st.markdown("*No items identified*")
                    
                    st.markdown("")  # مسافة فارغة

    # Tab 4: Charts & Visualizations
    with tabs[4]:
        st.subheader("📉 Price Analysis Charts")

        # ==== Layer Toggles ====
        col1, col2, col3, col4 = st.columns(4)
        show_bb     = col1.checkbox("Bollinger Bands", value=False)
        show_sma50  = col2.checkbox("SMA-50",           value=True)
        show_macd   = col3.checkbox("MACD Panel",       value=True)
        show_volume = col4.checkbox("Volume",           value=True)

        # ---------- 1) Unified Interactive Chart ----------
        interactive_fig = create_interactive_chart(
            analysis, analysis["technical_data"], stock_symbol,
            show_bb=show_bb, show_sma50=show_sma50,
            show_macd=show_macd, show_volume=show_volume
        )
        interactive_fig.update_layout(title_font=dict(size=28))
        st.plotly_chart(interactive_fig, use_container_width=True)

        st.divider()   # فاصل واضح بين التشارتات

        # ---------- 2) Price Comparison Chart ----------
        tech_df = analysis["technical_data"]

        nearest_res = analysis.get(
            "nearest_resistance_1y",
            tech_df["High"].rolling(window=252, min_periods=1).max().iloc[-1]
        )
        nearest_sup = analysis.get(
            "nearest_support_1y",
            tech_df["Low"].rolling(window=252, min_periods=1).min().iloc[-1]
        )

        fi = analysis.get("fundamental_info", {})
        analyst_min = fi.get("targetLowPrice")
        analyst_avg = fi.get("targetMeanPrice")
        analyst_max = fi.get("targetHighPrice")

        levels = {
            "analyst_min":        analyst_min,
            "nearest_resistance": nearest_res,
            "current":            analysis["current_price"],
            "analyst_avg":        analyst_avg,
            "nearest_support":    nearest_sup,
            "analyst_max":        analyst_max,
        }

        comparison_fig = create_comparison_chart(levels, stock_symbol)
        comparison_fig.update_layout(title_font=dict(size=28))
        st.plotly_chart(comparison_fig, use_container_width=True)

        st.divider()   # فاصل بين Price-Comparison و MACD Histogram

        # ---------- 3) MACD Histogram (custom HTML) ----------
        st.markdown(                 # ← العنوان الجديد
            f"<h2 style='font-size:32px;font-weight:600;margin:0 0 12px 0;'>"
            f"📊 MACD Histogram</h2>",
            unsafe_allow_html=True
        )
        macd_path = os.path.join(os.path.dirname(__file__), "macd_histogram.html")
        hist_df      = tech_df[['Date', 'MACD_Histogram']].tail(90)
        values_json  = json.dumps(hist_df['MACD_Histogram'].round(3).tolist())
        labels_json  = json.dumps(hist_df['Date'].dt.strftime('%d/%m').tolist())

        context = {
            "MACD_VALUES": values_json,
            "MACD_LABELS": labels_json,
        }
        html_raw = render_html_template(macd_path, context)

        # (تأكّد أنك كبّرت عنوان الـ HTML داخل الملف نفسه:
        #  font-size:28px أو options.plugins.title.font.size = 28)
        components.html(html_raw, height=650, scrolling=True)

        st.divider()   # فاصل قبل لوحة الـ S/R

        # ---------- 4) Support & Resistance Dashboard ----------
        st.markdown("### 🛠️ Support and Resistance")
        render_sr_dashboard_inline(
            analysis,
            current_price=analysis.get('current_price', 0)
        )


    # Tab 5: Investment Decision
    with tabs[5]:
        st.subheader("✅ Investment Recommendation Summary")

        # ---------- Banner ----------
        decision      = analysis["decision"]                 # Buy / Sell / Hold
        confidence    = analysis["confidence"]               # % confidence
        risk_score    = analysis.get("risk_rating", "N/A")   # Low / Medium / High
        reward_ratio  = analysis.get("reward_to_risk", np.nan)

        decision_map  = {"Buy": ("🟢", "#27ae60"),
                         "Sell": ("🔴", "#e74c3c"),
                         "Hold": ("🟡", "#f1c40f")}
        emoji, color  = decision_map.get(decision, ("🔵", "#3498db"))

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{color}44 0%,{color}22 100%);
                    padding:20px;border-radius:12px;text-align:center;margin-bottom:25px;
                    border:2px solid {color};">
            <h2 style="margin:0;font-size:38px;">{emoji} <span style="color:{color};">
                {decision}</span></h2>
            <p style="margin:5px 0 0;color:#cccccc;font-size:18px;">
                System confidence <strong>{confidence:.1f}%</strong> &middot;
                Risk level <strong>{risk_score}</strong> &middot;
                Reward-to-Risk <strong>{'—' if np.isnan(reward_ratio) else f'{reward_ratio:.2f}:1'}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ---------- Key Numbers Grid ----------
        k1,k2,k3,k4,k5,k6 = st.columns(6)

        k1.metric("💵 Current",   f"${analysis['current_price']:.2f}")
        k2.metric("🎯 Entry",     f"${analysis['entry_point']:.2f}")
        k3.metric("🛑 Stop-Loss", f"${analysis['stop_loss']:.2f}")
        # حساب نسبة التغيّر لكل هدف
        pct1 = (analysis['up_targets'][0] / analysis['current_price'] - 1) * 100
        pct2 = (analysis['up_targets'][1] / analysis['current_price'] - 1) * 100
        k4.metric("🎯 Target 1",
                  f"${analysis['up_targets'][0]:.2f}",
                  delta=f"{pct1:+.1f}%",
                  delta_color="normal")

        k5.metric("🎯 Target 2",
                  f"${analysis['up_targets'][1]:.2f}",
                  delta=f"{pct2:+.1f}%",
                  delta_color="normal")

        # Analyst average و نسبة الربح حتى ذلك الهدف
        avg_target = analysis.get("analyst_avg_target", np.nan)
        if not np.isnan(avg_target):
            profit_pct = (avg_target / analysis['current_price'] - 1) * 100
            k6.metric("📊 Analyst Avg", f"${avg_target:.2f}",
                      delta=f"{profit_pct:+.1f}%", delta_color="normal")
        else:
            k6.metric("📊 Analyst Avg", "N/A")

        # ---------- Score Bars ----------
        tech_score = analysis["technical_score"]                             # 0-100
        fund_score = analysis["financial_analysis"]["overall_score"]         # 0-100
        swot_score = analysis["swot_score"]                                  # 0-100

        def progress_bar(label, val, color):
            st.markdown(f"""
            <div style="margin:12px 0;">
              <span style="display:inline-block;width:120px;">{label}</span>
              <div style="background:#2c2f33;border-radius:6px;height:16px;
                          display:inline-block;width:60%;">
                <div style="background:{color};height:16px;border-radius:6px;
                            width:{val}%;"></div>
              </div>
              <span style="margin-left:8px;">{val:.0f}%</span>
            </div>
            """, unsafe_allow_html=True)

        progress_bar("Technical",   tech_score, "#3498db")
        progress_bar("Fundamental", fund_score, "#1abc9c")
        progress_bar("SWOT",        swot_score, "#9b59b6")

        # ---------- Build Technical Summary ----------
        technical_summary = analysis.get(
            "technical_summary",
            f"Net Score {analysis['avg_net_score']:+.2f} → "
            f"{analysis.get('prediction', 'No clear trend')}"
        )

        # ---------- Narrative Summary ----------
        st.markdown("### 📋 Why this recommendation?")
        st.info(f"""
        **Technical outlook:** {technical_summary}  
        **Financial health:** {analysis.get('financial_analysis', {}).get('health_description', 'N/A')}  
        **Valuation:** Stock trades at **{analysis.get('financial_analysis', {}).get('valuation_analysis', {}).get('valuation_flag', 'N/A')}** compared to peers.  
        **Catalysts:** {analysis['swot']['Opportunities'][0] if analysis['swot']['Opportunities'] else '—'}  
        **Risks:** {analysis['swot']['Threats'][0] if analysis['swot']['Threats'] else '—'}
        """)


    # Download button
    if 'excel_file' in st.session_state and st.session_state['excel_file']:
        st.markdown("---")
        with open(st.session_state['excel_file'], "rb") as file:
            st.download_button(
                label="📥 Download Complete Analysis Report",
                data=file.read(),
                file_name=f"{stock_symbol}_Complete_Analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                type="primary"
            )

else:
    # Welcome page
    st.markdown("""
    <div class="welcome-section">
        <h2>🎯 Welcome to the Comprehensive Stock Analysis Platform</h2>
        <p>Get professional-grade stock analysis with advanced technical indicators, fundamental analysis, SWOT assessment, and AI-powered investment recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="features-grid">
        <div class="feature-card">
            <h3>🔍 Technical Analysis</h3>
            <p>Advanced technical indicators including RSI, MACD, Bollinger Bands, Fibonacci levels, and support/resistance zones.</p>
        </div>
        <div class="feature-card">
            <h3>📊 Fundamental Analysis</h3>
            <p>Deep dive into financial statements, ratios, growth metrics, and company valuation analysis.</p>
        </div>
        <div class="feature-card">
            <h3>🧠 SWOT Analysis</h3>
            <p>Comprehensive evaluation of Strengths, Weaknesses, Opportunities, and Threats affecting the stock.</p>
        </div>
        <div class="feature-card">
            <h3>🎯 Price Targets</h3>
            <p>AI-powered price predictions with entry points, stop losses, and profit-taking levels.</p>
        </div>
        <div class="feature-card">
            <h3>📈 Interactive Charts</h3>
            <p>Professional-grade charts with technical overlays and price projection visualizations.</p>
        </div>
        <div class="feature-card">
            <h3>📋 Detailed Reports</h3>
            <p>Downloadable Excel reports with comprehensive analysis and investment recommendations.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="instructions">
        <h3>📚 How to Use This Platform</h3>
        <ol>
            <li>Enter the stock symbol in the sidebar (e.g., AAPL for Apple Inc.)</li>
            <li>Select your preferred date range for historical analysis</li>
            <li>Set your investment amount and industry P/E ratio</li>
            <li>Click "Start Analysis" to begin the comprehensive evaluation</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔥 Try Sample Analysis (AAPL)", use_container_width=True, type="primary"):
            st.info("👆 Use the sidebar form to start your analysis with AAPL or any other stock symbol!")

# Footer
st.markdown("---")
st.markdown(
    "⚠️ **Disclaimer:** This analysis is for educational purposes only. "
    "Please consult with qualified financial professionals before making any investment decisions."
)