"""
Stock Comparison Module
Compare multiple stocks side-by-side with technical and fundamental analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Palette used across all charts – defined once, referenced everywhere
COLORS: Tuple[str, ...] = ('#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b')

# Financial constants
_RISK_FREE_RATE   = 0.04   # 4% – more accurate for current environment (was 2%)
_TRADING_DAYS     = 252
_RSI_PERIOD       = 14
_RSI_OVERBOUGHT   = 70
_RSI_OVERSOLD     = 30
_SMA_SHORT        = 20
_SMA_LONG         = 50
_MACD_FAST        = 12
_MACD_SLOW        = 26
_MACD_SIGNAL      = 9
_VAR_CONFIDENCE   = 5      # 5th-percentile → 95% VaR
_MAX_SYMBOLS      = 5
_INDUSTRY_AVG_PE  = 20     # sector-neutral benchmark

# Drawdown calculation: use log-returns for numerical stability on long series
_USE_LOG_DRAWDOWN = True


# ---------------------------------------------------------------------------
# Pure financial helpers (no Streamlit dependency – fully unit-testable)
# ---------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Return numerator / denominator; *fallback* on zero/None."""
    if not denominator:
        return fallback
    return numerator / denominator


def _compute_returns(close: pd.Series) -> pd.Series:
    """Simple daily percentage returns, NaNs dropped."""
    return close.pct_change().dropna()


def _compute_rsi(close: pd.Series, period: int = _RSI_PERIOD) -> pd.Series:
    """
    Wilder's smoothed RSI using EWM (adjust=False, com=period-1).

    FIX: the original code used a simple rolling mean for both gain and loss,
    which underestimates RSI momentum.  Wilder's method (EWM with
    alpha = 1/period) is the standard and matches TradingView / Bloomberg.
    """
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)  # avoid 0-division when loss == 0
    return (100 - (100 / (1 + rs))).fillna(50)  # neutral 50 when undefined


def _compute_macd(
    close: pd.Series,
    fast: int = _MACD_FAST,
    slow: int = _MACD_SLOW,
    signal: int = _MACD_SIGNAL,
) -> Tuple[pd.Series, pd.Series]:
    """Return (macd_line, signal_line)."""
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def _compute_max_drawdown(returns: pd.Series) -> float:
    """
    Maximum drawdown (as a negative %, e.g. -23.5).

    FIX: the original code computed drawdown on the *cumulative product*
    starting from an arbitrary window start.  Using the running maximum of
    the full series gives the true historical max drawdown.
    Uses log-return cumsum for numerical stability on long series.
    """
    if returns.empty:
        return 0.0
    if _USE_LOG_DRAWDOWN:
        log_r   = np.log1p(returns)
        cum     = log_r.cumsum()
        peak    = cum.cummax()
        dd      = (cum - peak)          # always ≤ 0
    else:
        cum     = (1 + returns).cumprod()
        peak    = cum.cummax()
        dd      = (cum - peak) / peak
    return float(dd.min() * 100)


def _compute_sharpe(returns: pd.Series, volatility_annualised: float) -> float:
    """
    Annualised Sharpe ratio.

    FIX: uses _RISK_FREE_RATE = 4% (was 2%), which is materially more
    accurate as of 2024-2025.
    Guard against near-zero volatility to avoid inf/nan.
    """
    if volatility_annualised < 0.01:
        return 0.0
    avg_return = float(returns.mean()) * _TRADING_DAYS
    return (avg_return - _RISK_FREE_RATE) / (volatility_annualised / 100)


def _compute_sortino(returns: pd.Series) -> float:
    """
    Sortino ratio (penalises only downside deviation).

    NEW metric – more meaningful than Sharpe for skewed return distributions.
    """
    avg_return     = float(returns.mean()) * _TRADING_DAYS
    downside       = returns[returns < 0]
    downside_std   = float(downside.std()) * np.sqrt(_TRADING_DAYS) if len(downside) > 1 else 1e-9
    return (avg_return - _RISK_FREE_RATE) / downside_std


def _compute_calmar(returns: pd.Series, max_drawdown_pct: float) -> float:
    """
    Calmar ratio = annualised return / |max drawdown|.

    NEW metric – widely used in fund comparisons.
    """
    if max_drawdown_pct >= 0:  # no drawdown (extremely unlikely but safe)
        return 0.0
    avg_return = float(returns.mean()) * _TRADING_DAYS * 100
    return avg_return / abs(max_drawdown_pct)


def _compute_var(returns: pd.Series, confidence_pct: int = _VAR_CONFIDENCE) -> float:
    """
    Historical VaR at given confidence level, returned as a % (negative).

    FIX: original used np.percentile directly on daily returns – correct.
    We add a guard for empty series.
    """
    if returns.empty:
        return 0.0
    return float(np.percentile(returns, confidence_pct) * 100)


def _compute_cvar(returns: pd.Series, confidence_pct: int = _VAR_CONFIDENCE) -> float:
    """
    Conditional VaR (Expected Shortfall) – average loss beyond VaR.

    NEW metric – more informative than VaR alone for tail-risk assessment.
    """
    if returns.empty:
        return 0.0
    threshold = np.percentile(returns, confidence_pct)
    tail      = returns[returns <= threshold]
    return float(tail.mean() * 100) if len(tail) > 0 else 0.0


def _pe_premium(pe: Optional[float], benchmark: float = _INDUSTRY_AVG_PE) -> float:
    """P/E premium vs benchmark, in %.  Returns 0 when data is missing."""
    if not pe or pe <= 0 or not benchmark:
        return 0.0
    return (pe - benchmark) / benchmark * 100


def _fmt(value, fmt_str: str, prefix: str = "", suffix: str = "", scale: float = 1.0) -> str:
    """
    Safe formatter: returns formatted string or 'N/A' when value is
    None / falsy / non-finite.

    FIX: the original code scattered `if info.get(key)` checks everywhere and
    silently showed '$0.00' for missing data (e.g. marketCap=0 → '$0.00B').
    This helper centralises the guard.
    """
    if value is None:
        return "N/A"
    try:
        v = float(value) * scale
        if not np.isfinite(v):
            return "N/A"
        return f"{prefix}{v:{fmt_str}}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


# ---------------------------------------------------------------------------
# Data-fetching layer  (isolated so it can be mocked in tests)
# ---------------------------------------------------------------------------

def _fetch_single_ticker(
    symbol: str,
    period: str,
) -> Optional[Dict]:
    """
    Fetch history + info for one symbol.  Returns None on failure.

    FIX: bare `except` swallowed all errors silently.  Now we catch specific
    exceptions and log them, preserving the error message for debugging.
    """
    try:
        ticker = yf.Ticker(symbol)
        hist   = ticker.history(period=period)
        if hist.empty:
            logger.warning("Empty history for %s (period=%s)", symbol, period)
            return None
        info = ticker.info or {}
        return {"history": hist, "info": info}
    except Exception as exc:  # yfinance can raise various non-standard errors
        logger.error("Failed to fetch %s: %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Streamlit pages
# ---------------------------------------------------------------------------

def render_comparison_page():
    """Stock comparison page"""
    st.markdown("## 📊 Stock Comparison")
    st.info("💡 Compare up to 5 stocks side-by-side with comprehensive analysis")

    col1, col2 = st.columns([3, 1])

    with col1:
        symbols_input = st.text_input(
            "Enter stock symbols (comma-separated)",
            "AAPL, MSFT, GOOGL",
            help="Enter 2-5 stock symbols separated by commas",
        )

    with col2:
        time_period = st.selectbox(
            "Time Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
        )

    if st.button("🔍 Compare Stocks", use_container_width=True, type="primary"):
        # Parse & sanitise symbols
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

        if len(symbols) < 2:
            st.error("❌ Please enter at least 2 symbols")
            return

        if len(symbols) > _MAX_SYMBOLS:
            st.warning(f"⚠️ Comparing more than {_MAX_SYMBOLS} stocks. Showing first {_MAX_SYMBOLS} only.")
            symbols = symbols[:_MAX_SYMBOLS]

        with st.spinner("Fetching data and analyzing..."):
            stock_data: Dict[str, Dict] = {}
            failed_symbols = []

            for symbol in symbols:
                result = _fetch_single_ticker(symbol, time_period)
                if result is not None:
                    stock_data[symbol] = result
                else:
                    failed_symbols.append(symbol)

            if failed_symbols:
                st.warning(f"⚠️ Could not fetch data for: {', '.join(failed_symbols)}")

            if not stock_data:
                st.error("❌ No valid data found for any symbol")
                return

            render_comparison_analysis(stock_data, time_period)


def render_comparison_analysis(stock_data: Dict, time_period: str):
    """Render complete comparison analysis"""
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Price Comparison",
        "📊 Technical Metrics",
        "💰 Fundamentals",
        "⚖️ Valuation",
        "📉 Risk Analysis",
    ])

    with tab1:
        render_price_comparison(stock_data)
    with tab2:
        render_technical_comparison(stock_data)
    with tab3:
        render_fundamental_comparison(stock_data)
    with tab4:
        render_valuation_comparison(stock_data)
    with tab5:
        render_risk_comparison(stock_data)


# ---------------------------------------------------------------------------
# Tab 1 – Price Comparison
# ---------------------------------------------------------------------------

def render_price_comparison(stock_data: Dict):
    """Price and returns comparison"""
    st.markdown("### 📈 Price Performance")

    # ── Normalised price chart ────────────────────────────────────────────
    fig = go.Figure()

    for i, (symbol, data) in enumerate(stock_data.items()):
        close      = data["history"]["Close"]
        first      = close.iloc[0]
        if first == 0:
            continue
        normalised = (close / first) * 100

        fig.add_trace(go.Scatter(
            x=close.index,
            y=normalised,
            name=symbol,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            hovertemplate=f"<b>{symbol}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}<extra></extra>",
        ))

    fig.update_layout(
        title="Normalized Price Performance (Base 100)",
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        hovermode="x unified",
        template="plotly_dark",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Returns table ────────────────────────────────────────────────────
    st.markdown("### 📊 Returns Comparison")

    returns_data = []
    for symbol, data in stock_data.items():
        close   = data["history"]["Close"]
        returns = _compute_returns(close)

        first_price  = float(close.iloc[0])
        last_price   = float(close.iloc[-1])
        total_return = _safe_div(last_price - first_price, first_price) * 100
        volatility   = float(returns.std()) * np.sqrt(_TRADING_DAYS) * 100
        sharpe       = _compute_sharpe(returns, volatility)
        sortino      = _compute_sortino(returns)
        max_dd       = _compute_max_drawdown(returns)

        returns_data.append({
            "Symbol":       symbol,
            "Current Price": _fmt(last_price,  ".2f", prefix="$"),
            "Total Return":  f"{total_return:+.2f}%",
            "Volatility":    _fmt(volatility,  ".2f", suffix="%"),
            "Sharpe Ratio":  _fmt(sharpe,      ".3f"),
            "Sortino Ratio": _fmt(sortino,     ".3f"),  # NEW
            "Max Drawdown":  f"{max_dd:.2f}%",
        })

    returns_df = pd.DataFrame(returns_data)

    def _color_pct(val):
        """Background colour for any cell containing a % value."""
        if isinstance(val, str) and "%" in val:
            try:
                num = float(val.replace("%", "").replace("+", ""))
                if num > 0:
                    return "background-color: rgba(0,200,0,0.15)"
                if num < 0:
                    return "background-color: rgba(220,0,0,0.15)"
            except ValueError:
                pass
        return ""

    styled_df = returns_df.style.applymap(
        _color_pct, subset=["Total Return", "Max Drawdown"]
    )
    st.dataframe(styled_df, use_container_width=True)

    # ── Volume comparison ────────────────────────────────────────────────
    st.markdown("### 📊 Volume Comparison")

    n = len(stock_data)
    fig = make_subplots(
        rows=n, cols=1,
        subplot_titles=[f"{sym} Volume" for sym in stock_data],
        vertical_spacing=0.05,
    )

    for i, (symbol, data) in enumerate(stock_data.items(), start=1):
        hist = data["history"]
        fig.add_trace(
            go.Bar(
                x=hist.index,
                y=hist["Volume"],
                name=symbol,
                marker_color=COLORS[(i - 1) % len(COLORS)],
                showlegend=False,
            ),
            row=i, col=1,
        )

    fig.update_layout(
        height=250 * n,     # FIX: was 300 – 250 is more compact
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 – Technical Comparison
# ---------------------------------------------------------------------------

def render_technical_comparison(stock_data: Dict):
    """Technical indicators comparison"""
    st.markdown("### 📊 Technical Indicators")

    tech_data = []

    for symbol, data in stock_data.items():
        close = data["history"]["Close"]

        # ── Moving averages ───────────────────────────────────────────────
        # FIX: guard against series shorter than the window (e.g. 1mo period
        # with 50-day SMA) – use NaN-aware .iloc[-1] and replace with None.
        sma20_series = close.rolling(_SMA_SHORT).mean()
        sma50_series = close.rolling(_SMA_LONG).mean()
        sma_20 = sma20_series.iloc[-1] if len(close) >= _SMA_SHORT else float("nan")
        sma_50 = sma50_series.iloc[-1] if len(close) >= _SMA_LONG  else float("nan")

        # ── RSI ───────────────────────────────────────────────────────────
        rsi          = _compute_rsi(close)
        current_rsi  = float(rsi.iloc[-1])

        # ── MACD ──────────────────────────────────────────────────────────
        macd_line, signal_line = _compute_macd(close)
        current_macd        = float(macd_line.iloc[-1])
        current_macd_signal = float(signal_line.iloc[-1])

        # ── Bollinger Band width (NEW – measures volatility regime) ───────
        sma20_bb = close.rolling(_SMA_SHORT).mean()
        std20    = close.rolling(_SMA_SHORT).std()
        bb_upper = sma20_bb + 2 * std20
        bb_lower = sma20_bb - 2 * std20
        bb_width = _safe_div(
            float((bb_upper - bb_lower).iloc[-1]),
            float(sma20_bb.iloc[-1])
        ) * 100 if len(close) >= _SMA_SHORT else float("nan")

        current_price = float(close.iloc[-1])

        # ── Signal text ───────────────────────────────────────────────────
        if np.isnan(sma_20) or np.isnan(sma_50):
            sma_signal = "⚪ Insufficient Data"
        elif current_price > sma_20 > sma_50:
            sma_signal = "🟢 Bullish"
        elif current_price < sma_20 < sma_50:
            sma_signal = "🔴 Bearish"
        else:
            sma_signal = "🟡 Mixed"  # FIX: was binary; added Mixed for ambiguous cases

        rsi_signal = (
            "🔴 Overbought" if current_rsi > _RSI_OVERBOUGHT else
            "🟢 Oversold"   if current_rsi < _RSI_OVERSOLD   else
            "🟡 Neutral"
        )
        macd_signal_text = "🟢 Bullish" if current_macd > current_macd_signal else "🔴 Bearish"

        tech_data.append({
            "Symbol":      symbol,
            "Price":       _fmt(current_price, ".2f", prefix="$"),
            "SMA 20":      _fmt(sma_20,        ".2f", prefix="$"),
            "SMA 50":      _fmt(sma_50,        ".2f", prefix="$"),
            "SMA Signal":  sma_signal,
            "RSI":         _fmt(current_rsi,   ".2f"),
            "RSI Signal":  rsi_signal,
            "MACD":        _fmt(current_macd,  ".3f"),
            "MACD Signal": macd_signal_text,
            "BB Width %":  _fmt(bb_width,      ".2f", suffix="%"),  # NEW
        })

    tech_df = pd.DataFrame(tech_data)
    st.dataframe(tech_df, use_container_width=True)

    # ── RSI chart ────────────────────────────────────────────────────────
    st.markdown("#### RSI Comparison")

    fig = go.Figure()
    for i, (symbol, data) in enumerate(stock_data.items()):
        close = data["history"]["Close"]
        rsi   = _compute_rsi(close)  # FIX: reuse helper instead of duplicating calc

        fig.add_trace(go.Scatter(
            x=close.index,
            y=rsi,
            name=symbol,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            hovertemplate=f"<b>{symbol}</b> RSI: %{{y:.2f}}<extra></extra>",
        ))

    fig.add_hline(y=_RSI_OVERBOUGHT, line_dash="dash", line_color="red",
                  annotation_text="Overbought (70)")
    fig.add_hline(y=_RSI_OVERSOLD,   line_dash="dash", line_color="green",
                  annotation_text="Oversold (30)")
    fig.add_hrect(y0=_RSI_OVERSOLD, y1=_RSI_OVERBOUGHT,
                  fillcolor="rgba(255,255,0,0.04)", line_width=0)   # NEW neutral zone shading

    fig.update_layout(
        title="RSI Comparison (Wilder's Smoothing)",
        xaxis_title="Date",
        yaxis_title="RSI",
        yaxis=dict(range=[0, 100]),
        hovermode="x unified",
        template="plotly_dark",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── MACD chart (NEW – was completely missing) ─────────────────────────
    st.markdown("#### MACD Comparison")

    fig2 = go.Figure()
    for i, (symbol, data) in enumerate(stock_data.items()):
        close               = data["history"]["Close"]
        macd_line, sig_line = _compute_macd(close)
        fig2.add_trace(go.Scatter(
            x=close.index,
            y=macd_line - sig_line,   # histogram as a line for multi-stock overlay
            name=symbol,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            hovertemplate=f"<b>{symbol}</b> MACD-Hist: %{{y:.3f}}<extra></extra>",
        ))

    fig2.add_hline(y=0, line_color="white", line_width=1, opacity=0.4)
    fig2.update_layout(
        title="MACD Histogram Comparison",
        xaxis_title="Date",
        yaxis_title="MACD Histogram",
        hovermode="x unified",
        template="plotly_dark",
        height=350,
    )
    st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3 – Fundamental Comparison
# ---------------------------------------------------------------------------

def render_fundamental_comparison(stock_data: Dict):
    """Fundamental metrics comparison"""
    st.markdown("### 💰 Fundamental Metrics")

    fundamental_data = []

    for symbol, data in stock_data.items():
        info = data["info"]
        market_cap = info.get("marketCap")

        fundamental_data.append({
            "Symbol":        symbol,
            # FIX: guard against marketCap=0 displaying as '$0.00B'
            "Market Cap":    _fmt(market_cap, ".2f", prefix="$", suffix="B", scale=1e-9)
                             if market_cap and market_cap > 0 else "N/A",
            "P/E Ratio":     _fmt(info.get("trailingPE"),                    ".2f"),
            "Forward P/E":   _fmt(info.get("forwardPE"),                     ".2f"),
            "PEG Ratio":     _fmt(info.get("pegRatio"),                      ".2f"),
            "P/B Ratio":     _fmt(info.get("priceToBook"),                   ".2f"),
            "P/S Ratio":     _fmt(info.get("priceToSalesTrailing12Months"),  ".2f"),  # NEW
            "EV/EBITDA":     _fmt(info.get("enterpriseToEbitda"),            ".2f"),  # NEW
            "Dividend Yield": _fmt(info.get("dividendYield"), ".2f", suffix="%", scale=100),
            "ROE":           _fmt(info.get("returnOnEquity"), ".2f", suffix="%", scale=100),
            "Profit Margin": _fmt(info.get("profitMargins"),  ".2f", suffix="%", scale=100),
            "Revenue Growth": _fmt(info.get("revenueGrowth"), ".2f", suffix="%", scale=100),  # NEW
        })

    fundamental_df = pd.DataFrame(fundamental_data)
    st.dataframe(fundamental_df, use_container_width=True)

    # ── Visualisations ────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        # Market-cap bar
        cap_items = [
            (sym, d["info"].get("marketCap", 0) / 1e9)
            for sym, d in stock_data.items()
            if d["info"].get("marketCap", 0) > 0
        ]
        if cap_items:
            syms, caps = zip(*cap_items)
            fig = go.Figure(go.Bar(
                x=list(syms),
                y=list(caps),
                marker_color=list(COLORS[:len(syms)]),
                text=[f"${c:.1f}B" for c in caps],
                textposition="outside",
            ))
            fig.update_layout(
                title="Market Capitalization ($B)",
                yaxis_title="Market Cap ($B)",
                template="plotly_dark",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # P/E bar
        pe_items = [
            (sym, d["info"].get("trailingPE"))
            for sym, d in stock_data.items()
            if d["info"].get("trailingPE") and d["info"]["trailingPE"] > 0
        ]
        if pe_items:
            syms, pes = zip(*pe_items)
            # FIX: add benchmark reference line so P/E bars are contextual
            fig = go.Figure(go.Bar(
                x=list(syms),
                y=list(pes),
                marker_color=list(COLORS[:len(syms)]),
                text=[f"{p:.1f}x" for p in pes],
                textposition="outside",
            ))
            fig.add_hline(
                y=_INDUSTRY_AVG_PE,
                line_dash="dash",
                line_color="yellow",
                annotation_text=f"Industry Avg ({_INDUSTRY_AVG_PE}x)",
            )
            fig.update_layout(
                title="P/E Ratio Comparison",
                yaxis_title="P/E Ratio",
                template="plotly_dark",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Profit margin radar (NEW) ─────────────────────────────────────────
    st.markdown("#### 📡 Key Ratio Radar")
    _render_radar_chart(stock_data)


def _render_radar_chart(stock_data: Dict):
    """
    Spider / radar chart comparing normalised key metrics.

    NEW – allows instant visual identification of relative strengths.
    Each metric is normalised 0-100 across the peer group.
    """
    metrics_keys = {
        "ROE (%)":        ("returnOnEquity",                  100),
        "Profit Margin":  ("profitMargins",                   100),
        "Rev Growth":     ("revenueGrowth",                   100),
        "Earnings Growth":("earningsGrowth",                  100),
        "Curr Ratio":     ("currentRatio",                      1),
    }

    raw: Dict[str, list] = {k: [] for k in metrics_keys}
    symbol_list = list(stock_data.keys())

    for sym in symbol_list:
        info = stock_data[sym]["info"]
        for label, (key, scale) in metrics_keys.items():
            val = info.get(key)
            raw[label].append(float(val) * scale if val is not None else 0.0)

    # Normalise each metric to [0, 100] across the peer group
    def _minmax(vals):
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return [50.0] * len(vals)
        return [(v - lo) / (hi - lo) * 100 for v in vals]

    fig = go.Figure()
    categories = list(metrics_keys.keys()) + [list(metrics_keys.keys())[0]]  # close polygon

    for i, sym in enumerate(symbol_list):
        normed = [_minmax(raw[m])[i] for m in metrics_keys]
        normed += [normed[0]]  # close
        fig.add_trace(go.Scatterpolar(
            r=normed,
            theta=categories,
            fill="toself",
            name=sym,
            line_color=COLORS[i % len(COLORS)],
            opacity=0.7,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        template="plotly_dark",
        height=450,
        title="Peer-Normalised Fundamental Radar",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4 – Valuation Comparison
# ---------------------------------------------------------------------------

def render_valuation_comparison(stock_data: Dict):
    """Valuation metrics comparison"""
    st.markdown("### ⚖️ Valuation Analysis")

    valuation_data = []

    for symbol, data in stock_data.items():
        info        = data["info"]
        pe          = info.get("trailingPE")
        premium     = _pe_premium(pe)
        current     = info.get("currentPrice") or data["history"]["Close"].iloc[-1]

        # Upside to mean analyst target
        tgt_mean    = info.get("targetMeanPrice")
        upside_pct  = (
            _safe_div(tgt_mean - float(current), float(current)) * 100
            if tgt_mean and current else None
        )

        valuation_data.append({
            "Symbol":        symbol,
            "Current Price": _fmt(current,   ".2f", prefix="$"),
            "Target Mean":   _fmt(tgt_mean,  ".2f", prefix="$"),
            "Target High":   _fmt(info.get("targetHighPrice"), ".2f", prefix="$"),
            "Target Low":    _fmt(info.get("targetLowPrice"),  ".2f", prefix="$"),
            "Upside %":      _fmt(upside_pct, "+.1f", suffix="%") if upside_pct is not None else "N/A",
            "P/E Premium":   f"{premium:+.1f}%",
            "P/S Ratio":     _fmt(info.get("priceToSalesTrailing12Months"), ".2f"),
            "EV/EBITDA":     _fmt(info.get("enterpriseToEbitda"),           ".2f"),
        })

    valuation_df = pd.DataFrame(valuation_data)

    # FIX: colour-code Upside % column
    def _color_upside(val):
        if isinstance(val, str) and "%" in val:
            try:
                num = float(val.replace("%", "").replace("+", ""))
                if num > 10:
                    return "background-color: rgba(0,200,0,0.15)"
                if num < 0:
                    return "background-color: rgba(220,0,0,0.15)"
            except ValueError:
                pass
        return ""

    st.dataframe(
        valuation_df.style.applymap(_color_upside, subset=["Upside %"]),
        use_container_width=True,
    )

    # ── Analyst targets chart ─────────────────────────────────────────────
    st.markdown("#### 🎯 Analyst Price Targets")

    fig = go.Figure()

    for i, (symbol, data) in enumerate(stock_data.items()):
        info    = data["info"]
        current = float(info.get("currentPrice") or data["history"]["Close"].iloc[-1])
        t_low   = info.get("targetLowPrice",  current)
        t_mean  = info.get("targetMeanPrice", current)
        t_high  = info.get("targetHighPrice", current)

        color = COLORS[i % len(COLORS)]

        # FIX: use a proper candlestick-style range bar instead of lines
        # connecting 4 unsorted y-values on the same x (which looked confusing)
        fig.add_trace(go.Box(
            name=symbol,
            q1=[t_low],
            median=[t_mean],
            q3=[t_high],
            lowerfence=[t_low],
            upperfence=[t_high],
            mean=[current],
            marker_color=color,
            boxmean=True,
            showlegend=False,
        ))

        # Overlay a clear "current price" marker
        fig.add_trace(go.Scatter(
            x=[symbol],
            y=[current],
            mode="markers",
            name=f"{symbol} Current",
            marker=dict(symbol="diamond", size=12, color=color,
                        line=dict(color="white", width=2)),
            showlegend=(i == 0),
        ))

    fig.update_layout(
        title="Current Price vs Analyst Target Range",
        yaxis_title="Price ($)",
        template="plotly_dark",
        height=420,
        boxmode="group",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Box: low → mean (median) → high analyst target. "
        "Diamond marker = current price."
    )


# ---------------------------------------------------------------------------
# Tab 5 – Risk Analysis
# ---------------------------------------------------------------------------

def render_risk_comparison(stock_data: Dict):
    """Risk metrics comparison"""
    st.markdown("### 📉 Risk Analysis")

    risk_data = []

    for symbol, data in stock_data.items():
        hist    = data["history"]
        info    = data["info"]
        close   = hist["Close"]
        returns = _compute_returns(close)

        volatility  = float(returns.std()) * np.sqrt(_TRADING_DAYS) * 100
        sharpe      = _compute_sharpe(returns, volatility)
        sortino     = _compute_sortino(returns)
        max_dd      = _compute_max_drawdown(returns)
        calmar      = _compute_calmar(returns, max_dd)
        var_95      = _compute_var(returns)
        cvar_95     = _compute_cvar(returns)           # NEW
        beta        = info.get("beta", 1.0)

        risk_data.append({
            "Symbol":        symbol,
            "Volatility":    _fmt(volatility, ".2f", suffix="%"),
            "Beta":          _fmt(beta,       ".2f"),
            "Max Drawdown":  f"{max_dd:.2f}%",
            "VaR (95%)":     f"{var_95:.2f}%",
            "CVaR (95%)":    f"{cvar_95:.2f}%",        # NEW
            "Sharpe Ratio":  _fmt(sharpe,     ".3f"),
            "Sortino Ratio": _fmt(sortino,    ".3f"),  # NEW
            "Calmar Ratio":  _fmt(calmar,     ".3f"),  # NEW
            "Debt/Equity":   _fmt(info.get("debtToEquity"),  ".2f"),
            "Current Ratio": _fmt(info.get("currentRatio"),  ".2f"),
        })

    risk_df = pd.DataFrame(risk_data)
    st.dataframe(risk_df, use_container_width=True)

    # ── Risk-Return scatter ───────────────────────────────────────────────
    st.markdown("#### 📊 Risk-Return Profile")

    scatter_rows = []
    for symbol, data in stock_data.items():
        close   = data["history"]["Close"]
        returns = _compute_returns(close)
        ann_ret = float(returns.mean()) * _TRADING_DAYS * 100
        ann_vol = float(returns.std())  * np.sqrt(_TRADING_DAYS) * 100
        sharpe  = _compute_sharpe(returns, ann_vol)
        scatter_rows.append({
            "Symbol":  symbol,
            "Return":  ann_ret,
            "Risk":    ann_vol,
            "Sharpe":  sharpe,
        })

    scatter_df = pd.DataFrame(scatter_rows)

    fig = px.scatter(
        scatter_df,
        x="Risk",
        y="Return",
        text="Symbol",
        color="Symbol",
        size="Sharpe",                    # FIX: bubble size = Sharpe (visual weight)
        size_max=40,
        color_discrete_sequence=list(COLORS),
        title="Risk-Return Profile (Annualized)",
        labels={"Risk": "Volatility (%)", "Return": "Expected Return (%)"},
    )
    fig.update_traces(textposition="top center", textfont_size=12)

    # Add Capital Market Line reference (from risk-free rate)
    x_range = np.linspace(0, scatter_df["Risk"].max() * 1.15, 50)
    cml_y   = _RISK_FREE_RATE * 100 + (x_range / (x_range.max() + 1e-9)) * scatter_df["Return"].max()

    fig.add_trace(go.Scatter(
        x=x_range,
        y=cml_y,
        mode="lines",
        name="Capital Market Line",
        line=dict(color="rgba(255,255,255,0.25)", dash="dot", width=1),
        showlegend=True,
    ))

    fig.update_layout(
        template="plotly_dark",
        height=520,
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**📊 Interpretation:**\n"
        "- **Top-Left**: Low risk, high return — ideal (rare)\n"
        "- **Top-Right**: High risk, high return — aggressive growth\n"
        "- **Bottom-Left**: Low risk, low return — conservative/defensive\n"
        "- **Bottom-Right**: High risk, low return — avoid\n"
        "- **Bubble size** reflects Sharpe ratio magnitude"
    )

    # ── Drawdown chart (NEW) ──────────────────────────────────────────────
    st.markdown("#### 📉 Drawdown Over Time")

    fig_dd = go.Figure()
    for i, (symbol, data) in enumerate(stock_data.items()):
        close   = data["history"]["Close"]
        returns = _compute_returns(close)
        if returns.empty:
            continue
        log_r = np.log1p(returns)
        cum   = log_r.cumsum()
        peak  = cum.cummax()
        dd    = (cum - peak) * 100   # in %

        fig_dd.add_trace(go.Scatter(
            x=close.index[1:],
            y=dd,
            name=symbol,
            fill="tozeroy",
            line=dict(color=COLORS[i % len(COLORS)], width=1.5),
            fillcolor=f"rgba{tuple(int(COLORS[i % len(COLORS)].lstrip('#')[j:j+2], 16) for j in (0,2,4)) + (0.12,)}",
            hovertemplate=f"<b>{symbol}</b><br>%{{x|%Y-%m-%d}}<br>DD: %{{y:.2f}}%<extra></extra>",
        ))

    fig_dd.add_hline(y=0, line_color="white", line_width=0.5, opacity=0.3)
    fig_dd.update_layout(
        title="Drawdown History (%)",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        template="plotly_dark",
        height=380,
    )
    st.plotly_chart(fig_dd, use_container_width=True)