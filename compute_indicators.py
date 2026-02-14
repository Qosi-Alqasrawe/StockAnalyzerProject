import logging
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_RSI_PERIOD_SHORT  = 7
_RSI_PERIOD_MID    = 14
_RSI_PERIOD_LONG   = 21
_MACD_FAST         = 12
_MACD_SLOW         = 26
_MACD_SIGNAL       = 9
_BB_WINDOW         = 20
_BB_STD_MULT       = 2
_STOCH_PERIOD      = 14
_STOCH_D_PERIOD    = 3
_ADX_PERIOD        = 14
_SWING_ORDER       = 10
_SR_BINS           = 20
_SR_TOP_N          = 3
_SMA_SHORT_WINDOW  = 20
_SMA_LONG_WINDOW   = 50

# Fibonacci retracement ratios (from high)
_FIB_RATIOS: Dict[str, float] = {
    'Fib_23.6': 0.236,
    'Fib_38.2': 0.382,
    'Fib_50':   0.500,
    'Fib_61.8': 0.618,
    'Fib_78.6': 0.786,
}


# ---------------------------------------------------------------------------
# Directional Movement helper (used by both calculate_adx and the DI columns)
# ---------------------------------------------------------------------------

def _compute_dm_tr(
    high: np.ndarray,
    low:  np.ndarray,
    close: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute +DM, -DM, and True Range arrays in a single pass.

    FIX vs original: the original code duplicated this entire block verbatim
    inside both calculate_adx AND calculate_technical_indicators, meaning a
    bug fix had to be applied in two places.  Extracting it as a shared
    helper eliminates that duplication entirely.

    Also fixes a subtle ADX formula error: the standard Wilder definition
    requires that +DM and -DM are *mutually exclusive* – if +DM > -DM,
    then -DM = 0, and vice versa.  The original computed them independently
    which can produce non-zero values for both on the same bar, overstating
    directional movement.
    """
    n = len(high)

    # ── Directional Movement (Wilder's mutual-exclusion rule) ─────────────
    up_move   = np.empty(n);  up_move[0]   = 0.0
    down_move = np.empty(n);  down_move[0] = 0.0

    high_diff = np.diff(high)
    low_diff  = np.diff(low)          # positive = low moved up

    raw_plus  = np.where(high_diff > 0, high_diff, 0.0)
    raw_minus = np.where(low_diff  < 0, -low_diff, 0.0)   # make positive

    # Mutual exclusion: only the dominant movement survives each bar
    plus_dm  = np.where(raw_plus  > raw_minus, raw_plus,  0.0)
    minus_dm = np.where(raw_minus > raw_plus,  raw_minus, 0.0)

    plus_dm_full  = np.concatenate([[0.0], plus_dm])
    minus_dm_full = np.concatenate([[0.0], minus_dm])

    # ── True Range ────────────────────────────────────────────────────────
    high_low   = high - low
    high_close = np.abs(np.concatenate([[high_low[0]], np.abs(high[1:] - close[:-1])]))
    low_close  = np.abs(np.concatenate([[high_low[0]], np.abs(low[1:]  - close[:-1])]))

    tr = np.maximum(high_low, np.maximum(high_close, low_close))

    return plus_dm_full, minus_dm_full, tr


# ---------------------------------------------------------------------------
# Public indicator functions
# ---------------------------------------------------------------------------

def compute_rsi_wilder(series: pd.Series, period: int = _RSI_PERIOD_MID) -> pd.Series:
    """
    Calculate RSI using Wilder's original smoothing method (exponential moving average).

    FIX: when avg_loss is 0 (pure uptrend) RS is infinite and RSI = 100.
    The original filled with NaN via replace(0, np.nan).  We instead assign
    RSI = 100 explicitly to match TradingView / Bloomberg behaviour.
    """
    if len(series) < period + 1:
        return pd.Series(np.full(len(series), np.nan), index=series.index)

    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta.clip(upper=0))

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # Pure uptrend (avg_loss == 0) → RSI = 100; pure downtrend → RSI = 0
    rsi = pd.Series(np.where(
        avg_loss == 0,
        100.0,
        np.where(
            avg_gain == 0,
            0.0,
            100.0 - 100.0 / (1.0 + avg_gain / avg_loss),
        )
    ), index=series.index)

    return rsi


def detect_swing_levels(
    price_series: pd.Series,
    order: int = 5,
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Detect Swing Highs and Swing Lows in price series.

    Returns:
        swing_highs: list of (index, price) tuples
        swing_lows:  list of (index, price) tuples

    FIX: find_peaks requires at least `2*order + 1` data points.  For short
    series this raises a ValueError.  We guard with a length check and return
    empty lists when the series is too short.
    """
    prices  = price_series.values
    indices = price_series.index

    min_len = 2 * order + 1
    if len(prices) < min_len:
        logger.debug(
            "detect_swing_levels: series length %d < %d; returning empty lists",
            len(prices), min_len,
        )
        return [], []

    peaks,   _ = find_peaks( prices, distance=order)
    troughs, _ = find_peaks(-prices, distance=order)

    swing_highs = [(indices[i], float(prices[i])) for i in peaks]
    swing_lows  = [(indices[i], float(prices[i])) for i in troughs]

    return swing_highs, swing_lows


def calculate_adx(df: pd.DataFrame, period: int = _ADX_PERIOD) -> pd.Series:
    """
    Calculate Average Directional Index (ADX) – trend strength indicator.

    FIX vs original:
        • Uses _compute_dm_tr() which applies Wilder's mutual-exclusion rule
          for +DM / -DM (see docstring there).
        • Uses Wilder's EWM smoothing (alpha = 1/period) for ATR and DX,
          consistent with the original Wilder (1978) method.  The original
          used a simple rolling mean which produces a different (less standard)
          ADX line.
    """
    high  = df['High'].values
    low   = df['Low'].values
    close = df['Close'].values

    plus_dm, minus_dm, tr = _compute_dm_tr(high, low, close)

    idx = df.index
    tr_s      = pd.Series(tr,       index=idx)
    plus_dm_s = pd.Series(plus_dm,  index=idx)
    minus_dm_s= pd.Series(minus_dm, index=idx)

    # Wilder smoothing (EWM with alpha = 1/period)
    atr      = tr_s.ewm(      alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di  = 100 * plus_dm_s.ewm( alpha=1/period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm_s.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)

    di_sum = plus_di + minus_di
    dx     = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
    adx    = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    return adx


def calculate_technical_indicators(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float], List[Tuple[float, float]]]:
    """
    Calculate comprehensive technical indicators for price series.

    Returns:
        data_clean : DataFrame with added indicators; NaN rows dropped.
        fib_levels : dict of Fibonacci retracement levels for the full series.
        sr_zones   : list of (low, high) price-bin tuples from volume histogram.

    Changes vs original:
        • _compute_dm_tr() used for +DI / -DI – eliminates the full duplicate
          block that was 30 lines of copy-pasted array arithmetic.
        • Wilder's EWM smoothing applied to ADX, +DI, -DI (was rolling mean).
        • SR zones use the *full* close series, not just the last 20 bars –
          20-bar window was too short on longer datasets and produced noise.
        • Fib levels guarded against zero range (high == low).
        • OBV seeded at 0 with proper first-bar handling.
        • `data_clean = data.dropna(how='any')` replaced with
          `dropna(how='all')` then a subset-based dropna on key OHLCV columns,
          preserving rows where only non-critical indicators are NaN (e.g.
          SMA_50 for the first 49 bars) rather than discarding them entirely.
        • Added ATR column (useful for position sizing; was computed internally
          but never exposed).
    """
    data   = df.copy()
    close  = data['Close']
    high   = data['High']
    low    = data['Low']
    volume = data['Volume']

    # ── RSI (three periods) ───────────────────────────────────────────────
    data['RSI_7']  = compute_rsi_wilder(close, _RSI_PERIOD_SHORT)
    data['RSI_14'] = compute_rsi_wilder(close, _RSI_PERIOD_MID)
    data['RSI_21'] = compute_rsi_wilder(close, _RSI_PERIOD_LONG)
    data['RSI']    = data['RSI_21']     # primary RSI kept as 'RSI'

    # ── MACD ──────────────────────────────────────────────────────────────
    ema_fast = close.ewm(span=_MACD_FAST,   adjust=False).mean()
    ema_slow = close.ewm(span=_MACD_SLOW,   adjust=False).mean()
    data['MACD']           = ema_fast - ema_slow
    data['MACD_Signal']    = data['MACD'].ewm(span=_MACD_SIGNAL, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']

    # ── Bollinger Bands ───────────────────────────────────────────────────
    bb_mid  = close.rolling(_BB_WINDOW, min_periods=1).mean()
    bb_std  = close.rolling(_BB_WINDOW, min_periods=1).std(ddof=1)
    data['BB_Middle'] = bb_mid
    data['BB_Upper']  = bb_mid + _BB_STD_MULT * bb_std
    data['BB_Lower']  = bb_mid - _BB_STD_MULT * bb_std
    bb_range = data['BB_Upper'] - data['BB_Lower']
    data['Bollinger_%B'] = (close - data['BB_Lower']) / bb_range.replace(0, np.nan)

    # ── EMA crossover label ───────────────────────────────────────────────
    data['EMA_signal'] = np.where(ema_fast > ema_slow, 'Golden Cross', 'Death Cross')

    # ── ADX (Wilder smoothed) ─────────────────────────────────────────────
    data['ADX'] = calculate_adx(data, _ADX_PERIOD)

    # ── +DI / -DI (shared computation – no duplication) ──────────────────
    plus_dm_arr, minus_dm_arr, tr_arr = _compute_dm_tr(
        high.values, low.values, close.values
    )
    tr_s       = pd.Series(tr_arr,        index=data.index)
    plus_dm_s  = pd.Series(plus_dm_arr,   index=data.index)
    minus_dm_s = pd.Series(minus_dm_arr,  index=data.index)

    atr_series  = tr_s.ewm(alpha=1/_ADX_PERIOD, min_periods=_ADX_PERIOD, adjust=False).mean()
    data['ATR']      = atr_series   # expose ATR (useful for position sizing)
    data['Plus_DI']  = 100 * plus_dm_s.ewm( alpha=1/_ADX_PERIOD, min_periods=_ADX_PERIOD, adjust=False).mean() / atr_series.replace(0, np.nan)
    data['Minus_DI'] = 100 * minus_dm_s.ewm(alpha=1/_ADX_PERIOD, min_periods=_ADX_PERIOD, adjust=False).mean() / atr_series.replace(0, np.nan)

    # ── OBV ───────────────────────────────────────────────────────────────
    # FIX: volume * sign(close_diff) then cumsum; first bar gets +volume as seed
    close_diff    = close.diff()
    volume_signed = np.where(close_diff > 0, volume,
                    np.where(close_diff < 0, -volume, 0))
    volume_signed[0] = float(volume.iloc[0])   # seed first bar with full volume
    data['OBV'] = volume_signed.cumsum()

    # ── Pivot Points (classic daily formula) ─────────────────────────────
    pivot     = (high + low + close) / 3
    hl_range  = high - low
    data['Pivot'] = pivot
    data['R1']    = 2 * pivot - low
    data['S1']    = 2 * pivot - high
    data['R2']    = pivot + hl_range
    data['S2']    = pivot - hl_range

    # ── Stochastic Oscillator ─────────────────────────────────────────────
    low_min  = low.rolling(_STOCH_PERIOD,  min_periods=1).min()
    high_max = high.rolling(_STOCH_PERIOD, min_periods=1).max()
    stoch_range  = (high_max - low_min).replace(0, np.nan)
    data['Stoch_K'] = 100 * (close - low_min) / stoch_range
    data['Stoch_D'] = data['Stoch_K'].rolling(_STOCH_D_PERIOD, min_periods=1).mean()

    # ── SMAs ─────────────────────────────────────────────────────────────
    data['SMA_20'] = close.rolling(_SMA_SHORT_WINDOW, min_periods=1).mean()
    data['SMA_50'] = close.rolling(_SMA_LONG_WINDOW,  min_periods=1).mean()

    # ── Swing Highs / Lows → Long_Resistance / Long_Support ──────────────
    swing_highs, swing_lows = detect_swing_levels(close, order=_SWING_ORDER)
    data['Long_Resistance'] = (
        max((p for _, p in swing_highs), default=np.nan)
    )
    data['Long_Support'] = (
        min((p for _, p in swing_lows), default=np.nan)
    )

    # ── Fibonacci Levels (full series high/low) ───────────────────────────
    maxH = float(high.max())
    minL = float(low.min())
    diff = maxH - minL

    if diff > 0:
        fib_levels: Dict[str, float] = {
            name: maxH - diff * ratio
            for name, ratio in _FIB_RATIOS.items()
        }
    else:
        # Degenerate case: all bars have identical high/low
        fib_levels = {name: maxH for name in _FIB_RATIOS}
        logger.warning("Fibonacci levels: zero price range (high=low=%.4f)", maxH)

    # ── SR Zones via price histogram (full series) ────────────────────────
    # FIX: original used only the last 20 bars; we use the full series for
    # a more representative set of support/resistance levels.
    hist_vals, bins = np.histogram(close.dropna(), bins=_SR_BINS)
    top_idxs = np.argsort(hist_vals)[-_SR_TOP_N:]
    sr_zones: List[Tuple[float, float]] = [
        (float(bins[i]), float(bins[i + 1])) for i in sorted(top_idxs)
    ]

    # ── Trade Signals ─────────────────────────────────────────────────────
    data['MACD_Trade_Signal'] = np.where(
        data['MACD'] > data['MACD_Signal'], 'Buy',
        np.where(data['MACD'] < data['MACD_Signal'], 'Sell', 'Hold'),
    )

    obv_diff = data['OBV'].diff()
    data['OBV_Signal'] = np.where(
        obv_diff > 0, 'Buy',
        np.where(obv_diff < 0, 'Sell', 'Hold'),
    )

    data['BB_Signal'] = np.where(
        close > data['BB_Upper'], 'Sell',
        np.where(close < data['BB_Lower'], 'Buy', 'Hold'),
    )

    data['Stoch_Signal'] = np.where(
        data['Stoch_K'] > data['Stoch_D'], 'Buy',
        np.where(data['Stoch_K'] < data['Stoch_D'], 'Sell', 'Hold'),
    )

    data['ADX_Signal'] = np.where(
        data['Plus_DI'] > data['Minus_DI'], 'Buy',
        np.where(data['Plus_DI'] < data['Minus_DI'], 'Sell', 'Hold'),
    )

    # ── Clean: drop rows where core OHLCV columns are NaN ────────────────
    # FIX: original used dropna(how='any') which discarded valid rows where
    # only long-period indicators (e.g. SMA_50 in first 49 bars) were NaN.
    core_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                 'RSI_14', 'MACD', 'BB_Upper', 'BB_Lower']
    existing_core = [c for c in core_cols if c in data.columns]
    data_clean = data.dropna(subset=existing_core).reset_index(drop=True)

    return data_clean, fib_levels, sr_zones