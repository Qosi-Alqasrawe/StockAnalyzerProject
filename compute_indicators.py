import pandas as pd
import numpy as np
from scipy.signal import find_peaks


def compute_rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    """
    حساب RSI بطريقة Wilder الأصلية (تنعيم الأساس القديم).
    """
    delta = series.diff()
    gain = (
        delta.where(delta > 0, 0)
        .ewm(alpha=1/period, min_periods=period, adjust=False)
        .mean()
    )
    loss = (
        -delta.where(delta < 0, 0)
        .ewm(alpha=1/period, min_periods=period, adjust=False)
        .mean()
    )
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def detect_swing_levels(price_series: pd.Series, order: int = 5):
    """
    يكتشف Swing Highs و Swing Lows في سلسلة الأسعار.

    Returns:
    - swing_highs: list of tuples (index, price)
    - swing_lows: list of tuples (index, price)
    """
    peaks, _ = find_peaks(price_series.values, distance=order)
    troughs, _ = find_peaks(-price_series.values, distance=order)
    swing_highs = [(price_series.index[i], price_series.iat[i]) for i in peaks]
    swing_lows = [(price_series.index[i], price_series.iat[i]) for i in troughs]
    return swing_highs, swing_lows


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    حساب مؤشر قوة الاتجاه (ADX)
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    plus_dm = high.diff().clip(lower=0)
    minus_dm = low.diff().abs().clip(lower=0)
    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=period).mean()
    return adx


def calculate_technical_indicators(df: pd.DataFrame):
    """
    حساب المؤشرات الفنية الكاملة لسلسلة أسعار.

    Returns:
    - data_clean: DataFrame بعد إضافة المؤشرات وتنظيف القيم الفارغة
    - fib_levels: dict بمستويات فيبوناتشي للسلسلة كاملة
    - sr_zones: قائمة بمناطق الدعم/المقاومة المكتشفة عبر histogram
    """
    data = df.copy()

    # ----- RSI بفترات مختلفة -----
    data['RSI_7'] = compute_rsi_wilder(data['Close'], 7)
    data['RSI_14'] = compute_rsi_wilder(data['Close'], 14)
    data['RSI_21'] = compute_rsi_wilder(data['Close'], 21)
    data['RSI'] = data['RSI_21']

    # ----- MACD -----
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']

    # ----- Bollinger Bands -----
    window = 20
    data['BB_Middle'] = data['Close'].rolling(window).mean()
    std = data['Close'].rolling(window).std()
    data['BB_Upper'] = data['BB_Middle'] + 2 * std
    data['BB_Lower'] = data['BB_Middle'] - 2 * std
    data['Bollinger_%B'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])

    # ----- EMA Crossover Signal -----
    data['EMA_signal'] = np.where(data['EMA_12'] > data['EMA_26'], 'Golden Cross', 'Death Cross')
    data.drop(['EMA_12', 'EMA_26'], axis=1, inplace=True)

    # ----- ADX -----
    data['ADX'] = calculate_adx(data)

    # ----- OBV -----
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'].iat[i] > data['Close'].iat[i-1]:
            obv.append(obv[-1] + data['Volume'].iat[i])
        elif data['Close'].iat[i] < data['Close'].iat[i-1]:
            obv.append(obv[-1] - data['Volume'].iat[i])
        else:
            obv.append(obv[-1])
    data['OBV'] = obv

    # ----- Pivot Points -----
    pivot = (data['High'] + data['Low'] + data['Close']) / 3
    data['Pivot'] = pivot
    data['R1'] = 2 * pivot - data['Low']
    data['S1'] = 2 * pivot - data['High']
    data['R2'] = pivot + (data['High'] - data['Low'])
    data['S2'] = pivot - (data['High'] - data['Low'])

    # ----- SR Zones -----
    prices_hist = data['Close'].tail(window)
    hist, bins = np.histogram(prices_hist, bins=20)
    top_idxs = np.argsort(hist)[-3:]
    sr_zones = [(bins[i], bins[i+1]) for i in sorted(top_idxs)]

    # ----- Swing High/Low -----
    swing_highs, swing_lows = detect_swing_levels(data['Close'], order=10)
    data['Long_Resistance'] = max((price for _, price in swing_highs), default=np.nan)
    data['Long_Support'] = min((price for _, price in swing_lows), default=np.nan)

    # ----- Fibonacci Levels -----
    maxH, minL = data['High'].max(), data['Low'].min()
    diff = maxH - minL
    fib_levels = {
        'Fib_23.6': maxH - diff * 0.236,
        'Fib_38.2': maxH - diff * 0.382,
        'Fib_50': maxH - diff * 0.5,
        'Fib_61.8': maxH - diff * 0.618,
        'Fib_78.6': maxH - diff * 0.786
    }

    # ----- Signals -----
    data['MACD_Trade_Signal'] = np.where(data['MACD'] > data['MACD_Signal'], 'Buy',
                                       np.where(data['MACD'] < data['MACD_Signal'], 'Sell', 'Hold'))
    data['OBV_Signal'] = np.where(data['OBV'].diff() > 0, 'Buy',
                                  np.where(data['OBV'].diff() < 0, 'Sell', 'Hold'))
    data['BB_Signal'] = np.where(data['Close'] > data['BB_Upper'], 'Sell',
                                  np.where(data['Close'] < data['BB_Lower'], 'Buy', 'Hold'))
    low_min = data['Low'].rolling(14).min()
    high_max = data['High'].rolling(14).max()
    data['Stoch_K'] = 100 * (data['Close'] - low_min) / (high_max - low_min)
    data['Stoch_D'] = data['Stoch_K'].rolling(3).mean()
    data['Stoch_Signal'] = np.where(data['Stoch_K'] > data['Stoch_D'], 'Buy',
                                    np.where(data['Stoch_K'] < data['Stoch_D'], 'Sell', 'Hold'))
    plus_dm = data['High'].diff().clip(lower=0)
    minus_dm = data['Low'].diff().abs().clip(lower=0)
    tr1 = (data['High'] - data['Low']).abs()
    tr2 = (data['High'] - data['Close'].shift()).abs()
    tr3 = (data['Low'] - data['Close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    data['Plus_DI'] = 100 * (plus_dm.rolling(14).mean() / atr)
    data['Minus_DI'] = 100 * (minus_dm.rolling(14).mean() / atr)
    data['ADX_Signal'] = np.where(data['Plus_DI'] > data['Minus_DI'], 'Buy',
                                  np.where(data['Plus_DI'] < data['Minus_DI'], 'Sell', 'Hold'))

    # ----- SMAs -----
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    # ----- Clean and return -----
    data_clean = data.dropna(how='any').reset_index(drop=True)
    return data_clean, fib_levels, sr_zones
