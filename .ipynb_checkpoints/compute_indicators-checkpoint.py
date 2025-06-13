import pandas as pd
import numpy as np


def calculate_technical_indicators(data):
    """Calculate technical indicators for stock data."""
    df = data.copy()

    # 1) SMA 20, SMA 50
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Position_vs_SMA20'] = df.apply(
        lambda row: 'Above Average' if row['Close'] > row['SMA_20'] else 'Below Average',
        axis=1
    )
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Position_vs_SMA50'] = df.apply(
        lambda row: 'Above Average' if row['Close'] > row['SMA_50'] else 'Below Average',
        axis=1
    )

    # 2) EMA 12, EMA 26, Trend Signal
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['Trend_Signal'] = df.apply(
        lambda row: 'Bullish Signal' if row['EMA_12'] > row['EMA_26'] else 'Bearish Signal',
        axis=1
    )

    # 3) RSI
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    df['RSI'] = calculate_rsi(df['Close'])
    df['Market_Condition'] = df['RSI'].apply(
        lambda x: 'Overbought (Price Correction Down)'
        if x >= 70 else ('Oversold (Upward Bounce)' if x <= 30 else 'Normal Condition')
    )

    # 4) MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    # <- اضفنا هنا عمود MACD_Trade_Signal مع ترتيب المسافة البادئة بشكل صحيح ->
    df['MACD_Trade_Signal'] = df.apply(
        lambda row: 'Buy Signal' if row['MACD'] > row['MACD_Signal']
        else ('Sell Signal' if row['MACD'] < row['MACD_Signal'] else 'No Signal'),
        axis=1
    )

    # 5) Bollinger Bands
    sma = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma + (2 * std)
    df['BB_Middle'] = sma
    df['BB_Lower'] = sma - (2 * std)
    df['BB_Signal'] = df.apply(
        lambda row: 'Buy Signal' if row['Close'] < row['BB_Lower']
        else ('Sell Signal' if row['Close'] > row['BB_Upper'] else 'No Signal'),
        axis=1
    )

    # 6) ADX, +DI, -DI
    def calculate_adx_di(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        plus_dm = high.diff().where(high.diff() > 0, 0)
        minus_dm = -low.diff().where(low.diff() < 0, 0)
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=period).mean()
        return adx, plus_di, minus_di

    df['ADX'], df['Plus_DI'], df['Minus_DI'] = calculate_adx_di(
        df['High'], df['Low'], df['Close']
    )
    df['ADX_Signal'] = df.apply(
        lambda row: 'Strong Uptrend' if row['Plus_DI'] > row['Minus_DI'] and row['ADX'] > 20
        else ('Strong Downtrend' if row['Minus_DI'] > row['Plus_DI'] and row['ADX'] > 20 else 'Weak/Neutral'),
        axis=1
    )

    # 7) Stochastic Oscillator (K, D)
    lowest_low = df['Low'].rolling(window=14).min()
    highest_high = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    df['Stoch_Signal'] = df.apply(
        lambda row: 'Buy Signal' if row['Stoch_K'] < 20
        else ('Sell Signal' if row['Stoch_K'] > 80 else 'No Signal'),
        axis=1
    )

    # 8) On-Balance Volume (OBV)
    obv = [0]
    for i in range(1, len(df)):
        volume = df['Volume'].iloc[i] if pd.notna(df['Volume'].iloc[i]) else 0
        if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
            obv.append(obv[-1] + volume)
        elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
            obv.append(obv[-1] - volume)
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    df['OBV_Signal'] = df['OBV'].diff().apply(
        lambda x: 'Buy Signal' if x > 0 else ('Sell Signal' if x < 0 else 'No Signal')
    )

    # 9) Pivot Points
    def calculate_pivot_points(df_inner):
        df_inner['Pivot'] = (df_inner['High'].shift(1) + df_inner['Low'].shift(1) + df_inner['Close'].shift(1)) / 3
        df_inner['R1'] = 2 * df_inner['Pivot'] - df_inner['Low'].shift(1)
        df_inner['S1'] = 2 * df_inner['Pivot'] - df_inner['High'].shift(1)
        df_inner['R2'] = df_inner['Pivot'] + (df_inner['High'].shift(1) - df_inner['Low'].shift(1))
        df_inner['S2'] = df_inner['Pivot'] - (df_inner['High'].shift(1) - df_inner['Low'].shift(1))
        df_inner['Pivot_Signal'] = df_inner.apply(
            lambda row: 'Near Support S1' if abs(row['Close'] - row['S1']) < abs(row['Close'] - row['R1']) and row['Close'] > row['S2']
            else ('Near Resistance R1' if abs(row['Close'] - row['R1']) < abs(row['Close'] - row['S1']) and row['Close'] < row['R2'] else 'Neutral'),
            axis=1
        )
        return df_inner

    df = calculate_pivot_points(df)

    # 10) Fibonacci Levels
    def calculate_fibonacci_levels(df_inner, period=20):
        recent = df_inner.tail(period)
        high = recent['High'].max()
        low = recent['Low'].min()
        if high == low:
            return df_inner, {}
        diff = high - low
        fib_levels = {
            'Fib_23.6': high - 0.236 * diff,
            'Fib_38.2': high - 0.382 * diff,
            'Fib_50': high - 0.5 * diff,
            'Fib_61.8': high - 0.618 * diff,
            'Fib_78.6': high - 0.786 * diff
        }
        for level, value in fib_levels.items():
            df_inner[level] = value
        df_inner['Fib_Signal'] = df_inner.apply(
            lambda row: 'Near Fibonacci Support' if any(abs(row['Close'] - row[level]) < 0.02 * row['Close'] for level in fib_levels) and row['Close'] > low
            else ('Near Fibonacci Resistance' if any(abs(row['Close'] - row[level]) < 0.02 * row['Close'] for level in fib_levels) and row['Close'] < high else 'Neutral'),
            axis=1
        )
        return df_inner, fib_levels

    df, fib_levels = calculate_fibonacci_levels(df)

    # 11) Support/Resistance Zones
    def calculate_support_resistance_zones(df_inner, window=20, bin_size=0.02):
        prices = df_inner['Close'].tail(window)
        price_range = prices.max() - prices.min()
        if price_range == 0:
            df_inner['SR_Zone'] = 'Outside Zone'
            return df_inner, []
        bins = np.arange(prices.min(), prices.max(), price_range * bin_size)
        hist, edges = np.histogram(prices, bins=bins, density=True)
        zones = []
        for i in range(len(hist)):
            if hist[i] > np.percentile(hist, 75):
                zones.append((edges[i], edges[i + 1]))
        current_price = df_inner['Close'].iloc[-1]
        df_inner['SR_Zone'] = 'Outside Zone'
        for zone in zones:
            if zone[0] <= current_price <= zone[1]:
                df_inner.at[df_inner.index[-1], 'SR_Zone'] = f'Support/Resistance Zone ({zone[0]:.2f} - {zone[1]:.2f})'
        return df_inner, zones

    df, sr_zones = calculate_support_resistance_zones(df)

    return df, fib_levels, sr_zones
