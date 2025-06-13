# import_fetch_technical.py
import pandas as pd
import yfinance as yf

def fetch_technical_data(symbol, start, end):
    """Fetch historical stock data using yfinance."""
    try:
        data = yf.download(symbol, start=start, end=end)
        if data.empty:
            raise ValueError(f"No data available for stock {symbol}. Check symbol, date range, or network connection.")
        data = data.reset_index()
        # إعادة تسمية الأعمدة إن لزم لتحويلها: 
        if len(data.columns) == 6:
            data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        elif len(data.columns) == 7:
            data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        else:
            raise ValueError(f"Unexpected data format for {symbol}: {len(data.columns)} columns")
        data = data.dropna().reset_index(drop=True)
        return data
    except Exception as e:
        raise ValueError(f"Error fetching data for {symbol}: {str(e)}")