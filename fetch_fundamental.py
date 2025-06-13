import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Tuple, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')

def fetch_fundamental_data(symbol: str) -> Tuple[Optional[dict], str]:
    """
    متوافق مع ui.py: يجلب البيانات المالية الأساسية والبيانات التفصيلية المطلوبة.
    """
    try:
        symbol = symbol.upper().strip()
        stock = yf.Ticker(symbol)
        info = stock.info

        if not info or (not info.get('longName') and not info.get('shortName')):
            return None, f"Invalid symbol or no data available for: {symbol}"

        # (1) بيانات أساسية (للواجهة)
        basic_info = {
            'company_name': info.get('longName') or info.get('shortName', 'N/A'),
            'symbol': symbol,
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'country': info.get('country', 'N/A'),
            'website': info.get('website', 'N/A'),
            'summary': info.get('longBusinessSummary', 'N/A')[:200] + '...' if info.get('longBusinessSummary') else 'N/A',
            'current_price': info.get('currentPrice', info.get('regularMarketPrice', None)),
            'previous_close': info.get('previousClose', None),
            'market_cap': info.get('marketCap', None),
            'volume': info.get('volume', None),
            'average_volume': info.get('averageVolume', None),
            'day_range': f"{info.get('dayLow', 'N/A')} - {info.get('dayHigh', 'N/A')}",
            'week_52_range': f"{info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}",
            'trailingPE': info.get('trailingPE', None),
            'forwardPE': info.get('forwardPE', None),
            'priceToBook': info.get('priceToBook', None),
            'priceToSalesTrailing12Months': info.get('priceToSalesTrailing12Months', None),
            'pegRatio': info.get('pegRatio', None),
            'currentRatio': info.get('currentRatio', None),
            'debtToEquity': info.get('debtToEquity', None),
            'returnOnEquity': info.get('returnOnEquity', None),
            'returnOnAssets': info.get('returnOnAssets', None),
            'grossMargins': info.get('grossMargins', None),
            'operatingMargins': info.get('operatingMargins', None),
            'profitMargins': info.get('profitMargins', None),
            'dividendYield': info.get('dividendYield', None),
            'dividendRate': info.get('dividendRate', None),
            'payoutRatio': info.get('payoutRatio', None),
            'revenueGrowth': info.get('revenueGrowth', None),
            'earningsGrowth': info.get('earningsGrowth', None),
            'beta': info.get('beta', None),
            'recommendationMean': info.get('recommendationMean', None),
            'targetMeanPrice': info.get('targetMeanPrice', None),
            'targetLowPrice'    : info.get('targetLowPrice', None),   # ⬅️ أضف هذا
            'targetHighPrice'   : info.get('targetHighPrice', None),  # ⬅️ وأضف هذا
        }

        # (2) القوائم المالية (لتحليل الأداء المالي)
        try:
            financials = stock.financials
        except: financials = pd.DataFrame()
        try:
            balance_sheet = stock.balance_sheet
        except: balance_sheet = pd.DataFrame()
        try:
            cashflow = stock.cashflow
        except: cashflow = pd.DataFrame()
        try:
            quarterly_financials = stock.quarterly_financials
        except: quarterly_financials = pd.DataFrame()
        try:
            quarterly_balance_sheet = stock.quarterly_balance_sheet
        except: quarterly_balance_sheet = pd.DataFrame()
        try:
            quarterly_cashflow = stock.quarterly_cashflow
        except: quarterly_cashflow = pd.DataFrame()

        # (3) إعداد الدكتشنري النهائي بنفس هيكلية ui.py
        result = {
            "basic_info": basic_info,
            "financials": financials,
            "balance_sheet": balance_sheet,
            "cashflow": cashflow,
            "quarterly_financials": quarterly_financials,
            "quarterly_balance_sheet": quarterly_balance_sheet,
            "quarterly_cashflow": quarterly_cashflow,
        }
        return result, f"Successfully fetched fundamental data for {symbol}"

    except Exception as e:
        return None, f"Error fetching fundamental data for {symbol}: {str(e)}"
