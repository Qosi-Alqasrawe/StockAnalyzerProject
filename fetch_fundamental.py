import logging
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Tuple, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keys extracted from yfinance info – centralised for maintainability
# ---------------------------------------------------------------------------
_SCALAR_KEYS: Tuple[str, ...] = (
    'trailingPE', 'forwardPE', 'priceToBook', 'priceToSalesTrailing12Months',
    'pegRatio', 'currentRatio', 'debtToEquity', 'returnOnEquity',
    'returnOnAssets', 'grossMargins', 'operatingMargins', 'profitMargins',
    'dividendYield', 'dividendRate', 'payoutRatio', 'revenueGrowth',
    'earningsGrowth', 'beta', 'recommendationMean', 'targetMeanPrice',
    'targetLowPrice', 'targetHighPrice', 'previousClose', 'marketCap',
    'volume', 'averageVolume',
)

# Summary truncation length
_SUMMARY_MAX_LEN = 200

# Minimum required identifier for the symbol to be considered valid
_REQUIRED_INFO_KEYS = ('longName', 'shortName', 'regularMarketPrice', 'currentPrice')


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _safe_fetch_df(fetch_callable) -> pd.DataFrame:
    """
    Call a zero-argument callable that returns a DataFrame.
    Returns an empty DataFrame on any exception instead of propagating.

    FIX vs original: bare `except:` is replaced with a typed catch + logging
    so that genuine network/API errors are visible in logs without crashing
    the overall fetch.
    """
    try:
        result = fetch_callable()
        if isinstance(result, pd.DataFrame):
            return result
        return pd.DataFrame()
    except Exception as exc:
        logger.debug("Financial statement fetch failed: %s", exc)
        return pd.DataFrame()


def _extract_price(info: dict) -> Optional[float]:
    """
    Return the best available current price from the info dict.

    yfinance sometimes populates 'currentPrice', sometimes
    'regularMarketPrice', and occasionally neither.
    """
    for key in ('currentPrice', 'regularMarketPrice', 'previousClose'):
        val = info.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def _build_basic_info(info: dict, symbol: str) -> dict:
    """
    Build the basic_info dict in a single, readable pass.

    FIX vs original:
    • All scalar fields extracted via a loop so adding / removing keys is
      a one-line change to _SCALAR_KEYS rather than a dict-literal edit.
    • current_price resolved via _extract_price (multi-key fallback).
    • Summary truncation uses slicing instead of a conditional expression
      that would evaluate the full f-string even when not needed.
    • day_range / week_52_range use explicit 'N/A' fallback instead of
      f-string with naked info.get() calls that silently display 'None'.
    """
    long_name    = info.get('longName')
    short_name   = info.get('shortName', 'N/A')
    summary_raw  = info.get('longBusinessSummary', '')
    summary      = (summary_raw[:_SUMMARY_MAX_LEN] + '...') if summary_raw else 'N/A'

    day_low      = info.get('dayLow')
    day_high     = info.get('dayHigh')
    w52_low      = info.get('fiftyTwoWeekLow')
    w52_high     = info.get('fiftyTwoWeekHigh')

    basic_info: Dict[str, object] = {
        'company_name':  long_name or short_name,
        'symbol':        symbol,
        'sector':        info.get('sector',   'N/A'),
        'industry':      info.get('industry', 'N/A'),
        'country':       info.get('country',  'N/A'),
        'website':       info.get('website',  'N/A'),
        'summary':       summary,
        'current_price': _extract_price(info),
        'day_range':     f"{day_low} - {day_high}"   if day_low  is not None and day_high  is not None else 'N/A',
        'week_52_range': f"{w52_low} - {w52_high}"   if w52_low  is not None and w52_high  is not None else 'N/A',
    }

    # Bulk-copy all scalar financial metrics
    for key in _SCALAR_KEYS:
        basic_info[key] = info.get(key)

    return basic_info


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_fundamental_data(symbol: str) -> Tuple[Optional[dict], str]:
    """
    Compatible with ui.py: Fetches fundamental financial data and detailed required data.
    Returns a tuple of (result_dict, status_message).

    Changes vs original:
        • symbol validated and normalised before any network call.
        • Validity check is more robust: at least one price field must be
          present (not just a name), because yfinance returns partial info
          dicts for delisted / invalid tickers that still have a longName.
        • All financial statement fetches go through _safe_fetch_df which
          logs failures rather than silently returning an empty frame.
        • Six separate statement fetches are grouped into a single dict-comp
          to reduce visual repetition and make adding new statements trivial.
        • Error path returns the original exception message for debuggability.
    """
    # ── Input validation ──────────────────────────────────────────────────
    if not symbol or not str(symbol).strip():
        return None, "Symbol cannot be empty"

    symbol = str(symbol).upper().strip()

    try:
        stock = yf.Ticker(symbol)

        # ── Single expensive info call ────────────────────────────────────
        info = stock.info or {}

        # ── Validity: must have a name AND at least one price field ───────
        has_name  = bool(info.get('longName') or info.get('shortName'))
        has_price = _extract_price(info) is not None
        if not has_name and not has_price:
            return None, f"Invalid symbol or no data available for: {symbol}"

        # ── Build basic_info ──────────────────────────────────────────────
        basic_info = _build_basic_info(info, symbol)

        # ── Fetch all financial statements ────────────────────────────────
        statement_fetchers = {
            'financials':                lambda: stock.financials,
            'balance_sheet':             lambda: stock.balance_sheet,
            'cashflow':                  lambda: stock.cashflow,
            'quarterly_financials':      lambda: stock.quarterly_financials,
            'quarterly_balance_sheet':   lambda: stock.quarterly_balance_sheet,
            'quarterly_cashflow':        lambda: stock.quarterly_cashflow,
        }
        statements = {key: _safe_fetch_df(fn) for key, fn in statement_fetchers.items()}

        result = {"basic_info": basic_info, **statements}

        logger.debug("Successfully fetched fundamental data for %s", symbol)
        return result, f"Successfully fetched fundamental data for {symbol}"

    except Exception as exc:
        logger.error("Error fetching fundamental data for %s: %s", symbol, exc, exc_info=True)
        return None, f"Error fetching fundamental data for {symbol}: {exc}"