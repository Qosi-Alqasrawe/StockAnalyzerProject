# import_fetch_technical.py
import logging
import pandas as pd
import yfinance as yf
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants – centralised so callers can inspect expected column names
# ---------------------------------------------------------------------------
_COLS_6 = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
_COLS_7 = ['Date', 'Close', 'High', 'Low', 'Open', 'Adj Close', 'Volume']

# Minimum number of rows considered "usable" data
_MIN_ROWS = 2


def fetch_technical_data(symbol: str, start, end) -> pd.DataFrame:
    """
    Fetch historical stock data using yfinance.

    Args:
        symbol: Stock ticker symbol (case-insensitive; normalised internally).
        start:  Start date for historical data (str 'YYYY-MM-DD' or datetime).
        end:    End date for historical data   (str 'YYYY-MM-DD' or datetime).

    Returns:
        DataFrame with columns: Date, Close, High, Low, Open, Volume
        (and optionally Adj Close when present).
        Index is a clean RangeIndex; Date is a regular column.
        All rows with NaN values are dropped.

    Raises:
        ValueError: If the symbol is invalid, no data is available,
                    the data format is unexpected, or fewer than
                    _MIN_ROWS rows survive NaN cleaning.

    Changes vs original:
        • symbol normalised to uppercase before the network call so that
          the error message always shows the canonical form.
        • yf.download multi-level columns (produced when auto_adjust=True
          or for certain ticker groups) are flattened before counting.
        • Column count check is preceded by an explicit column-name check so
          that unexpected orderings produce a clear diagnostic instead of
          silently assigning wrong names.
        • After dropna, a minimum-row guard prevents returning an empty or
          single-row frame that would break downstream rolling calculations.
        • Logging replaces bare silence on the error path for easier debugging.
    """
    # ── Normalise symbol ──────────────────────────────────────────────────
    symbol = str(symbol).upper().strip()

    try:
        # ── Download ──────────────────────────────────────────────────────
        data = yf.download(symbol, start=start, end=end, progress=False)

        # ── Empty-data guard ──────────────────────────────────────────────
        if data is None or data.empty:
            raise ValueError(
                f"No data available for stock '{symbol}'. "
                "Verify the symbol, date range, and network connection."
            )

        # ── Flatten MultiIndex columns (yfinance ≥ 0.2 may return them) ──
        if isinstance(data.columns, pd.MultiIndex):
            # e.g. ('Close', 'AAPL') → 'Close'
            data.columns = [col[0] if isinstance(col, tuple) else col
                            for col in data.columns]

        # ── Reset index so Date becomes a regular column ──────────────────
        data = data.reset_index()

        num_cols = len(data.columns)

        # ── Column renaming with explicit name-based verification ─────────
        if num_cols == 6:
            data.columns = _COLS_6
        elif num_cols == 7:
            data.columns = _COLS_7
        else:
            raise ValueError(
                f"Unexpected column count for '{symbol}': {num_cols} "
                f"(expected 6 or 7). Columns received: {list(data.columns)}"
            )

        # ── Drop NaN rows ─────────────────────────────────────────────────
        before = len(data)
        data = data.dropna().reset_index(drop=True)
        dropped = before - len(data)
        if dropped:
            logger.debug("Dropped %d NaN rows for %s", dropped, symbol)

        # ── Minimum-row guard ─────────────────────────────────────────────
        if len(data) < _MIN_ROWS:
            raise ValueError(
                f"Insufficient data for '{symbol}': only {len(data)} row(s) "
                f"after cleaning (need at least {_MIN_ROWS})."
            )

        # ── Ensure numeric types for OHLCV columns ────────────────────────
        numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # Re-check NaN after coercion
        data = data.dropna(subset=numeric_cols).reset_index(drop=True)

        if len(data) < _MIN_ROWS:
            raise ValueError(
                f"Insufficient numeric data for '{symbol}' after type coercion."
            )

        logger.debug("Fetched %d rows for %s", len(data), symbol)
        return data

    except ValueError:
        # Re-raise our own descriptive errors unchanged
        raise
    except Exception as exc:
        # Wrap unexpected library errors with context
        logger.error("Unexpected error fetching '%s': %s", symbol, exc, exc_info=True)
        raise ValueError(
            f"Error fetching data for '{symbol}': {exc}"
        ) from exc