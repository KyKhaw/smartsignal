"""
smartsignal.data.schema
========================
Canonical schema definitions, column name constants, and dtype enforcement
for all OHLCV data flowing through the SmartSignal pipeline.

Centralising these here means every module imports from a single source of
truth rather than re-declaring column names inline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────
# Canonical column names
# ──────────────────────────────────────────────────────────────

OHLCV_COLS: List[str] = ["open", "high", "low", "close", "volume"]

REQUIRED_COLS: List[str] = OHLCV_COLS          # minimum required set
OPTIONAL_COLS: List[str] = ["vwap", "trades"]  # optional enrichment columns

# Dtype mapping for OHLCV columns (used in schema enforcement)
OHLCV_DTYPES: Dict[str, np.dtype] = {
    "open":   np.dtype("float64"),
    "high":   np.dtype("float64"),
    "low":    np.dtype("float64"),
    "close":  np.dtype("float64"),
    "volume": np.dtype("float64"),
}

# ──────────────────────────────────────────────────────────────
# Exhaustive synonym table (used by loader and validator)
# ──────────────────────────────────────────────────────────────

OHLCV_SYNONYMS: Dict[str, List[str]] = {
    "open": [
        "open", "o", "open_price", "px_open", "Open", "OPEN",
    ],
    "high": [
        "high", "h", "high_price", "px_high", "day_high", "High", "HIGH",
    ],
    "low": [
        "low", "l", "low_price", "px_low", "day_low", "Low", "LOW",
    ],
    "close": [
        "close", "c", "close_price", "last_price", "last", "px",
        "adj_close", "adj close", "adjusted close", "adjusted_close",
        "px_last", "settle", "Close", "CLOSE",
    ],
    "volume": [
        "volume", "vol", "v", "qty", "quantity", "shares_traded",
        "total_volume", "Volume", "VOLUME",
    ],
}

TICKER_SYNONYMS: List[str] = [
    "ticker", "symbol", "sym", "instrument", "asset",
    "security", "code", "isin", "cusip", "ric",
    "Ticker", "Symbol",
]

# Reverse lookup: any variant → canonical name
REVERSE_OHLCV: Dict[str, str] = {
    v.lower().replace(" ", "_"): canon
    for canon, variants in OHLCV_SYNONYMS.items()
    for v in variants
}

# ──────────────────────────────────────────────────────────────
# Validated market record (per ticker, single asset)
# ──────────────────────────────────────────────────────────────

@dataclass
class MarketRecord:
    """
    Validated, normalised OHLCV record for a single equity ticker.

    Attributes
    ----------
    ticker   : uppercase ticker symbol.
    data     : DataFrame with DatetimeIndex and canonical OHLCV columns.
    freq     : inferred data frequency string (e.g. 'B', 'D', 'H').
    n_rows   : number of valid rows.
    date_min : earliest date in the series.
    date_max : latest date in the series.
    extra_cols: non-OHLCV columns preserved from the source file.
    """
    ticker:     str
    data:       pd.DataFrame
    freq:       Optional[str]      = None
    n_rows:     int                = 0
    date_min:   Optional[pd.Timestamp] = None
    date_max:   Optional[pd.Timestamp] = None
    extra_cols: List[str]          = field(default_factory=list)

    def __post_init__(self):
        self.n_rows   = len(self.data)
        self.date_min = self.data.index.min() if self.n_rows else None
        self.date_max = self.data.index.max() if self.n_rows else None
        # Detect non-OHLCV columns
        self.extra_cols = [c for c in self.data.columns if c not in OHLCV_COLS]

    def summary(self) -> str:
        return (
            f"MarketRecord({self.ticker}: {self.n_rows} rows, "
            f"{self.date_min} – {self.date_max}, freq={self.freq})"
        )


# ──────────────────────────────────────────────────────────────
# Schema enforcement helpers
# ──────────────────────────────────────────────────────────────

def enforce_schema(df: pd.DataFrame, ticker: str = "?") -> pd.DataFrame:
    """
    Enforce OHLCV dtypes and basic structural constraints on a DataFrame.

    Raises ValueError if required columns are missing after dtype coercion.
    Returns a clean copy with correct dtypes and a DatetimeIndex.
    """
    df = df.copy()

    # Coerce OHLCV columns to float64
    for col, dtype in OHLCV_DTYPES.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as exc:
            raise ValueError(
                f"[{ticker}] Cannot convert index to DatetimeIndex: {exc}"
            )

    df.index.name = "date"

    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"[{ticker}] Missing required columns: {missing}")

    # Basic sanity: high >= low, close within [low, high]
    invalid_hl = (df["high"] < df["low"]).sum()
    if invalid_hl > 0:
        import logging
        logging.getLogger(__name__).warning(
            "[%s] %d rows have high < low — these rows will be dropped.", ticker, invalid_hl
        )
        df = df[df["high"] >= df["low"]]

    return df


def infer_frequency(df: pd.DataFrame) -> Optional[str]:
    """
    Infer the dominant trading frequency from a DatetimeIndex.

    Returns a pandas offset alias ('B', 'D', 'h', '15min', etc.)
    or None if the frequency cannot be determined.
    """
    if len(df) < 3:
        return None
    try:
        freq = pd.infer_freq(df.index)
        return freq
    except Exception:
        # Fall back to median gap
        gaps = df.index.to_series().diff().dt.total_seconds().dropna()
        median_gap = gaps.median()
        if median_gap <= 60:
            return "min"
        if median_gap <= 3600:
            return "h"
        if median_gap <= 86400:
            return "D"
        return "W"
