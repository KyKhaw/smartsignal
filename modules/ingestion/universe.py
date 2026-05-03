"""
smartsignal.data.loader
=======================
Equity OHLCV data ingestion layer.

Supports three input modes:
  1. Dict[str, pd.DataFrame]  – pre-loaded per-ticker DataFrames
  2. Directory path           – folder of <TICKER>.csv or <TICKER>.parquet files
  3. Single stacked CSV/Parquet with a ticker-identifier column

All outputs are normalised to:
  - DatetimeIndex (ascending, no duplicates)
  - Lowercase OHLCV column names: open, high, low, close, volume
  - One DataFrame per ticker stored in a plain dict

Column synonym resolution handles the most common vendor naming variants
(e.g. adj_close, last_price, px, Adj Close, etc.).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Synonym dictionaries
# ──────────────────────────────────────────────────────────────

_OHLCV_SYNONYMS: Dict[str, List[str]] = {
    "open":   ["open", "o", "open_price", "px_open"],
    "high":   ["high", "h", "high_price", "px_high", "day_high"],
    "low":    ["low",  "l", "low_price",  "px_low",  "day_low"],
    "close":  [
        "close", "c", "close_price", "last_price", "last", "px",
        "adj_close", "adj close", "adjusted close", "adjusted_close",
        "px_last", "settle",
    ],
    "volume": ["volume", "vol", "v", "qty", "quantity", "shares_traded",
               "total_volume"],
}

_TICKER_SYNONYMS: List[str] = [
    "ticker", "symbol", "sym", "instrument", "asset", "security",
    "code", "isin", "cusip", "ric",
]

# Build reverse-lookup: normalised_variant -> canonical OHLCV name
_REVERSE_OHLCV: Dict[str, str] = {}
for _canonical, _variants in _OHLCV_SYNONYMS.items():
    for _v in _variants:
        _REVERSE_OHLCV[_v.lower().replace(" ", "_")] = _canonical


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _resolve_ohlcv_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Map DataFrame columns to canonical OHLCV names using the synonym table.

    Returns
    -------
    df_mapped   : DataFrame with renamed columns.
    mapping     : {original_col -> canonical_col} for logging.
    """
    normalised = {c: c.lower().strip().replace(" ", "_") for c in df.columns}
    rename_map: Dict[str, str] = {}
    mapping: Dict[str, str] = {}

    for orig, norm in normalised.items():
        if norm in _REVERSE_OHLCV:
            canonical = _REVERSE_OHLCV[norm]
            if canonical not in rename_map.values():   # first match wins
                rename_map[orig] = canonical
                mapping[orig] = canonical

    df_mapped = df.rename(columns=rename_map)
    return df_mapped, mapping


def _resolve_ticker_column(df: pd.DataFrame) -> Optional[str]:
    """
    Identify the column that encodes per-instrument labels.

    Strategy
    --------
    1. Exact match against _TICKER_SYNONYMS.
    2. Cardinality heuristic: string columns with few unique values
       (< 5 % of rows, at most 2000 distinct values).
    """
    cols_lower = {c.lower(): c for c in df.columns}

    # Exact synonym match
    for syn in _TICKER_SYNONYMS:
        if syn in cols_lower:
            return cols_lower[syn]

    # Cardinality heuristic
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == "category":
            n_unique = df[col].nunique()
            ratio    = n_unique / max(len(df), 1)
            if ratio < 0.05 and n_unique <= 2000:
                return col

    return None


def _clean_single_df(df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """
    Standardise a single-ticker DataFrame:
      - Map columns to OHLCV
      - Parse / set DatetimeIndex
      - Drop all-NaN rows, deduplicate, sort ascending
      - Validate required columns
    """
    df, col_map = _resolve_ohlcv_columns(df)

    # Try to set a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        # Look for a date-like column
        for cand in ["date", "datetime", "timestamp", "time", "Date", "Datetime"]:
            if cand in df.columns:
                try:
                    df = df.set_index(pd.to_datetime(df[cand]))
                    df = df.drop(columns=[cand], errors="ignore")
                    break
                except Exception:
                    pass
        else:
            try:
                df.index = pd.to_datetime(df.index, infer_datetime_format=True)
            except Exception:
                logger.warning("  [%s] Could not parse index as DatetimeIndex — skipping.", ticker)
                return None

    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    # Drop all-NaN rows, deduplicate (keep last), sort
    df = df.dropna(how="all")
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    # Validate required OHLCV columns
    missing = {"open", "high", "low", "close", "volume"} - set(df.columns)
    if missing:
        logger.warning("  [%s] Missing OHLCV columns after mapping: %s — skipping.", ticker, missing)
        return None

    # Guard against accidental multi-column close
    if df["close"].ndim > 1:
        logger.warning("  [%s] Multiple 'close' columns detected — skipping.", ticker)
        return None

    # Cast OHLCV to float; volume to float (large ints overflow some operations)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.debug("  [%s] Loaded %d rows (%s → %s), columns mapped: %s",
                 ticker, len(df),
                 df.index.min().date() if len(df) else "?",
                 df.index.max().date() if len(df) else "?",
                 col_map)
    return df


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

class EquityLoader:
    """
    Loads and normalises equity OHLCV data from multiple sources.

    Parameters
    ----------
    min_rows : int
        Minimum rows required to retain a ticker (default 252 ~ 1 year).
    verbose  : bool
        Print per-ticker load summaries.
    """

    def __init__(self, min_rows: int = 252, verbose: bool = True):
        self.min_rows = min_rows
        self.verbose  = verbose

    # ── Entry points ───────────────────────────────────────────

    def from_dict(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Load from a {ticker: DataFrame} dictionary."""
        return self._process_dict(dfs)

    def from_directory(
        self,
        directory: Union[str, Path],
        extension: str = "csv",
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all <TICKER>.<extension> files from a directory.

        Parameters
        ----------
        directory : path to folder containing per-ticker files.
        extension : 'csv' or 'parquet'.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")

        raw: Dict[str, pd.DataFrame] = {}
        for fp in sorted(directory.glob(f"*.{extension}")):
            ticker = fp.stem.upper()
            try:
                df = pd.read_csv(fp) if extension == "csv" else pd.read_parquet(fp)
                raw[ticker] = df
            except Exception as exc:
                logger.warning("  Could not read %s: %s", fp.name, exc)

        if not raw:
            raise ValueError(f"No .{extension} files found in {directory}")

        return self._process_dict(raw)

    def from_stacked(
        self,
        source: Union[str, Path, pd.DataFrame],
        ticker_col: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load from a single stacked file where all tickers share the same rows.

        Parameters
        ----------
        source     : path to CSV/Parquet, or an already-loaded DataFrame.
        ticker_col : column name identifying the instrument; auto-detected if None.
        """
        if isinstance(source, (str, Path)):
            source = Path(source)
            df_all = pd.read_csv(source) if source.suffix == ".csv" else pd.read_parquet(source)
        else:
            df_all = source.copy()

        if ticker_col is None:
            ticker_col = _resolve_ticker_column(df_all)
        if ticker_col is None:
            raise ValueError(
                "Could not identify a ticker column. "
                "Pass ticker_col='<column_name>' explicitly."
            )

        raw: Dict[str, pd.DataFrame] = {}
        for ticker, grp in df_all.groupby(ticker_col):
            raw[str(ticker).upper()] = grp.drop(columns=[ticker_col])

        return self._process_dict(raw)

    def from_yfinance(
        self,
        tickers: List[str],
        start: str = "2015-01-01",
        end: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Download OHLCV data for a list of tickers via yfinance.

        Requires: pip install yfinance
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required: pip install yfinance")

        tickers_yf = [t.replace(".", "-") for t in tickers]
        end_str    = end or pd.Timestamp.today().strftime("%Y-%m-%d")

        if self.verbose:
            print(f"[EquityLoader] Downloading {len(tickers_yf)} tickers from yfinance …")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = yf.download(
                tickers_yf,
                start=start,
                end=end_str,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
            )

        raw: Dict[str, pd.DataFrame] = {}
        for orig, yf_ticker in zip(tickers, tickers_yf):
            try:
                if len(tickers_yf) == 1:
                    df = data.copy()
                else:
                    df = data[yf_ticker].copy()

                df = df.dropna(how="all")
                if df.empty:
                    continue
                df.columns = [c.lower() for c in df.columns]
                # yfinance already uses open/high/low/close/volume
                raw[orig.upper()] = df
            except Exception as exc:
                logger.warning("  [%s] yfinance extract failed: %s", orig, exc)

        return self._process_dict(raw)

    # ── Internal processing ────────────────────────────────────

    def _process_dict(self, raw: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        result: Dict[str, pd.DataFrame] = {}
        skipped: List[str] = []

        for ticker, df in raw.items():
            cleaned = _clean_single_df(df, ticker)
            if cleaned is None:
                skipped.append(ticker)
                continue
            if len(cleaned) < self.min_rows:
                logger.warning(
                    "  [%s] Only %d rows (min=%d) — skipping.",
                    ticker, len(cleaned), self.min_rows
                )
                skipped.append(ticker)
                continue
            result[ticker] = cleaned

        if self.verbose:
            print(
                f"[EquityLoader] Loaded {len(result)} tickers "
                f"({len(skipped)} skipped)."
            )
            if skipped:
                print(f"  Skipped: {skipped[:20]}" +
                      (" …" if len(skipped) > 20 else ""))

        if not result:
            raise ValueError("No valid ticker data loaded.")

        return result


# ── Convenience function ───────────────────────────────────────

def load_equity_data(
    source,
    *,
    ticker_col: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    start: str = "2015-01-01",
    end: Optional[str] = None,
    min_rows: int = 252,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Unified entry point for equity data loading.

    Parameters
    ----------
    source : one of:
        - Dict[str, pd.DataFrame]  → passed directly
        - str / Path (directory)   → reads all CSV/Parquet files in folder
        - str / Path (single file) → stacked panel, needs ticker_col
        - 'yfinance'               → downloads via yfinance; requires `tickers`
    ticker_col : column name for stacked-panel files (auto-detected if None).
    tickers    : list of tickers for yfinance download mode.
    start / end: date range for yfinance download.
    min_rows   : minimum rows to retain a ticker.
    verbose    : print loading summary.
    """
    loader = EquityLoader(min_rows=min_rows, verbose=verbose)

    if isinstance(source, dict):
        return loader.from_dict(source)

    if source == "yfinance":
        if not tickers:
            raise ValueError("Pass tickers=['AAPL', …] when source='yfinance'.")
        return loader.from_yfinance(tickers, start=start, end=end)

    path = Path(source)
    if path.is_dir():
        ext = "parquet" if any(path.glob("*.parquet")) else "csv"
        return loader.from_directory(path, extension=ext)

    # Single stacked file
    return loader.from_stacked(path, ticker_col=ticker_col)
