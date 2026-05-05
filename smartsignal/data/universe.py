"""
smartsignal.data.universe
=========================
S&P 500 (and broader equity) universe management.

Provides:
  - fetch_sp500_tickers()  : download current S&P 500 constituent list
  - filter_universe()      : apply minimum history and liquidity screens
  - align_universe()       : align all tickers to a common calendar
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SP500_CSV_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/sp500.csv"
)


def fetch_sp500_tickers(url: str = _SP500_CSV_URL) -> List[str]:
    """
    Download the current S&P 500 constituent list from GitHub.

    Returns
    -------
    tickers : list of Yahoo-Finance-compatible ticker strings.
    """
    try:
        df = pd.read_csv(url)
    except Exception as exc:
        raise RuntimeError(f"Failed to download S&P 500 tickers: {exc}") from exc

    col_candidates = ["Symbol", "symbol", "Ticker", "ticker"]
    for col in col_candidates:
        if col in df.columns:
            tickers = df[col].dropna().unique().tolist()
            # Yahoo Finance uses '-' not '.' for class shares
            tickers = [t.replace(".", "-") for t in tickers]
            logger.info("Fetched %d S&P 500 tickers.", len(tickers))
            return tickers

    raise ValueError(
        f"Could not find a ticker column in the S&P 500 CSV. "
        f"Available columns: {list(df.columns)}"
    )


def filter_universe(
    dfs: Dict[str, pd.DataFrame],
    *,
    min_history_days: int = 504,       # ~2 trading years
    min_avg_volume: float = 1e6,       # 1M shares/day average
    min_avg_price: float = 5.0,        # exclude penny stocks
    max_missing_frac: float = 0.02,    # max 2 % missing close prices
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Apply data-quality and liquidity filters to a universe of OHLCV DataFrames.

    Parameters
    ----------
    dfs              : raw loaded DataFrames from EquityLoader.
    min_history_days : drop tickers with fewer trading days.
    min_avg_volume   : drop tickers with median daily volume below this.
    min_avg_price    : drop tickers with median close below this (penny-stock filter).
    max_missing_frac : drop tickers with more than this fraction of missing closes.
    start_date       : clip data to this start date (ISO format).
    end_date         : clip data to this end date.
    verbose          : log filter stats.

    Returns
    -------
    filtered : dict of DataFrames that passed all filters.
    """
    filtered: Dict[str, pd.DataFrame] = {}
    reasons: Dict[str, str] = {}

    for ticker, df in dfs.items():
        # Date clipping
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]

        # History length
        if len(df) < min_history_days:
            reasons[ticker] = f"insufficient history ({len(df)} < {min_history_days})"
            continue

        # Missing close
        missing_frac = df["close"].isna().mean()
        if missing_frac > max_missing_frac:
            reasons[ticker] = f"too many missing closes ({missing_frac:.1%})"
            continue

        # Average volume
        avg_vol = df["volume"].median()
        if avg_vol < min_avg_volume:
            reasons[ticker] = f"low avg volume ({avg_vol:,.0f})"
            continue

        # Average price (penny-stock filter)
        avg_price = df["close"].median()
        if avg_price < min_avg_price:
            reasons[ticker] = f"price below minimum ({avg_price:.2f})"
            continue

        filtered[ticker] = df

    if verbose:
        print(
            f"[Universe] {len(filtered)}/{len(dfs)} tickers passed filters "
            f"({len(reasons)} removed)."
        )
        if reasons and verbose:
            sample = dict(list(reasons.items())[:5])
            print(f"  Sample removals: {sample}")

    return filtered


def align_universe(
    dfs: Dict[str, pd.DataFrame],
    *,
    method: str = "ffill",
    max_consecutive_fill: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Align all tickers to a common trading calendar (union of all dates),
    then forward-fill short gaps to handle corporate-action halts.

    Parameters
    ----------
    dfs                   : filtered DataFrames.
    method                : 'ffill' (forward-fill) or 'drop' (drop dates
                            not present in all tickers).
    max_consecutive_fill  : maximum number of consecutive NaNs to forward-fill;
                            longer gaps remain as NaN.

    Returns
    -------
    aligned : dict with all DataFrames sharing the same DatetimeIndex.
    """
    if not dfs:
        return {}

    if method == "drop":
        # Intersection of all indices
        common_idx = None
        for df in dfs.values():
            common_idx = df.index if common_idx is None else common_idx.intersection(df.index)
        return {tk: df.reindex(common_idx) for tk, df in dfs.items()}

    # Union calendar + forward-fill
    all_dates = pd.DatetimeIndex(
        sorted(set().union(*[df.index for df in dfs.values()]))
    )

    aligned: Dict[str, pd.DataFrame] = {}
    for ticker, df in dfs.items():
        reindexed = df.reindex(all_dates)
        reindexed = reindexed.ffill(limit=max_consecutive_fill)
        aligned[ticker] = reindexed

    logger.info(
        "Aligned %d tickers to %d calendar dates (%s – %s).",
        len(aligned),
        len(all_dates),
        all_dates.min().date(),
        all_dates.max().date(),
    )
    return aligned


def build_equity_universe(
    tickers: Optional[List[str]] = None,
    source: str = "yfinance",
    start: str = "2015-01-01",
    end: Optional[str] = None,
    min_history_days: int = 504,
    min_avg_volume: float = 1e6,
    min_avg_price: float = 5.0,
    align: bool = True,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    End-to-end helper: fetch → load → filter → align.

    Parameters
    ----------
    tickers           : list of tickers; if None, downloads S&P 500 list.
    source            : 'yfinance' (default) or a directory path.
    start / end       : date range.
    min_history_days  : passed to filter_universe.
    min_avg_volume    : passed to filter_universe.
    min_avg_price     : passed to filter_universe.
    align             : whether to align all tickers to a common calendar.
    verbose           : print progress.
    """
    from smartsignal.data.loader import load_equity_data

    if tickers is None:
        tickers = fetch_sp500_tickers()
        if verbose:
            print(f"[Universe] Using S&P 500 universe: {len(tickers)} tickers.")

    dfs = load_equity_data(
        source,
        tickers=tickers,
        start=start,
        end=end,
        verbose=verbose,
    )

    dfs = filter_universe(
        dfs,
        min_history_days=min_history_days,
        min_avg_volume=min_avg_volume,
        min_avg_price=min_avg_price,
        start_date=start,
        end_date=end,
        verbose=verbose,
    )

    if align:
        dfs = align_universe(dfs, verbose=False)

    return dfs