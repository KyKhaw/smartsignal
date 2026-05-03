"""
smartsignal.backtesting.baselines
===================================
Benchmark strategies for comparison against the LambdaMART pipeline.

Three baselines are implemented (matching those in the CSM notebook):

1. equal_weight_buyhold()
      Equal-weight, rebalanced monthly, long-only portfolio of all
      tickers in the universe.  The natural passive benchmark.

2. cross_sectional_momentum()
      Classic 12-1 momentum (Jegadeesh & Titman, 1993): rank stocks by
      their past-12-month return (excluding the most recent month),
      long the top decile, short the bottom decile, dollar-neutral.

3. run_all_baselines()
      Convenience wrapper that returns a dict of pd.Series (daily returns)
      for both baselines, ready for plotting alongside the ML strategy.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def equal_weight_buyhold(
    dfs: Dict[str, pd.DataFrame],
    rebalance_freq: str = "ME",
    transaction_cost: float = 0.001,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> pd.Series:
    """
    Equal-weight buy-and-hold benchmark.

    Parameters
    ----------
    dfs              : per-ticker OHLCV DataFrames.
    rebalance_freq   : pandas offset alias for rebalancing (default 'ME').
    transaction_cost : one-way cost applied on each rebalance.
    start_date       : optional clip start.
    end_date         : optional clip end.

    Returns
    -------
    daily_returns : pd.Series of daily portfolio returns.
    """
    # Build close-price matrix
    close = pd.DataFrame(
        {tk: df["close"] for tk, df in dfs.items()}
    ).sort_index().ffill()

    if start_date:
        close = close[close.index >= pd.Timestamp(start_date)]
    if end_date:
        close = close[close.index <= pd.Timestamp(end_date)]

    ret = close.pct_change()

    # Rebalance dates (equal-weight → just mark turnover dates)
    rebal_dates = ret.resample(rebalance_freq).last().index

    n = close.shape[1]
    port_ret = pd.Series(0.0, index=ret.index)

    for i, dt in enumerate(ret.index):
        day_ret = ret.loc[dt].mean()
        if dt in rebal_dates and i > 0:
            # Transaction cost on rebalance: assume full turnover for simplicity
            day_ret -= transaction_cost * 0.5   # half-turn approximation
        port_ret.loc[dt] = day_ret

    port_ret.name = "EW_BuyHold"
    return port_ret


def cross_sectional_momentum(
    dfs: Dict[str, pd.DataFrame],
    lookback_months: int = 12,
    skip_months:     int = 1,
    top_pct:         float = 0.1,
    bot_pct:         float = 0.1,
    rebalance_freq:  str   = "ME",
    transaction_cost:float = 0.001,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> pd.Series:
    """
    Cross-sectional momentum strategy (Jegadeesh & Titman 1993).

    At each rebalance date:
      - Rank stocks by their return over [t-lookback, t-skip] months ago.
      - Long top_pct, short bot_pct; dollar-neutral.

    Parameters
    ----------
    lookback_months  : formation window in months.
    skip_months      : recent months to skip (avoids short-term reversal).
    top_pct / bot_pct: fraction of universe to long/short.
    rebalance_freq   : pandas offset alias (default 'ME').
    transaction_cost : one-way cost.
    start_date / end_date : optional clip.

    Returns
    -------
    daily_returns : pd.Series of daily portfolio returns.
    """
    close = pd.DataFrame(
        {tk: df["close"] for tk, df in dfs.items()}
    ).sort_index().ffill()

    if start_date:
        close = close[close.index >= pd.Timestamp(start_date)]
    if end_date:
        close = close[close.index <= pd.Timestamp(end_date)]

    ret = close.pct_change()
    all_dates = close.index

    # Monthly rebalance dates
    rebal_dates = close.resample(rebalance_freq).last().index

    positions = pd.DataFrame(0.0, index=all_dates, columns=close.columns)
    prev_pos  = pd.Series(0.0, index=close.columns)

    for dt in rebal_dates:
        lookback_start = dt - pd.DateOffset(months=lookback_months)
        skip_end       = dt - pd.DateOffset(months=skip_months)

        if lookback_start < close.index[0]:
            continue

        window = close.loc[lookback_start:skip_end]
        if window.empty:
            continue

        # Past return over formation window
        past_ret = (window.iloc[-1] / window.iloc[0] - 1).dropna()
        if len(past_ret) < 5:
            continue

        n_long  = max(1, int(len(past_ret) * top_pct))
        n_short = max(1, int(len(past_ret) * bot_pct))

        ranked   = past_ret.sort_values(ascending=False)
        long_t   = ranked.index[:n_long]
        short_t  = ranked.index[-n_short:]

        new_pos = pd.Series(0.0, index=close.columns)
        new_pos[long_t]  = 1.0 / n_long
        new_pos[short_t] = -1.0 / n_short

        # Apply cost on turnover
        turnover     = (new_pos - prev_pos).abs().sum()
        cost_this_dt = transaction_cost * turnover

        # Forward-fill position until next rebalance
        future_dates = all_dates[(all_dates > dt)]
        next_rebal   = rebal_dates[rebal_dates > dt]
        fill_until   = next_rebal[0] if len(next_rebal) > 0 else all_dates[-1]
        fill_idx     = future_dates[future_dates <= fill_until]

        positions.loc[dt]       = new_pos
        positions.loc[fill_idx] = np.tile(new_pos.values, (len(fill_idx), 1))

        prev_pos = new_pos

    # Compute daily returns
    daily_pnl = (positions.shift(1) * ret).sum(axis=1)

    # Rough cost per rebalance date
    for dt in rebal_dates:
        if dt in daily_pnl.index:
            n_long  = max(1, int(len(close.columns) * top_pct))
            daily_pnl.loc[dt] -= transaction_cost * (n_long * 2) / len(close.columns)

    daily_pnl.name = "CSMomentum"
    return daily_pnl


def run_all_baselines(
    dfs: Dict[str, pd.DataFrame],
    transaction_cost: float = 0.001,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, pd.Series]:
    """
    Compute all benchmark strategies and return as a labelled dict.

    Returns
    -------
    {'EW_BuyHold': pd.Series, 'CSMomentum': pd.Series}
    """
    if verbose:
        print("[Baselines] Computing equal-weight buy-and-hold …")
    bh = equal_weight_buyhold(
        dfs, transaction_cost=transaction_cost,
        start_date=start_date, end_date=end_date,
    )

    if verbose:
        print("[Baselines] Computing cross-sectional momentum …")
    mom = cross_sectional_momentum(
        dfs, transaction_cost=transaction_cost,
        start_date=start_date, end_date=end_date,
    )

    return {"EW_BuyHold": bh, "CSMomentum": mom}
