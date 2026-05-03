"""
smartsignal.backtesting.engine
================================
Backtesting engine: converts rank scores into positions and computes P&L.

Pipeline
--------
1. build_daily_positions()
      Converts the ranked panel (rank_score) into a {date: {ticker: ±1}}
      position matrix, applying:
        - rebalance frequency filter (daily / weekly / monthly)
        - ADX-based regime filter (suppress signals in choppy markets)
        - minimum-hold filter (per-ticker position lock-in)
        - universe-size guard

2. compute_portfolio_returns()
      Translates position changes into daily P&L with separate transaction
      costs for long and short legs.

3. BacktestResult
      Dataclass bundling all outputs: equity curves, positions, metrics.

Design notes
------------
- Dollar-neutral: long and short legs are equal-weight and equal-sized.
- Transaction cost model: proportional cost applied to gross position change
  on each rebalance date.
- ADX regime filter: suppresses all signals when the cross-sectional median
  ADX < adx_threshold (default 20).  This is the same filter applied in the
  CSM LambdaMART notebook after the first round of refinements (§3.3).
- Minimum-hold filter: once a position is entered it cannot be reversed for
  at least min_hold_days bars, reducing high-frequency turnover.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Data class for backtest outputs
# ──────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Container for all backtesting outputs."""
    strategy_returns:  pd.Series    # daily net returns of the L/S strategy
    long_returns:      pd.Series    # daily returns of the long leg only
    short_returns:     pd.Series    # daily returns of the short leg only
    positions:         pd.DataFrame # date × ticker position matrix (±1, 0)
    equity_curve:      pd.Series    # cumulative growth of $1 invested
    metrics:           Dict         # annualised performance metrics
    raw_scores:        pd.DataFrame = field(repr=False, default=None)


# ──────────────────────────────────────────────────────────────
# Step 1: position construction
# ──────────────────────────────────────────────────────────────

def build_daily_positions(
    panel_scored: pd.DataFrame,
    n_long:            int   = 10,
    n_short:           int   = 10,
    rebalance_freq:    str   = "D",
    regime_filter:     bool  = True,
    adx_threshold:     float = 20.0,
    min_universe_size: int   = 10,
    min_hold_days:     int   = 1,
    verbose:           bool  = False,
) -> pd.DataFrame:
    """
    Convert rank scores into a long-short position matrix.

    Parameters
    ----------
    panel_scored     : output of run_walk_forward, with 'rank_score' column.
    n_long           : number of stocks in the long leg (top-ranked).
    n_short          : number of stocks in the short leg (bottom-ranked).
    rebalance_freq   : 'D' daily, 'W' weekly, 'ME' month-end.
    regime_filter    : suppress signals when cross-sectional median ADX < threshold.
    adx_threshold    : ADX level below which market is considered low-conviction.
    min_universe_size: skip dates with fewer valid tickers than this.
    min_hold_days    : minimum bars to hold a position before reversal allowed.

    Returns
    -------
    positions : DataFrame, index=date, columns=ticker, values in {-1, 0, +1}.
    """
    scored = panel_scored.dropna(subset=["rank_score"]).copy()
    scored = scored.sort_index()

    all_dates = scored.index.unique().sort_values()

    # Determine rebalance dates
    if rebalance_freq == "D":
        rebalance_dates = all_dates
    else:
        tmp = pd.Series(all_dates, index=all_dates)
        period_ends     = tmp.resample(rebalance_freq).last().dropna()
        rebalance_dates = pd.DatetimeIndex(period_ends.values)
        rebalance_dates = rebalance_dates[rebalance_dates.isin(all_dates)]

    # Per-ticker hold timer
    hold_timer: Dict[str, int] = {}
    current_pos: Dict[str, int] = {}

    raw_positions: Dict[pd.Timestamp, Dict[str, int]] = {}

    for dt in rebalance_dates:
        day_data = scored.loc[dt]
        if isinstance(day_data, pd.Series):
            day_data = day_data.to_frame().T

        # Universe size guard
        if len(day_data) < min_universe_size:
            if verbose:
                logger.debug("%s: universe too small (%d) — skipping.", dt.date(), len(day_data))
            raw_positions[dt] = {t: 0 for t in day_data["ticker"].values}
            continue

        # Regime filter: cross-sectional median ADX
        if regime_filter and "adx_14" in day_data.columns:
            median_adx = day_data["adx_14"].median()
            if pd.notna(median_adx) and median_adx < adx_threshold:
                if verbose:
                    logger.debug("%s: low ADX (%.1f) — suppressing signals.", dt.date(), median_adx)
                raw_positions[dt] = {t: 0 for t in day_data["ticker"].values}
                continue

        # Rank stocks by model score
        day_data = day_data.sort_values("rank_score", ascending=False)
        tickers  = day_data["ticker"].values
        scores   = day_data["rank_score"].values

        new_pos: Dict[str, int] = {t: 0 for t in tickers}

        # Long: top n_long
        for i, t in enumerate(tickers[:n_long]):
            # Minimum-hold: if already short, enforce lock
            if hold_timer.get(t, 0) > 0 and current_pos.get(t, 0) == -1:
                new_pos[t] = -1
            else:
                new_pos[t] = 1

        # Short: bottom n_short
        for t in tickers[-n_short:]:
            if hold_timer.get(t, 0) > 0 and current_pos.get(t, 0) == 1:
                new_pos[t] = 1
            else:
                new_pos[t] = -1

        # Update hold timers
        for t, pos in new_pos.items():
            prev = current_pos.get(t, 0)
            if pos != prev:               # position changed
                hold_timer[t] = min_hold_days
            elif hold_timer.get(t, 0) > 0:
                hold_timer[t] -= 1

        current_pos = new_pos.copy()
        raw_positions[dt] = new_pos

    # Build position DataFrame (forward-fill to non-rebalance days)
    all_tickers = scored["ticker"].unique().tolist()
    pos_df = pd.DataFrame(
        index=all_dates, columns=all_tickers, data=0.0, dtype=float
    )
    for dt, pos_dict in raw_positions.items():
        for t, v in pos_dict.items():
            if t in pos_df.columns:
                pos_df.loc[dt, t] = float(v)

    if rebalance_freq != "D":
        pos_df = pos_df.replace(0.0, np.nan).ffill().fillna(0.0)

    return pos_df


# ──────────────────────────────────────────────────────────────
# Step 2: P&L computation
# ──────────────────────────────────────────────────────────────

def compute_portfolio_returns(
    positions:        pd.DataFrame,
    dfs:              Dict[str, pd.DataFrame],
    transaction_cost: float = 0.001,
    slippage:         float = 0.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Translate position changes into daily P&L.

    Parameters
    ----------
    positions        : date × ticker DataFrame of {-1, 0, +1} positions.
    dfs              : original per-ticker OHLCV DataFrames (for close prices).
    transaction_cost : one-way proportional transaction cost (e.g. 0.001 = 10 bps).
    slippage         : additional one-way slippage per trade (proportional).

    Returns
    -------
    strat_ret  : daily net strategy returns.
    long_ret   : daily returns of the long leg.
    short_ret  : daily returns of the short leg.
    """
    # Build close-price matrix
    close_matrix = pd.DataFrame(
        {tk: df["close"] for tk, df in dfs.items() if tk in positions.columns}
    ).reindex(positions.index).ffill()

    # Daily returns matrix
    ret_matrix = close_matrix.pct_change()

    pos_prev = positions.shift(1).fillna(0)
    turnover = (positions - pos_prev).abs()

    # Cost = (transaction_cost + slippage) × turnover × 0.5 (two-sided)
    total_cost = (transaction_cost + slippage) * turnover

    # Gross P&L per ticker: position × return
    gross_pnl = positions.shift(1) * ret_matrix

    # Net P&L (deduct costs)
    net_pnl = gross_pnl - total_cost

    # Separate long and short legs
    long_mask  = positions.shift(1) > 0
    short_mask = positions.shift(1) < 0

    n_long_active  = long_mask.sum(axis=1).replace(0, np.nan)
    n_short_active = short_mask.sum(axis=1).replace(0, np.nan)

    long_ret  = (net_pnl.where(long_mask,  0.0).sum(axis=1) / n_long_active).fillna(0)
    short_ret = (net_pnl.where(short_mask, 0.0).sum(axis=1) / n_short_active).fillna(0)

    # Dollar-neutral: equal weight long and short
    strat_ret = 0.5 * (long_ret - short_ret)

    return strat_ret, long_ret, short_ret


# ──────────────────────────────────────────────────────────────
# Main backtesting entry point
# ──────────────────────────────────────────────────────────────

def run_backtest(
    panel_scored:     pd.DataFrame,
    dfs:              Dict[str, pd.DataFrame],
    n_long:           int   = 10,
    n_short:          int   = 10,
    rebalance_freq:   str   = "D",
    regime_filter:    bool  = True,
    adx_threshold:    float = 20.0,
    min_universe_size:int   = 10,
    min_hold_days:    int   = 1,
    transaction_cost: float = 0.001,
    slippage:         float = 0.0,
    verbose:          bool  = True,
) -> BacktestResult:
    """
    Run the full LambdaMART long-short backtest.

    Parameters
    ----------
    panel_scored     : walk-forward scored panel (rank_score column).
    dfs              : per-ticker OHLCV DataFrames.
    n_long / n_short : long / short leg sizes.
    rebalance_freq   : 'D', 'W', or 'ME'.
    regime_filter    : ADX regime gate.
    adx_threshold    : ADX cutoff.
    min_universe_size: minimum active tickers.
    min_hold_days    : minimum holding period.
    transaction_cost : one-way trading cost.
    slippage         : additional one-way slippage.
    verbose          : print summary statistics.

    Returns
    -------
    BacktestResult with all outputs.
    """
    from smartsignal.utils.metrics import compute_metrics

    if verbose:
        print("[Backtest] Building daily positions …")

    positions = build_daily_positions(
        panel_scored,
        n_long=n_long, n_short=n_short,
        rebalance_freq=rebalance_freq,
        regime_filter=regime_filter,
        adx_threshold=adx_threshold,
        min_universe_size=min_universe_size,
        min_hold_days=min_hold_days,
    )

    if verbose:
        print("[Backtest] Computing portfolio returns …")

    strat_ret, long_ret, short_ret = compute_portfolio_returns(
        positions, dfs,
        transaction_cost=transaction_cost,
        slippage=slippage,
    )

    equity_curve = (1 + strat_ret).cumprod()
    metrics      = compute_metrics(strat_ret)

    if verbose:
        print(
            f"\n[Backtest] Results\n"
            f"  Annualised return : {metrics['ann_return']:+.2%}\n"
            f"  Sharpe ratio      : {metrics['sharpe']:.3f}\n"
            f"  Max drawdown      : {metrics['max_drawdown']:.2%}\n"
            f"  Win rate          : {metrics['win_rate']:.2%}\n"
            f"  Calmar ratio      : {metrics['calmar']:.3f}\n"
        )

    return BacktestResult(
        strategy_returns = strat_ret,
        long_returns     = long_ret,
        short_returns    = short_ret,
        positions        = positions,
        equity_curve     = equity_curve,
        metrics          = metrics,
        raw_scores       = panel_scored,
    )
