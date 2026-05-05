"""
smartsignal.backtesting.cross_section
=======================================
Cross-sectional (per-date) return attribution and signal analysis.

Provides
--------
daily_long_short_spread()  : per-date return spread between long and short legs.
quintile_returns()         : mean return per model-score quintile bin.
cross_sectional_ic()       : per-date Spearman IC between scores and returns.
hit_rate_by_decile()       : fraction of correct directional predictions per decile.
spread_decomposition()     : decompose L/S spread into selection vs timing.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────
# Daily long/short return spread
# ──────────────────────────────────────────────────────────────

def daily_long_short_spread(
    panel_scored: pd.DataFrame,
    n_long:       int = 10,
    n_short:      int = 10,
    ret_col:      str = "fwd_ret",
    score_col:    str = "rank_score",
) -> pd.DataFrame:
    """
    Compute the daily L/S return spread directly from the scored panel.

    Useful for diagnosing whether the model's directional skill is
    consistent over time without running the full position-construction
    pipeline.

    Returns
    -------
    DataFrame with columns: long_ret, short_ret, spread, date.
    """
    scored = panel_scored.dropna(subset=[score_col, ret_col]).copy()
    records = []

    for dt, day in scored.groupby(level=0):
        day_sorted   = day.sort_values(score_col, ascending=False)
        long_stocks  = day_sorted.iloc[:n_long]
        short_stocks = day_sorted.iloc[-n_short:]

        long_ret  = long_stocks[ret_col].mean()
        short_ret = short_stocks[ret_col].mean()
        spread    = long_ret - short_ret

        records.append({
            "date":      dt,
            "long_ret":  long_ret,
            "short_ret": short_ret,
            "spread":    spread,
            "n_long":    len(long_stocks),
            "n_short":   len(short_stocks),
        })

    result = pd.DataFrame(records).set_index("date")
    return result


# ──────────────────────────────────────────────────────────────
# Quintile return analysis
# ──────────────────────────────────────────────────────────────

def quintile_returns(
    panel_scored: pd.DataFrame,
    n_bins:       int = 5,
    ret_col:      str = "fwd_ret",
    score_col:    str = "rank_score",
) -> pd.DataFrame:
    """
    Bin stocks into n_bins score quintiles and compute mean return per bin.

    Returns
    -------
    DataFrame indexed by quintile (0 = lowest score, n_bins-1 = highest),
    columns: mean_return, median_return, std_return, count.
    """
    scored = panel_scored.dropna(subset=[score_col, ret_col]).copy()

    scored["quintile"] = (
        scored.groupby(level=0)[score_col]
              .transform(lambda s: pd.qcut(
                  s.rank(method="first"),
                  q=n_bins, labels=False, duplicates="drop"
              ))
    )

    summary = (
        scored.groupby("quintile")[ret_col]
              .agg(mean_return="mean", median_return="median",
                   std_return="std", count="count")
    )

    return summary


# ──────────────────────────────────────────────────────────────
# Per-date IC
# ──────────────────────────────────────────────────────────────

def cross_sectional_ic(
    panel_scored: pd.DataFrame,
    ret_col:      str = "fwd_ret",
    score_col:    str = "rank_score",
    method:       str = "spearman",
    rolling:      Optional[int] = None,
) -> pd.Series:
    """
    Compute per-date Information Coefficient (Spearman or Pearson).

    Parameters
    ----------
    panel_scored : scored panel with rank_score and fwd_ret.
    ret_col      : forward return column.
    score_col    : model score column.
    method       : 'spearman' (default) or 'pearson'.
    rolling      : if provided, return a rolling-window mean IC.

    Returns
    -------
    pd.Series of daily IC values.
    """
    scored = panel_scored.dropna(subset=[score_col, ret_col])

    daily_ic = (
        scored.groupby(level=0)
              .apply(lambda g: g[score_col].corr(g[ret_col], method=method))
    )

    if rolling:
        return daily_ic.rolling(rolling).mean().rename(f"IC_{rolling}d_rolling")

    return daily_ic.rename("IC")


# ──────────────────────────────────────────────────────────────
# Hit rate by decile
# ──────────────────────────────────────────────────────────────

def hit_rate_by_decile(
    panel_scored: pd.DataFrame,
    n_bins:       int = 10,
    ret_col:      str = "fwd_ret",
    score_col:    str = "rank_score",
) -> pd.DataFrame:
    """
    Compute fraction of positive forward returns in each score decile.

    Returns
    -------
    DataFrame with decile (0-based, 0=lowest score), hit_rate, mean_return.
    """
    scored = panel_scored.dropna(subset=[score_col, ret_col]).copy()
    scored["decile"] = (
        scored.groupby(level=0)[score_col]
              .transform(lambda s: pd.qcut(
                  s.rank(method="first"),
                  q=n_bins, labels=False, duplicates="drop"
              ))
    )

    summary = scored.groupby("decile").agg(
        hit_rate   = (ret_col, lambda s: (s > 0).mean()),
        mean_return= (ret_col, "mean"),
        count      = (ret_col, "count"),
    )
    return summary


# ──────────────────────────────────────────────────────────────
# Spread decomposition
# ──────────────────────────────────────────────────────────────

def spread_decomposition(
    panel_scored:    pd.DataFrame,
    n_long:          int = 10,
    n_short:         int = 10,
    ret_col:         str = "fwd_ret",
    score_col:       str = "rank_score",
) -> Dict[str, float]:
    """
    Decompose L/S spread into selection skill and timing skill.

    Selection skill : average return of top-N stocks averaged over all dates
                      (the model's ability to pick outperformers).
    Timing skill    : correlation between spread magnitude and market vol
                      (the model's ability to time its conviction).

    Returns
    -------
    dict with selection_skill, short_skill, timing_correlation, mean_spread.
    """
    spread_df = daily_long_short_spread(panel_scored, n_long, n_short, ret_col, score_col)

    selection_skill  = float(spread_df["long_ret"].mean())
    short_skill      = float(-spread_df["short_ret"].mean())  # sign-flipped
    mean_spread      = float(spread_df["spread"].mean())

    # Timing: does the model produce wider spreads when market vol is high?
    market_vol = panel_scored.groupby(level=0)[ret_col].std()
    timing_corr = float(
        spread_df["spread"].corr(market_vol.reindex(spread_df.index))
    )

    return {
        "selection_skill":  selection_skill,
        "short_skill":      short_skill,
        "mean_spread":      mean_spread,
        "timing_correlation": timing_corr,
    }