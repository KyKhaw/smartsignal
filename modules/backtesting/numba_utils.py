"""
smartsignal.backtesting.numba_utils
=====================================
Optional Numba-accelerated helper functions for performance-critical
backtesting computations.

These functions provide significant speedups for large universes (500+
tickers, 10+ years of daily data) where pure-pandas implementations
become the pipeline bottleneck.

All functions degrade gracefully to pure-numpy implementations when
Numba is not installed, so the module is safe to import unconditionally.

Functions
---------
rolling_sharpe          : fast rolling Sharpe ratio computation.
expanding_max_drawdown  : expanding-window maximum drawdown.
compute_turnover        : daily turnover from position changes.
rank_cross_section      : cross-sectional percentile ranking (fast path).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Try to import numba; fall back silently if unavailable
try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

    # Stub decorators so the function definitions below still work
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator if args and callable(args[0]) else decorator

    def prange(n):
        return range(n)


# ──────────────────────────────────────────────────────────────
# Rolling Sharpe ratio
# ──────────────────────────────────────────────────────────────

if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _rolling_sharpe_nb(returns: np.ndarray, window: int) -> np.ndarray:
        n = len(returns)
        result = np.full(n, np.nan)
        for i in prange(window - 1, n):
            w = returns[i - window + 1: i + 1]
            m = w.mean()
            s = w.std()
            if s > 1e-12:
                result[i] = m / s * np.sqrt(252)
        return result
else:
    def _rolling_sharpe_nb(returns: np.ndarray, window: int) -> np.ndarray:
        n = len(returns)
        result = np.full(n, np.nan)
        for i in range(window - 1, n):
            w = returns[i - window + 1: i + 1]
            m = w.mean()
            s = w.std()
            if s > 1e-12:
                result[i] = m / s * np.sqrt(252)
        return result


def rolling_sharpe(
    returns: pd.Series,
    window:  int = 63,
) -> pd.Series:
    """
    Rolling annualised Sharpe ratio.

    Parameters
    ----------
    returns : daily return series.
    window  : rolling window in bars (default 63 ~ 3 months).

    Returns
    -------
    pd.Series of rolling Sharpe ratios.
    """
    arr = returns.fillna(0).values.astype(np.float64)
    out = _rolling_sharpe_nb(arr, window)
    return pd.Series(out, index=returns.index, name="rolling_sharpe")


# ──────────────────────────────────────────────────────────────
# Expanding maximum drawdown
# ──────────────────────────────────────────────────────────────

if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _expanding_max_dd_nb(equity: np.ndarray) -> np.ndarray:
        n    = len(equity)
        peak = equity[0]
        dd   = np.zeros(n)
        for i in range(n):
            if equity[i] > peak:
                peak = equity[i]
            dd[i] = (equity[i] - peak) / peak if peak > 0 else 0.0
        return dd
else:
    def _expanding_max_dd_nb(equity: np.ndarray) -> np.ndarray:
        n    = len(equity)
        peak = equity[0]
        dd   = np.zeros(n)
        for i in range(n):
            if equity[i] > peak:
                peak = equity[i]
            dd[i] = (equity[i] - peak) / peak if peak > 0 else 0.0
        return dd


def expanding_drawdown(returns: pd.Series) -> pd.Series:
    """
    Expanding-window drawdown series.

    Parameters
    ----------
    returns : daily return series.

    Returns
    -------
    pd.Series of drawdown values (non-positive).
    """
    equity = (1 + returns.fillna(0)).cumprod().values.astype(np.float64)
    dd     = _expanding_max_dd_nb(equity)
    return pd.Series(dd, index=returns.index, name="drawdown")


# ──────────────────────────────────────────────────────────────
# Daily portfolio turnover
# ──────────────────────────────────────────────────────────────

def compute_turnover(positions: pd.DataFrame) -> pd.Series:
    """
    Compute daily one-way portfolio turnover as fraction of gross exposure.

    Turnover[t] = sum(|pos[t] - pos[t-1]|) / (2 × n_active[t])

    Parameters
    ----------
    positions : date × ticker position matrix ({-1, 0, +1}).

    Returns
    -------
    pd.Series of daily turnover fractions.
    """
    pos_prev = positions.shift(1).fillna(0)
    change   = (positions - pos_prev).abs().sum(axis=1)
    n_active = (positions.abs() > 0).sum(axis=1).replace(0, np.nan)
    turnover = change / (2 * n_active)
    return turnover.fillna(0).rename("turnover")


# ──────────────────────────────────────────────────────────────
# Cross-sectional rank (fast numpy path)
# ──────────────────────────────────────────────────────────────

def rank_cross_section(scores: np.ndarray) -> np.ndarray:
    """
    Percentile rank a 1-D score array (cross-section for one date).

    Returns
    -------
    ranks in [0, 1].
    """
    n = len(scores)
    if n == 0:
        return scores
    order = np.argsort(scores)
    ranks = np.empty(n)
    ranks[order] = np.arange(n) / max(n - 1, 1)
    return ranks
