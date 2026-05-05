"""
smartsignal.utils.metrics
==========================
Standard risk-adjusted performance metrics for financial strategies.

All metrics assume daily returns as input and annualise using 252 trading
days per year unless otherwise noted.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def compute_metrics(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of risk-adjusted performance metrics.

    Parameters
    ----------
    daily_returns    : pd.Series of daily net returns (not log returns).
    risk_free_rate   : annualised risk-free rate (default 0.0).
    periods_per_year : trading days per year for annualisation.

    Returns
    -------
    dict with keys:
        ann_return, ann_volatility, sharpe, sortino, max_drawdown,
        calmar, win_rate, avg_win, avg_loss, profit_factor,
        skewness, kurtosis, var_95, cvar_95,
        longest_drawdown_days, recovery_factor
    """
    r = daily_returns.dropna()

    if len(r) < 2:
        return {k: np.nan for k in [
            "ann_return", "ann_volatility", "sharpe", "sortino",
            "max_drawdown", "calmar", "win_rate", "avg_win", "avg_loss",
            "profit_factor", "skewness", "kurtosis", "var_95", "cvar_95",
            "longest_drawdown_days", "recovery_factor",
        ]}

    rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    # ── Return metrics ────────────────────────────────────────
    total_return = (1 + r).prod() - 1
    n_years      = len(r) / periods_per_year
    ann_return   = (1 + total_return) ** (1 / max(n_years, 1e-6)) - 1

    ann_vol = r.std() * np.sqrt(periods_per_year)

    # ── Sharpe ────────────────────────────────────────────────
    excess = r - rf_daily
    sharpe = (excess.mean() / excess.std() * np.sqrt(periods_per_year)
              if excess.std() > 1e-12 else np.nan)

    # ── Sortino ───────────────────────────────────────────────
    downside = excess[excess < 0]
    sortino_denom = downside.std() * np.sqrt(periods_per_year)
    sortino = (excess.mean() * periods_per_year / sortino_denom
               if sortino_denom > 1e-12 else np.nan)

    # ── Drawdown ──────────────────────────────────────────────
    cum    = (1 + r).cumprod()
    peak   = cum.cummax()
    dd     = (cum - peak) / peak
    max_dd = dd.min()

    # Longest drawdown duration
    in_dd  = dd < 0
    streaks = (in_dd != in_dd.shift()).cumsum()
    longest = in_dd.groupby(streaks).sum().max()

    calmar = (ann_return / abs(max_dd)
              if abs(max_dd) > 1e-12 else np.nan)

    # ── Win rate ──────────────────────────────────────────────
    wins     = r[r > 0]
    losses   = r[r < 0]
    win_rate = len(wins) / max(len(r), 1)

    avg_win  = wins.mean()  if len(wins)   > 0 else np.nan
    avg_loss = losses.mean() if len(losses) > 0 else np.nan

    profit_factor = (
        wins.sum() / abs(losses.sum())
        if len(losses) > 0 and losses.sum() != 0 else np.nan
    )

    # ── Distribution ──────────────────────────────────────────
    skewness = float(r.skew())
    kurtosis = float(r.kurtosis())   # excess kurtosis

    # ── VaR / CVaR (95 %) ────────────────────────────────────
    var_95  = float(r.quantile(0.05))
    cvar_95 = float(r[r <= var_95].mean()) if (r <= var_95).any() else var_95

    # ── Recovery factor ───────────────────────────────────────
    recovery = (total_return / abs(max_dd)
                if abs(max_dd) > 1e-12 else np.nan)

    return {
        "ann_return":            float(ann_return),
        "ann_volatility":        float(ann_vol),
        "sharpe":                float(sharpe),
        "sortino":               float(sortino),
        "max_drawdown":          float(max_dd),
        "calmar":                float(calmar),
        "win_rate":              float(win_rate),
        "avg_win":               float(avg_win),
        "avg_loss":              float(avg_loss),
        "profit_factor":         float(profit_factor),
        "skewness":              skewness,
        "kurtosis":              kurtosis,
        "var_95":                var_95,
        "cvar_95":               cvar_95,
        "longest_drawdown_days": int(longest),
        "recovery_factor":       float(recovery),
    }


def compare_strategies(
    strategies: Dict[str, pd.Series],
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Build a comparison table of metrics for multiple strategies.

    Parameters
    ----------
    strategies : {label: daily_returns_series}
    risk_free_rate : annualised risk-free rate.

    Returns
    -------
    DataFrame with strategies as columns and metrics as rows,
    formatted as human-readable strings.
    """
    records = {}
    for label, ret in strategies.items():
        records[label] = compute_metrics(ret, risk_free_rate=risk_free_rate)

    # Keep raw floats for programmatic use
    df_raw = pd.DataFrame(records)

    # Build a formatted display copy (string dtype throughout)
    pct_rows = [
        "ann_return", "ann_volatility", "max_drawdown", "win_rate",
        "avg_win", "avg_loss", "var_95", "cvar_95",
    ]
    float_rows = [
        "sharpe", "sortino", "calmar", "profit_factor",
        "skewness", "kurtosis", "recovery_factor",
    ]
    int_rows = ["longest_drawdown_days"]

    display_data = {}
    for col in df_raw.columns:
        col_display = {}
        for row in df_raw.index:
            val = df_raw.loc[row, col]
            if row in pct_rows:
                col_display[row] = f"{val:.2%}" if pd.notna(val) else "NaN"
            elif row in float_rows:
                col_display[row] = f"{val:.3f}" if pd.notna(val) else "NaN"
            elif row in int_rows:
                col_display[row] = str(int(val)) if pd.notna(val) else "NaN"
            else:
                col_display[row] = str(val)
        display_data[col] = col_display

    return pd.DataFrame(display_data)