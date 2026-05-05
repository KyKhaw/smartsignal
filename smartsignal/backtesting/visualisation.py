"""
smartsignal.backtesting.visualisation
=======================================
Dedicated visualisation module for backtesting outputs.

Separates plotting logic from utils/plotting.py by providing
charts that require backtesting-specific data structures
(position matrices, IC series, quintile returns, etc.).

All functions return matplotlib Figure objects and optionally
save to disk.  matplotlib is an optional dependency; an
ImportError with a helpful message is raised if absent.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _check_mpl():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualisation: pip install matplotlib"
        )


# ──────────────────────────────────────────────────────────────
# Quintile bar chart
# ──────────────────────────────────────────────────────────────

def plot_quintile_returns(
    panel_scored: pd.DataFrame,
    n_bins:       int = 5,
    ret_col:      str = "fwd_ret",
    score_col:    str = "rank_score",
    title:        str = "Mean Forward Return by Score Quintile",
    figsize:      tuple = (8, 4),
    save_path:    Optional[str] = None,
):
    """Bar chart of mean forward return per model-score quintile."""
    from smartsignal.backtesting.cross_section import quintile_returns
    plt = _check_mpl()

    qr = quintile_returns(panel_scored, n_bins=n_bins, ret_col=ret_col, score_col=score_col)

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in qr["mean_return"]]
    ax.bar(qr.index, qr["mean_return"] * 100, color=colors, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Score Quintile (0 = lowest, N-1 = highest)")
    ax.set_ylabel("Mean Forward Return (%)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────────────────────
# IC time series
# ──────────────────────────────────────────────────────────────

def plot_ic_series(
    panel_scored: pd.DataFrame,
    ret_col:      str = "fwd_ret",
    score_col:    str = "rank_score",
    rolling:      int = 21,
    title:        str = "Daily & Rolling IC",
    figsize:      tuple = (12, 4),
    save_path:    Optional[str] = None,
):
    """Plot daily IC and a rolling-window smoothed IC."""
    from smartsignal.backtesting.cross_section import cross_sectional_ic
    plt = _check_mpl()

    ic_daily   = cross_sectional_ic(panel_scored, ret_col=ret_col, score_col=score_col)
    ic_rolling = cross_sectional_ic(panel_scored, ret_col=ret_col, score_col=score_col,
                                    rolling=rolling)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(ic_daily.index, ic_daily.values, color="#9467bd", alpha=0.3, label="Daily IC")
    ax.plot(ic_rolling.index, ic_rolling.values, color="#1f77b4", lw=1.5,
            label=f"{rolling}-day Rolling IC")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("IC (Spearman)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────────────────────
# Monthly return heatmap
# ──────────────────────────────────────────────────────────────

def plot_monthly_returns(
    returns:   pd.Series,
    title:     str   = "Monthly Returns (%)",
    figsize:   tuple = (14, 6),
    save_path: Optional[str] = None,
):
    """Heatmap of monthly returns (years × months)."""
    from smartsignal.backtesting.performance import PerformanceAnalyser
    plt = _check_mpl()

    analyser = PerformanceAnalyser(returns)
    monthly  = analyser.monthly_returns()
    numeric  = monthly.drop(columns=["Annual"], errors="ignore")

    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            numeric * 100,
            ax=ax, annot=True, fmt=".1f",
            cmap="RdYlGn", center=0,
            linewidths=0.5, cbar_kws={"label": "Return (%)"},
        )
        ax.set_title(title)
    except ImportError:
        # Fallback: simple table plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        table = ax.table(
            cellText=(numeric * 100).round(1).values.tolist(),
            rowLabels=numeric.index.tolist(),
            colLabels=numeric.columns.tolist(),
            loc="center",
        )
        table.auto_set_font_size(True)
        ax.set_title(title, pad=20)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────────────────────
# Position exposure over time
# ──────────────────────────────────────────────────────────────

def plot_exposure(
    positions: pd.DataFrame,
    title:     str   = "Long / Short Exposure Over Time",
    figsize:   tuple = (12, 4),
    save_path: Optional[str] = None,
):
    """Stacked area chart showing long and short exposure over time."""
    plt = _check_mpl()

    n_long  = (positions > 0).sum(axis=1)
    n_short = (positions < 0).sum(axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(n_long.index,  n_long.values,  alpha=0.55, color="#2ca02c", label="# Long")
    ax.fill_between(n_short.index, -n_short.values, alpha=0.55, color="#d62728", label="# Short")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("# Positions")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────────────────────
# Turnover chart
# ──────────────────────────────────────────────────────────────

def plot_turnover(
    positions: pd.DataFrame,
    title:     str   = "Daily Portfolio Turnover",
    figsize:   tuple = (12, 3),
    save_path: Optional[str] = None,
):
    """Line chart of daily one-way portfolio turnover."""
    from smartsignal.backtesting.numba_utils import compute_turnover
    plt = _check_mpl()

    turnover = compute_turnover(positions)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(turnover.index, turnover.values * 100, color="#ff7f0e", lw=1)
    ax.set_ylabel("One-way Turnover (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
