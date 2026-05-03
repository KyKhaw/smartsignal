"""
smartsignal.utils.plotting
============================
Standard visualisation helpers for SmartSignal backtesting results.

Produces a 5-panel performance dashboard identical in structure to the
LambdaMART notebook charts (Figure 1 & 2 in the midterm report):

  Panel 1 – Equity curves (strategy vs baselines)
  Panel 2 – Long / short leg equity curves
  Panel 3 – Rolling drawdown
  Panel 4 – Position analysis (long/short ratio)
  Panel 5 – Feature importance bar chart (if provided)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _MATPLOTLIB = True
except ImportError:
    _MATPLOTLIB = False


def _require_matplotlib():
    if not _MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting: pip install matplotlib")


# ──────────────────────────────────────────────────────────────
# Main dashboard
# ──────────────────────────────────────────────────────────────

def plot_performance(
    strategy_returns:  pd.Series,
    baselines:         Optional[Dict[str, pd.Series]] = None,
    long_returns:      Optional[pd.Series] = None,
    short_returns:     Optional[pd.Series] = None,
    positions:         Optional[pd.DataFrame] = None,
    feature_importance:Optional[pd.DataFrame] = None,
    title:             str = "SmartSignal — LambdaMART Cross-Sectional L/S Strategy",
    figsize:           tuple = (16, 14),
    save_path:         Optional[str] = None,
) -> "plt.Figure":
    """
    Draw the 5-panel performance dashboard.

    Parameters
    ----------
    strategy_returns   : daily returns of the ML strategy.
    baselines          : dict {name: daily_returns} for benchmark overlays.
    long_returns       : daily returns of the long leg.
    short_returns      : daily returns of the short leg.
    positions          : date × ticker position matrix.
    feature_importance : DataFrame with 'feature' and 'importance' columns.
    title              : figure suptitle.
    figsize            : matplotlib figure size.
    save_path          : if provided, save the figure to this path.

    Returns
    -------
    matplotlib Figure object.
    """
    _require_matplotlib()

    n_panels = 4 + (1 if feature_importance is not None else 0)
    fig      = plt.figure(figsize=figsize, constrained_layout=True)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(n_panels, 1, figure=fig, hspace=0.35)

    axes = [fig.add_subplot(gs[i]) for i in range(n_panels)]

    # ── Panel 1: equity curves ────────────────────────────────
    ax = axes[0]
    _plot_equity(ax, strategy_returns, label="LambdaMART L/S", color="#1f77b4", lw=2)
    if baselines:
        colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        for (name, ret), col in zip(baselines.items(), colors):
            _plot_equity(ax, ret.reindex(strategy_returns.index).ffill(),
                         label=name, color=col, lw=1.2, alpha=0.8)
    ax.set_ylabel("Portfolio Value ($1 invested)")
    ax.set_title("Equity Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: long/short legs ──────────────────────────────
    ax = axes[1]
    if long_returns is not None:
        _plot_equity(ax, long_returns, label="Long leg", color="#2ca02c", lw=1.5)
    if short_returns is not None:
        _plot_equity(ax, short_returns, label="Short leg", color="#d62728", lw=1.5)
    ax.set_ylabel("Value ($1)")
    ax.set_title("Long / Short Leg")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: drawdown ─────────────────────────────────────
    ax = axes[2]
    _plot_drawdown(ax, strategy_returns)
    if baselines:
        colors = ["#ff7f0e", "#2ca02c"]
        for (name, ret), col in zip(list(baselines.items())[:2], colors):
            _plot_drawdown(ax, ret.reindex(strategy_returns.index).ffill(),
                           label=name, color=col, alpha=0.35)
    ax.set_ylabel("Drawdown")
    ax.set_title("Underwater Chart")
    ax.grid(True, alpha=0.3)

    # ── Panel 4: position analysis ────────────────────────────
    ax = axes[3]
    if positions is not None:
        n_long  = (positions > 0).sum(axis=1)
        n_short = (positions < 0).sum(axis=1)
        ax.fill_between(n_long.index, n_long.values,  alpha=0.5,
                        color="#2ca02c", label="# Long")
        ax.fill_between(n_short.index, -n_short.values, alpha=0.5,
                        color="#d62728", label="# Short")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("# Positions")
        ax.legend(fontsize=8)
    else:
        rolling_vol = strategy_returns.rolling(21).std() * np.sqrt(252)
        ax.plot(rolling_vol.index, rolling_vol.values, color="#9467bd", lw=1.2)
        ax.set_ylabel("21-day Rolling Vol (ann.)")
    ax.set_title("Position Analysis")
    ax.grid(True, alpha=0.3)

    # ── Panel 5: feature importance (optional) ────────────────
    if feature_importance is not None:
        ax = axes[4]
        top20  = feature_importance.head(20)
        colors = [_CATEGORY_COLORS.get(
            top20["category"].iloc[i] if "category" in top20.columns else "other",
            "#7f7f7f"
        ) for i in range(len(top20))]
        ax.barh(
            top20["feature"][::-1],
            top20["importance"][::-1],
            color=colors[::-1], edgecolor="white", linewidth=0.5,
        )
        ax.set_xlabel("Feature Importance (gain)")
        ax.set_title("Top-20 Feature Importance")
        ax.grid(True, axis="x", alpha=0.3)

        # Legend for categories
        from matplotlib.patches import Patch
        handles = [
            Patch(color=col, label=cat)
            for cat, col in _CATEGORY_COLORS.items()
        ]
        ax.legend(handles=handles, fontsize=7, loc="lower right")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

_CATEGORY_COLORS = {
    "overlap":         "#1f77b4",
    "momentum":        "#ff7f0e",
    "volatility":      "#2ca02c",
    "volume":          "#d62728",
    "price_transform": "#9467bd",
    "other":           "#7f7f7f",
}


def _plot_equity(ax, returns, label, color, lw=1.5, alpha=1.0):
    eq = (1 + returns.dropna()).cumprod()
    ax.plot(eq.index, eq.values, label=label, color=color, lw=lw, alpha=alpha)


def _plot_drawdown(ax, returns, label="Strategy", color="#1f77b4", alpha=0.5):
    cum  = (1 + returns.dropna()).cumprod()
    peak = cum.cummax()
    dd   = (cum - peak) / peak
    ax.fill_between(dd.index, dd.values, 0,
                    color=color, alpha=alpha, label=label)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_k: int = 30,
    title: str = "Feature Importance by Category",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Standalone bar chart of feature importances coloured by category.

    Parameters
    ----------
    importance_df : DataFrame with 'feature', 'importance', 'category' columns.
    top_k         : number of features to display.
    """
    _require_matplotlib()

    top = importance_df.head(top_k)
    colors = [_CATEGORY_COLORS.get(
        top["category"].iloc[i] if "category" in top.columns else "other",
        "#7f7f7f"
    ) for i in range(len(top))]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(top["feature"][::-1], top["importance"][::-1],
            color=colors[::-1], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(title, fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    handles = [
        Patch(color=col, label=cat)
        for cat, col in _CATEGORY_COLORS.items()
    ]
    ax.legend(handles=handles, fontsize=8)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
