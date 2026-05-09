"""
smartsignal.utils.report
==========================
Full multi-figure visualisation report for SmartSignal pipeline results.

Generates four publication-quality figures:

  Figure 1 – Performance Dashboard
      Row 1: Equity curves (strategy vs baselines)
      Row 2: Long / Short leg equity curves
      Row 3: Rolling drawdown (underwater chart)
      Row 4: Portfolio long/short exposure over time
      Row 5: Feature importance bar chart (coloured by category)

  Figure 2 – Signal Quality
      Panel 1: Mean return by score quintile (bar chart)
      Panel 2: Directional hit rate by quintile
      Row 2:   Daily IC + rolling IC time series
      Row 3:   Walk-forward fold validation Sharpe bar chart
      Row 4:   63-day rolling annualised Sharpe

  Figure 3 – Monthly Returns Heatmap
      Heatmap for each strategy (years x months) with annual totals

  Figure 4 – Risk Analytics
      Panel 1: Return distribution histogram vs normal fit
      Panel 2: Q-Q plot (requires scipy) or rolling vol fallback
      Row 2:   Rolling 21-day and 63-day volatility
      Row 2:   Daily portfolio turnover

Usage
-----
    from smartsignal.utils.report import generate_report
    result = pipe.run(dfs=my_dfs)
    generate_report(result, save_dir="./charts", show=True)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────
# Matplotlib setup
# ──────────────────────────────────────────────────────────────

def _mpl():
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.ticker as mticker
        matplotlib.rcParams.update({
            "figure.facecolor":  "white",
            "axes.facecolor":    "#f8f9fa",
            "axes.edgecolor":    "#cccccc",
            "axes.grid":         True,
            "grid.color":        "#e0e0e0",
            "grid.linewidth":    0.6,
            "font.family":       "sans-serif",
            "font.size":         9,
            "axes.titlesize":    10,
            "axes.titleweight":  "bold",
            "axes.labelsize":    8,
            "xtick.labelsize":   7,
            "ytick.labelsize":   7,
            "legend.fontsize":   7,
            "lines.linewidth":   1.4,
        })
        return plt, gridspec, mticker
    except ImportError:
        raise ImportError("matplotlib is required: pip install matplotlib")


_PALETTE = {
    "strategy":  "#1f77b4",
    "long":      "#2ca02c",
    "short":     "#d62728",
    "baseline1": "#ff7f0e",
    "baseline2": "#9467bd",
    "neutral":   "#7f7f7f",
    "pos":       "#2ca02c",
    "neg":       "#d62728",
}

_CAT_COLORS = {
    "overlap":         "#1f77b4",
    "momentum":        "#ff7f0e",
    "volatility":      "#2ca02c",
    "volume":          "#d62728",
    "price_transform": "#9467bd",
    "other":           "#7f7f7f",
}


# ──────────────────────────────────────────────────────────────
# Internal draw helpers
# ──────────────────────────────────────────────────────────────

def _plot_equity(ax, returns, label, color, lw=1.5, alpha=1.0):
    eq = (1 + returns.dropna()).cumprod()
    ax.plot(eq.index, eq.values, label=label, color=color, lw=lw, alpha=alpha)


def _plot_drawdown(ax, returns, label, color, alpha=0.5):
    cum  = (1 + returns.dropna()).cumprod()
    peak = cum.cummax()
    dd   = (cum - peak) / peak
    ax.fill_between(dd.index, dd.values, 0,
                    color=color, alpha=alpha, label=label)


# ══════════════════════════════════════════════════════════════
# Figure 1 – Performance Dashboard
# ══════════════════════════════════════════════════════════════

def figure_performance(result, title_suffix: str = ""):
    plt, gs_mod, ticker_mod = _mpl()

    bt = result.backtest_result
    bl = result.baselines
    fi = result.feature_importance

    n_rows = 5 if fi is not None else 4
    fig    = plt.figure(figsize=(14, 4 * n_rows), constrained_layout=True)
    fig.suptitle(f"SmartSignal — Performance Dashboard{title_suffix}",
                 fontsize=12, fontweight="bold")
    gs   = gs_mod.GridSpec(n_rows, 1, figure=fig, hspace=0.4)
    axes = [fig.add_subplot(gs[i]) for i in range(n_rows)]

    # Row 1: Equity curves
    ax = axes[0]
    _plot_equity(ax, bt.strategy_returns, "LambdaMART L/S",
                 _PALETTE["strategy"], lw=2.0)
    bl_colors = [_PALETTE["baseline1"], _PALETTE["baseline2"]]
    for (name, ret), col in zip(bl.items(), bl_colors):
        _plot_equity(ax,
                     ret.reindex(bt.strategy_returns.index).ffill(),
                     name, col, lw=1.2, alpha=0.8)
    m = bt.metrics
    ax.set_title(
        f"Equity Curves   |   Sharpe {m['sharpe']:.3f}  "
        f"Ann.Ret {m['ann_return']:+.1%}  MaxDD {m['max_drawdown']:.1%}"
    )
    ax.set_ylabel("Portfolio Value ($1 invested)")
    ax.yaxis.set_major_formatter(ticker_mod.FormatStrFormatter("$%.2f"))
    ax.legend(loc="upper left")

    # Row 2: Long/short legs
    ax = axes[1]
    if bt.long_returns is not None:
        _plot_equity(ax, bt.long_returns,  "Long leg",  _PALETTE["long"])
    if bt.short_returns is not None:
        _plot_equity(ax, bt.short_returns, "Short leg", _PALETTE["short"])
    ax.set_title("Long / Short Leg Equity")
    ax.set_ylabel("Value ($1)")
    ax.legend(loc="upper left")

    # Row 3: Drawdown
    ax = axes[2]
    _plot_drawdown(ax, bt.strategy_returns,
                   "LambdaMART L/S", _PALETTE["strategy"])
    for (name, ret), col in zip(bl.items(), bl_colors):
        _plot_drawdown(ax,
                       ret.reindex(bt.strategy_returns.index).ffill(),
                       name, col, alpha=0.3)
    ax.set_title("Underwater Chart (Drawdown from Peak)")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(ticker_mod.PercentFormatter(1.0))
    ax.legend(loc="lower left")

    # Row 4: Position exposure
    ax = axes[3]
    if bt.positions is not None:
        n_long  = (bt.positions > 0).sum(axis=1)
        n_short = (bt.positions < 0).sum(axis=1)
        ax.fill_between(n_long.index,  n_long.values,   alpha=0.55,
                        color=_PALETTE["long"],  label="# Long")
        ax.fill_between(n_short.index, -n_short.values, alpha=0.55,
                        color=_PALETTE["short"], label="# Short")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("# Positions")
        ax.legend(loc="upper right")
    ax.set_title("Long / Short Position Count")

    # Row 5: Feature importance (optional)
    if fi is not None and n_rows == 5:
        ax  = axes[4]
        top = fi.head(20)
        cols = [_CAT_COLORS.get(
            top["category"].iloc[i] if "category" in top.columns else "other",
            "#7f7f7f") for i in range(len(top))]
        ax.barh(top["feature"][::-1], top["importance"][::-1],
                color=cols[::-1], edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Feature Importance (gain)")
        ax.set_title("Top-20 Feature Importance by Category")
        from matplotlib.patches import Patch
        handles = [Patch(color=c, label=cat) for cat, c in _CAT_COLORS.items()
                   if "category" in top.columns and cat in top["category"].values]
        ax.legend(handles=handles, loc="lower right")

    return fig


# ══════════════════════════════════════════════════════════════
# Figure 2 – Signal Quality
# ══════════════════════════════════════════════════════════════

def figure_signal_quality(result, training_results=None):
    plt, gs_mod, ticker_mod = _mpl()
    from smartsignal.backtesting.cross_section import (
        quintile_returns, cross_sectional_ic, hit_rate_by_decile
    )
    from smartsignal.backtesting.numba_utils import rolling_sharpe

    panel = result.panel_scored
    bt    = result.backtest_result

    fig = plt.figure(figsize=(14, 14), constrained_layout=True)
    fig.suptitle("SmartSignal — Signal Quality Analysis",
                 fontsize=12, fontweight="bold")
    gs   = gs_mod.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, :])
    ax20 = fig.add_subplot(gs[2, :])
    ax30 = fig.add_subplot(gs[3, :])

    # Panel (0,0): Quintile mean returns
    ax = ax00
    try:
        qr     = quintile_returns(panel, n_bins=5)
        colors = [_PALETTE["neg"] if v < 0 else _PALETTE["pos"]
                  for v in qr["mean_return"]]
        ax.bar(qr.index.astype(str), qr["mean_return"] * 100,
               color=colors, edgecolor="white", width=0.6)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Score Quintile (0 = Lowest)")
        ax.set_ylabel("Mean Fwd Return (%)")
        ax.set_title("Mean Return by Score Quintile")
    except Exception as e:
        ax.text(0.5, 0.5, f"Unavailable\n{e}",
                ha="center", va="center", transform=ax.transAxes)

    # Panel (0,1): Hit rate by quintile
    ax = ax01
    try:
        hr     = hit_rate_by_decile(panel, n_bins=5)
        colors = [_PALETTE["pos"] if v > 0.5 else _PALETTE["neg"]
                  for v in hr["hit_rate"]]
        ax.bar(hr.index.astype(str), hr["hit_rate"] * 100,
               color=colors, edgecolor="white", width=0.6)
        ax.axhline(50, color="black", lw=0.8, linestyle="--",
                   label="50% baseline")
        ax.set_xlabel("Score Quintile")
        ax.set_ylabel("Directional Hit Rate (%)")
        ax.set_title("Directional Hit Rate by Quintile")
        ax.legend()
    except Exception as e:
        ax.text(0.5, 0.5, f"Unavailable\n{e}",
                ha="center", va="center", transform=ax.transAxes)

    # Row 2: IC time series
    ax = ax10
    try:
        ic_daily   = cross_sectional_ic(panel)
        ic_rolling = cross_sectional_ic(panel, rolling=21)
        ax.bar(ic_daily.index, ic_daily.values, color="#9467bd",
               alpha=0.25, label="Daily IC", width=1.5)
        ax.plot(ic_rolling.index, ic_rolling.values,
                color=_PALETTE["strategy"], lw=1.5,
                label="21-day Rolling IC")
        ax.axhline(0, color="black", lw=0.8)
        ic_mean = ic_daily.mean()
        ax.axhline(ic_mean, color=_PALETTE["baseline1"],
                   lw=1.0, linestyle="--",
                   label=f"Mean IC = {ic_mean:.4f}")
        ax.set_ylabel("IC (Spearman)")
        ax.set_title("Daily Information Coefficient and 21-day Rolling Mean")
        ax.legend(loc="upper right")
    except Exception as e:
        ax.text(0.5, 0.5, f"IC chart unavailable\n{e}",
                ha="center", va="center", transform=ax.transAxes)

    # Row 3: Fold-by-fold validation Sharpe
    ax = ax20
    if training_results:
        try:
            sharpes   = [r.val_sharpe for r in training_results]
            test_ends = [r.test_end.strftime("%Y-%m")
                         for r in training_results]
            colors    = [_PALETTE["pos"] if s > 0 else _PALETTE["neg"]
                         for s in sharpes]
            ax.bar(range(len(sharpes)), sharpes, color=colors,
                   edgecolor="white", width=0.7)
            ax.axhline(0, color="black", lw=0.8)
            mean_sh = float(np.nanmean(sharpes))
            ax.axhline(mean_sh, color=_PALETTE["baseline1"],
                       lw=1.2, linestyle="--",
                       label=f"Mean = {mean_sh:.3f}")
            ax.set_xticks(range(len(sharpes)))
            ax.set_xticklabels(test_ends, rotation=45,
                               ha="right", fontsize=6)
            ax.set_xlabel("Fold (Test Window End Date)")
            ax.set_ylabel("IC-Sharpe (validation)")
            ax.set_title("Walk-Forward Fold Validation IC-Sharpe")
            ax.legend()
        except Exception as e:
            ax.text(0.5, 0.5, f"Fold chart unavailable\n{e}",
                    ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5,
                "No fold data.\nPass training_results to generate_report().",
                ha="center", va="center", transform=ax.transAxes)

    # Row 4: Rolling 63-day strategy Sharpe
    ax = ax30
    try:
        rs = rolling_sharpe(bt.strategy_returns, window=63)
        ax.plot(rs.index, rs.values,
                color=_PALETTE["strategy"], lw=1.3)
        ax.axhline(0, color="black", lw=0.8)
        ax.fill_between(rs.index, rs.values, 0,
                        where=(rs.values >= 0),
                        color=_PALETTE["pos"],  alpha=0.2)
        ax.fill_between(rs.index, rs.values, 0,
                        where=(rs.values < 0),
                        color=_PALETTE["neg"],  alpha=0.2)
        ax.set_ylabel("Rolling Sharpe (annualised)")
        ax.set_title("63-day Rolling Annualised Sharpe Ratio")
    except Exception as e:
        ax.text(0.5, 0.5, f"Rolling Sharpe unavailable\n{e}",
                ha="center", va="center", transform=ax.transAxes)

    return fig


# ══════════════════════════════════════════════════════════════
# Figure 3 – Monthly Returns Heatmap
# ══════════════════════════════════════════════════════════════

def figure_monthly_heatmap(result):
    plt, gs_mod, ticker_mod = _mpl()
    from smartsignal.backtesting.performance import PerformanceAnalyser

    bt = result.backtest_result
    bl = result.baselines

    strategies = {"LambdaMART L/S": bt.strategy_returns}
    strategies.update(bl)
    n = len(strategies)

    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n),
                              constrained_layout=True)
    fig.suptitle("SmartSignal — Monthly Returns Heatmap",
                 fontsize=12, fontweight="bold")
    if n == 1:
        axes = [axes]

    for ax, (name, ret) in zip(axes, strategies.items()):
        pa      = PerformanceAnalyser(ret, name=name)
        monthly = pa.monthly_returns()
        numeric = monthly.drop(columns=["Annual"], errors="ignore")

        try:
            import seaborn as sns
            sns.heatmap(
                numeric * 100, ax=ax,
                annot=True, fmt=".1f",
                cmap="RdYlGn", center=0,
                linewidths=0.4,
                cbar_kws={"label": "Return (%)", "shrink": 0.6},
                annot_kws={"size": 7},
            )
        except ImportError:
            im = ax.imshow(numeric.values * 100, cmap="RdYlGn",
                           aspect="auto", vmin=-10, vmax=10)
            ax.set_xticks(range(len(numeric.columns)))
            ax.set_xticklabels(numeric.columns, fontsize=7)
            ax.set_yticks(range(len(numeric.index)))
            ax.set_yticklabels(numeric.index, fontsize=7)
            for i in range(len(numeric.index)):
                for j in range(len(numeric.columns)):
                    v = numeric.values[i, j] * 100
                    ax.text(j, i, f"{v:.1f}",
                            ha="center", va="center", fontsize=6)
            fig.colorbar(im, ax=ax, shrink=0.5, label="Return (%)")

        # Annotate annual column on the right
        if "Annual" in monthly.columns:
            for i, (_, row) in enumerate(monthly.iterrows()):
                ann = row["Annual"]
                if pd.notna(ann):
                    col = _PALETTE["pos"] if ann > 0 else _PALETTE["neg"]
                    ax.text(len(numeric.columns) + 0.3, i,
                            f"{ann:+.1%}", va="center",
                            fontsize=6.5, color=col, fontweight="bold")

        m = PerformanceAnalyser(ret).metrics()
        ax.set_title(
            f"{name}   |   Ann. Return {m['ann_return']:+.1%}"
            f"   Sharpe {m['sharpe']:.3f}",
            fontsize=10
        )

    return fig


# ══════════════════════════════════════════════════════════════
# Figure 4 – Risk Analytics
# ══════════════════════════════════════════════════════════════

def figure_risk(result):
    plt, gs_mod, ticker_mod = _mpl()
    from smartsignal.backtesting.numba_utils import compute_turnover

    bt = result.backtest_result
    r  = bt.strategy_returns.dropna()

    fig = plt.figure(figsize=(14, 12), constrained_layout=True)
    fig.suptitle("SmartSignal — Risk Analytics",
                 fontsize=12, fontweight="bold")
    gs   = gs_mod.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])

    # Panel (0,0): Return distribution
    ax = ax00
    ax.hist(r * 100, bins=60, color=_PALETTE["strategy"],
            alpha=0.7, edgecolor="white", density=True)
    try:
        from scipy.stats import norm as _norm
        xs = np.linspace(r.min() * 100, r.max() * 100, 300)
        ax.plot(xs, _norm.pdf(xs, r.mean() * 100, r.std() * 100),
                color="black", lw=1.2, linestyle="--", label="Normal fit")
    except ImportError:
        pass
    ax.axvline(0, color="black", lw=0.8)
    var95 = float(r.quantile(0.05)) * 100
    ax.axvline(var95, color=_PALETTE["neg"], lw=1.2, linestyle=":",
               label=f"VaR 95% = {var95:.2f}%")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Density")
    ax.set_title("Daily Return Distribution")
    ax.legend()

    # Panel (0,1): Q-Q plot or rolling vol fallback
    ax = ax01
    try:
        from scipy.stats import probplot
        (osm, osr), (slope, intercept, r_val) = probplot(r, dist="norm",
                                                          fit=True)
        ax.scatter(osm, osr * 100, s=5,
                   color=_PALETTE["strategy"], alpha=0.5)
        line_x = np.array([osm.min(), osm.max()])
        ax.plot(line_x, (slope * line_x + intercept) * 100,
                color="black", lw=1.2,
                label=f"Normal line (R\u00b2={r_val**2:.3f})")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles (%)")
        ax.set_title("Q-Q Plot vs Normal Distribution")
        ax.legend()
    except ImportError:
        rv = r.rolling(21).std() * np.sqrt(252)
        ax.plot(rv.index, rv * 100, color=_PALETTE["strategy"])
        ax.set_ylabel("Annualised Vol (%)")
        ax.set_title("21-day Rolling Volatility (install scipy for Q-Q)")

    # Panel (1,0): Rolling volatility
    ax = ax10
    rv21 = r.rolling(21).std() * np.sqrt(252)
    rv63 = r.rolling(63).std() * np.sqrt(252)
    ax.plot(rv21.index, rv21 * 100, color=_PALETTE["strategy"],
            lw=1.2, alpha=0.8, label="21-day")
    ax.plot(rv63.index, rv63 * 100, color=_PALETTE["baseline1"],
            lw=1.5, label="63-day")
    ax.set_ylabel("Annualised Vol (%)")
    ax.set_title("Rolling Volatility")
    ax.legend()

    # Panel (1,1): Turnover
    ax = ax11
    if bt.positions is not None:
        try:
            to   = compute_turnover(bt.positions)
            roll = to.rolling(21).mean()
            ax.bar(to.index, to.values * 100,
                   color=_PALETTE["neutral"], alpha=0.4, width=1.5)
            ax.plot(roll.index, roll.values * 100,
                    color=_PALETTE["strategy"], lw=1.4,
                    label="21-day rolling mean")
            ax.set_ylabel("One-way Turnover (%)")
            ax.set_title("Daily Portfolio Turnover")
            ax.legend()
        except Exception as e:
            ax.text(0.5, 0.5, f"Turnover unavailable\n{e}",
                    ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "Position data unavailable",
                ha="center", va="center", transform=ax.transAxes)

    return fig


# ══════════════════════════════════════════════════════════════
# Master generate_report()
# ══════════════════════════════════════════════════════════════

def generate_report(
    result,
    save_dir:         Optional[str]  = None,
    show:             bool            = True,
    training_results: Optional[list] = None,
    title_suffix:     str             = "",
    dpi:              int             = 150,
    fmt:              str             = "png",
) -> Dict[str, object]:
    """
    Generate the full SmartSignal visualisation report (4 figures).

    Parameters
    ----------
    result           : PipelineResult from SmartSignalPipeline.run().
    save_dir         : directory to save figure files; None = don't save.
    show             : whether to display figures interactively.
    training_results : list of TrainingResult (enables fold Sharpe chart).
    title_suffix     : text appended to each figure's suptitle.
    dpi              : resolution for saved images.
    fmt              : file format ('png', 'pdf', 'svg').

    Returns
    -------
    dict of {figure_name: matplotlib Figure}.
    """
    plt, _, _ = _mpl()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # If training_results not passed explicitly, try getting from result
    if training_results is None:
        training_results = getattr(result, "training_results", None)

    steps = [
        ("performance",     lambda: figure_performance(result, title_suffix)),
        ("signal_quality",  lambda: figure_signal_quality(result,
                                                           training_results)),
        ("monthly_heatmap", lambda: figure_monthly_heatmap(result)),
        ("risk_analytics",  lambda: figure_risk(result)),
    ]

    figs = {}
    for name, fn in steps:
        print(f"[Report] Rendering {name} ...")
        try:
            fig = fn()
            figs[name] = fig
            if save_dir:
                path = Path(save_dir) / f"smartsignal_{name}.{fmt}"
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                print(f"         Saved -> {path}")
        except Exception as e:
            import traceback
            print(f"         [WARN] {name} failed: {e}")
            traceback.print_exc()

    if show:
        plt.show()

    return figs