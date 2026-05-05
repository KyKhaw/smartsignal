"""
smartsignal.backtesting.performance
=====================================
Detailed performance analytics extending the core metrics in utils/metrics.py.

Provides:
  - PerformanceAnalyser   : full analytics object for one strategy.
  - rolling_performance() : rolling-window Sharpe, vol, and drawdown.
  - period_returns()      : monthly and annual return decomposition.
  - information_coefficient(): IC and ICIR between rank scores and realised returns.
  - attribution_by_category(): per-feature-category performance attribution.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from smartsignal.utils.metrics import compute_metrics


# ──────────────────────────────────────────────────────────────
# PerformanceAnalyser
# ──────────────────────────────────────────────────────────────

class PerformanceAnalyser:
    """
    Comprehensive performance analytics for a single strategy.

    Parameters
    ----------
    returns  : daily net return series.
    name     : strategy label for display.
    rf_rate  : annualised risk-free rate.
    """

    def __init__(
        self,
        returns:  pd.Series,
        name:     str   = "Strategy",
        rf_rate:  float = 0.0,
    ):
        self.returns = returns.dropna()
        self.name    = name
        self.rf_rate = rf_rate
        self._metrics: Optional[Dict] = None

    # ── Core metrics ──────────────────────────────────────────

    def metrics(self) -> Dict:
        if self._metrics is None:
            self._metrics = compute_metrics(self.returns, risk_free_rate=self.rf_rate)
        return self._metrics

    def print_summary(self) -> None:
        m = self.metrics()
        print(f"\n{'─'*50}")
        print(f"  {self.name}")
        print(f"{'─'*50}")
        print(f"  Annualised return   : {m['ann_return']:+.2%}")
        print(f"  Annualised vol      : {m['ann_volatility']:.2%}")
        print(f"  Sharpe ratio        : {m['sharpe']:.3f}")
        print(f"  Sortino ratio       : {m['sortino']:.3f}")
        print(f"  Max drawdown        : {m['max_drawdown']:.2%}")
        print(f"  Calmar ratio        : {m['calmar']:.3f}")
        print(f"  Win rate            : {m['win_rate']:.2%}")
        print(f"  Profit factor       : {m['profit_factor']:.2f}")
        print(f"  Longest DD (days)   : {m['longest_drawdown_days']}")
        print(f"{'─'*50}\n")

    # ── Rolling analytics ─────────────────────────────────────

    def rolling_sharpe(self, window: int = 63) -> pd.Series:
        """Rolling annualised Sharpe ratio over `window` trading days."""
        from smartsignal.backtesting.numba_utils import rolling_sharpe
        return rolling_sharpe(self.returns, window=window)

    def rolling_volatility(self, window: int = 21) -> pd.Series:
        """Rolling annualised volatility."""
        return (
            self.returns.rolling(window).std()
            * np.sqrt(252)
        ).rename("rolling_vol")

    def drawdown_series(self) -> pd.Series:
        """Time series of drawdown from peak equity."""
        from smartsignal.backtesting.numba_utils import expanding_drawdown
        return expanding_drawdown(self.returns)

    # ── Period return breakdown ───────────────────────────────

    def monthly_returns(self) -> pd.DataFrame:
        """
        Monthly return table (years × months).

        Returns
        -------
        DataFrame with years as rows, months (1-12) as columns.
        """
        monthly = (1 + self.returns).resample("M").prod() - 1
        table   = monthly.to_frame("ret")
        table["year"]  = table.index.year
        table["month"] = table.index.month
        pivot = table.pivot(index="year", columns="month", values="ret")
        pivot.columns = [
            "Jan","Feb","Mar","Apr","May","Jun",
            "Jul","Aug","Sep","Oct","Nov","Dec"
        ][:len(pivot.columns)]
        pivot["Annual"] = (1 + monthly).resample("Y").prod() - 1
        return pivot

    def annual_returns(self) -> pd.Series:
        """Annual compounded returns."""
        return ((1 + self.returns).resample("Y").prod() - 1).rename("annual_return")


# ──────────────────────────────────────────────────────────────
# Information Coefficient (IC)
# ──────────────────────────────────────────────────────────────

def information_coefficient(
    scores:    pd.Series,
    fwd_ret:   pd.Series,
    by_date:   bool = True,
) -> Dict[str, float]:
    """
    Compute Information Coefficient (Spearman rank correlation) between
    rank scores and realised forward returns.

    Parameters
    ----------
    scores   : model rank scores aligned to (date, ticker).
    fwd_ret  : realised forward returns aligned to (date, ticker).
    by_date  : if True, compute daily IC and return mean/std/ICIR.

    Returns
    -------
    dict with keys: ic_mean, ic_std, icir, ic_positive_pct.
    """
    # Both series share the same (possibly duplicate) DatetimeIndex.
    # Align positionally rather than by label to avoid reindex errors.
    combined = pd.DataFrame({
        "score":   scores.values,
        "fwd_ret": fwd_ret.values,
        "date":    scores.index,
    }).dropna()

    if not by_date:
        ic = float(combined["score"].corr(combined["fwd_ret"], method="spearman"))
        return {"ic_mean": ic, "ic_std": np.nan, "icir": np.nan, "ic_positive_pct": np.nan}

    daily_ic = (
        combined.groupby("date")
                .apply(lambda g: g["score"].corr(g["fwd_ret"], method="spearman"))
    )

    return {
        "ic_mean":         float(daily_ic.mean()),
        "ic_std":          float(daily_ic.std()),
        "icir":            float(daily_ic.mean() / daily_ic.std() * np.sqrt(252))
                           if daily_ic.std() > 1e-12 else np.nan,
        "ic_positive_pct": float((daily_ic > 0).mean()),
    }


# ──────────────────────────────────────────────────────────────
# Category attribution
# ──────────────────────────────────────────────────────────────

def attribution_by_category(
    feature_importance_df: pd.DataFrame,
    strategy_returns:      pd.Series,
) -> pd.DataFrame:
    """
    Summarise feature importances by category alongside strategy Sharpe.

    Parameters
    ----------
    feature_importance_df : DataFrame with 'feature', 'category', 'importance'.
    strategy_returns      : daily strategy returns.

    Returns
    -------
    summary DataFrame with category-level importance shares.
    """
    m      = compute_metrics(strategy_returns)
    sharpe = m["sharpe"]

    cat_imp = (
        feature_importance_df.groupby("category")["importance"]
                              .sum()
                              .sort_values(ascending=False)
    )
    total = cat_imp.sum()
    cat_share = (cat_imp / total).rename("importance_share")

    summary = cat_share.to_frame()
    summary["strategy_sharpe"] = sharpe
    return summary