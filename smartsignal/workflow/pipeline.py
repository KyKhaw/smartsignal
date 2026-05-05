"""
smartsignal.workflow.pipeline
===============================
Top-level SmartSignal pipeline orchestrator.

SmartSignalPipeline ties together all modules into a single configurable
object.  The pipeline runs in six sequential stages:

  Stage 1 – Data ingestion    : load and validate equity OHLCV data
  Stage 2 – Feature engineering: compute 42-feature cross-sectional panel
  Stage 3 – Label generation  : cross-sectional quintile relevance labels
  Stage 4 – Walk-forward train: expanding-window LambdaMART training
  Stage 5 – Backtesting       : position construction and P&L computation
  Stage 6 – Reporting         : performance metrics and baseline comparison

Usage
-----
    from smartsignal import SmartSignalPipeline

    pipe = SmartSignalPipeline(
        n_long=10, n_short=10,
        train_years=3, test_months=3,
        rebalance_freq="W",
        regime_filter=True,
        min_hold_days=3,
        transaction_cost=0.001,
    )

    # Run on a dict of pre-loaded DataFrames
    result = pipe.run(dfs=my_dfs)

    # Or download S&P 500 automatically
    result = pipe.run(
        tickers=["AAPL", "MSFT", "GOOG", ...],
        start="2018-01-01",
    )

    result.print_summary()
    result.plot()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Pipeline result container
# ──────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Container for all pipeline outputs."""
    backtest_result:    object                    # BacktestResult
    baselines:          Dict[str, pd.Series]
    panel_scored:       pd.DataFrame = field(repr=False)
    feature_importance: Optional[pd.DataFrame] = None
    selected_features:  Optional[List[str]] = None
    metrics_table:      Optional[pd.DataFrame] = None

    def print_summary(self) -> None:
        """Print a formatted performance summary."""
        from smartsignal.utils.metrics import compare_strategies

        strategies = {"LambdaMART L/S": self.backtest_result.strategy_returns}
        strategies.update(self.baselines)

        print("\n" + "=" * 60)
        print("  SmartSignal — Performance Summary")
        print("=" * 60)
        table = compare_strategies(strategies)
        self.metrics_table = table
        print(table.to_string())
        print("=" * 60)

        m = self.backtest_result.metrics
        print(f"\n  Sharpe : {m['sharpe']:.3f}  |  "
              f"Ann. Return : {m['ann_return']:+.2%}  |  "
              f"Max DD : {m['max_drawdown']:.2%}")

        if self.selected_features:
            print(f"\n  Selected features ({len(self.selected_features)}): "
                  f"{', '.join(self.selected_features[:8])}"
                  + (" …" if len(self.selected_features) > 8 else ""))

    def plot(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """Render the 5-panel performance dashboard."""
        from smartsignal.utils.plotting import plot_performance

        fig = plot_performance(
            strategy_returns   = self.backtest_result.strategy_returns,
            baselines          = self.baselines,
            long_returns       = self.backtest_result.long_returns,
            short_returns      = self.backtest_result.short_returns,
            positions          = self.backtest_result.positions,
            feature_importance = self.feature_importance,
            save_path          = save_path,
        )
        try:
            import matplotlib.pyplot as plt
            if show:
                plt.show()
        except ImportError:
            pass
        return fig


# ──────────────────────────────────────────────────────────────
# Main pipeline class
# ──────────────────────────────────────────────────────────────

class SmartSignalPipeline:
    """
    Adaptive ML pipeline for equity trading signal generation.

    Parameters
    ----------
    Data
    ----
    start_date        : earliest date to include (ISO format).
    end_date          : latest date to include.
    min_history_days  : minimum trading-day history per ticker.
    min_avg_volume    : minimum median daily volume (liquidity filter).
    min_avg_price     : minimum median price (penny-stock filter).

    Features
    --------
    execution_lag     : bars to shift features forward (prevents look-ahead).
    forward_days      : forward-return horizon for label construction.

    Model
    -----
    top_k_features    : number of features to retain after importance selection.
    train_years       : initial training window (years).
    test_months       : test window per fold (months).
    mode              : 'expanding' or 'rolling' walk-forward mode.
    embargo_days      : trading-day embargo gap between train and test.
    ranker_params     : dict of LGBMRanker hyper-parameters.

    Backtesting
    -----------
    n_long / n_short  : number of stocks in each leg.
    rebalance_freq    : 'D', 'W', 'ME'.
    regime_filter     : suppress signals when median ADX < adx_threshold.
    adx_threshold     : ADX cutoff for regime filter.
    min_hold_days     : minimum position holding period.
    transaction_cost  : one-way trading cost.
    slippage          : additional one-way slippage.
    """

    def __init__(
        self,
        # Data
        start_date:        str   = "2015-01-01",
        end_date:          Optional[str] = None,
        min_history_days:  int   = 504,
        min_avg_volume:    float = 1e6,
        min_avg_price:     float = 5.0,
        # Features
        execution_lag:     int   = 1,
        forward_days:      int   = 5,
        # Model
        top_k_features:    int   = 25,
        train_years:       int   = 3,
        test_months:       int   = 3,
        mode:              str   = "expanding",
        embargo_days:      int   = 5,
        ranker_params:     Optional[Dict] = None,
        # Backtesting
        n_long:            int   = 10,
        n_short:           int   = 10,
        rebalance_freq:    str   = "W",
        regime_filter:     bool  = True,
        adx_threshold:     float = 20.0,
        min_hold_days:     int   = 3,
        transaction_cost:  float = 0.001,
        slippage:          float = 0.0,
        verbose:           bool  = True,
    ):
        self.start_date       = start_date
        self.end_date         = end_date
        self.min_history_days = min_history_days
        self.min_avg_volume   = min_avg_volume
        self.min_avg_price    = min_avg_price

        self.execution_lag    = execution_lag
        self.forward_days     = forward_days

        self.top_k_features   = top_k_features
        self.train_years      = train_years
        self.test_months      = test_months
        self.mode             = mode
        self.embargo_days     = embargo_days
        self.ranker_params    = ranker_params

        self.n_long           = n_long
        self.n_short          = n_short
        self.rebalance_freq   = rebalance_freq
        self.regime_filter    = regime_filter
        self.adx_threshold    = adx_threshold
        self.min_hold_days    = min_hold_days
        self.transaction_cost = transaction_cost
        self.slippage         = slippage
        self.verbose          = verbose

    # ── Main entry point ───────────────────────────────────────

    def run(
        self,
        dfs:       Optional[Dict[str, pd.DataFrame]] = None,
        tickers:   Optional[List[str]] = None,
        data_dir:  Optional[str] = None,
        compute_baselines: bool = True,
    ) -> PipelineResult:
        """
        Execute the full six-stage SmartSignal pipeline.

        Input sources (choose one):
          dfs      : pre-loaded {ticker: DataFrame} dict.
          tickers  : list of tickers to download via yfinance.
          data_dir : directory of per-ticker CSV / Parquet files.

        Parameters
        ----------
        compute_baselines : whether to run benchmark strategies.

        Returns
        -------
        PipelineResult with backtest, baselines, and model artefacts.
        """
        t0 = time.perf_counter()

        # ── Stage 1: Data ──────────────────────────────────────
        dfs = self._stage_data(dfs, tickers, data_dir)

        # ── Stage 2: Features ──────────────────────────────────
        panel = self._stage_features(dfs)

        # ── Stage 3: Labels ────────────────────────────────────
        panel = self._stage_labels(panel)

        # ── Stage 4: Walk-forward training ────────────────────
        panel_scored, model = self._stage_train(panel)

        # ── Stage 5: Backtesting ───────────────────────────────
        backtest_result = self._stage_backtest(panel_scored, dfs)

        # ── Stage 6: Baselines & reporting ────────────────────
        baselines = {}
        if compute_baselines:
            baselines = self._stage_baselines(dfs)

        # Feature importance
        fi_df = None
        if hasattr(model, "feature_importance_df"):
            fi_df = model.feature_importance_df()
            # Add category info
            from smartsignal.features.equity_features import FEATURE_CATEGORIES
            fi_df["category"] = fi_df["feature"].map(FEATURE_CATEGORIES).fillna("other")

        elapsed = time.perf_counter() - t0
        if self.verbose:
            print(f"\n[Pipeline] Completed in {elapsed:.1f}s.")

        return PipelineResult(
            backtest_result   = backtest_result,
            baselines         = baselines,
            panel_scored      = panel_scored,
            feature_importance= fi_df,
            selected_features = getattr(model, "selected_features_", None),
        )

    # ── Stage implementations ──────────────────────────────────

    def _stage_data(
        self,
        dfs:      Optional[Dict],
        tickers:  Optional[List[str]],
        data_dir: Optional[str],
    ) -> Dict[str, pd.DataFrame]:
        from smartsignal.data.loader import load_equity_data
        from smartsignal.data.universe import filter_universe

        if self.verbose:
            print("\n[Stage 1/6] Data ingestion …")

        if dfs is not None:
            pass   # already loaded
        elif tickers is not None:
            dfs = load_equity_data(
                "yfinance",
                tickers=tickers,
                start=self.start_date,
                end=self.end_date,
                verbose=self.verbose,
            )
        elif data_dir is not None:
            dfs = load_equity_data(data_dir, verbose=self.verbose)
        else:
            raise ValueError(
                "Provide one of: dfs, tickers, or data_dir."
            )

        dfs = filter_universe(
            dfs,
            min_history_days=self.min_history_days,
            min_avg_volume=self.min_avg_volume,
            min_avg_price=self.min_avg_price,
            start_date=self.start_date,
            end_date=self.end_date,
            verbose=self.verbose,
        )
        return dfs

    def _stage_features(self, dfs: Dict) -> pd.DataFrame:
        from smartsignal.features.equity_features import build_feature_panel

        if self.verbose:
            print("\n[Stage 2/6] Feature engineering …")

        return build_feature_panel(
            dfs,
            execution_lag=self.execution_lag,
            forward_days=self.forward_days,
            verbose=self.verbose,
        )

    def _stage_labels(self, panel: pd.DataFrame) -> pd.DataFrame:
        from smartsignal.labels.generator import generate_labels

        if self.verbose:
            print("\n[Stage 3/6] Label generation …")

        return generate_labels(panel, label_type="quintile", n_bins=4)

    def _stage_train(self, panel: pd.DataFrame):
        from smartsignal.models.lambdamart   import LambdaMARTRanker
        from smartsignal.models.panel_trainer import PanelTrainer
        from smartsignal.features.equity_features import FEATURE_COLS

        if self.verbose:
            print("\n[Stage 4/6] Walk-forward training …")

        model = LambdaMARTRanker(
            feature_cols   = FEATURE_COLS,
            top_k_features = self.top_k_features,
            ranker_params  = self.ranker_params,
        )

        trainer = PanelTrainer(
            model                  = model,
            train_years            = self.train_years,
            test_months            = self.test_months,
            embargo_days           = self.embargo_days,
            mode                   = self.mode,
            forward_days           = self.forward_days,
            min_tickers_per_date   = None,   # auto-detect from universe
            feature_selection_freq = 0,      # select once on fold 0
            verbose                = self.verbose,
        )

        panel_scored, _ = trainer.fit_predict(panel)
        return panel_scored, model

    def _stage_backtest(
        self, panel_scored: pd.DataFrame, dfs: Dict
    ):
        from smartsignal.backtesting.engine import run_backtest

        if self.verbose:
            print("\n[Stage 5/6] Backtesting …")

        return run_backtest(
            panel_scored,
            dfs,
            n_long            = self.n_long,
            n_short           = self.n_short,
            rebalance_freq    = self.rebalance_freq,
            regime_filter     = self.regime_filter,
            adx_threshold     = self.adx_threshold,
            min_universe_size = self.n_long + self.n_short,  # minimum sensible floor
            min_hold_days     = self.min_hold_days,
            transaction_cost  = self.transaction_cost,
            slippage          = self.slippage,
            verbose           = self.verbose,
        )

    def _stage_baselines(self, dfs: Dict) -> Dict[str, pd.Series]:
        from smartsignal.backtesting.baselines import run_all_baselines

        if self.verbose:
            print("\n[Stage 6/6] Computing baselines …")

        return run_all_baselines(
            dfs,
            transaction_cost=self.transaction_cost,
            start_date=self.start_date,
            end_date=self.end_date,
            verbose=self.verbose,
        )