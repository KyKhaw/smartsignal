"""
smartsignal.workflow.combined
==============================
Multi-model and multi-strategy combined pipeline runs.

Provides two high-level workflows:

  ModelComparisonRun
  ------------------
  Runs the full SmartSignal pipeline for each of several model families
  (LambdaMART, LGBM classifier, Random Forest, Ridge) on the same dataset
  and collects BacktestResult objects for each.  Generates a combined
  performance comparison table.

  StrategyGridRun
  ---------------
  Runs the LambdaMART pipeline over a grid of strategy parameters
  (n_long, n_short, rebalance_freq, regime_filter) and collects Sharpe
  ratios to identify the most robust configuration.

Both classes use the shared helper utilities from helpers.py and the
standard SmartSignalPipeline orchestrator from pipeline.py.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from smartsignal.workflow.helpers import PipelineConfig, make_run_id

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Model comparison run
# ──────────────────────────────────────────────────────────────

class ModelComparisonRun:
    """
    Run the SmartSignal pipeline for multiple model families on the same data.

    Parameters
    ----------
    dfs    : pre-loaded {ticker: DataFrame} dict (shared across all runs).
    cfg    : base PipelineConfig.
    models : list of model family names to compare.
             Defaults to ['lambdamart', 'lgbm_classifier', 'random_forest', 'ridge'].
    """

    DEFAULT_MODELS = ["lambdamart", "lgbm_classifier", "random_forest", "ridge"]

    def __init__(
        self,
        dfs:    Dict[str, pd.DataFrame],
        cfg:    Optional[PipelineConfig] = None,
        models: Optional[List[str]] = None,
    ):
        self.dfs    = dfs
        self.cfg    = cfg or PipelineConfig()
        self.models = models or self.DEFAULT_MODELS
        self.results: Dict[str, Any] = {}

    def run(self, verbose: bool = True) -> pd.DataFrame:
        """
        Execute all model runs.

        Returns
        -------
        comparison_table : DataFrame of performance metrics (strategies × metrics).
        """
        from smartsignal.workflow.pipeline import SmartSignalPipeline
        from smartsignal.utils.metrics    import compare_strategies
        from smartsignal.features.equity_features import FEATURE_COLS
        from smartsignal.models.registry  import get_model

        # Build panel once (shared feature engineering)
        if verbose:
            print("[ModelComparison] Computing shared feature panel …")
        base_pipe = SmartSignalPipeline(**{
            k: v for k, v in self.cfg.to_dict().items()
            if k in SmartSignalPipeline.__init__.__code__.co_varnames
        })
        # We only want the data + feature stage from base_pipe
        dfs_filtered = base_pipe._stage_data(self.dfs, None, None)
        panel        = base_pipe._stage_features(dfs_filtered)
        panel        = base_pipe._stage_labels(panel)

        strat_returns: Dict[str, pd.Series] = {}

        for model_name in self.models:
            if verbose:
                print(f"\n[ModelComparison] Running model: {model_name} …")
            try:
                model = get_model(
                    model_name,
                    feature_cols   = FEATURE_COLS,
                    top_k_features = self.cfg.top_k_features,
                )
                from smartsignal.models.panel_trainer import PanelTrainer
                trainer = PanelTrainer(
                    model         = model,
                    train_years   = self.cfg.train_years,
                    test_months   = self.cfg.test_months,
                    embargo_days  = self.cfg.embargo_days,
                    forward_days  = self.cfg.forward_days,
                    verbose       = verbose,
                )
                panel_scored, _ = trainer.fit_predict(panel)
                bt = base_pipe._stage_backtest(panel_scored, dfs_filtered)
                strat_returns[model_name] = bt.strategy_returns
                self.results[model_name]  = bt
            except Exception as exc:
                logger.warning("Model %s failed: %s", model_name, exc)
                strat_returns[model_name] = pd.Series(dtype=float)

        comparison = compare_strategies(strat_returns)
        self.comparison_table = comparison
        return comparison


# ──────────────────────────────────────────────────────────────
# Strategy parameter grid search
# ──────────────────────────────────────────────────────────────

class StrategyGridRun:
    """
    Grid search over LambdaMART strategy parameters.

    Parameters
    ----------
    panel_scored : pre-computed scored panel (avoids re-training).
    dfs          : raw OHLCV data for P&L computation.
    param_grid   : dict of {param_name: [values_to_try]}.
    """

    DEFAULT_GRID: Dict[str, List] = {
        "n_long":         [5, 10, 20],
        "n_short":        [5, 10, 20],
        "rebalance_freq": ["D", "W", "M"],
        "min_hold_days":  [1, 3, 5],
    }

    def __init__(
        self,
        panel_scored: pd.DataFrame,
        dfs:          Dict[str, pd.DataFrame],
        param_grid:   Optional[Dict[str, List]] = None,
        transaction_cost: float = 0.001,
    ):
        self.panel_scored     = panel_scored
        self.dfs              = dfs
        self.param_grid       = param_grid or self.DEFAULT_GRID
        self.transaction_cost = transaction_cost
        self.grid_results: List[Dict] = []

    def run(self, verbose: bool = True) -> pd.DataFrame:
        """
        Evaluate all parameter combinations.

        Returns
        -------
        DataFrame with one row per parameter combination,
        columns for all parameters + Sharpe, Ann.Return, Max DD.
        """
        from smartsignal.backtesting.engine import run_backtest

        keys   = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combos = list(product(*values))

        if verbose:
            print(f"[GridSearch] {len(combos)} parameter combinations.")

        for combo in combos:
            params = dict(zip(keys, combo))
            try:
                bt = run_backtest(
                    self.panel_scored,
                    self.dfs,
                    transaction_cost = self.transaction_cost,
                    verbose          = False,
                    **params,
                )
                row = {**params,
                       "sharpe":      bt.metrics["sharpe"],
                       "ann_return":  bt.metrics["ann_return"],
                       "max_drawdown":bt.metrics["max_drawdown"]}
            except Exception as exc:
                logger.debug("Grid combo %s failed: %s", params, exc)
                row = {**params, "sharpe": np.nan, "ann_return": np.nan,
                       "max_drawdown": np.nan}

            self.grid_results.append(row)

        results_df = pd.DataFrame(self.grid_results)
        self.grid_results_df = results_df.sort_values("sharpe", ascending=False)

        if verbose:
            print("\n[GridSearch] Top-5 configurations:")
            print(self.grid_results_df.head(5).to_string(index=False))

        return self.grid_results_df

    def best_params(self, metric: str = "sharpe") -> Dict:
        """Return the parameter combination with the best metric value."""
        if not self.grid_results:
            raise RuntimeError("Call run() first.")
        best = self.grid_results_df.sort_values(metric, ascending=False).iloc[0]
        return {k: best[k] for k in self.param_grid}
