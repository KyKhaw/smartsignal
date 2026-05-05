"""
smartsignal.models.panel_trainer
==================================
Walk-forward training engine for cross-sectional panel models (LambdaMART).

This is the primary training engine for SmartSignal.  It coordinates:

  1. PanelWalkForwardSplitter  – produces (train_panel, test_panel) fold pairs.
  2. LambdaMARTRanker          – fits on train, predicts on test.
  3. Per-fold feature selection – importance-based selection; optionally re-run
     at each fold boundary (rolling feature selection).
  4. TrainingResult collection – returns a complete audit trail.

Usage
-----
    from smartsignal.models.panel_trainer import PanelTrainer
    from smartsignal.models.lambdamart   import LambdaMARTRanker
    from smartsignal.features.equity_features import FEATURE_COLS

    model   = LambdaMARTRanker(feature_cols=FEATURE_COLS, top_k_features=25)
    trainer = PanelTrainer(model=model, train_years=3, test_months=3)
    panel_scored, results = trainer.fit_predict(labelled_panel)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from smartsignal.models.base          import BaseModel
from smartsignal.models.types         import ModelPrediction, TrainingResult
from smartsignal.models.splits_panel  import PanelWalkForwardSplitter

logger = logging.getLogger(__name__)


class PanelTrainer:
    """
    Walk-forward training engine for cross-sectional panel models.

    Parameters
    ----------
    model                 : a LambdaMARTRanker or any BaseModel.
    train_years           : initial training window (years).
    test_months           : test window per fold (months).
    embargo_days          : purge gap between train_end and test_start.
    mode                  : 'expanding' or 'rolling'.
    forward_days          : label horizon (embargo safety check).
    min_tickers_per_date  : minimum cross-sectional breadth per fold.
    feature_selection_freq: how often to re-run feature selection:
                              0 = first fold only (default)
                              1 = every fold
                              N = every N folds
    verbose               : print fold summaries.
    """

    def __init__(
        self,
        model:                  BaseModel,
        train_years:            int  = 3,
        test_months:            int  = 3,
        embargo_days:           int  = 5,
        mode:                   str  = "expanding",
        forward_days:           int  = 5,
        min_tickers_per_date:   Optional[int] = None,
        feature_selection_freq: int  = 0,
        verbose:                bool = True,
    ):
        self.model       = model
        self.verbose     = verbose
        self.feat_sel_freq = feature_selection_freq

        self.splitter = PanelWalkForwardSplitter(
            train_years          = train_years,
            test_months          = test_months,
            embargo_days         = embargo_days,
            mode                 = mode,
            min_tickers_per_date = min_tickers_per_date,
            forward_days         = forward_days,
        )

    def fit_predict(
        self,
        panel: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[TrainingResult]]:
        """
        Run the full walk-forward loop over the labelled panel.

        Parameters
        ----------
        panel : labelled cross-sectional panel (output of label workflow).

        Returns
        -------
        panel_scored : original panel with 'rank_score' column added
                       (NaN in training rows, filled in test rows).
        results      : list of TrainingResult per fold.
        """
        results:      List[TrainingResult] = []
        score_chunks: List[pd.Series]      = []
        # Track integer positions of each test row in the original panel
        # so we can use iloc for the final merge (avoids duplicate-date .loc issues)
        score_positions: List[np.ndarray]  = []
        panel_reset = panel.reset_index(drop=False)   # gives a clean 0-based int index
        fold_id = 0

        for train_panel, test_panel in self.splitter.split(panel):
            # Decide whether to run feature selection this fold
            run_sel = (fold_id == 0) or (
                self.feat_sel_freq > 0 and fold_id % self.feat_sel_freq == 0
            )

            # Fit model
            self.model.fit(train_panel, run_feature_selection=run_sel)

            # Predict test window — scores come back sorted by (date, ticker)
            scores = self.model.predict(test_panel)

            # Map scores back to their integer positions in the original panel.
            # test_panel rows are a subset of panel; find their iloc positions.
            test_sorted = test_panel.sort_index().sort_values("ticker", kind="stable")
            # Build a (date, ticker) key to locate rows in the original panel
            orig_keys  = list(zip(panel.index, panel["ticker"]))
            test_keys  = list(zip(test_sorted.index, test_sorted["ticker"]))
            key_to_pos = {k: i for i, k in enumerate(orig_keys)}
            positions  = np.array([key_to_pos.get(k, -1) for k in test_keys], dtype=int)
            valid_mask = positions >= 0

            # Compute validation Sharpe (IC-based proxy)
            val_sharpe = self._ic_sharpe(scores, test_panel)

            pred = ModelPrediction(
                scores       = scores,
                fold_id      = fold_id,
                feature_cols = getattr(self.model, "selected_features_", []) or [],
                model_family = getattr(self.model, "model_family", "unknown"),
            )

            result = TrainingResult(
                fold_id      = fold_id,
                train_start  = train_panel.index.min(),
                train_end    = train_panel.index.max(),
                test_start   = test_panel.index.min(),
                test_end     = test_panel.index.max(),
                predictions  = pred,
                val_sharpe   = val_sharpe,
                n_train_rows = len(train_panel),
                n_test_rows  = len(test_panel),
                selected_features = pred.feature_cols,
            )

            if self.verbose:
                print(f"  {result.summary()}")

            results.append(result)
            score_chunks.append(scores)
            score_positions.append((positions[valid_mask], scores.values[valid_mask]))
            fold_id += 1

        if not score_chunks:
            raise ValueError("Walk-forward produced no folds.")

        # Merge scores back using integer positions (avoids duplicate-date .loc issues)
        panel_out = panel.copy()
        panel_out["rank_score"] = np.nan
        score_arr = panel_out["rank_score"].values.copy()
        for positions, vals in score_positions:
            score_arr[positions] = vals
        panel_out["rank_score"] = score_arr

        all_scores = pd.concat(score_chunks)

        if self.verbose:
            n_scored = all_scores.notna().sum()
            print(
                f"\n[PanelTrainer] {n_scored:,} rows scored across "
                f"{len(results)} folds."
            )

        return panel_out, results

    @staticmethod
    def _ic_sharpe(scores: pd.Series, panel: pd.DataFrame) -> float:
        """Annualised IC as a quick Sharpe proxy for validation."""
        if "fwd_ret" not in panel.columns:
            return np.nan
        try:
            # Panel has duplicate DatetimeIndex (one row per ticker per date).
            # Align by resetting index to avoid duplicate-label reindex errors.
            panel_reset  = panel[["fwd_ret"]].reset_index(drop=True)
            scores_reset = scores.reset_index(drop=True)
            combined     = pd.DataFrame({"score": scores_reset, "fwd_ret": panel_reset["fwd_ret"]})
            combined     = combined.dropna()
            if len(combined) < 10:
                return np.nan
            ic = combined["score"].corr(combined["fwd_ret"])
            return float(ic * np.sqrt(252)) if pd.notna(ic) else np.nan
        except Exception:
            return np.nan