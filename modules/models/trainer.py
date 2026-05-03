"""
smartsignal.models.trainer
============================
Walk-forward training engine for single-asset (time-series) models.

Complements panel_trainer.py which handles the cross-sectional case.
Used for:
  - per-ticker binary/regression models
  - hyperparameter search within a fold
  - validation Sharpe computation per fold

The trainer wraps any BaseModel and drives it through a walk-forward loop,
collecting TrainingResult objects for each fold.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from smartsignal.models.base  import BaseModel
from smartsignal.models.types import ModelPrediction, TrainingResult
from smartsignal.models.splits import PurgedTimeSeriesSplit

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Walk-forward training engine for a single-asset dataset.

    Parameters
    ----------
    model         : any BaseModel subclass.
    feature_cols  : feature columns in the DataFrame.
    label_col     : target column.
    n_splits      : number of walk-forward folds.
    test_size     : bars per test window.
    embargo_days  : purge gap between train and test.
    """

    def __init__(
        self,
        model:        BaseModel,
        feature_cols: List[str],
        label_col:    str  = "relevance",
        n_splits:     int  = 5,
        test_size:    int  = 63,
        embargo_days: int  = 5,
    ):
        self.model        = model
        self.feature_cols = feature_cols
        self.label_col    = label_col
        self.splitter     = PurgedTimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            embargo_days=embargo_days,
        )

    def fit_predict(
        self,
        df: pd.DataFrame,
        verbose: bool = True,
    ) -> Tuple[List[TrainingResult], pd.Series]:
        """
        Run the walk-forward loop on a single-ticker DataFrame.

        Returns
        -------
        results     : list of TrainingResult per fold.
        all_scores  : concatenated test-window scores (pd.Series).
        """
        from smartsignal.models.dataset import TabularDataset

        dataset     = TabularDataset(self.feature_cols, self.label_col, scale=True)
        results: List[TrainingResult] = []
        score_chunks: List[pd.Series] = []

        X_all = df[self.feature_cols].values
        y_all = df[self.label_col].values   if self.label_col in df.columns \
                else np.zeros(len(df))

        for fold_id, (tr_idx, te_idx) in enumerate(
            self.splitter.split(X_all, y_all)
        ):
            tr_df = df.iloc[tr_idx]
            te_df = df.iloc[te_idx]

            X_tr, y_tr = dataset.fit_transform(tr_df)
            X_te, _    = dataset.transform(te_df)

            self.model.fit(tr_df)
            scores = self.model.predict(te_df)

            pred = ModelPrediction(
                scores       = scores,
                fold_id      = fold_id,
                feature_cols = self.feature_cols,
                model_family = self.model.model_family,
            )

            val_sharpe = self._compute_val_sharpe(scores, te_df)

            result = TrainingResult(
                fold_id      = fold_id,
                train_start  = tr_df.index[0],
                train_end    = tr_df.index[-1],
                test_start   = te_df.index[0],
                test_end     = te_df.index[-1],
                predictions  = pred,
                val_sharpe   = val_sharpe,
                n_train_rows = len(tr_df),
                n_test_rows  = len(te_df),
            )

            if verbose:
                print(f"  {result.summary()}")

            results.append(result)
            score_chunks.append(scores)

        all_scores = pd.concat(score_chunks).sort_index() if score_chunks else pd.Series(dtype=float)
        return results, all_scores

    @staticmethod
    def _compute_val_sharpe(scores: pd.Series, panel: pd.DataFrame) -> float:
        """Simple Sharpe proxy: IC × sqrt(252)."""
        if "fwd_ret" not in panel.columns:
            return np.nan
        fwd = panel.reindex(scores.index)["fwd_ret"].dropna()
        sc  = scores.reindex(fwd.index).dropna()
        if len(sc) < 10:
            return np.nan
        ic  = sc.corr(fwd)
        return float(ic * np.sqrt(252)) if pd.notna(ic) else np.nan
