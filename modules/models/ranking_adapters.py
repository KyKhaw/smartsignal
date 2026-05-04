"""
smartsignal.models.ranking_adapters
=====================================
Adapter classes that wrap standard classifiers and regressors so they
conform to the same (X, y, groups) interface used by LGBMRanker.

Two adapter strategies:

1. ScoreToRankAdapter
   Wraps any model that outputs a continuous score (probability, regression
   value) and converts it to a cross-sectional rank signal.  The model does
   NOT need to know about "queries" — the adapter handles the date-level
   aggregation.

2. PointwiseRankingAdapter
   Trains a pointwise model (classifier or regressor) per fold, then at
   prediction time ranks stocks within each date by their score.  Equivalent
   to a "soft" ranker that doesn't optimise NDCG directly but is simpler
   to tune and often competitive.

These adapters make it trivial to plug LightGBM classifiers, Random Forest,
Ridge, or any sklearn-compatible model into the SmartSignal walk-forward loop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from smartsignal.models.base  import BaseModel
from smartsignal.models.types import ModelFamily


# ──────────────────────────────────────────────────────────────
# Base adapter
# ──────────────────────────────────────────────────────────────

class _BaseAdapter(BaseModel):
    """Internal base for ranking adapters."""

    def __init__(
        self,
        estimator,
        feature_cols: List[str],
        label_col:    str  = "relevance",
        scale:        bool = True,
    ):
        self.estimator    = estimator
        self.feature_cols = feature_cols
        self.label_col    = label_col
        self.scale        = scale
        self._scaler: Optional[StandardScaler] = None
        self._fitted      = False

    def _get_X(self, panel: pd.DataFrame, fit: bool = False) -> np.ndarray:
        X = panel[self.feature_cols].values.astype(np.float32)
        if self.scale:
            if fit:
                self._scaler = StandardScaler()
                X = self._scaler.fit_transform(X)
            elif self._scaler is not None:
                X = self._scaler.transform(X)
        return X

    def get_params(self) -> Dict[str, Any]:
        return {"estimator": self.estimator, "feature_cols": self.feature_cols}


# ──────────────────────────────────────────────────────────────
# Score-to-rank adapter
# ──────────────────────────────────────────────────────────────

class ScoreToRankAdapter(_BaseAdapter):
    """
    Wraps any continuous-output model and converts scores to cross-sectional ranks.

    The model is trained pointwise (ignoring date structure) and at prediction
    time stocks are ranked within each date by their raw score.

    Works with: LGBMClassifier, LGBMRegressor, RandomForestClassifier,
                Ridge, Lasso, ElasticNet, SVR …
    """

    model_family: str = ModelFamily.LGBM

    def fit(
        self,
        train_panel: pd.DataFrame,
        run_feature_selection: bool = True,
    ) -> "ScoreToRankAdapter":
        X = self._get_X(train_panel, fit=True)
        y = train_panel[self.label_col].values
        self.estimator.fit(X, y)
        self._fitted = True
        return self

    def predict(self, test_panel: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        test_sorted = test_panel.sort_index().sort_values("ticker", kind="stable")
        X = self._get_X(test_sorted, fit=False)

        # Raw scores (probability of positive class or regression value)
        if hasattr(self.estimator, "predict_proba"):
            raw = self.estimator.predict_proba(X)
            # Use probability of the highest class
            scores = raw[:, -1]
        else:
            scores = self.estimator.predict(X)

        score_series = pd.Series(scores, index=test_sorted.index, name="rank_score")

        # Cross-sectional rank within each date (optional but improves stability)
        ranked = (
            score_series.groupby(level=0)
                        .rank(method="average", pct=True)
        )
        return ranked


# ──────────────────────────────────────────────────────────────
# Pointwise ranking adapter
# ──────────────────────────────────────────────────────────────

class PointwiseRankingAdapter(_BaseAdapter):
    """
    Pointwise ranker: train on (feature, label) pairs ignoring query structure.

    This is a simplified alternative to LambdaMART when you want to use a
    standard sklearn/LightGBM model without modifying its training objective.

    The raw predictions are cross-sectionally ranked within each date at
    inference time to produce a signal consistent with the L/S strategy.
    """

    model_family: str = ModelFamily.LGBM

    def __init__(
        self,
        estimator,
        feature_cols:   List[str],
        label_col:      str   = "relevance",
        scale:          bool  = True,
        regression_mode:bool  = False,
    ):
        super().__init__(estimator, feature_cols, label_col, scale)
        self.regression_mode = regression_mode

    def fit(
        self,
        train_panel: pd.DataFrame,
        run_feature_selection: bool = True,
    ) -> "PointwiseRankingAdapter":
        X = self._get_X(train_panel, fit=True)
        y = train_panel[self.label_col].values
        if self.regression_mode and "fwd_ret" in train_panel.columns:
            y = train_panel["fwd_ret"].values
        self.estimator.fit(X, y)
        self._fitted = True
        return self

    def predict(self, test_panel: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        test_sorted = test_panel.sort_index().sort_values("ticker", kind="stable")
        X = self._get_X(test_sorted, fit=False)

        if hasattr(self.estimator, "predict_proba") and not self.regression_mode:
            raw = self.estimator.predict_proba(X)[:, -1]
        else:
            raw = self.estimator.predict(X)

        return (
            pd.Series(raw, index=test_sorted.index, name="rank_score")
              .groupby(level=0)
              .rank(method="average", pct=True)
        )
