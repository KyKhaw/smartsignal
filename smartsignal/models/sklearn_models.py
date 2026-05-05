"""
smartsignal.models.sklearn_models
===================================
Scikit-learn and LightGBM classifier/regressor models wrapped as
SmartSignal BaseModel subclasses using the PointwiseRankingAdapter.

Available models
----------------
LGBMClassifierModel   – LightGBM binary/multiclass classifier
RandomForestRanker    – sklearn RandomForestClassifier as a ranker
RidgeRanker           – Ridge regression score → cross-sectional rank
LassoRanker           – Lasso regression score → cross-sectional rank
ElasticNetRanker      – ElasticNet score → rank
SklearnRankerAdapter  – generic wrapper for any sklearn-compatible estimator

All models share the same fit(panel) / predict(panel) → pd.Series interface
defined in BaseModel, making them drop-in replacements for LambdaMARTRanker
in the walk-forward loop.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from smartsignal.models.base             import BaseModel
from smartsignal.models.ranking_adapters import PointwiseRankingAdapter
from smartsignal.models.types            import ModelFamily


# ──────────────────────────────────────────────────────────────
# Generic sklearn adapter
# ──────────────────────────────────────────────────────────────

class SklearnRankerAdapter(BaseModel):
    """
    Generic wrapper: accepts any sklearn-compatible estimator.

    Parameters
    ----------
    estimator    : an sklearn-compatible estimator instance.
    feature_cols : feature column names.
    label_col    : label column name.
    scale        : whether to StandardScaler the features.
    """

    model_family: str = ModelFamily.LGBM

    def __init__(
        self,
        estimator,
        feature_cols: List[str],
        label_col:    str  = "relevance",
        scale:        bool = True,
    ):
        self._adapter = PointwiseRankingAdapter(
            estimator    = estimator,
            feature_cols = feature_cols,
            label_col    = label_col,
            scale        = scale,
        )
        self.feature_cols = feature_cols
        self.label_col    = label_col

    def fit(self, train_panel: pd.DataFrame, run_feature_selection: bool = True) -> "SklearnRankerAdapter":
        self._adapter.fit(train_panel, run_feature_selection)
        return self

    def predict(self, test_panel: pd.DataFrame) -> pd.Series:
        return self._adapter.predict(test_panel)

    def get_params(self) -> Dict[str, Any]:
        return {"estimator": self._adapter.estimator}


# ──────────────────────────────────────────────────────────────
# LightGBM classifier
# ──────────────────────────────────────────────────────────────

class LGBMClassifierModel(BaseModel):
    """
    LightGBM binary classifier used as a cross-sectional ranker.

    Parameters
    ----------
    feature_cols      : features to use.
    label_col         : relevance label column.
    top_k_features    : placeholder for API consistency (no selection here).
    n_estimators      : number of boosting rounds.
    learning_rate     : shrinkage rate.
    max_depth         : max tree depth.
    """

    model_family: str = ModelFamily.LGBM

    def __init__(
        self,
        feature_cols:   List[str],
        label_col:      str   = "relevance",
        top_k_features: int   = 25,
        n_estimators:   int   = 300,
        learning_rate:  float = 0.05,
        max_depth:      int   = 4,
        num_leaves:     int   = 31,
        subsample:      float = 0.8,
        colsample_bytree: float = 0.8,
        random_state:   int   = 42,
    ):
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            raise ImportError("lightgbm required: pip install lightgbm")

        self.feature_cols = feature_cols
        self.label_col    = label_col

        estimator = LGBMClassifier(
            n_estimators     = n_estimators,
            learning_rate    = learning_rate,
            max_depth        = max_depth,
            num_leaves       = num_leaves,
            subsample        = subsample,
            colsample_bytree = colsample_bytree,
            random_state     = random_state,
            verbose          = -1,
            n_jobs           = -1,
        )
        self._adapter = PointwiseRankingAdapter(estimator, feature_cols, label_col)

    def fit(self, train_panel: pd.DataFrame, run_feature_selection: bool = True) -> "LGBMClassifierModel":
        self._adapter.fit(train_panel)
        return self

    def predict(self, test_panel: pd.DataFrame) -> pd.Series:
        return self._adapter.predict(test_panel)


# ──────────────────────────────────────────────────────────────
# Random Forest ranker
# ──────────────────────────────────────────────────────────────

class RandomForestRanker(BaseModel):
    """RandomForestClassifier wrapped as a cross-sectional ranker."""

    model_family: str = ModelFamily.RANDOM_FOREST

    def __init__(
        self,
        feature_cols:     List[str],
        label_col:        str  = "relevance",
        top_k_features:   int  = 25,
        n_estimators:     int  = 200,
        max_depth:        Optional[int] = 8,
        min_samples_leaf: int  = 10,
        random_state:     int  = 42,
    ):
        from sklearn.ensemble import RandomForestClassifier

        self.feature_cols = feature_cols
        self.label_col    = label_col

        estimator = RandomForestClassifier(
            n_estimators     = n_estimators,
            max_depth        = max_depth,
            min_samples_leaf = min_samples_leaf,
            random_state     = random_state,
            n_jobs           = -1,
        )
        self._adapter = PointwiseRankingAdapter(estimator, feature_cols, label_col)

    def fit(self, train_panel: pd.DataFrame, run_feature_selection: bool = True) -> "RandomForestRanker":
        self._adapter.fit(train_panel)
        return self

    def predict(self, test_panel: pd.DataFrame) -> pd.Series:
        return self._adapter.predict(test_panel)


# ──────────────────────────────────────────────────────────────
# Ridge ranker
# ──────────────────────────────────────────────────────────────

class RidgeRanker(BaseModel):
    """Ridge regression score → cross-sectional rank."""

    model_family: str = ModelFamily.RIDGE

    def __init__(
        self,
        feature_cols:   List[str],
        label_col:      str   = "relevance",
        top_k_features: int   = 25,
        alpha:          float = 1.0,
    ):
        from sklearn.linear_model import Ridge

        self.feature_cols = feature_cols
        self.label_col    = label_col

        estimator = Ridge(alpha=alpha, fit_intercept=True)
        self._adapter = PointwiseRankingAdapter(
            estimator, feature_cols, label_col, regression_mode=True
        )

    def fit(self, train_panel: pd.DataFrame, run_feature_selection: bool = True) -> "RidgeRanker":
        self._adapter.fit(train_panel)
        return self

    def predict(self, test_panel: pd.DataFrame) -> pd.Series:
        return self._adapter.predict(test_panel)