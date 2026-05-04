"""
smartsignal.models.advanced_models
=====================================
Ensemble and stacked ranking models built on top of the base model primitives.

EnsembleRanker
--------------
Combines predictions from multiple base rankers by averaging their
cross-sectional rank scores.  Each member model is trained independently
on the same training panel and its raw score is percentile-ranked before
averaging, making the ensemble robust to score-scale differences.

StackedRanker
-------------
Two-stage meta-learner:
  Stage 1: Train N base rankers and generate out-of-fold rank scores.
  Stage 2: Train a Ridge meta-learner on the stage-1 scores to produce
           the final combined signal.

Both models implement the standard BaseModel interface (fit / predict).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from smartsignal.models.base  import BaseModel
from smartsignal.models.types import ModelFamily

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# EnsembleRanker
# ──────────────────────────────────────────────────────────────

class EnsembleRanker(BaseModel):
    """
    Simple averaging ensemble of multiple base rankers.

    Parameters
    ----------
    members      : list of (name, BaseModel) tuples.
    weights      : optional per-member weights; defaults to equal weight.
    feature_cols : needed for interface consistency (members manage their own).
    label_col    : label column name.
    """

    model_family: str = ModelFamily.LGBM   # representative

    def __init__(
        self,
        members:      List[BaseModel],
        weights:      Optional[List[float]] = None,
        feature_cols: Optional[List[str]]   = None,
        label_col:    str = "relevance",
    ):
        if not members:
            raise ValueError("EnsembleRanker requires at least one member model.")

        self.members      = members
        self.weights      = weights or [1.0 / len(members)] * len(members)
        self.feature_cols = feature_cols or []
        self.label_col    = label_col

        if len(self.weights) != len(members):
            raise ValueError("len(weights) must equal len(members).")

        # Normalise weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def fit(
        self,
        train_panel: pd.DataFrame,
        run_feature_selection: bool = True,
    ) -> "EnsembleRanker":
        for i, model in enumerate(self.members):
            logger.debug("EnsembleRanker: fitting member %d/%d …", i + 1, len(self.members))
            model.fit(train_panel, run_feature_selection=run_feature_selection)
        return self

    def predict(self, test_panel: pd.DataFrame) -> pd.Series:
        all_scores: List[pd.Series] = []

        for model in self.members:
            raw = model.predict(test_panel)
            # Percentile rank within each date to normalise scales
            ranked = raw.groupby(level=0).rank(method="average", pct=True)
            all_scores.append(ranked)

        # Weighted average
        combined = sum(w * s for w, s in zip(self.weights, all_scores))
        return combined.rename("rank_score")


# ──────────────────────────────────────────────────────────────
# StackedRanker
# ──────────────────────────────────────────────────────────────

class StackedRanker(BaseModel):
    """
    Two-stage stacked ranker (meta-learner on out-of-fold predictions).

    Stage 1: Train each base ranker and collect their test-fold rank scores
             using an internal walk-forward split on the training panel.
    Stage 2: Train a Ridge meta-learner on those scores.
    Final:   At test time, average Stage-1 scores and apply the Ridge weight.

    Parameters
    ----------
    base_models  : list of BaseModel instances (stage-1 learners).
    feature_cols : feature columns.
    label_col    : label column.
    meta_alpha   : Ridge regularisation for the meta-learner.
    n_inner_folds: number of inner folds for out-of-fold generation.
    """

    model_family: str = ModelFamily.LGBM

    def __init__(
        self,
        base_models:   List[BaseModel],
        feature_cols:  List[str],
        label_col:     str   = "relevance",
        meta_alpha:    float = 1.0,
        n_inner_folds: int   = 3,
    ):
        if not base_models:
            raise ValueError("StackedRanker requires at least one base model.")

        self.base_models    = base_models
        self.feature_cols   = feature_cols
        self.label_col      = label_col
        self.n_inner_folds  = n_inner_folds
        self._meta_learner  = Ridge(alpha=meta_alpha, fit_intercept=True)
        self._meta_scaler   = StandardScaler()
        self._fitted        = False

    def fit(
        self,
        train_panel: pd.DataFrame,
        run_feature_selection: bool = True,
    ) -> "StackedRanker":
        from smartsignal.models.splits_panel import PanelWalkForwardSplitter

        dates = train_panel.index.unique().sort_values()
        n_dates = len(dates)

        # Generate out-of-fold stage-1 predictions on the training panel
        oof_chunks: List[pd.DataFrame] = []
        inner_splitter = PanelWalkForwardSplitter(
            train_years = max(1, n_dates // (252 * (self.n_inner_folds + 1))),
            test_months = max(1, n_dates // (21 * self.n_inner_folds)),
            embargo_days = 5,
        )

        for inner_train, inner_test in inner_splitter.split(train_panel):
            fold_scores = {}
            for i, model in enumerate(self.base_models):
                model.fit(inner_train, run_feature_selection=run_feature_selection)
                s = model.predict(inner_test)
                fold_scores[f"model_{i}"] = (
                    s.groupby(level=0).rank(pct=True)
                )
            chunk = pd.DataFrame(fold_scores)
            if self.label_col in inner_test.columns:
                chunk["__label__"] = inner_test[self.label_col]
            oof_chunks.append(chunk)

        if oof_chunks:
            oof = pd.concat(oof_chunks).dropna()
            feat_cols = [c for c in oof.columns if c != "__label__"]
            X_meta = self._meta_scaler.fit_transform(oof[feat_cols].values.astype(np.float32))
            y_meta = oof["__label__"].values.astype(float)
            self._meta_learner.fit(X_meta, y_meta)
        else:
            logger.warning("StackedRanker: no inner folds produced; meta-learner skipped.")

        # Refit base models on the full training panel
        for model in self.base_models:
            model.fit(train_panel, run_feature_selection=run_feature_selection)

        self._feat_cols_meta = [f"model_{i}" for i in range(len(self.base_models))]
        self._fitted = True
        return self

    def predict(self, test_panel: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        stage1: Dict[str, pd.Series] = {}
        for i, model in enumerate(self.base_models):
            s = model.predict(test_panel)
            stage1[f"model_{i}"] = s.groupby(level=0).rank(pct=True)

        meta_df = pd.DataFrame(stage1).dropna()
        if meta_df.empty:
            return pd.Series(np.nan, index=test_panel.index, name="rank_score")

        X_meta = self._meta_scaler.transform(meta_df.values.astype(np.float32))
        final  = self._meta_learner.predict(X_meta)

        return pd.Series(final, index=meta_df.index, name="rank_score")
