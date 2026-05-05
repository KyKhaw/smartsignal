"""
smartsignal.models.types
==========================
Shared type aliases, enums, and lightweight dataclasses used throughout
the models module.

Keeping types in one place prevents circular imports and makes the
interfaces of trainers, adapters, and registries explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────

class ModelFamily(str, Enum):
    """Supported model families."""
    LAMBDAMART  = "lambdamart"   # LightGBM LambdaMART ranker (primary)
    LGBM        = "lgbm"         # LightGBM classifier/regressor
    XGBOOST     = "xgboost"      # XGBoost classifier/regressor
    RANDOM_FOREST = "random_forest"
    RIDGE       = "ridge"
    LASSO       = "lasso"
    ELASTIC_NET = "elastic_net"
    SVM         = "svm"


class LabelType(str, Enum):
    """Supported label/target types."""
    QUINTILE          = "quintile"
    BINARY_DIRECTION  = "binary_direction"
    REGRESSION        = "regression"
    MOMENTUM          = "momentum"


class ValidationMode(str, Enum):
    """Walk-forward validation mode."""
    EXPANDING = "expanding"
    ROLLING   = "rolling"


class RebalanceFreq(str, Enum):
    """Portfolio rebalancing frequency."""
    DAILY   = "D"
    WEEKLY  = "W"
    MONTHLY = "M"


# ──────────────────────────────────────────────────────────────
# Prediction output dataclass
# ──────────────────────────────────────────────────────────────

@dataclass
class ModelPrediction:
    """
    Standardised prediction output from any SmartSignal model.

    Attributes
    ----------
    scores      : raw model output (rank score, probability, or regression value).
    fold_id     : walk-forward fold this prediction belongs to.
    feature_cols: features used to produce this prediction.
    model_family: family of the underlying model.
    meta        : arbitrary metadata (feature importances, attention weights …).
    """
    scores:       pd.Series
    fold_id:      int                     = 0
    feature_cols: List[str]               = field(default_factory=list)
    model_family: str                     = ModelFamily.LAMBDAMART
    meta:         Dict[str, Any]          = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Convert to a DataFrame suitable for merging with the panel."""
        return self.scores.rename("rank_score").to_frame()


# ──────────────────────────────────────────────────────────────
# Training result dataclass
# ──────────────────────────────────────────────────────────────

@dataclass
class TrainingResult:
    """
    Container for the outputs of a single walk-forward training fold.

    Attributes
    ----------
    fold_id           : index of this fold.
    train_start/end   : training period bounds.
    test_start/end    : test period bounds.
    predictions       : ModelPrediction for the test window.
    val_sharpe        : validation Sharpe ratio (negative = bad).
    selected_features : feature subset used in this fold.
    n_train_rows      : number of training rows.
    n_test_rows       : number of test rows.
    """
    fold_id:           int
    train_start:       pd.Timestamp
    train_end:         pd.Timestamp
    test_start:        pd.Timestamp
    test_end:          pd.Timestamp
    predictions:       ModelPrediction
    val_sharpe:        float              = np.nan
    selected_features: List[str]          = field(default_factory=list)
    n_train_rows:      int                = 0
    n_test_rows:       int                = 0

    def summary(self) -> str:
        return (
            f"Fold {self.fold_id:02d} | "
            f"Train: {self.train_start.date()}–{self.train_end.date()} "
            f"({self.n_train_rows:,} rows) | "
            f"Test:  {self.test_start.date()}–{self.test_end.date()} "
            f"({self.n_test_rows:,} rows) | "
            f"Val Sharpe: {self.val_sharpe:.3f}"
        )