"""
smartsignal.models.base
========================
Abstract base class (interface) that all SmartSignal models must implement.

Enforcing a uniform interface means the training engine, walk-forward loop,
and backtesting pipeline can treat any model interchangeably — whether it is
a LambdaMART ranker, a scikit-learn classifier, or a future LSTM model.

Required methods
----------------
fit(train_panel)        → self
predict(test_panel)     → pd.Series  (rank scores / probabilities / returns)

Optional methods
----------------
feature_importance_df() → pd.DataFrame
get_params()            → dict
set_params(**params)    → self
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd

from smartsignal.models.types import ModelFamily


class BaseModel(ABC):
    """
    Abstract base for all SmartSignal predictive models.

    Subclasses must implement `fit` and `predict`.
    Everything else has sensible defaults or raises NotImplementedError.
    """

    # Subclasses should set this to the appropriate ModelFamily enum value
    model_family: str = ModelFamily.LAMBDAMART

    # ── Required ──────────────────────────────────────────────

    @abstractmethod
    def fit(
        self,
        train_panel: pd.DataFrame,
        run_feature_selection: bool = True,
    ) -> "BaseModel":
        """
        Fit the model on a training panel slice.

        Parameters
        ----------
        train_panel            : labelled cross-sectional panel.
        run_feature_selection  : whether to perform feature selection.

        Returns
        -------
        self (for chaining).
        """

    @abstractmethod
    def predict(self, test_panel: pd.DataFrame) -> pd.Series:
        """
        Generate predictions (scores / probabilities / return estimates).

        Parameters
        ----------
        test_panel : panel slice for which to generate predictions.

        Returns
        -------
        pd.Series indexed like test_panel, named 'rank_score'.
        """

    # ── Optional ──────────────────────────────────────────────

    def feature_importance_df(self) -> pd.DataFrame:
        """Return a DataFrame of feature importances, if supported."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement feature_importance_df()."
        )

    def get_params(self) -> Dict[str, Any]:
        """Return a dict of model hyper-parameters."""
        return {}

    def set_params(self, **params) -> "BaseModel":
        """Set hyper-parameters (for use with Optuna or sklearn grid search)."""
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:
        params = self.get_params()
        param_str = ", ".join(f"{k}={v!r}" for k, v in list(params.items())[:5])
        return f"{self.__class__.__name__}({param_str})"
