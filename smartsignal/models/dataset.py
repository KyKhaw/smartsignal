"""
smartsignal.models.dataset
============================
Dataset preparation utilities for single-asset (time-series) models.

For cross-sectional panel datasets used by the LambdaMART ranker, see
panel_dataset.py.

Responsibilities
----------------
- Convert raw DataFrames into numpy arrays ready for scikit-learn / LightGBM.
- Apply StandardScaler per fold (fit on train, transform test).
- Provide SequenceDataset for sequence-based models (future LSTM support).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────────────────────
# Tabular dataset (cross-sectional, single-date)
# ──────────────────────────────────────────────────────────────

class TabularDataset:
    """
    Converts a panel slice to (X, y) numpy arrays for standard ML models.

    Parameters
    ----------
    feature_cols : list of feature column names.
    label_col    : target column name.
    scale        : whether to StandardScaler the features.
    """

    def __init__(
        self,
        feature_cols: List[str],
        label_col:    str  = "relevance",
        scale:        bool = True,
    ):
        self.feature_cols = feature_cols
        self.label_col    = label_col
        self.scale        = scale
        self._scaler: Optional[StandardScaler] = None

    def fit_transform(
        self, panel: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the scaler on the panel and return (X, y).
        Call this on the training slice.
        """
        X = panel[self.feature_cols].values.astype(np.float32)
        y = panel[self.label_col].values

        if self.scale:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        return X, y

    def transform(
        self, panel: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform a new panel using the already-fitted scaler.
        Call this on the test/validation slice.
        """
        X = panel[self.feature_cols].values.astype(np.float32)

        if self.scale:
            if self._scaler is None:
                raise RuntimeError("Call fit_transform() before transform().")
            X = self._scaler.transform(X)

        y = panel[self.label_col].values if self.label_col in panel.columns else None
        return X, y

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse StandardScaler transformation."""
        if self._scaler is not None:
            return self._scaler.inverse_transform(X)
        return X


# ──────────────────────────────────────────────────────────────
# Time-series sequence dataset (for future LSTM / sequence models)
# ──────────────────────────────────────────────────────────────

class SequenceDataset:
    """
    Builds lookback-window sequences from a single-ticker time series.

    For each time step t, creates a feature matrix of shape
    (lookback_window, n_features) representing bars [t-lookback, t-1].

    Parameters
    ----------
    feature_cols    : list of feature column names.
    label_col       : target column name.
    lookback_window : number of past bars to include in each sequence.
    scale           : whether to StandardScaler the features.
    """

    def __init__(
        self,
        feature_cols:    List[str],
        label_col:       str  = "direction",
        lookback_window: int  = 20,
        scale:           bool = True,
    ):
        self.feature_cols    = feature_cols
        self.label_col       = label_col
        self.lookback_window = lookback_window
        self.scale           = scale
        self._scaler: Optional[StandardScaler] = None

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build sequences from a training DataFrame.

        Returns
        -------
        X : shape (n_samples, lookback_window, n_features)
        y : shape (n_samples,)
        """
        feat = df[self.feature_cols].values.astype(np.float32)

        if self.scale:
            self._scaler = StandardScaler()
            feat = self._scaler.fit_transform(feat)

        X, y = self._build_sequences(feat, df[self.label_col].values)
        return X, y

    def transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        feat = df[self.feature_cols].values.astype(np.float32)

        if self.scale and self._scaler is not None:
            feat = self._scaler.transform(feat)

        y = df[self.label_col].values if self.label_col in df.columns else None
        if y is not None:
            X, y = self._build_sequences(feat, y)
        else:
            X, _ = self._build_sequences(feat, np.zeros(len(feat)))
        return X, y

    def _build_sequences(
        self,
        feat: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n  = len(feat)
        lb = self.lookback_window
        Xs = []
        ys = []
        for i in range(lb, n):
            Xs.append(feat[i - lb: i])
            ys.append(labels[i])
        return np.array(Xs, dtype=np.float32), np.array(ys)