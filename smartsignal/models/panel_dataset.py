"""
smartsignal.models.panel_dataset
==================================
Cross-sectional panel dataset utilities specifically for the LambdaMART
learning-to-rank framework.

The key difference from TabularDataset:
  LGBMRanker requires a `group` array that specifies the number of items
  (stocks) per query (trading day).  This module handles the conversion
  from a panel DataFrame to (X, y, groups) arrays with correct ordering.

PanelDataset also manages:
  - consistent sorting (date → ticker) required by LGBMRanker
  - index tracking so predictions can be re-aligned to the original panel
  - per-fold StandardScaler fitting with selected feature sub-indexing
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class PanelDataset:
    """
    Prepares a cross-sectional panel for LambdaMART training.

    Parameters
    ----------
    feature_cols : full list of candidate feature columns.
    label_col    : integer relevance label column.
    scale        : whether to StandardScaler features (recommended).
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

        self._scaler:   Optional[StandardScaler] = None
        self._sel_idx:  Optional[List[int]]      = None   # selected feature indices

    # ── Fit (training panel) ──────────────────────────────────

    def fit_transform(
        self,
        panel: pd.DataFrame,
        selected_features: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sort the training panel, fit a scaler, and return (X, y, groups).

        Parameters
        ----------
        panel             : labelled training slice.
        selected_features : subset of feature_cols to use; uses all if None.

        Returns
        -------
        X      : (n_rows, n_features) float32 array.
        y      : (n_rows,) int32 label array.
        groups : (n_dates,) int32 array of per-date stock counts.
        """
        panel = self._sort_panel(panel)
        X_raw = panel[self.feature_cols].values.astype(np.float32)

        if self.scale:
            self._scaler = StandardScaler()
            X_scaled     = self._scaler.fit_transform(X_raw)
        else:
            X_scaled = X_raw

        # Feature sub-selection
        if selected_features is not None:
            self._sel_idx = [self.feature_cols.index(f) for f in selected_features]
        else:
            self._sel_idx = list(range(len(self.feature_cols)))

        X      = X_scaled[:, self._sel_idx].astype(np.float32)
        y      = panel[self.label_col].values.astype(np.int32)
        groups = panel.groupby(level=0).size().values.astype(np.int32)

        return X, y, groups

    # ── Transform (test panel) ────────────────────────────────

    def transform(
        self,
        panel: pd.DataFrame,
    ) -> Tuple[np.ndarray, pd.Index]:
        """
        Transform a test panel using the fitted scaler and feature selection.

        Returns
        -------
        X     : (n_rows, n_selected_features) float32 array.
        index : DatetimeIndex to re-align predictions.
        """
        panel  = self._sort_panel(panel)
        X_raw  = panel[self.feature_cols].values.astype(np.float32)

        if self.scale:
            if self._scaler is None:
                raise RuntimeError("Call fit_transform() before transform().")
            X_scaled = self._scaler.transform(X_raw)
        else:
            X_scaled = X_raw

        idx = self._sel_idx or list(range(len(self.feature_cols)))
        X   = X_scaled[:, idx].astype(np.float32)

        return X, panel.index

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _sort_panel(panel: pd.DataFrame) -> pd.DataFrame:
        """Sort panel by date first, then by ticker — required by LGBMRanker."""
        return panel.sort_index().sort_values("ticker", kind="stable")

    def selected_feature_names(self) -> List[str]:
        """Return the currently selected feature column names."""
        if self._sel_idx is None:
            return self.feature_cols
        return [self.feature_cols[i] for i in self._sel_idx]

    def n_features(self) -> int:
        """Number of active features after selection."""
        if self._sel_idx is None:
            return len(self.feature_cols)
        return len(self._sel_idx)
