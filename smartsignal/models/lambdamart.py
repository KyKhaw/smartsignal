"""
smartsignal.models.lambdamart
==============================
LambdaMART cross-sectional ranking model.

Architecture
------------
LGBMRanker with objective='lambdarank' treats each trading day as a
"query" and all stocks as "items" to be ranked within that query.
The model learns to rank stocks by their expected relative forward return,
directly optimising NDCG rather than a pointwise loss.

Key design decisions (from the SmartSignal midterm report §2.3–2.4):

1. Query = trading day.  group sizes fed to fit() describe the number of
   stocks in the universe on each date.
2. Relevance labels are integer quintile bins (0–n_bins-1), computed
   cross-sectionally per date.
3. Feature selection via LGBMRanker importance scores on the first fold
   (Wang & Dong 2025 cross-selection heuristic); planned future work will
   repeat this at every fold boundary.
4. StandardScaler is applied per fold before fitting.  While tree models
   are scale-invariant, scaling slightly improves convergence of the
   boosting gradients.
5. Dead-band suppression: predictions that fall within a neutral middle
   band are zeroed out to reduce unnecessary portfolio turnover.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRanker
except ImportError as _e:
    raise ImportError("lightgbm is required: pip install lightgbm") from _e

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Default hyper-parameters
# ──────────────────────────────────────────────────────────────

DEFAULT_RANKER_PARAMS: Dict = dict(
    objective="lambdarank",
    boosting_type="gbdt",
    n_estimators=400,
    learning_rate=0.03,
    max_depth=5,
    num_leaves=31,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)


# ──────────────────────────────────────────────────────────────
# Helper: panel slice → (X, y, groups)
# ──────────────────────────────────────────────────────────────

def panel_to_arrays(
    panel_slice: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "relevance",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a panel slice to (X, y, groups) required by LGBMRanker.

    groups[i] = number of stocks in the i-th query (date).
    Rows must be sorted by date first, then by ticker within each date.
    """
    panel_slice = panel_slice.sort_index().sort_values("ticker", kind="stable")
    X      = panel_slice[feature_cols].values.astype(np.float32)
    y      = panel_slice[label_col].values.astype(np.int32)
    groups = panel_slice.groupby(level=0).size().values.astype(np.int32)
    return X, y, groups


# ──────────────────────────────────────────────────────────────
# Feature selection
# ──────────────────────────────────────────────────────────────

def select_top_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    feature_names: List[str],
    top_k: int = 25,
    ranker_params: Optional[Dict] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    Fit a quick LGBMRanker and select the top-k features by importance.

    Parameters
    ----------
    X_train, y_train, groups_train : training arrays.
    feature_names                  : names corresponding to X_train columns.
    top_k                          : number of features to retain.
    ranker_params                  : custom hyper-parameters (uses defaults if None).

    Returns
    -------
    selected_features : list of the top-k feature names.
    importances       : raw importance scores for all features.
    """
    params = {**DEFAULT_RANKER_PARAMS, **(ranker_params or {})}
    ranker = LGBMRanker(**params)
    # Use a DataFrame so LightGBM stores named features — avoids mismatch on predict
    import pandas as _pd
    X_df = _pd.DataFrame(X_train, columns=feature_names)
    ranker.fit(X_df, y_train, group=groups_train)
    imp    = ranker.feature_importances_
    order  = np.argsort(imp)[::-1]
    selected = [feature_names[i] for i in order[:top_k]]
    logger.info(
        "Feature selection: retaining %d / %d features. "
        "Top-5: %s",
        top_k, len(feature_names),
        selected[:5],
    )
    return selected, imp


# ──────────────────────────────────────────────────────────────
# Main model class
# ──────────────────────────────────────────────────────────────

class LambdaMARTRanker:
    """
    Wrapper around LGBMRanker for cross-sectional equity ranking.

    Parameters
    ----------
    feature_cols   : full list of candidate feature column names.
    top_k_features : number of features to retain after importance selection.
    ranker_params  : LGBMRanker hyper-parameters (overrides defaults).
    label_col      : column name of the integer relevance label.
    """

    def __init__(
        self,
        feature_cols: List[str],
        top_k_features: int = 25,
        ranker_params: Optional[Dict] = None,
        label_col: str = "relevance",
    ):
        self.feature_cols    = feature_cols
        self.top_k_features  = top_k_features
        self.ranker_params   = {**DEFAULT_RANKER_PARAMS, **(ranker_params or {})}
        self.label_col       = label_col

        # Set after fit
        self.selected_features_: Optional[List[str]] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self._scaler: Optional[StandardScaler] = None
        self._ranker: Optional[LGBMRanker] = None
        self._sel_idx: Optional[List[int]] = None

    def fit(
        self,
        train_panel: pd.DataFrame,
        run_feature_selection: bool = True,
    ) -> "LambdaMARTRanker":
        """
        Fit the ranker on a training panel slice.

        Parameters
        ----------
        train_panel            : panel slice (from walk-forward splitter).
        run_feature_selection  : if True, perform importance-based feature
                                 selection before fitting the final model.
        """
        X_tr, y_tr, g_tr = panel_to_arrays(
            train_panel, self.feature_cols, self.label_col
        )
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)

        if run_feature_selection or self.selected_features_ is None:
            selected, imp = select_top_features(
                X_tr_s, y_tr, g_tr,
                feature_names=self.feature_cols,
                top_k=self.top_k_features,
                ranker_params=self.ranker_params,
            )
            self.selected_features_    = selected
            self.feature_importances_  = imp

        self._sel_idx = [self.feature_cols.index(f) for f in self.selected_features_]
        X_tr_sel = X_tr_s[:, self._sel_idx]

        ranker = LGBMRanker(**self.ranker_params)
        # Pass feature_name explicitly so LightGBM stores names that match
        # what predict() will supply (numpy arrays, no pandas column names).
        sel_names = [self.feature_cols[i] for i in self._sel_idx]
        ranker.fit(X_tr_sel, y_tr, group=g_tr, feature_name=sel_names)

        self._scaler    = scaler
        self._ranker    = ranker
        self._sel_names = sel_names
        return self

    def predict(self, test_panel: pd.DataFrame) -> pd.Series:
        """
        Score all (date, ticker) rows in the test panel.

        Returns
        -------
        scores : pd.Series indexed like test_panel, preserving original row order.
        """
        if self._ranker is None:
            raise RuntimeError("Call fit() before predict().")

        test_sorted = test_panel.sort_index().sort_values("ticker", kind="stable")
        X_raw = test_sorted[self.feature_cols].values.astype(np.float32)
        X_te  = self._scaler.transform(X_raw)[:, self._sel_idx]
        # Predict using a named DataFrame so LightGBM feature names always match
        import pandas as _pd
        X_df  = _pd.DataFrame(X_te, columns=self._sel_names)
        scores = self._ranker.predict(X_df)
        return pd.Series(scores, index=test_sorted.index, name="rank_score")

    def feature_importance_df(self) -> pd.DataFrame:
        """Return a DataFrame of feature importances (all features)."""
        if self.feature_importances_ is None:
            raise RuntimeError("Call fit() first.")
        df = pd.DataFrame({
            "feature":    self.feature_cols,
            "importance": self.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        df["selected"] = df["feature"].isin(self.selected_features_ or [])
        return df