"""
smartsignal.features.transforms
================================
Cross-sectional normalisation and transformation utilities.

All transforms operate per-date across the ticker universe (cross-sectional),
consistent with how LambdaMART ranks stocks within each daily query.

Functions
---------
cs_zscore          : cross-sectional z-score normalisation
cs_minmax          : cross-sectional min-max scaling
cs_rank            : cross-sectional percentile rank (0–1)
cs_winsorise       : winsorise extreme values before normalisation
apply_cs_transforms: apply a configurable transform pipeline to a panel
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────
# Individual transforms (operate on a single cross-section)
# ──────────────────────────────────────────────────────────────

def cs_zscore(s: pd.Series, min_obs: int = 5) -> pd.Series:
    """
    Cross-sectional z-score: (x - mean) / std.

    Returns NaN for the whole cross-section if fewer than `min_obs`
    non-NaN values are available.
    """
    valid = s.dropna()
    if len(valid) < min_obs:
        return pd.Series(np.nan, index=s.index)
    mu  = valid.mean()
    sig = valid.std()
    if sig < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sig


def cs_minmax(s: pd.Series, min_obs: int = 5) -> pd.Series:
    """Cross-sectional min-max scaling → [0, 1]."""
    valid = s.dropna()
    if len(valid) < min_obs:
        return pd.Series(np.nan, index=s.index)
    lo, hi = valid.min(), valid.max()
    if (hi - lo) < 1e-12:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def cs_rank(s: pd.Series, min_obs: int = 5) -> pd.Series:
    """
    Cross-sectional percentile rank, scaled to [0, 1].
    Ties are broken by averaging (method='average').
    """
    valid = s.dropna()
    if len(valid) < min_obs:
        return pd.Series(np.nan, index=s.index)
    ranked = s.rank(method="average", na_option="keep")
    return (ranked - 1) / (len(valid) - 1 + 1e-12)


def cs_winsorise(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Winsorise values at the given quantile bounds, cross-sectionally.
    Call this before z-scoring to reduce the influence of outliers.
    """
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


# ──────────────────────────────────────────────────────────────
# Panel-level transform pipeline
# ──────────────────────────────────────────────────────────────

def apply_cs_transforms(
    panel: pd.DataFrame,
    feature_cols: List[str],
    method: str = "zscore",
    winsorise: bool = True,
    winsorise_bounds: Tuple[float, float] = (0.01, 0.99),
    min_obs: int = 5,
) -> pd.DataFrame:
    """
    Apply cross-sectional transforms to all feature columns in the panel,
    grouping by date.

    Parameters
    ----------
    panel            : DataFrame with DatetimeIndex and feature columns.
                       Must also contain a 'ticker' column.
    feature_cols     : list of columns to transform (in-place copy).
    method           : 'zscore', 'minmax', or 'rank'.
    winsorise        : whether to winsorise before normalising.
    winsorise_bounds : (lower_quantile, upper_quantile) for winsorising.
    min_obs          : minimum cross-section size to apply transforms.

    Returns
    -------
    panel_t : transformed copy of the panel.
    """
    _transforms = {
        "zscore": cs_zscore,
        "minmax": cs_minmax,
        "rank":   cs_rank,
    }
    if method not in _transforms:
        raise ValueError(f"method must be one of {list(_transforms)}; got '{method}'.")

    transform_fn = _transforms[method]
    panel_t = panel.copy()

    valid_cols = [c for c in feature_cols if c in panel_t.columns]

    def _apply_date(grp: pd.DataFrame) -> pd.DataFrame:
        for col in valid_cols:
            s = grp[col]
            if winsorise:
                s = cs_winsorise(s, *winsorise_bounds)
            grp[col] = transform_fn(s, min_obs=min_obs).values
        return grp

    panel_t = panel_t.groupby(level=0, group_keys=False).apply(_apply_date)
    return panel_t


def forward_fill_panel(
    panel: pd.DataFrame,
    feature_cols: List[str],
    limit: int = 5,
) -> pd.DataFrame:
    """
    Forward-fill missing feature values within each ticker's time series.

    Parameters
    ----------
    panel        : stacked panel with 'ticker' column and DatetimeIndex.
    feature_cols : columns to forward-fill.
    limit        : maximum number of consecutive NaNs to fill.
    """
    panel = panel.copy()
    for ticker, grp in panel.groupby("ticker"):
        idx = grp.index
        for col in feature_cols:
            if col in panel.columns:
                panel.loc[idx, col] = grp[col].ffill(limit=limit).values
    return panel


def add_cross_sectional_features(
    panel: pd.DataFrame,
    base_features: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Augment the panel with cross-sectional rank versions of selected features.

    For each feature in `base_features`, adds a '<feature>_csrank' column
    containing the cross-sectional percentile rank (0–1) on that date.
    These rank features are stable across price-level regimes and complement
    the raw indicator values.

    Returns
    -------
    panel_augmented : panel with rank columns added.
    new_cols        : list of the newly added column names.
    """
    panel = panel.copy()
    new_cols: List[str] = []

    for feat in base_features:
        if feat not in panel.columns:
            continue
        rank_col = f"{feat}_csrank"
        panel[rank_col] = (
            panel.groupby(level=0)[feat]
                 .transform(lambda s: cs_rank(s, min_obs=3))
        )
        new_cols.append(rank_col)

    return panel, new_cols
