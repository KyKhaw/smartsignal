"""
smartsignal.labels.generator
=============================
Cross-sectional relevance label generation for the LambdaMART ranker.

LGBMRanker requires INTEGER relevance scores where higher = better.
We bin stocks by their forward return rank within each trading day's
cross-section into Q equal-width quantile buckets:

  n_bins=4 (default) → labels 0,1,2,3
    label 3 = top-quartile expected performer  → LONG candidates
    label 0 = bottom-quartile expected performer → SHORT candidates

This is the same labelling scheme used in the CSM LambdaMART notebook
(Section 3.3 of the SmartSignal midterm report).

Additional label types supported:
  - binary_direction : 1 if forward_return > 0 else 0 (simple direction)
  - regression       : raw forward return (for regression models)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Core label functions
# ──────────────────────────────────────────────────────────────

def _quintile_label(s: pd.Series, n_bins: int) -> pd.Series:
    """
    Assign cross-sectional quantile bin labels within a single date's returns.

    Returns NaN for the whole cross-section if fewer than n_bins stocks
    have valid forward returns.
    """
    valid = s.dropna()
    if len(valid) < n_bins:
        return pd.Series(np.nan, index=s.index)
    return pd.qcut(
        s.rank(method="first"),
        q=n_bins,
        labels=False,
        duplicates="drop",
    )


def add_quintile_labels(
    panel: pd.DataFrame,
    n_bins: int = 4,
    fwd_ret_col: str = "fwd_ret",
    label_col: str = "relevance",
    drop_unlabelled: bool = True,
) -> pd.DataFrame:
    """
    Add integer cross-sectional quantile relevance labels to the panel.

    Parameters
    ----------
    panel           : feature panel with DatetimeIndex and 'fwd_ret' column.
    n_bins          : number of quantile buckets (default 4 → quartile labels 0-3).
    fwd_ret_col     : name of the forward-return column (already in the panel).
    label_col       : name of the output label column.
    drop_unlabelled : drop rows where the label is NaN.

    Returns
    -------
    panel with `label_col` column added, dtype int.
    """
    panel = panel.copy()

    panel[label_col] = (
        panel.groupby(level=0)[fwd_ret_col]
             .transform(lambda s: _quintile_label(s, n_bins))
             .astype("Int64")
    )

    if drop_unlabelled:
        before = len(panel)
        panel = panel.dropna(subset=[label_col, fwd_ret_col])
        after  = len(panel)
        if before - after > 0:
            logger.debug(
                "Dropped %d rows with missing relevance labels.", before - after
            )

    panel[label_col] = panel[label_col].astype(int)

    label_dist = panel[label_col].value_counts().sort_index().to_dict()
    logger.info("Relevance label distribution: %s", label_dist)

    return panel


def add_binary_direction_labels(
    panel: pd.DataFrame,
    fwd_ret_col: str = "fwd_ret",
    label_col: str = "direction",
    drop_unlabelled: bool = True,
) -> pd.DataFrame:
    """
    Add binary direction labels: 1 if forward return > 0 else 0.

    Useful as an alternative target for binary classifiers.
    """
    panel = panel.copy()
    panel[label_col] = (panel[fwd_ret_col] > 0).astype("Int64")
    if drop_unlabelled:
        panel = panel.dropna(subset=[fwd_ret_col])
    panel[label_col] = panel[label_col].fillna(0).astype(int)
    return panel


def add_regression_labels(
    panel: pd.DataFrame,
    fwd_ret_col: str = "fwd_ret",
    winsorise: bool = True,
    winsorise_bounds: tuple = (0.01, 0.99),
    drop_unlabelled: bool = True,
) -> pd.DataFrame:
    """
    Add raw forward-return regression targets (optionally winsorised).

    Used when training regression models instead of rankers.
    """
    panel = panel.copy()
    if winsorise and fwd_ret_col in panel.columns:
        lo = panel[fwd_ret_col].quantile(winsorise_bounds[0])
        hi = panel[fwd_ret_col].quantile(winsorise_bounds[1])
        panel["target_ret"] = panel[fwd_ret_col].clip(lo, hi)
    else:
        panel["target_ret"] = panel[fwd_ret_col]

    if drop_unlabelled:
        panel = panel.dropna(subset=["target_ret"])
    return panel


# ──────────────────────────────────────────────────────────────
# Convenience wrapper
# ──────────────────────────────────────────────────────────────

def generate_labels(
    panel: pd.DataFrame,
    label_type: str = "quintile",
    n_bins: int = 4,
    forward_days: Optional[int] = None,
    fwd_ret_col: str = "fwd_ret",
    drop_unlabelled: bool = True,
) -> pd.DataFrame:
    """
    Unified label generation entry point.

    Parameters
    ----------
    panel          : feature panel (output of build_feature_panel).
    label_type     : 'quintile' | 'binary_direction' | 'regression'.
    n_bins         : quantile bins for 'quintile' mode.
    forward_days   : if provided and fwd_ret is missing, compute it here
                     as close.pct_change(forward_days).shift(-forward_days).
    fwd_ret_col    : column containing pre-computed forward returns.
    drop_unlabelled: drop rows where the label is NaN.

    Returns
    -------
    panel with label columns added.
    """
    # Optionally compute forward return on the fly
    if forward_days is not None and fwd_ret_col not in panel.columns:
        if "close" not in panel.columns:
            raise ValueError(
                "'close' column required to compute forward returns. "
                "Pass forward_days=None if 'fwd_ret' is already in the panel."
            )
        panel = panel.copy()
        panel[fwd_ret_col] = (
            panel.groupby("ticker")["close"]
                 .transform(lambda s: s.pct_change(forward_days).shift(-forward_days))
        )

    if label_type == "quintile":
        return add_quintile_labels(
            panel, n_bins=n_bins, fwd_ret_col=fwd_ret_col,
            drop_unlabelled=drop_unlabelled
        )
    elif label_type == "binary_direction":
        return add_binary_direction_labels(
            panel, fwd_ret_col=fwd_ret_col, drop_unlabelled=drop_unlabelled
        )
    elif label_type == "regression":
        return add_regression_labels(
            panel, fwd_ret_col=fwd_ret_col, drop_unlabelled=drop_unlabelled
        )
    else:
        raise ValueError(
            f"Unknown label_type '{label_type}'. "
            "Choose from: 'quintile', 'binary_direction', 'regression'."
        )
