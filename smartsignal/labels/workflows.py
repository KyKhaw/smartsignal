"""
smartsignal.labels.workflows
==============================
End-to-end label generation workflows that combine forward-return computation,
label building, timing trimming, and panel validation into single callable steps.

Workflows
---------
build_ranking_labels(panel)   : main workflow for LambdaMART quintile labels
build_classification_labels() : workflow for binary direction classifiers
build_regression_labels()     : workflow for regression targets
run_label_pipeline()          : configurable unified entry point
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from smartsignal.labels.builders import get_builder
from smartsignal.labels.timing  import (
    compute_panel_forward_returns,
    trim_label_horizon,
    min_embargo_days,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Workflow: LambdaMART quintile labels (primary)
# ──────────────────────────────────────────────────────────────

def build_ranking_labels(
    panel:        pd.DataFrame,
    forward_days: int = 5,
    n_bins:       int = 4,
    label_col:    str = "relevance",
    fwd_ret_col:  str = "fwd_ret",
    trim_horizon: bool = True,
    min_obs:      int  = 8,
    verbose:      bool = True,
) -> pd.DataFrame:
    """
    Build cross-sectional quintile relevance labels for LambdaMART.

    Steps
    -----
    1. Compute per-ticker forward returns (if not already present).
    2. Apply QuintileBuilder to assign cross-sectional quantile labels.
    3. Trim the final forward_days rows (no valid labels).
    4. Report label distribution.

    Parameters
    ----------
    panel        : feature panel with 'close' and 'ticker' columns.
    forward_days : forward return horizon in bars.
    n_bins       : number of quantile buckets (default 4 → labels 0-3).
    label_col    : output column name.
    fwd_ret_col  : column to use/store the forward return.
    trim_horizon : whether to drop unlabelled tail rows.
    min_obs      : minimum cross-section size to assign labels.
    verbose      : print diagnostics.

    Returns
    -------
    Labelled panel.
    """
    # Step 1: forward return
    if fwd_ret_col not in panel.columns:
        if verbose:
            print(f"[Labels] Computing {forward_days}-day forward returns …")
        panel = compute_panel_forward_returns(
            panel, horizons=[forward_days], price_col="close"
        )
        # rename to the expected fwd_ret_col
        panel = panel.rename(columns={f"fwd_ret_{forward_days}d": fwd_ret_col})

    # Step 2: quintile labels
    builder = get_builder("quintile", n_bins=n_bins, label_col=label_col, min_obs=min_obs)
    panel   = builder.build(panel, fwd_ret_col=fwd_ret_col)

    # Step 3: trim horizon
    if trim_horizon:
        panel = trim_label_horizon(panel, forward_days=forward_days, fwd_ret_col=fwd_ret_col)

    if verbose:
        n_labelled = panel[label_col].notna().sum()
        dist       = panel[label_col].value_counts().sort_index().to_dict()
        print(
            f"[Labels] {n_labelled:,} labelled rows | "
            f"distribution: {dist} | "
            f"recommended embargo: ≥{min_embargo_days(forward_days)} days"
        )

    return panel


# ──────────────────────────────────────────────────────────────
# Workflow: binary direction labels
# ──────────────────────────────────────────────────────────────

def build_classification_labels(
    panel:        pd.DataFrame,
    forward_days: int   = 5,
    threshold:    float = 0.0,
    label_col:    str   = "direction",
    fwd_ret_col:  str   = "fwd_ret",
    trim_horizon: bool  = True,
    verbose:      bool  = True,
) -> pd.DataFrame:
    """Binary direction label workflow."""
    if fwd_ret_col not in panel.columns:
        panel = compute_panel_forward_returns(panel, horizons=[forward_days])
        panel = panel.rename(columns={f"fwd_ret_{forward_days}d": fwd_ret_col})

    builder = get_builder("binary_direction", threshold=threshold, label_col=label_col)
    panel   = builder.build(panel, fwd_ret_col=fwd_ret_col)

    if trim_horizon:
        panel = trim_label_horizon(panel, forward_days=forward_days)

    if verbose:
        dist = panel[label_col].value_counts().sort_index().to_dict()
        print(f"[Labels] Binary direction labels: {dist}")

    return panel


# ──────────────────────────────────────────────────────────────
# Workflow: regression labels
# ──────────────────────────────────────────────────────────────

def build_regression_labels(
    panel:            pd.DataFrame,
    forward_days:     int   = 5,
    fwd_ret_col:      str   = "fwd_ret",
    label_col:        str   = "target_ret",
    winsorise:        bool  = True,
    winsorise_bounds: tuple = (0.01, 0.99),
    trim_horizon:     bool  = True,
    verbose:          bool  = True,
) -> pd.DataFrame:
    """Regression target label workflow."""
    if fwd_ret_col not in panel.columns:
        panel = compute_panel_forward_returns(panel, horizons=[forward_days])
        panel = panel.rename(columns={f"fwd_ret_{forward_days}d": fwd_ret_col})

    builder = get_builder(
        "regression",
        winsorise=winsorise,
        winsorise_bounds=winsorise_bounds,
        label_col=label_col,
    )
    panel = builder.build(panel, fwd_ret_col=fwd_ret_col)

    if trim_horizon:
        panel = trim_label_horizon(panel, forward_days=forward_days)

    if verbose:
        print(
            f"[Labels] Regression target '{label_col}': "
            f"mean={panel[label_col].mean():.4f}, "
            f"std={panel[label_col].std():.4f}"
        )

    return panel


# ──────────────────────────────────────────────────────────────
# Unified entry point
# ──────────────────────────────────────────────────────────────

def run_label_pipeline(
    panel:        pd.DataFrame,
    label_type:   str  = "quintile",
    forward_days: int  = 5,
    verbose:      bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Dispatch to the appropriate label workflow.

    Parameters
    ----------
    panel        : feature panel.
    label_type   : 'quintile', 'binary_direction', or 'regression'.
    forward_days : forward return horizon.
    **kwargs     : passed to the underlying workflow function.
    """
    workflows = {
        "quintile":         build_ranking_labels,
        "binary_direction": build_classification_labels,
        "regression":       build_regression_labels,
    }

    if label_type not in workflows:
        raise ValueError(
            f"Unknown label_type '{label_type}'. "
            f"Choose from: {list(workflows)}."
        )

    return workflows[label_type](
        panel, forward_days=forward_days, verbose=verbose, **kwargs
    )
