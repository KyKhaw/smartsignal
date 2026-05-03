"""
smartsignal.features.timing
=============================
Lookback period tracking and warmup utilities for the feature pipeline.

Every technical indicator requires a minimum number of historical bars
to produce a valid (non-NaN) output.  Tracking these requirements:

  1. Prevents training on rows that are within the warmup period.
  2. Lets the pipeline auto-compute the correct offset when joining
     feature data to label data (ensuring no label leakage through NaN rows).
  3. Supports the embargo logic in the walk-forward splitter by verifying
     that each fold's training window is long enough.

The FEATURE_LOOKBACKS dict is the single source of truth; all other helpers
derive from it.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from smartsignal.features.equity_features import FEATURE_LOOKBACKS

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Lookback queries
# ──────────────────────────────────────────────────────────────

def get_lookback(feature: str) -> int:
    """
    Return the minimum lookback (bars) required for a single feature.

    Raises KeyError if the feature is not in the registry.
    """
    if feature not in FEATURE_LOOKBACKS:
        raise KeyError(
            f"Feature '{feature}' not in FEATURE_LOOKBACKS. "
            "Register custom features via equity_features.FEATURE_LOOKBACKS."
        )
    return FEATURE_LOOKBACKS[feature]


def max_lookback(feature_list: Optional[List[str]] = None) -> int:
    """
    Return the maximum lookback across the given features (or all features).

    Parameters
    ----------
    feature_list : list of feature names; if None, uses all registered features.
    """
    if feature_list is None:
        return max(FEATURE_LOOKBACKS.values())
    return max(
        (FEATURE_LOOKBACKS.get(f, 0) for f in feature_list),
        default=0,
    )


def warmup_end_date(
    df: pd.DataFrame,
    feature_list: Optional[List[str]] = None,
    extra_bars: int = 0,
) -> pd.Timestamp:
    """
    Return the first date at which all features produce valid outputs.

    Parameters
    ----------
    df           : DataFrame with DatetimeIndex.
    feature_list : features to consider; defaults to all registered features.
    extra_bars   : additional buffer bars (e.g. for execution lag).

    Returns
    -------
    First valid date after the warmup period.
    """
    n_warmup = max_lookback(feature_list) + extra_bars
    if len(df) <= n_warmup:
        raise ValueError(
            f"DataFrame has only {len(df)} rows but {n_warmup} bars of warmup "
            "are required. Use a longer history."
        )
    return df.index[n_warmup]


# ──────────────────────────────────────────────────────────────
# Training window checks
# ──────────────────────────────────────────────────────────────

def check_fold_warmup(
    train_panel: pd.DataFrame,
    feature_list: Optional[List[str]] = None,
    execution_lag: int = 1,
) -> Tuple[bool, int, int]:
    """
    Check whether a walk-forward training fold has enough bars after warmup.

    Returns
    -------
    (ok, n_valid_rows, n_warmup_rows)
    """
    n_warmup   = max_lookback(feature_list) + execution_lag
    n_total    = train_panel.index.nunique()   # unique dates
    n_valid    = max(n_total - n_warmup, 0)
    ok         = n_valid > 30                  # at least 30 tradable dates
    return ok, n_valid, n_warmup


def validate_history_length(
    df: pd.DataFrame,
    train_years: int,
    feature_list: Optional[List[str]] = None,
    execution_lag: int = 1,
) -> bool:
    """
    Return True if the DataFrame has enough history for the given training window.

    Accounts for warmup bars so the effective training period is:
        train_years × 252 + max_lookback + execution_lag
    """
    required = int(train_years * 252) + max_lookback(feature_list) + execution_lag
    return len(df) >= required


# ──────────────────────────────────────────────────────────────
# Panel warmup trimming
# ──────────────────────────────────────────────────────────────

def trim_panel_warmup(
    panel: pd.DataFrame,
    feature_list: Optional[List[str]] = None,
    execution_lag: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Drop the initial warmup rows from a cross-sectional panel.

    After feature computation, the first MAX_LOOKBACK rows of each ticker
    contain NaNs from rolling windows.  This function removes those rows
    globally (since all tickers share the same date index).

    Parameters
    ----------
    panel        : feature panel with DatetimeIndex.
    feature_list : features to compute the warmup period for.
    execution_lag: additional bars to trim for execution lag.
    verbose      : print how many rows were trimmed.

    Returns
    -------
    trimmed panel.
    """
    n_warmup   = max_lookback(feature_list) + execution_lag
    all_dates  = panel.index.unique().sort_values()

    if len(all_dates) <= n_warmup:
        logger.warning(
            "Panel has only %d unique dates but warmup requires %d. "
            "Returning empty panel.", len(all_dates), n_warmup
        )
        return panel.iloc[0:0]

    cutoff    = all_dates[n_warmup]
    trimmed   = panel[panel.index >= cutoff]

    if verbose:
        n_before = len(panel)
        n_after  = len(trimmed)
        print(
            f"[Timing] Trimmed {n_before - n_after:,} warmup rows "
            f"(cutoff: {cutoff.date()}, warmup={n_warmup} bars)."
        )

    return trimmed
