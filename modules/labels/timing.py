"""
smartsignal.labels.timing
===========================
Forward-return horizon management and label timing utilities.

Responsible for:
  1. Computing forward returns over configurable horizons.
  2. Ensuring the forward-return shift does not introduce look-ahead leakage.
  3. Tracking the "label horizon" so the walk-forward splitter can apply
     an appropriate embargo gap between training and test sets.

The embargo gap must be at least as large as forward_days to prevent
forward-return information from the training window appearing in test labels.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Forward-return computation
# ──────────────────────────────────────────────────────────────

def compute_forward_returns(
    df: pd.DataFrame,
    horizons: List[int] = (1, 5, 10, 21),
    price_col: str = "close",
    shift: bool = True,
) -> pd.DataFrame:
    """
    Compute forward returns over multiple horizons for a single ticker.

    Parameters
    ----------
    df        : single-ticker DataFrame with DatetimeIndex.
    horizons  : list of forward horizons in bars (trading days).
    price_col : price column to compute returns from.
    shift     : if True, shift returns so each row holds the forward return
                that will be realised *after* that bar (no look-ahead).

    Returns
    -------
    DataFrame with 'fwd_ret_Nd' columns added for each horizon.
    """
    df = df.copy()
    c  = df[price_col]

    for h in horizons:
        col = f"fwd_ret_{h}d"
        raw = c.pct_change(h)
        if shift:
            raw = raw.shift(-h)   # align: row t holds return for [t, t+h]
        df[col] = raw

    return df


def compute_panel_forward_returns(
    panel: pd.DataFrame,
    horizons: List[int] = (5,),
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Compute forward returns for a cross-sectional panel, per ticker.

    Each ticker's returns are computed independently to avoid cross-ticker
    contamination.

    Parameters
    ----------
    panel    : stacked panel with DatetimeIndex and 'ticker' column.
    horizons : forward horizons in bars.
    price_col: price column.

    Returns
    -------
    panel with 'fwd_ret_Nd' columns added.
    """
    panel = panel.copy()

    for h in horizons:
        col = f"fwd_ret_{h}d"
        panel[col] = (
            panel.groupby("ticker")[price_col]
                 .transform(lambda s: s.pct_change(h).shift(-h))
        )

    return panel


# ──────────────────────────────────────────────────────────────
# Label horizon / embargo utilities
# ──────────────────────────────────────────────────────────────

def min_embargo_days(forward_days: int, extra_buffer: int = 0) -> int:
    """
    Return the minimum embargo gap that should be inserted between a training
    fold's end and the test fold's start.

    The embargo must be at least forward_days so that forward-return labels
    computed at the end of the training window do not leak into the test set.

    Parameters
    ----------
    forward_days  : forward-return horizon used for label construction.
    extra_buffer  : additional calendar-day buffer (default 0).

    Returns
    -------
    minimum embargo in trading days.
    """
    return forward_days + extra_buffer


def last_valid_label_date(
    panel: pd.DataFrame,
    forward_days: int,
) -> pd.Timestamp:
    """
    Return the last date in the panel that has a valid forward-return label.

    Forward returns for the last `forward_days` rows will be NaN because there
    is no future price data.  This date is the effective end of the labelled
    dataset.
    """
    dates  = panel.index.unique().sort_values()
    cutoff = len(dates) - forward_days
    if cutoff <= 0:
        raise ValueError(
            f"Panel has only {len(dates)} dates but forward_days={forward_days}. "
            "Use a longer date range."
        )
    return dates[cutoff - 1]


def trim_label_horizon(
    panel: pd.DataFrame,
    forward_days: int,
    fwd_ret_col: str = "fwd_ret",
) -> pd.DataFrame:
    """
    Drop the last `forward_days` rows of the panel where forward returns are NaN.

    This should be called after label generation to avoid NaN labels entering
    model training.
    """
    dates         = panel.index.unique().sort_values()
    last_valid    = last_valid_label_date(panel, forward_days)
    trimmed       = panel[panel.index <= last_valid]
    n_dropped     = len(panel) - len(trimmed)

    if n_dropped > 0:
        logger.debug(
            "Trimmed %d rows beyond label horizon (last valid: %s).",
            n_dropped, last_valid.date(),
        )

    return trimmed


# ──────────────────────────────────────────────────────────────
# Horizon metadata
# ──────────────────────────────────────────────────────────────

HORIZON_METADATA: Dict[int, Dict] = {
    1:  {"name": "1-day",   "category": "intraday",  "min_universe_days": 126},
    5:  {"name": "1-week",  "category": "short",     "min_universe_days": 252},
    10: {"name": "2-week",  "category": "short",     "min_universe_days": 252},
    21: {"name": "1-month", "category": "medium",    "min_universe_days": 504},
    63: {"name": "3-month", "category": "medium",    "min_universe_days": 756},
}


def describe_horizon(forward_days: int) -> str:
    """Human-readable description of a forward horizon."""
    meta = HORIZON_METADATA.get(forward_days)
    if meta:
        return f"{forward_days}-day horizon ({meta['name']}, {meta['category']})"
    return f"{forward_days}-day horizon"
