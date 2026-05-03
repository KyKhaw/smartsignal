"""
smartsignal.models.splits
==========================
Time-series-aware data splitting utilities for single-asset (tabular) data.

For cross-sectional panel splitting, see splits_panel.py.

These splitters complement the walk-forward logic in validation/walk_forward.py
by providing lower-level split iterators that can be used:
  - independently for single-ticker model experiments
  - as building blocks inside the WalkForwardSplitter
  - for hyperparameter tuning inside a fold
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────
# PurgedTimeSeriesSplit (single-asset)
# ──────────────────────────────────────────────────────────────

class PurgedTimeSeriesSplit:
    """
    Time-series cross-validation with an embargo gap (purging).

    Identical in spirit to sklearn's TimeSeriesSplit but with:
      - a configurable embargo_days gap between train and test
      - expanding or rolling window mode
      - yields (train_idx, test_idx) arrays

    Parameters
    ----------
    n_splits    : number of folds.
    test_size   : number of test samples per fold.
    embargo_days: number of samples to skip between train and test end.
    mode        : 'expanding' (default) or 'rolling'.
    """

    def __init__(
        self,
        n_splits:     int  = 5,
        test_size:    int  = 63,     # ~3 months of trading days
        embargo_days: int  = 5,
        mode:         str  = "expanding",
    ):
        self.n_splits     = n_splits
        self.test_size    = test_size
        self.embargo_days = embargo_days
        self.mode         = mode

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield (train_indices, test_indices) for each fold.

        Parameters
        ----------
        X : feature array of shape (n_samples, n_features).
        y : ignored (present for sklearn compatibility).
        """
        n = len(X)
        test_starts = np.linspace(
            n - self.n_splits * self.test_size,
            n - self.test_size,
            self.n_splits,
            dtype=int,
        )

        for fold, test_start in enumerate(test_starts):
            test_end   = test_start + self.test_size
            train_end  = test_start - self.embargo_days

            if self.mode == "rolling":
                train_start = max(0, train_end - self.test_size * self.n_splits)
            else:
                train_start = 0

            if train_end <= train_start:
                continue

            train_idx = np.arange(train_start, train_end)
            test_idx  = np.arange(test_start, min(test_end, n))

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# ──────────────────────────────────────────────────────────────
# Walk-forward date splitter (date-based, not index-based)
# ──────────────────────────────────────────────────────────────

class DateSplitter:
    """
    Splits a DatetimeIndex into (train_dates, test_dates) pairs.

    Unlike PurgedTimeSeriesSplit which works on array indices, DateSplitter
    works directly on pd.DatetimeIndex objects.  Useful when you need to
    slice a panel by date rather than position.

    Parameters
    ----------
    train_years  : initial training window in years.
    test_months  : test window per fold in months.
    embargo_days : calendar days to skip between train_end and test_start.
    mode         : 'expanding' or 'rolling'.
    """

    def __init__(
        self,
        train_years:  int = 2,
        test_months:  int = 3,
        embargo_days: int = 5,
        mode:         str = "expanding",
    ):
        self.train_years  = train_years
        self.test_months  = test_months
        self.embargo_days = embargo_days
        self.mode         = mode

    def split(
        self,
        dates: pd.DatetimeIndex,
    ) -> Iterator[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Yield (train_dates, test_dates) for each fold."""
        dates = dates.sort_values()
        start = dates[0]
        end   = dates[-1]

        train_end   = start + pd.DateOffset(years=self.train_years)
        train_start = start

        while train_end < end:
            embargo_end = train_end + pd.Timedelta(days=self.embargo_days)
            test_end    = min(
                embargo_end + pd.DateOffset(months=self.test_months), end
            )

            train_dates = dates[(dates >= train_start) & (dates <= train_end)]
            test_dates  = dates[(dates > embargo_end)  & (dates <= test_end)]

            if len(train_dates) > 0 and len(test_dates) > 0:
                yield train_dates, test_dates

            if self.mode == "expanding":
                train_end = test_end
            else:
                train_start = train_start + pd.DateOffset(months=self.test_months)
                train_end   = train_start + pd.DateOffset(years=self.train_years)
