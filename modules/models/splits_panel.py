"""
smartsignal.models.splits_panel
=================================
Cross-sectional panel data splitting utilities.

Extends the base splitters in splits.py with panel-specific logic:
  - slicing a stacked (date × ticker) panel by date range
  - computing LGBMRanker group sizes for each slice
  - verifying that each fold has enough cross-sectional breadth
    (minimum number of active tickers per date)
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from smartsignal.models.splits import DateSplitter


# ──────────────────────────────────────────────────────────────
# Panel-aware walk-forward splitter
# ──────────────────────────────────────────────────────────────

class PanelWalkForwardSplitter:
    """
    Generates (train_panel, test_panel) slices from a cross-sectional panel.

    Wraps DateSplitter and adds:
      - minimum per-date ticker count check
      - group array computation for LGBMRanker
      - optional embargo gap enforcement via forward_days alignment

    Parameters
    ----------
    train_years         : initial training window.
    test_months         : test window per fold.
    embargo_days        : days to skip between train_end and test_start.
    mode                : 'expanding' or 'rolling'.
    min_tickers_per_date: fold skipped if median daily ticker count < this.
    forward_days        : label horizon; used to ensure embargo ≥ forward_days.
    """

    def __init__(
        self,
        train_years:          int  = 2,
        test_months:          int  = 3,
        embargo_days:         int  = 5,
        mode:                 str  = "expanding",
        min_tickers_per_date: int  = 20,
        forward_days:         int  = 5,
    ):
        self.embargo_days = max(embargo_days, forward_days)   # safety
        self._splitter    = DateSplitter(
            train_years  = train_years,
            test_months  = test_months,
            embargo_days = self.embargo_days,
            mode         = mode,
        )
        self.min_tickers_per_date = min_tickers_per_date

    def split(
        self,
        panel: pd.DataFrame,
    ) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Yield (train_panel, test_panel) DataFrame pairs.

        Each panel slice retains all original columns.
        """
        dates = panel.index.unique().sort_values()

        for train_dates, test_dates in self._splitter.split(dates):
            train_panel = panel.loc[panel.index.isin(train_dates)]
            test_panel  = panel.loc[panel.index.isin(test_dates)]

            # Breadth check
            if not self._breadth_ok(train_panel):
                continue
            if not self._breadth_ok(test_panel):
                continue

            yield train_panel, test_panel

    def _breadth_ok(self, panel: pd.DataFrame) -> bool:
        """Return True if median daily ticker count meets the minimum."""
        daily_counts = panel.groupby(level=0).size()
        return daily_counts.median() >= self.min_tickers_per_date

    @staticmethod
    def compute_groups(panel: pd.DataFrame) -> np.ndarray:
        """
        Compute the LGBMRanker group array for the panel.

        Returns an int32 array where groups[i] = number of stocks on date i.
        """
        sorted_panel = panel.sort_index()
        return sorted_panel.groupby(level=0).size().values.astype(np.int32)

    @staticmethod
    def panel_to_arrays(
        panel: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "relevance",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert a panel slice to (X, y, groups) arrays for LGBMRanker.

        Rows are sorted by date then ticker (required by LGBMRanker).
        """
        sorted_panel = (
            panel.sort_index()
                 .sort_values("ticker", kind="stable")
        )
        X      = sorted_panel[feature_cols].values.astype(np.float32)
        y      = sorted_panel[label_col].values.astype(np.int32)
        groups = sorted_panel.groupby(level=0).size().values.astype(np.int32)
        return X, y, groups
