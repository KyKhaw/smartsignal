"""
smartsignal.validation.walk_forward
=====================================
Walk-forward cross-validation for time-series financial data.

Standard k-fold cross-validation violates temporal ordering by allowing
future data to appear in training sets.  Walk-forward validation instead
generates chronologically-ordered folds:

  Expanding window (default):
    Fold 1: train [T0, T1], test [T1+embargo, T2]
    Fold 2: train [T0, T2], test [T2+embargo, T3]
    ...
    Training window grows with each fold; test window is fixed length.

  Rolling window:
    Fold 1: train [T0,      T1], test [T1+embargo, T2]
    Fold 2: train [T1_roll, T2], test [T2+embargo, T3]
    ...
    Training window shifts forward with a fixed size.

An embargo gap between training and test prevents autocorrelation from
adjacent bars inflating out-of-sample metrics (as recommended by López
de Prado, 2018, and confirmed by Swetha & Arya, 2025, for equity ML).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Fold data class
# ──────────────────────────────────────────────────────────────

@dataclass
class WalkForwardFold:
    """One chronological train/test split."""
    fold_id:       int
    train_start:   pd.Timestamp
    train_end:     pd.Timestamp
    test_start:    pd.Timestamp
    test_end:      pd.Timestamp
    train_panel:   pd.DataFrame = field(repr=False)
    test_panel:    pd.DataFrame = field(repr=False)

    def summary(self) -> str:
        return (
            f"Fold {self.fold_id:02d} | "
            f"Train: {self.train_start.date()} – {self.train_end.date()} "
            f"({len(self.train_panel):,} rows) | "
            f"Test:  {self.test_start.date()} – {self.test_end.date()} "
            f"({len(self.test_panel):,} rows)"
        )


# ──────────────────────────────────────────────────────────────
# Main splitter
# ──────────────────────────────────────────────────────────────

class WalkForwardSplitter:
    """
    Generate chronologically-ordered walk-forward train/test folds.

    Parameters
    ----------
    train_years       : length of the initial (and minimum) training window.
    test_months       : length of each test window.
    mode              : 'expanding' (growing train) or 'rolling' (fixed-size train).
    rolling_years     : fixed training window size for 'rolling' mode.
    embargo_days      : trading-day buffer between train end and test start.
    min_train_dates   : minimum number of unique dates in the training window.
    min_test_dates    : skip a fold if the test window has fewer dates.
    verbose           : print fold summaries.
    """

    def __init__(
        self,
        train_years:     int   = 2,
        test_months:     int   = 3,
        mode:            str   = "expanding",
        rolling_years:   int   = 3,
        embargo_days:    int   = 5,
        min_train_dates: int   = 60,
        min_test_dates:  int   = 5,
        verbose:         bool  = True,
    ):
        if mode not in ("expanding", "rolling"):
            raise ValueError("mode must be 'expanding' or 'rolling'.")
        self.train_years     = train_years
        self.test_months     = test_months
        self.mode            = mode
        self.rolling_years   = rolling_years
        self.embargo_days    = embargo_days
        self.min_train_dates = min_train_dates
        self.min_test_dates  = min_test_dates
        self.verbose         = verbose

    def split(self, panel: pd.DataFrame) -> Iterator[WalkForwardFold]:
        """
        Yield WalkForwardFold objects for the given panel.

        Parameters
        ----------
        panel : cross-sectional panel with DatetimeIndex and 'ticker' column.

        Yields
        ------
        WalkForwardFold for each valid chronological fold.
        """
        panel  = panel.sort_index()
        dates  = panel.index.unique().sort_values()
        start  = dates[0]
        end    = dates[-1]

        # Initial training window end
        train_end = start + pd.DateOffset(years=self.train_years) - pd.Timedelta(days=1)

        if train_end >= end:
            raise ValueError(
                f"Insufficient data for a {self.train_years}-year initial "
                f"training window ({start.date()} – {end.date()}). "
                f"Use a longer date range or reduce train_years."
            )

        fold_id   = 0
        train_start = start

        while train_end < end:
            # Test window
            test_start = train_end + pd.Timedelta(days=self.embargo_days)
            test_end   = min(
                test_start + pd.DateOffset(months=self.test_months),
                end,
            )

            # Slice panels
            train_panel = panel.loc[train_start:train_end]
            test_panel  = panel.loc[
                (panel.index > train_end) &
                (panel.index >= test_start) &
                (panel.index <= test_end)
            ]

            train_dates = train_panel.index.unique()
            test_dates  = test_panel.index.unique()

            # Validation checks
            if len(train_dates) < self.min_train_dates:
                logger.debug(
                    "Fold %d: train has %d dates (< %d) — advancing.",
                    fold_id, len(train_dates), self.min_train_dates
                )
                train_end = test_end
                continue

            if len(test_dates) < self.min_test_dates:
                logger.debug(
                    "Fold %d: test has %d dates (< %d) — skipping.",
                    fold_id, len(test_dates), self.min_test_dates
                )
                train_end = test_end
                continue

            fold = WalkForwardFold(
                fold_id     = fold_id,
                train_start = train_start,
                train_end   = train_end,
                test_start  = test_start,
                test_end    = test_end,
                train_panel = train_panel,
                test_panel  = test_panel,
            )

            if self.verbose:
                print(f"  {fold.summary()}")

            yield fold

            # Advance windows
            if self.mode == "expanding":
                train_end = test_end
            else:  # rolling
                train_start = train_start + pd.DateOffset(months=self.test_months)
                train_end   = train_start + pd.DateOffset(years=self.rolling_years)
                if train_end > test_end:
                    train_end = test_end

            fold_id += 1


# ──────────────────────────────────────────────────────────────
# Convenience: run full walk-forward ranking loop
# ──────────────────────────────────────────────────────────────

def run_walk_forward(
    panel: pd.DataFrame,
    model,
    feature_cols: List[str],
    splitter: Optional[WalkForwardSplitter] = None,
    run_feature_selection_each_fold: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Execute a complete walk-forward loop and return a scored panel.

    Parameters
    ----------
    panel          : labelled cross-sectional panel.
    model          : instance of LambdaMARTRanker (or any model with
                     fit(panel) / predict(panel) → pd.Series).
    feature_cols   : features to pass to the model.
    splitter       : WalkForwardSplitter; created with defaults if None.
    run_feature_selection_each_fold : re-run feature selection at each fold.
    verbose        : print progress.

    Returns
    -------
    panel with 'rank_score' column added (NaN in the training window).
    """
    if splitter is None:
        splitter = WalkForwardSplitter(verbose=verbose)

    score_chunks: List[pd.DataFrame] = []

    for fold in splitter.split(panel):
        # Feature selection: always on fold 0, optionally on later folds
        run_sel = (fold.fold_id == 0) or run_feature_selection_each_fold

        model.fit(fold.train_panel, run_feature_selection=run_sel)

        scores = model.predict(fold.test_panel)
        chunk  = fold.test_panel[["ticker", "fwd_ret", "relevance"]].copy()
        chunk["rank_score"] = scores.values

        score_chunks.append(chunk)

    if not score_chunks:
        raise ValueError("Walk-forward produced no scored folds.")

    scored = pd.concat(score_chunks).sort_index()

    # Merge back into full panel (training rows get NaN rank_score)
    panel_out = panel.copy()
    panel_out["rank_score"] = np.nan
    panel_out.loc[scored.index, "rank_score"] = scored["rank_score"].values

    if verbose:
        n_scored = scored["rank_score"].notna().sum()
        print(f"\n[WalkForward] {n_scored:,} rows scored across "
              f"{len(score_chunks)} folds.")

    return panel_out
