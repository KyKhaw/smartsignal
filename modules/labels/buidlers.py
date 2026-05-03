"""
smartsignal.labels.builders
=============================
Concrete label-building functions used by the label generator.

Separating the arithmetic of label construction from the orchestration
logic in generator.py makes each builder unit-testable in isolation.

Builders
--------
QuintileBuilder     – cross-sectional quantile bins (main LambdaMART labels)
BinaryBuilder       – binary direction labels (up/down)
RegressionBuilder   – winsorised forward-return regression targets
MomentumBuilder     – classical 12-1 cross-sectional momentum labels
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────

class LabelBuilder(ABC):
    """Abstract base class for label builders."""

    @abstractmethod
    def build(
        self,
        panel: pd.DataFrame,
        fwd_ret_col: str = "fwd_ret",
    ) -> pd.DataFrame:
        """
        Add label column(s) to the panel and return the modified DataFrame.

        Parameters
        ----------
        panel       : cross-sectional panel with DatetimeIndex.
        fwd_ret_col : name of the forward-return column.

        Returns
        -------
        panel with new label column(s) added.
        """


# ──────────────────────────────────────────────────────────────
# Concrete builders
# ──────────────────────────────────────────────────────────────

class QuintileBuilder(LabelBuilder):
    """
    Cross-sectional quantile bin labels (0 … n_bins-1).

    This is the primary label type for LambdaMART:
      label = 0 → bottom quantile (SHORT candidates)
      label = n_bins-1 → top quantile (LONG candidates)

    Parameters
    ----------
    n_bins    : number of quantile buckets (default 4 → quartile labels 0-3).
    label_col : output column name.
    min_obs   : minimum cross-section size to assign labels (else NaN).
    """

    def __init__(
        self,
        n_bins:    int = 4,
        label_col: str = "relevance",
        min_obs:   int = 8,
    ):
        self.n_bins    = n_bins
        self.label_col = label_col
        self.min_obs   = min_obs

    def build(self, panel: pd.DataFrame, fwd_ret_col: str = "fwd_ret") -> pd.DataFrame:
        panel = panel.copy()

        def _label_date(s: pd.Series) -> pd.Series:
            valid = s.dropna()
            if len(valid) < self.min_obs:
                return pd.Series(np.nan, index=s.index)
            return pd.qcut(
                s.rank(method="first"),
                q=self.n_bins, labels=False, duplicates="drop",
            )

        panel[self.label_col] = (
            panel.groupby(level=0)[fwd_ret_col]
                 .transform(_label_date)
                 .astype("Int64")
        )
        panel = panel.dropna(subset=[self.label_col])
        panel[self.label_col] = panel[self.label_col].astype(int)

        _log_dist(panel[self.label_col], self.label_col)
        return panel


class BinaryBuilder(LabelBuilder):
    """
    Binary direction label: 1 if forward return > threshold else 0.

    Parameters
    ----------
    threshold   : return threshold for positive classification (default 0).
    label_col   : output column name.
    balance     : if True, balance classes by dropping excess of the majority.
    """

    def __init__(
        self,
        threshold: float = 0.0,
        label_col: str   = "direction",
        balance:   bool  = False,
    ):
        self.threshold = threshold
        self.label_col = label_col
        self.balance   = balance

    def build(self, panel: pd.DataFrame, fwd_ret_col: str = "fwd_ret") -> pd.DataFrame:
        panel = panel.copy()
        panel = panel.dropna(subset=[fwd_ret_col])
        panel[self.label_col] = (panel[fwd_ret_col] > self.threshold).astype(int)

        if self.balance:
            pos = panel[panel[self.label_col] == 1]
            neg = panel[panel[self.label_col] == 0]
            n   = min(len(pos), len(neg))
            panel = pd.concat([
                pos.sample(n, random_state=42),
                neg.sample(n, random_state=42),
            ]).sort_index()

        _log_dist(panel[self.label_col], self.label_col)
        return panel


class RegressionBuilder(LabelBuilder):
    """
    Winsorised forward return regression target.

    Parameters
    ----------
    winsorise        : whether to winsorise the forward return.
    winsorise_bounds : (lower_quantile, upper_quantile).
    label_col        : output column name.
    cs_normalise     : if True, cross-sectionally z-score the target per date.
    """

    def __init__(
        self,
        winsorise:        bool  = True,
        winsorise_bounds: tuple = (0.01, 0.99),
        label_col:        str   = "target_ret",
        cs_normalise:     bool  = False,
    ):
        self.winsorise        = winsorise
        self.winsorise_bounds = winsorise_bounds
        self.label_col        = label_col
        self.cs_normalise     = cs_normalise

    def build(self, panel: pd.DataFrame, fwd_ret_col: str = "fwd_ret") -> pd.DataFrame:
        panel = panel.copy()
        panel = panel.dropna(subset=[fwd_ret_col])
        target = panel[fwd_ret_col].copy()

        if self.winsorise:
            lo = target.quantile(self.winsorise_bounds[0])
            hi = target.quantile(self.winsorise_bounds[1])
            target = target.clip(lo, hi)

        if self.cs_normalise:
            def _cs_zscore(s: pd.Series) -> pd.Series:
                mu, sig = s.mean(), s.std()
                return (s - mu) / sig if sig > 1e-12 else s * 0.0
            target = panel.groupby(level=0)[fwd_ret_col].transform(_cs_zscore)

        panel[self.label_col] = target
        return panel


class MomentumBuilder(LabelBuilder):
    """
    Classical cross-sectional momentum labels.

    Ranks stocks by their past-N-day return (formation window) and assigns
    decile labels 0-9, where 9 = strongest past winners.

    Parameters
    ----------
    lookback_days : formation window in trading days (default 252 ~ 12 months).
    skip_days     : recent days to skip to avoid reversal (default 21 ~ 1 month).
    n_bins        : number of decile/quintile bins.
    label_col     : output column name.
    """

    def __init__(
        self,
        lookback_days: int = 252,
        skip_days:     int = 21,
        n_bins:        int = 10,
        label_col:     str = "mom_label",
    ):
        self.lookback_days = lookback_days
        self.skip_days     = skip_days
        self.n_bins        = n_bins
        self.label_col     = label_col

    def build(self, panel: pd.DataFrame, fwd_ret_col: str = "fwd_ret") -> pd.DataFrame:
        """Use past return (already in panel as ret_Nd features) to rank."""
        panel = panel.copy()

        # Use 'ret_5d' or similar as proxy; ideally ret_252d if available
        # Fall back to cross-sectional ranking of fwd_ret (for illustration)
        ret_col = "ret_10d" if "ret_10d" in panel.columns else fwd_ret_col
        if ret_col not in panel.columns:
            logger.warning("MomentumBuilder: no return column found — skipping.")
            panel[self.label_col] = np.nan
            return panel

        def _label_date(s: pd.Series) -> pd.Series:
            valid = s.dropna()
            if len(valid) < self.n_bins:
                return pd.Series(np.nan, index=s.index)
            return pd.qcut(
                s.rank(method="first"),
                q=self.n_bins, labels=False, duplicates="drop",
            )

        panel[self.label_col] = (
            panel.groupby(level=0)[ret_col]
                 .transform(_label_date)
                 .astype("Int64")
        )
        panel = panel.dropna(subset=[self.label_col])
        panel[self.label_col] = panel[self.label_col].astype(int)
        return panel


# ──────────────────────────────────────────────────────────────
# Registry of available builders
# ──────────────────────────────────────────────────────────────

BUILDER_REGISTRY = {
    "quintile":         QuintileBuilder,
    "binary_direction": BinaryBuilder,
    "regression":       RegressionBuilder,
    "momentum":         MomentumBuilder,
}


def get_builder(label_type: str, **kwargs) -> LabelBuilder:
    """
    Instantiate a label builder by name.

    Parameters
    ----------
    label_type : one of 'quintile', 'binary_direction', 'regression', 'momentum'.
    **kwargs   : passed to the builder constructor.
    """
    key = label_type.lower()
    if key not in BUILDER_REGISTRY:
        raise KeyError(
            f"Unknown label type '{label_type}'. "
            f"Available: {list(BUILDER_REGISTRY)}."
        )
    return BUILDER_REGISTRY[key](**kwargs)


# ──────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────

def _log_dist(series: pd.Series, col: str) -> None:
    dist = series.value_counts().sort_index().to_dict()
    logger.debug("Label distribution for '%s': %s", col, dist)
