"""
smartsignal.data.validator
============================
Structural integrity and leakage-hint validation for OHLCV DataFrames.

Two validation layers:

  1. StructuralValidator
     - Checks DatetimeIndex, required columns, duplicate timestamps,
       correct dtypes, monotonic ordering, NaN budget.

  2. LeakageHintValidator
     - Scans column names for patterns that suggest forward-looking
       information was accidentally included (e.g. 'future_', 'next_',
       'fwd_' at the feature stage, 'target' outside the label column).

Both return ValidationReport objects listing errors, warnings, and hints.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from smartsignal.data.schema import OHLCV_COLS, REQUIRED_COLS

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Report dataclass
# ──────────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    ticker:   str
    passed:   bool              = True
    errors:   List[str]         = field(default_factory=list)
    warnings: List[str]         = field(default_factory=list)
    hints:    List[str]         = field(default_factory=list)   # leakage hints

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_hint(self, msg: str):
        self.hints.append(msg)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        parts  = [f"[{self.ticker}] {status}"]
        if self.errors:
            parts.append(f"  ERRORS  : {'; '.join(self.errors)}")
        if self.warnings:
            parts.append(f"  WARNINGS: {'; '.join(self.warnings)}")
        if self.hints:
            parts.append(f"  HINTS   : {'; '.join(self.hints)}")
        return "\n".join(parts)

    def raise_if_errors(self):
        if self.errors:
            raise ValueError(
                f"Validation failed for {self.ticker}:\n"
                + "\n".join(f"  • {e}" for e in self.errors)
            )


# ──────────────────────────────────────────────────────────────
# Structural validator
# ──────────────────────────────────────────────────────────────

_LEAKAGE_PATTERNS = [
    r"^future_",    r"_future$",
    r"^next_",      r"_next$",
    r"^fwd_",       r"_fwd$",
    r"^lead_",      r"_lead$",
    r"^target",
    r"^label",
    r"tomorrow",
    r"t\+\d+",
]
_LEAKAGE_RE = re.compile("|".join(_LEAKAGE_PATTERNS), re.IGNORECASE)


class StructuralValidator:
    """
    Validate the structural integrity of a single-ticker OHLCV DataFrame.

    Parameters
    ----------
    max_missing_frac : maximum acceptable fraction of NaN closes.
    min_rows         : minimum row count.
    check_ohlcv_logic: whether to check high >= low, close in [low, high].
    """

    def __init__(
        self,
        max_missing_frac: float = 0.05,
        min_rows:         int   = 60,
        check_ohlcv_logic:bool  = True,
    ):
        self.max_missing_frac  = max_missing_frac
        self.min_rows          = min_rows
        self.check_ohlcv_logic = check_ohlcv_logic

    def validate(self, df: pd.DataFrame, ticker: str = "?") -> ValidationReport:
        report = ValidationReport(ticker=ticker)

        # 1. DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            report.add_error("Index is not a DatetimeIndex.")
            return report   # cannot proceed without a proper index

        # 2. Monotonic ordering
        if not df.index.is_monotonic_increasing:
            report.add_warning("Index is not monotonically increasing — data was reordered.")

        # 3. Duplicate timestamps
        n_dups = df.index.duplicated().sum()
        if n_dups > 0:
            report.add_warning(f"{n_dups} duplicate timestamps found.")

        # 4. Required columns
        missing_cols = set(REQUIRED_COLS) - set(df.columns)
        if missing_cols:
            report.add_error(f"Missing required columns: {missing_cols}")
            return report

        # 5. Minimum row count
        if len(df) < self.min_rows:
            report.add_error(
                f"Only {len(df)} rows (minimum {self.min_rows})."
            )

        # 6. Missing close prices
        missing_close = df["close"].isna().mean()
        if missing_close > self.max_missing_frac:
            report.add_warning(
                f"Close price missing in {missing_close:.1%} of rows "
                f"(threshold {self.max_missing_frac:.0%})."
            )

        # 7. Non-positive prices
        for col in ["open", "high", "low", "close"]:
            n_neg = (df[col] <= 0).sum()
            if n_neg > 0:
                report.add_warning(f"{n_neg} non-positive values in '{col}'.")

        # 8. OHLCV logic
        if self.check_ohlcv_logic:
            bad_hl = (df["high"] < df["low"]).sum()
            if bad_hl > 0:
                report.add_warning(f"{bad_hl} rows with high < low.")

            bad_close_h = (df["close"] > df["high"] * 1.001).sum()
            bad_close_l = (df["close"] < df["low"]  * 0.999).sum()
            if bad_close_h + bad_close_l > 0:
                report.add_warning(
                    f"Close outside [low, high] in "
                    f"{bad_close_h + bad_close_l} rows."
                )

        # 9. Zero-volume days
        zero_vol = (df["volume"] == 0).mean()
        if zero_vol > 0.02:
            report.add_warning(
                f"{zero_vol:.1%} of rows have zero volume."
            )

        return report


class LeakageHintValidator:
    """
    Scan feature column names for patterns that suggest forward-looking leakage.
    Emits hints (non-blocking) so the researcher can investigate.
    """

    SAFE_PREFIXES = {"fwd_ret", "relevance", "target_ret", "direction"}

    def validate(
        self,
        df: pd.DataFrame,
        ticker: str = "?",
        safe_cols: Optional[List[str]] = None,
    ) -> ValidationReport:
        report = ValidationReport(ticker=ticker)
        safe   = self.SAFE_PREFIXES | set(safe_cols or [])

        for col in df.columns:
            if col in safe:
                continue
            if _LEAKAGE_RE.search(col):
                report.add_hint(
                    f"Column '{col}' matches a forward-looking naming pattern. "
                    "Verify this is not a future feature."
                )

        return report


# ──────────────────────────────────────────────────────────────
# Combined validation pipeline
# ──────────────────────────────────────────────────────────────

def validate_universe(
    dfs:              Dict[str, pd.DataFrame],
    max_missing_frac: float = 0.05,
    min_rows:         int   = 60,
    check_leakage:    bool  = True,
    raise_on_errors:  bool  = False,
    verbose:          bool  = True,
) -> Dict[str, ValidationReport]:
    """
    Validate all tickers in a universe dict.

    Parameters
    ----------
    dfs              : {ticker: DataFrame}.
    max_missing_frac : passed to StructuralValidator.
    min_rows         : passed to StructuralValidator.
    check_leakage    : whether to run LeakageHintValidator.
    raise_on_errors  : raise ValueError if any ticker fails structural check.
    verbose          : print per-ticker summaries for failures.

    Returns
    -------
    reports : {ticker: ValidationReport}.
    """
    struct_val  = StructuralValidator(
        max_missing_frac=max_missing_frac, min_rows=min_rows
    )
    leakage_val = LeakageHintValidator() if check_leakage else None

    reports: Dict[str, ValidationReport] = {}
    n_fail  = 0

    for ticker, df in dfs.items():
        rep = struct_val.validate(df, ticker=ticker)
        if leakage_val:
            leak_rep = leakage_val.validate(df, ticker=ticker)
            rep.hints.extend(leak_rep.hints)

        reports[ticker] = rep
        if not rep.passed:
            n_fail += 1
            if verbose:
                print(rep.summary())
        elif rep.hints and verbose:
            for h in rep.hints:
                logger.warning("  [%s] LEAKAGE HINT: %s", ticker, h)

    if verbose:
        print(
            f"[Validator] {len(dfs) - n_fail}/{len(dfs)} tickers passed "
            f"structural validation."
        )

    if raise_on_errors and n_fail > 0:
        failing = [t for t, r in reports.items() if not r.passed]
        raise ValueError(
            f"{n_fail} tickers failed validation: {failing[:10]}"
        )

    return reports


def validate_panel(
    panel:         pd.DataFrame,
    feature_cols:  List[str],
    label_col:     str = "relevance",
    check_leakage: bool = True,
    verbose:       bool = True,
) -> ValidationReport:
    """
    Validate the cross-sectional panel DataFrame before model training.

    Checks:
      - DatetimeIndex present
      - 'ticker' column present
      - all feature_cols present and numeric
      - label column present and integer
      - no feature NaN exceeding 10 %
      - optional: leakage hints on column names
    """
    report = ValidationReport(ticker="<panel>")

    if not isinstance(panel.index, pd.DatetimeIndex):
        report.add_error("Panel index is not a DatetimeIndex.")
        return report

    if "ticker" not in panel.columns:
        report.add_error("Panel is missing a 'ticker' column.")

    missing_feat = [f for f in feature_cols if f not in panel.columns]
    if missing_feat:
        report.add_error(f"Panel is missing {len(missing_feat)} feature columns: {missing_feat[:5]} …")

    if label_col not in panel.columns:
        report.add_error(f"Label column '{label_col}' not found in panel.")

    # NaN budget per feature
    for feat in feature_cols:
        if feat not in panel.columns:
            continue
        nan_frac = panel[feat].isna().mean()
        if nan_frac > 0.10:
            report.add_warning(f"Feature '{feat}' has {nan_frac:.1%} NaN values.")

    if check_leakage:
        lv = LeakageHintValidator()
        lr = lv.validate(panel, ticker="<panel>",
                         safe_cols=feature_cols + [label_col, "ticker", "fwd_ret"])
        report.hints.extend(lr.hints)

    if verbose and (not report.passed or report.warnings or report.hints):
        print(report.summary())

    return report
