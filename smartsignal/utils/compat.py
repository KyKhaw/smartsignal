"""
smartsignal.utils.compat
=========================
Not Currently Used, kept for future, all compat issues are currently manually fixed
more or less just to handle compatibility issues from Pandas, since the pipelien reqs Pandas 1.xx, 
however pandas 2.2+ renamed several frequency aliases:
    'M'  -> 'ME'  (month-end)
    'A'  -> 'YE'  (year-end)
    'Q'  -> 'QE'  (quarter-end)
"""

from __future__ import annotations
import pandas as pd

_PD_VERSION  = tuple(int(x) for x in pd.__version__.split(".")[:2])
_NEW_ALIASES = _PD_VERSION >= (2, 2)

_COMPAT_MAP: dict = {
    "ME":  "M",
    "YE":  "A",
    "QE":  "Q",
    "BME": "BM",
    "BYE": "BA",
}


def freq(alias: str) -> str:
    """
    Return the correct pandas frequency alias for the installed version.

    Pass modern aliases ('ME', 'YE') — downgrades automatically on
    pandas < 2.2.

    Examples
    --------
    >>> freq("ME")   # "ME" on pandas >= 2.2, "M" on older
    >>> freq("YE")   # "YE" on pandas >= 2.2, "A" on older
    >>> freq("W")    # unchanged on all versions
    """
    if _NEW_ALIASES:
        return alias
    return _COMPAT_MAP.get(alias, alias)


def resample(series_or_df, rule: str, **kwargs):
    """Version-safe resample wrapper."""
    return series_or_df.resample(freq(rule), **kwargs)