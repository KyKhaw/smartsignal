"""
smartsignal.features.presets
==============================
Named feature preset configurations for the equity feature pipeline.

Presets define which features are active and what parameter variants to use.
Two built-in presets:

  'base'   – one variant per indicator (42 features).
             Fast to compute; good baseline for initial experiments.

  'heavy'  – short/medium/long variants for oscillators and moving averages
             (extends to ~70 features).  Better coverage of multiple regimes.

Custom presets can be registered via register_preset().

Usage
-----
    from smartsignal.features.presets import get_preset, list_presets

    spec = get_preset("base")
    # spec is a dict: {feature_name: True/False}

    spec = get_preset("heavy")

    # Filter to only momentum and volatility features
    spec = get_preset("base", categories=["momentum", "volatility"])
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional

from smartsignal.features.equity_features import FEATURE_COLS, FEATURE_CATEGORIES

# ──────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, Dict] = {}


def register_preset(name: str, spec: Dict) -> None:
    """Register a custom feature preset."""
    _REGISTRY[name.lower()] = spec


def list_presets() -> List[str]:
    """Return names of all registered presets."""
    return sorted(_REGISTRY.keys())


# ──────────────────────────────────────────────────────────────
# Built-in presets
# ──────────────────────────────────────────────────────────────

# BASE: the 42-feature set from the CSM LambdaMART notebook
_BASE_SPEC: Dict = {
    "active_features": FEATURE_COLS,
    "description": (
        "Standard 42-feature set (Wang & Dong 2025). "
        "One parameter variant per indicator."
    ),
    "extra_params": {},
}

# HEAVY: extends base with additional MA / oscillator period variants
_HEAVY_EXTRA_FEATURES = [
    # Additional SMA periods
    "sma_3", "sma_100",
    # Additional EMA periods
    "ema_5", "ema_50",
    # Additional RSI periods
    "rsi_7", "rsi_21",
    # Additional MACD variants (short fast/slow)
    "macd_fast",   # ema_5 - ema_13
    "macd_slow",   # ema_19 - ema_39
    # Additional vol windows
    "hvol_5", "hvol_60",
    # Additional return windows
    "ret_21d", "ret_63d",
    # Additional BB variants
    "bb_width_10", "bb_pct_10",
    # Additional vol ratio periods
    "vol_ratio_5", "vol_ratio_50",
]

_HEAVY_SPEC: Dict = {
    "active_features": FEATURE_COLS + _HEAVY_EXTRA_FEATURES,
    "description": (
        "Extended feature set with short/medium/long period variants. "
        "~70 features; better multi-regime coverage."
    ),
    "extra_params": {
        "sma_3": {"window": 3},
        "sma_100": {"window": 100},
        "ema_5": {"span": 5},
        "ema_50": {"span": 50},
        "rsi_7": {"period": 7},
        "rsi_21": {"period": 21},
        "macd_fast": {"fast": 5, "slow": 13, "signal": 4},
        "macd_slow": {"fast": 19, "slow": 39, "signal": 9},
        "hvol_5":  {"window": 5},
        "hvol_60": {"window": 60},
        "ret_21d": {"periods": 21},
        "ret_63d": {"periods": 63},
        "bb_width_10": {"window": 10},
        "bb_pct_10":   {"window": 10},
        "vol_ratio_5":  {"window": 5},
        "vol_ratio_50": {"window": 50},
    },
}

# MOMENTUM_ONLY: only momentum and overlap features (for signal isolation)
_MOMENTUM_SPEC: Dict = {
    "active_features": [
        f for f, cat in FEATURE_CATEGORIES.items()
        if cat in ("momentum", "overlap")
    ],
    "description": "Momentum and overlap (trend) features only.",
    "extra_params": {},
}

# VOLATILITY_VOLUME: volatility + volume features
_VOL_VOLUME_SPEC: Dict = {
    "active_features": [
        f for f, cat in FEATURE_CATEGORIES.items()
        if cat in ("volatility", "volume")
    ],
    "description": "Volatility and volume features only.",
    "extra_params": {},
}

register_preset("base",            _BASE_SPEC)
register_preset("heavy",           _HEAVY_SPEC)
register_preset("momentum",        _MOMENTUM_SPEC)
register_preset("volatility_volume", _VOL_VOLUME_SPEC)


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def get_preset(
    name: str,
    categories: Optional[List[str]] = None,
) -> Dict:
    """
    Retrieve a feature preset specification.

    Parameters
    ----------
    name       : preset name ('base', 'heavy', 'momentum', 'volatility_volume',
                 or any user-registered name).
    categories : optional list of feature categories to retain
                 ('overlap', 'momentum', 'volatility', 'volume', 'price_transform').
                 If None, all features in the preset are active.

    Returns
    -------
    spec : dict with keys:
        'active_features' : list of feature column names
        'description'     : human-readable description
        'extra_params'    : per-feature parameter overrides
    """
    key = name.lower()
    if key not in _REGISTRY:
        available = list_presets()
        raise KeyError(
            f"Unknown preset '{name}'. "
            f"Available: {available}. "
            f"Register custom presets with register_preset()."
        )

    spec = deepcopy(_REGISTRY[key])

    if categories is not None:
        categories_set = set(c.lower() for c in categories)
        spec["active_features"] = [
            f for f in spec["active_features"]
            if FEATURE_CATEGORIES.get(f, "other") in categories_set
        ]

    return spec


def preset_feature_list(
    name: str,
    categories: Optional[List[str]] = None,
) -> List[str]:
    """Convenience: return just the active feature list for a preset."""
    return get_preset(name, categories=categories)["active_features"]
