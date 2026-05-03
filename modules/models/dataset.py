"""
smartsignal.models.registry
=============================
Central registry that maps model-family names to their constructor classes.

Any new model family (e.g. an LSTM, XGBoost ranker, or custom sklearn model)
is added here and is immediately available everywhere in the pipeline.

Usage
-----
    from smartsignal.models.registry import get_model, list_models

    model = get_model("lambdamart", feature_cols=FEATURE_COLS, top_k_features=25)
    model.fit(train_panel)
    scores = model.predict(test_panel)
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

from smartsignal.models.base import BaseModel


# ──────────────────────────────────────────────────────────────
# Registry store
# ──────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register_model(name: str, cls: Type[BaseModel]) -> None:
    """Register a model class under the given name."""
    _REGISTRY[name.lower()] = cls


def list_models() -> List[str]:
    """Return all registered model names."""
    return sorted(_REGISTRY.keys())


# ──────────────────────────────────────────────────────────────
# Populate registry (lazy import to avoid circular deps)
# ──────────────────────────────────────────────────────────────

def _register_defaults() -> None:
    from smartsignal.models.lambdamart import LambdaMARTRanker
    register_model("lambdamart", LambdaMARTRanker)

    from smartsignal.models.sklearn_models import (
        SklearnRankerAdapter,
        LGBMClassifierModel,
        RandomForestRanker,
        RidgeRanker,
    )
    register_model("lgbm_classifier", LGBMClassifierModel)
    register_model("random_forest",   RandomForestRanker)
    register_model("ridge",           RidgeRanker)

    from smartsignal.models.advanced_models import (
        EnsembleRanker,
        StackedRanker,
    )
    register_model("ensemble", EnsembleRanker)
    register_model("stacked",  StackedRanker)


def get_model(name: str, **kwargs) -> BaseModel:
    """
    Instantiate a model by name.

    Parameters
    ----------
    name     : registered model name (case-insensitive).
    **kwargs : passed to the model constructor.

    Returns
    -------
    Instantiated model implementing BaseModel interface.
    """
    if not _REGISTRY:
        _register_defaults()

    key = name.lower()
    if key not in _REGISTRY:
        available = list_models()
        raise KeyError(
            f"Model '{name}' not in registry. "
            f"Available: {available}. "
            f"Register custom models with register_model()."
        )
    return _REGISTRY[key](**kwargs)
