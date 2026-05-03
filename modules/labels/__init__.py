"""
SmartSignal
===========
An adaptive machine-learning pipeline for automated financial trading
signal generation, focused on U.S. equity markets.

Modules
-------
data        – OHLCV ingestion and universe management
features    – Equity-specific feature engineering and cross-sectional transforms
labels      – Cross-sectional quintile relevance label generation
models      – LambdaMART ranker with importance-based feature selection
validation  – Walk-forward expanding/rolling window splitter
backtesting – Position construction, P&L computation, and performance metrics
utils       – Shared metrics, plotting helpers, and logging utilities
workflow    – Top-level pipeline orchestrator
"""

__version__ = "0.1.0"
__author__  = "SmartSignal"

from smartsignal.workflow.pipeline import SmartSignalPipeline

__all__ = ["SmartSignalPipeline"]
