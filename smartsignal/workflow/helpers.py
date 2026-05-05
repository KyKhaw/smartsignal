"""
smartsignal.workflow.helpers
==============================
Shared utilities and configuration helpers for the pipeline workflow layer.

Provides:
  - PipelineConfig       : typed dataclass for all pipeline parameters.
  - validate_pipeline_config() : pre-flight checks before a run.
  - make_run_id()        : generate reproducible run identifiers.
  - save_results()       : serialise PipelineResult to disk.
  - load_results()       : deserialise a saved PipelineResult.
  - setup_logging()      : configure root logger for pipeline runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Pipeline configuration dataclass
# ──────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """
    Typed, serialisable configuration for a SmartSignal pipeline run.

    Mirrors the parameters of SmartSignalPipeline.__init__ so that a run
    can be fully reproduced from its saved config JSON.
    """
    # Data
    start_date:        str   = "2015-01-01"
    end_date:          Optional[str] = None
    min_history_days:  int   = 504
    min_avg_volume:    float = 1e6
    min_avg_price:     float = 5.0
    # Features
    execution_lag:     int   = 1
    forward_days:      int   = 5
    # Model
    top_k_features:    int   = 25
    train_years:       int   = 3
    test_months:       int   = 3
    mode:              str   = "expanding"
    embargo_days:      int   = 5
    ranker_params:     Dict  = field(default_factory=dict)
    # Backtesting
    n_long:            int   = 10
    n_short:           int   = 10
    rebalance_freq:    str   = "W"
    regime_filter:     bool  = True
    adx_threshold:     float = 20.0
    min_hold_days:     int   = 3
    transaction_cost:  float = 0.001
    slippage:          float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: Optional[str] = None) -> str:
        d    = self.to_dict()
        text = json.dumps(d, indent=2, default=str)
        if path:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        data = json.loads(Path(path).read_text())
        return cls(**data)

    @classmethod
    def from_dict(cls, d: Dict) -> "PipelineConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ──────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────

def validate_pipeline_config(cfg: PipelineConfig) -> List[str]:
    """
    Pre-flight checks for a pipeline config.

    Returns a list of warning strings (empty = no issues).
    Raises ValueError for hard errors.
    """
    issues: List[str] = []

    if cfg.train_years < 1:
        raise ValueError("train_years must be ≥ 1.")

    if cfg.forward_days >= cfg.embargo_days:
        issues.append(
            f"embargo_days ({cfg.embargo_days}) should be > forward_days "
            f"({cfg.forward_days}) to prevent label leakage."
        )

    if cfg.transaction_cost < 0:
        raise ValueError("transaction_cost cannot be negative.")

    if cfg.n_long < 1 or cfg.n_short < 1:
        raise ValueError("n_long and n_short must each be ≥ 1.")

    if cfg.top_k_features < 5:
        issues.append(
            f"top_k_features={cfg.top_k_features} is very small. "
            "Consider ≥ 10 for stable feature selection."
        )

    return issues


# ──────────────────────────────────────────────────────────────
# Run ID and reproducibility
# ──────────────────────────────────────────────────────────────

def make_run_id(cfg: Optional[PipelineConfig] = None) -> str:
    """
    Generate a short reproducible run identifier from the config hash + timestamp.

    Format: YYYYMMDD_HHMMSS_<6-char config hash>
    """
    ts   = time.strftime("%Y%m%d_%H%M%S")
    if cfg is not None:
        cfg_str = json.dumps(cfg.to_dict(), sort_keys=True, default=str)
        h       = hashlib.md5(cfg_str.encode()).hexdigest()[:6]
    else:
        h = "000000"
    return f"{ts}_{h}"


# ──────────────────────────────────────────────────────────────
# Result persistence
# ──────────────────────────────────────────────────────────────

def save_results(
    result,
    output_dir:  str,
    run_id:      Optional[str] = None,
    cfg:         Optional[PipelineConfig] = None,
) -> str:
    """
    Save a PipelineResult to disk.

    Saves:
      - strategy_returns.csv
      - positions.parquet
      - metrics.json
      - config.json (if cfg provided)

    Returns the output directory path.
    """
    out_dir = Path(output_dir) / (run_id or make_run_id(cfg))
    out_dir.mkdir(parents=True, exist_ok=True)

    bt = result.backtest_result

    # Strategy returns
    bt.strategy_returns.to_csv(out_dir / "strategy_returns.csv", header=True)

    # Positions
    if bt.positions is not None:
        bt.positions.to_parquet(out_dir / "positions.parquet")

    # Metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(bt.metrics, f, indent=2, default=str)

    # Feature importance
    if result.feature_importance is not None:
        result.feature_importance.to_csv(out_dir / "feature_importance.csv", index=False)

    # Config
    if cfg is not None:
        cfg.to_json(str(out_dir / "config.json"))

    logger.info("Results saved to %s", out_dir)
    return str(out_dir)


def load_results(output_dir: str) -> Dict[str, Any]:
    """
    Load saved results from a run directory.

    Returns a dict with keys: strategy_returns, metrics, config (if present).
    """
    p = Path(output_dir)
    out: Dict[str, Any] = {}

    ret_path = p / "strategy_returns.csv"
    if ret_path.exists():
        out["strategy_returns"] = pd.read_csv(ret_path, index_col=0, parse_dates=True).squeeze()

    metrics_path = p / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            out["metrics"] = json.load(f)

    cfg_path = p / "config.json"
    if cfg_path.exists():
        out["config"] = PipelineConfig.from_json(str(cfg_path))

    fi_path = p / "feature_importance.csv"
    if fi_path.exists():
        import pandas as pd
        out["feature_importance"] = pd.read_csv(fi_path)

    return out


# ──────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """Configure the root logger for pipeline runs."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=handlers,
        force=True,
    )
    logging.getLogger("lightgbm").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
