"""
smartsignal.backtesting.execution
===================================
Trade execution cost model for the SmartSignal backtesting engine.

Models three components of transaction cost:

  1. Commission / brokerage fee     – proportional to notional traded.
  2. Market impact (slippage)       – proportional to order size relative
                                      to average daily volume (ADV).
  3. Bid-ask spread                 – half-spread per round-trip.

The default parameters are calibrated for U.S. equity markets:
  - Brokerage: ~5 bps one-way (institutional approximation).
  - Market impact: negligible at the portfolio sizes typical of academic
    backtests, but included for realism.
  - Spread: ~5 bps for liquid large-cap stocks.

All costs are returned as a DataFrame of the same shape as the position
matrix so they can be subtracted from the gross P&L on each date.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────
# Cost model parameters
# ──────────────────────────────────────────────────────────────

@dataclass
class CostModel:
    """
    Container for all transaction cost components.

    Parameters
    ----------
    commission    : one-way brokerage commission as a fraction of notional
                    (e.g. 0.0005 = 5 bps).
    slippage      : additional one-way slippage (e.g. 0.0005 = 5 bps).
    half_spread   : half of the bid-ask spread (e.g. 0.0003 = 3 bps).
    impact_factor : market impact coefficient; cost scales as
                    impact_factor × (trade_size / ADV)^0.5.
                    Set to 0 to disable impact modelling.
    """
    commission:    float = 0.0005
    slippage:      float = 0.0005
    half_spread:   float = 0.0003
    impact_factor: float = 0.0

    @property
    def total_one_way(self) -> float:
        """Fixed one-way cost (commission + slippage + half spread)."""
        return self.commission + self.slippage + self.half_spread


# ──────────────────────────────────────────────────────────────
# Execution engine
# ──────────────────────────────────────────────────────────────

class ExecutionModel:
    """
    Computes transaction costs from position changes.

    Parameters
    ----------
    cost_model : CostModel instance; uses defaults if None.
    """

    def __init__(self, cost_model: Optional[CostModel] = None):
        self.cost_model = cost_model or CostModel()

    def compute_costs(
        self,
        positions:    pd.DataFrame,
        close_prices: pd.DataFrame,
        adv:          Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute a cost matrix from position changes.

        Parameters
        ----------
        positions    : date × ticker position matrix ({-1, 0, +1}).
        close_prices : date × ticker close price matrix.
        adv          : date × ticker average daily volume matrix (optional;
                       required for impact modelling).

        Returns
        -------
        cost_matrix : date × ticker DataFrame of per-position costs
                      (as fraction of notional).  Positive = cost.
        """
        pos_prev = positions.shift(1).fillna(0)
        turnover = (positions - pos_prev).abs()

        fixed_cost = self.cost_model.total_one_way * turnover

        if self.cost_model.impact_factor > 0 and adv is not None:
            # Approximate trade size as 1 position unit = $1 of notional
            # Impact = factor × sqrt(trade_size / ADV)
            # Here we use turnover as a proxy for trade size fraction
            adv_aligned = adv.reindex_like(positions).ffill()
            impact = (
                self.cost_model.impact_factor
                * (turnover / adv_aligned.clip(lower=1)).pow(0.5)
            )
            impact = impact.fillna(0)
        else:
            impact = pd.DataFrame(0.0, index=positions.index, columns=positions.columns)

        return fixed_cost + impact

    def net_returns(
        self,
        positions:    pd.DataFrame,
        close_prices: pd.DataFrame,
        adv:          Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute gross P&L minus transaction costs per ticker per day.

        Returns
        -------
        net_pnl : date × ticker DataFrame of daily net returns per position.
        """
        ret_matrix = close_prices.pct_change()
        gross      = positions.shift(1) * ret_matrix
        costs      = self.compute_costs(positions, close_prices, adv)
        return gross - costs
