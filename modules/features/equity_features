"""
smartsignal.features.equity_features
=====================================
Equity-market feature engineering pipeline.

Implements a 42-feature cross-sectional panel following the 5-category
framework of Wang & Dong (2025) and Dey et al. (2025):

  Category 1 – Overlap Studies (trend / MA-based)
    sma_5, sma_10, sma_20, sma_50, ema_12, ema_26, dema_12, wma_10,
    bb_width, bb_pct,
    price_sma5_ratio, price_sma20_ratio, price_sma50_ratio, price_sma200_ratio

  Category 2 – Momentum (oscillators, trend strength)
    rsi_14, stoch_rsi,
    macd, macd_signal, macd_hist,
    adx_14, willr_14, aroon_osc,
    roc_10, mom_10, cmo_14

  Category 3 – Volatility
    atr_14, hvol_10, hvol_20, tr_norm

  Category 4 – Volume
    vol_ratio, force_idx, obv, ad_line, pvt

  Category 5 – Price Transform (candle structure + short returns)
    body_ratio, upper_shadow, lower_shadow, gap,
    ret_1d, ret_3d, ret_5d, ret_10d

Design principles:
  - All computations are strictly backward-looking (no future leakage).
  - A small epsilon guard (_safe_div) avoids divide-by-zero in all ratios.
  - Each feature function declares its minimum lookback via FEATURE_LOOKBACKS.
  - The registry maps feature name → category for grouped analysis.
"""

from __future__ import annotations

import math
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Feature catalogue
# ──────────────────────────────────────────────────────────────

FEATURE_COLS: List[str] = [
    # Category 1 – Overlap / Trend
    "sma_5", "sma_10", "sma_20", "sma_50",
    "ema_12", "ema_26", "dema_12", "wma_10",
    "bb_width", "bb_pct",
    "price_sma5_ratio", "price_sma20_ratio", "price_sma50_ratio", "price_sma200_ratio",
    # Category 2 – Momentum
    "rsi_14", "stoch_rsi",
    "macd", "macd_signal", "macd_hist",
    "adx_14", "willr_14", "aroon_osc",
    "roc_10", "mom_10", "cmo_14",
    # Category 3 – Volatility
    "atr_14", "hvol_10", "hvol_20", "tr_norm",
    # Category 4 – Volume
    "vol_ratio", "force_idx", "obv", "ad_line", "pvt",
    # Category 5 – Price Transform
    "body_ratio", "upper_shadow", "lower_shadow", "gap",
    "ret_1d", "ret_3d", "ret_5d", "ret_10d",
]

FEATURE_CATEGORIES: Dict[str, str] = {
    # Category 1
    **{f: "overlap" for f in [
        "sma_5", "sma_10", "sma_20", "sma_50",
        "ema_12", "ema_26", "dema_12", "wma_10",
        "bb_width", "bb_pct",
        "price_sma5_ratio", "price_sma20_ratio", "price_sma50_ratio", "price_sma200_ratio",
    ]},
    # Category 2
    **{f: "momentum" for f in [
        "rsi_14", "stoch_rsi",
        "macd", "macd_signal", "macd_hist",
        "adx_14", "willr_14", "aroon_osc",
        "roc_10", "mom_10", "cmo_14",
    ]},
    # Category 3
    **{f: "volatility" for f in ["atr_14", "hvol_10", "hvol_20", "tr_norm"]},
    # Category 4
    **{f: "volume" for f in ["vol_ratio", "force_idx", "obv", "ad_line", "pvt"]},
    # Category 5
    **{f: "price_transform" for f in [
        "body_ratio", "upper_shadow", "lower_shadow", "gap",
        "ret_1d", "ret_3d", "ret_5d", "ret_10d",
    ]},
}

# Minimum number of bars required for each feature to produce valid output
FEATURE_LOOKBACKS: Dict[str, int] = {
    "sma_5": 5, "sma_10": 10, "sma_20": 20, "sma_50": 50,
    "ema_12": 26, "ema_26": 26, "dema_12": 26, "wma_10": 10,
    "bb_width": 20, "bb_pct": 20,
    "price_sma5_ratio": 5, "price_sma20_ratio": 20,
    "price_sma50_ratio": 50, "price_sma200_ratio": 200,
    "rsi_14": 28, "stoch_rsi": 28,
    "macd": 26, "macd_signal": 35, "macd_hist": 35,
    "adx_14": 28, "willr_14": 14, "aroon_osc": 25,
    "roc_10": 10, "mom_10": 10, "cmo_14": 14,
    "atr_14": 14, "hvol_10": 10, "hvol_20": 20, "tr_norm": 1,
    "vol_ratio": 20, "force_idx": 1, "obv": 1, "ad_line": 1, "pvt": 1,
    "body_ratio": 1, "upper_shadow": 1, "lower_shadow": 1, "gap": 1,
    "ret_1d": 1, "ret_3d": 3, "ret_5d": 5, "ret_10d": 10,
}

# Overall warmup requirement (max lookback across all features)
MIN_WARMUP_BARS: int = max(FEATURE_LOOKBACKS.values())   # 200


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

def _safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    """Divide a by b, returning `fill` where |b| < epsilon."""
    eps = 1e-12
    mask = b.abs() > eps
    return a.where(mask, fill) / b.where(mask, 1.0)


def _wma(series: pd.Series, n: int) -> pd.Series:
    """Linearly-weighted moving average over n periods."""
    weights = np.arange(1, n + 1, dtype=float)
    return series.rolling(n).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


# ──────────────────────────────────────────────────────────────
# Per-ticker feature computation
# ──────────────────────────────────────────────────────────────

def compute_features(
    df: pd.DataFrame,
    execution_lag: int = 1,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Compute all 42 equity features for a single-ticker OHLCV DataFrame.

    Parameters
    ----------
    df            : DataFrame with columns open/high/low/close/volume and
                    a DatetimeIndex.
    execution_lag : forward-shift all features by this many bars to simulate
                    execution lag (1 = open-to-open lag, prevents look-ahead).
    drop_na       : drop rows that still contain NaN after warmup period.

    Returns
    -------
    DataFrame with original OHLCV columns plus all 42 feature columns.
    """
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    # Normalise column names
    col_map = {
        "adj close": "close", "adjusted close": "close",
        "adj_close": "close", "adjusted_close": "close",
    }
    df.rename(columns=col_map, inplace=True)

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    if df["close"].ndim > 1:
        raise ValueError("Multiple 'close' columns detected.")

    o, h, l, c, v = (df[k] for k in ["open", "high", "low", "close", "volume"])

    # ── Category 1: Overlap Studies ───────────────────────────
    for n in [5, 10, 20, 50, 200]:
        df[f"sma_{n}"] = c.rolling(n).mean()

    for n in [12, 26]:
        df[f"ema_{n}"] = c.ewm(span=n, adjust=False).mean()

    ema12 = df["ema_12"]
    df["dema_12"] = 2 * ema12 - ema12.ewm(span=12, adjust=False).mean()
    df["wma_10"]  = _wma(c, 10)

    bb_mid = df["sma_20"]
    bb_std = c.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_width"] = _safe_div(df["bb_upper"] - df["bb_lower"], bb_mid)
    df["bb_pct"]   = _safe_div(c - df["bb_lower"], df["bb_upper"] - df["bb_lower"])

    df["price_sma5_ratio"]   = _safe_div(c, df["sma_5"])   - 1
    df["price_sma20_ratio"]  = _safe_div(c, df["sma_20"])  - 1
    df["price_sma50_ratio"]  = _safe_div(c, df["sma_50"])  - 1
    df["price_sma200_ratio"] = _safe_div(c, df["sma_200"]) - 1

    # ── Category 2: Momentum ──────────────────────────────────
    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_l = loss.ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi_14"] = 100 - 100 / (1 + _safe_div(avg_g, avg_l, fill=1e9))

    rsi = df["rsi_14"]
    df["stoch_rsi"] = _safe_div(
        rsi - rsi.rolling(14).min(),
        rsi.rolling(14).max() - rsi.rolling(14).min(),
    )

    df["macd"]        = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # True Range & ATR (shared with Category 3)
    tr = pd.concat(
        [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1
    ).max(axis=1)

    dm_up   = (h - h.shift(1)).clip(lower=0)
    dm_down = (l.shift(1) - l).clip(lower=0)
    dm_up   = dm_up.where(dm_up > dm_down, 0)
    dm_down = dm_down.where(dm_down > dm_up, 0)
    atr14   = tr.ewm(span=14, adjust=False).mean()
    di_up   = 100 * _safe_div(dm_up.ewm(span=14, adjust=False).mean(),   atr14)
    di_down = 100 * _safe_div(dm_down.ewm(span=14, adjust=False).mean(), atr14)
    dx      = 100 * _safe_div((di_up - di_down).abs(), di_up + di_down)
    df["adx_14"] = dx.ewm(span=14, adjust=False).mean()

    hh = h.rolling(14).max()
    ll = l.rolling(14).min()
    df["willr_14"] = -100 * _safe_div(hh - c, hh - ll)

    aroon_up   = 100 * (25 - h.rolling(25).apply(
        lambda x: 25 - int(np.argmax(x)), raw=True)) / 25
    aroon_down = 100 * (25 - l.rolling(25).apply(
        lambda x: 25 - int(np.argmin(x)), raw=True)) / 25
    df["aroon_osc"] = aroon_up - aroon_down

    df["roc_10"] = _safe_div(c - c.shift(10), c.shift(10)) * 100
    df["mom_10"] = c - c.shift(10)

    pos_sum = gain.rolling(14).sum()
    neg_sum = loss.rolling(14).sum()
    df["cmo_14"] = 100 * _safe_div(pos_sum - neg_sum, pos_sum + neg_sum)

    # ── Category 3: Volatility ────────────────────────────────
    df["atr_14"]  = atr14
    ret1          = c.pct_change()
    df["hvol_10"] = ret1.rolling(10).std() * math.sqrt(252)
    df["hvol_20"] = ret1.rolling(20).std() * math.sqrt(252)
    df["tr_norm"] = _safe_div(tr, c)

    # ── Category 4: Volume ────────────────────────────────────
    sign        = np.sign(c.diff()).fillna(0)
    df["obv"]   = (sign * v).cumsum()

    clv             = _safe_div((c - l) - (h - c), h - l)
    df["ad_line"]   = (clv * v).cumsum()
    df["pvt"]       = (_safe_div(c - c.shift(1), c.shift(1)) * v).cumsum()
    df["vol_ratio"] = _safe_div(v, v.rolling(20).mean())
    df["force_idx"] = c.diff() * v

    # ── Category 5: Price Transform & Short Returns ───────────
    body   = (c - o).abs()
    range_ = (h - l).clip(lower=1e-9)
    df["body_ratio"]   = _safe_div(body, range_)
    df["upper_shadow"] = _safe_div(
        h - pd.concat([c, o], axis=1).max(axis=1), range_
    )
    df["lower_shadow"] = _safe_div(
        pd.concat([c, o], axis=1).min(axis=1) - l, range_
    )
    df["gap"] = _safe_div(o - c.shift(1), c.shift(1))

    for n in [1, 3, 5, 10]:
        df[f"ret_{n}d"] = c.pct_change(n)

    # ── Execution lag (prevent look-ahead) ────────────────────
    if execution_lag > 0:
        feature_only = [f for f in FEATURE_COLS if f in df.columns]
        df[feature_only] = df[feature_only].shift(execution_lag)

    if drop_na:
        df = df.dropna(subset=FEATURE_COLS)

    return df


# ──────────────────────────────────────────────────────────────
# Multi-ticker panel builder
# ──────────────────────────────────────────────────────────────

def build_feature_panel(
    dfs: Dict[str, pd.DataFrame],
    execution_lag: int = 1,
    forward_days: int = 5,
    drop_na: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute features for all tickers and stack into a cross-sectional panel.

    Each row is (date, ticker).  The forward return column 'fwd_ret' is
    added here for use by the label generator but is NOT included in
    FEATURE_COLS (no future leakage).

    Parameters
    ----------
    dfs           : dict of per-ticker OHLCV DataFrames (from EquityLoader).
    execution_lag : bars to shift features forward (execution lag).
    forward_days  : horizon for the forward return target.
    drop_na       : drop rows with missing features.
    verbose       : print progress stats.

    Returns
    -------
    panel : pd.DataFrame with MultiIndex (date, ticker) implied by the
            date index and a 'ticker' column.
    """
    records = []
    skipped = []

    for ticker, df_raw in dfs.items():
        try:
            df_feat = compute_features(df_raw, execution_lag=execution_lag, drop_na=False)
            df_feat["ticker"]  = ticker
            # Forward return — shifted AFTER feature computation to avoid leakage
            df_feat["fwd_ret"] = df_feat["close"].pct_change(forward_days).shift(-forward_days)
            if drop_na:
                df_feat = df_feat.dropna(subset=FEATURE_COLS)
            records.append(df_feat)
        except Exception as exc:
            skipped.append(f"{ticker}: {exc}")
            logger.warning("  [WARN] Skipped %s: %s", ticker, exc)

    if not records:
        raise ValueError("No valid ticker data after feature computation.")

    panel = pd.concat(records).sort_index()

    if verbose:
        dates   = panel.index.unique()
        tickers = panel["ticker"].unique()
        print(
            f"[FeaturePanel] {len(dates)} dates × {len(tickers)} tickers "
            f"= {len(panel):,} rows"
        )
        print(
            f"  Date range : {dates.min().date()} – {dates.max().date()}"
        )
        if skipped:
            print(f"  Skipped    : {len(skipped)} tickers")

    return panel


def get_feature_importance_by_category(
    feature_names: List[str],
    importances: np.ndarray,
) -> pd.DataFrame:
    """
    Summarise feature importances grouped by the 5 feature categories.

    Parameters
    ----------
    feature_names : list of feature names matching columns of the model.
    importances   : array of importances (e.g. LightGBM feature_importances_).

    Returns
    -------
    summary : DataFrame with columns [feature, category, importance, rank].
    """
    df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
    })
    df["category"] = df["feature"].map(FEATURE_CATEGORIES).fillna("other")
    df["rank"]     = df["importance"].rank(ascending=False).astype(int)
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df
