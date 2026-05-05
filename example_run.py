"""
example_run.py
==============
End-to-end SmartSignal usage examples.

Run from the smartsignal/ project root:
    python example_run.py

Three examples are provided:
    1. QUICK  – 5 tickers, 3 years, runs in ~60 seconds (no yfinance needed,
                uses synthetic data to demonstrate the full pipeline offline).
    2. SP500  – full S&P 500 universe, live yfinance download (requires internet).
    3. ADVANCED – model comparison + parameter grid search on your own data.
"""

import sys
import os

# Add the project root to sys.path if running as a script
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════
# HELPER: Synthetic OHLCV generator (no internet required)
# ══════════════════════════════════════════════════════════════

def make_synthetic_universe(
    n_tickers: int = 30,
    n_days: int = 1500,
    start: str = "2019-01-01",
    seed: int = 42,
) -> dict:
    """
    Generate synthetic OHLCV data for a small equity universe.
    Used by Example 1 so the pipeline can be tested offline.
    """
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    dfs   = {}

    for i in range(n_tickers):
        ticker  = f"SYN{i:03d}"
        log_ret = rng.normal(0.0003, 0.015, n_days)
        close   = 100 * np.exp(np.cumsum(log_ret))
        high    = close * (1 + rng.uniform(0.000, 0.012, n_days))
        low     = close * (1 - rng.uniform(0.000, 0.012, n_days))
        open_   = close * (1 + rng.normal(0, 0.005, n_days))
        volume  = rng.integers(500_000, 5_000_000, n_days).astype(float)

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low,
             "close": close, "volume": volume},
            index=dates,
        )
        dfs[ticker] = df

    return dfs


# ══════════════════════════════════════════════════════════════
# EXAMPLE 1 – Quick offline demo with synthetic data
# ══════════════════════════════════════════════════════════════

def example_quick():
    """
    Minimal end-to-end pipeline on synthetic data.
    Demonstrates every module without any internet connection.
    Takes ~30–60 seconds.
    """
    print("\n" + "═" * 60)
    print("  EXAMPLE 1 — Quick Demo (Synthetic Data)")
    print("═" * 60)

    from smartsignal.workflow.pipeline import SmartSignalPipeline

    # ── Generate synthetic universe ───────────────────────────
    print("\n[Step 0] Generating 30-ticker synthetic universe (1 500 days) …")
    dfs = make_synthetic_universe(n_tickers=30, n_days=1500)

    # ── Configure pipeline ────────────────────────────────────
    pipe = SmartSignalPipeline(
        start_date       = "2019-01-01",
        # Features
        execution_lag    = 1,
        forward_days     = 5,
        # Model
        top_k_features   = 20,
        train_years      = 2,
        test_months      = 3,
        mode             = "expanding",
        embargo_days     = 7,
        # Backtesting
        n_long           = 5,
        n_short          = 5,
        rebalance_freq   = "W",
        regime_filter    = True,
        adx_threshold    = 15.0,
        min_hold_days    = 2,
        transaction_cost = 0.001,
        verbose          = True,
    )

    # ── Run all 6 stages ──────────────────────────────────────
    result = pipe.run(dfs=dfs, compute_baselines=True)

    # ── Print summary ─────────────────────────────────────────
    result.print_summary()

    # ── Cross-sectional analytics ──────────────────────────────
    from smartsignal.backtesting.cross_section import (
        quintile_returns,
        cross_sectional_ic,
        spread_decomposition,
    )

    qr = quintile_returns(result.panel_scored)
    print("\n[Analytics] Mean return by score quintile:")
    print(qr[["mean_return"]].to_string())

    decomp = spread_decomposition(result.panel_scored)
    print(f"\n[Analytics] Spread decomposition:")
    for k, v in decomp.items():
        print(f"  {k:<25}: {v:+.4f}")

    # ── Feature importance ────────────────────────────────────
    if result.feature_importance is not None:
        print("\n[Analytics] Top-10 features:")
        top10 = result.feature_importance.head(10)[["feature", "category", "importance"]]
        print(top10.to_string(index=False))

    return result


# ══════════════════════════════════════════════════════════════
# EXAMPLE 2 – Live S&P 500 pipeline (requires internet)
# ══════════════════════════════════════════════════════════════

def example_sp500(
    n_tickers: int = 50,
    start: str = "2018-01-01",
):
    """
    Full pipeline on a sample of S&P 500 tickers downloaded via yfinance.

    Parameters
    ----------
    n_tickers : number of S&P 500 tickers to use (50 is fast, 500 is complete).
    start     : history start date.
    """
    print("\n" + "═" * 60)
    print("  EXAMPLE 2 — S&P 500 Live Pipeline")
    print("═" * 60)

    from smartsignal.data.universe   import fetch_sp500_tickers
    from smartsignal.workflow.pipeline import SmartSignalPipeline
    from smartsignal.workflow.helpers  import (
        PipelineConfig, validate_pipeline_config,
        make_run_id, save_results
    )

    # ── Fetch tickers ─────────────────────────────────────────
    print(f"\n[Step 0] Fetching S&P 500 ticker list …")
    try:
        all_tickers = fetch_sp500_tickers()
        tickers = all_tickers[:n_tickers]
        print(f"  Using {len(tickers)} tickers: {tickers[:5]} …")
    except Exception as e:
        print(f"  Warning: could not fetch S&P 500 list ({e}). Using hardcoded sample.")
        tickers = [
            "AAPL","MSFT","GOOGL","AMZN","NVDA","META","BRK-B","JPM","V","UNH",
            "XOM","LLY","AVGO","MA","HD","CVX","MRK","ABBV","KO","PEP",
            "BAC","COST","PFE","TMO","CSCO","ACN","MCD","DIS","WMT","ABT",
            "NKE","TXN","PM","UPS","CRM","NEE","HON","LIN","DHR","QCOM",
            "BMY","AMGN","LOW","INTC","SBUX","RTX","CAT","GS","IBM","BLK",
        ][:n_tickers]

    # ── Build typed config ─────────────────────────────────────
    cfg = PipelineConfig(
        start_date       = start,
        min_history_days = 504,
        min_avg_volume   = 500_000,
        min_avg_price    = 5.0,
        execution_lag    = 1,
        forward_days     = 5,
        top_k_features   = 25,
        train_years      = 3,
        test_months      = 3,
        mode             = "expanding",
        embargo_days     = 7,
        n_long           = 10,
        n_short          = 10,
        rebalance_freq   = "W",
        regime_filter    = True,
        adx_threshold    = 20.0,
        min_hold_days    = 3,
        transaction_cost = 0.001,
        slippage         = 0.0,
    )

    # Pre-flight validation
    issues = validate_pipeline_config(cfg)
    if issues:
        print("\n[Config] Warnings:")
        for w in issues:
            print(f"  ⚠  {w}")

    # ── Run pipeline ──────────────────────────────────────────
    pipe = SmartSignalPipeline(**{
        k: v for k, v in cfg.to_dict().items()
        if k in SmartSignalPipeline.__init__.__code__.co_varnames
    })
    result = pipe.run(tickers=tickers, compute_baselines=True)

    # ── Save results ──────────────────────────────────────────
    run_id   = make_run_id(cfg)
    out_path = save_results(result, output_dir="./runs", run_id=run_id, cfg=cfg)
    print(f"\n[Saved] Results at: {out_path}")

    # ── Performance summary ───────────────────────────────────
    result.print_summary()

    # ── IC analysis ───────────────────────────────────────────
    from smartsignal.backtesting.performance import information_coefficient
    ic_stats = information_coefficient(
        result.panel_scored["rank_score"],
        result.panel_scored["fwd_ret"],
    )
    print(f"\n[IC Analysis]")
    print(f"  IC Mean  : {ic_stats['ic_mean']:.4f}")
    print(f"  ICIR     : {ic_stats['icir']:.3f}")
    print(f"  IC +ve % : {ic_stats['ic_positive_pct']:.1%}")

    return result


# ══════════════════════════════════════════════════════════════
# EXAMPLE 3 – Advanced: model comparison + grid search
# ══════════════════════════════════════════════════════════════

def example_advanced():
    """
    Run multiple model families on the same synthetic data, then
    grid-search strategy parameters for the best LambdaMART config.
    """
    print("\n" + "═" * 60)
    print("  EXAMPLE 3 — Advanced: Model Comparison + Grid Search")
    print("═" * 60)

    from smartsignal.workflow.combined import ModelComparisonRun, StrategyGridRun
    from smartsignal.workflow.helpers  import PipelineConfig
    from smartsignal.workflow.pipeline import SmartSignalPipeline

    dfs = make_synthetic_universe(n_tickers=40, n_days=1500)
    cfg = PipelineConfig(train_years=2, test_months=3, top_k_features=20)

    # ── Model comparison ──────────────────────────────────────
    print("\n[Step 1] Model comparison across 4 families …")
    comp_run = ModelComparisonRun(
        dfs    = dfs,
        cfg    = cfg,
        models = ["lambdamart", "lgbm_classifier", "random_forest", "ridge"],
    )
    table = comp_run.run(verbose=True)
    print("\n[Model Comparison Table]")
    print(table.to_string())

    # ── Pre-compute LambdaMART scored panel for grid search ───
    print("\n[Step 2] Pre-computing LambdaMART scored panel for grid search …")
    base_pipe = SmartSignalPipeline(train_years=2, test_months=3, verbose=False)
    dfs_f     = base_pipe._stage_data(dfs, None, None)
    panel     = base_pipe._stage_features(dfs_f)
    panel     = base_pipe._stage_labels(panel)
    panel_scored, _ = SmartSignalPipeline(
        train_years=2, test_months=3, verbose=True
    )._stage_train(panel)

    # ── Strategy grid search ──────────────────────────────────
    print("\n[Step 3] Grid-searching strategy parameters …")
    grid_run = StrategyGridRun(
        panel_scored = panel_scored,
        dfs          = dfs_f,
        param_grid   = {
            "n_long":         [5, 10],
            "n_short":        [5, 10],
            "rebalance_freq": ["W", "ME"],
            "min_hold_days":  [1, 3],
        },
    )
    grid_df    = grid_run.run(verbose=True)
    best_params = grid_run.best_params()
    print(f"\n[Grid Search] Best parameters: {best_params}")

    return comp_run, grid_run


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SmartSignal pipeline examples")
    parser.add_argument(
        "--example", choices=["quick", "sp500", "advanced"], default="quick",
        help="Which example to run (default: quick)"
    )
    parser.add_argument(
        "--tickers", type=int, default=50,
        help="Number of S&P 500 tickers to use in the sp500 example"
    )
    parser.add_argument(
        "--start", default="2018-01-01",
        help="Start date for the sp500 example (YYYY-MM-DD)"
    )
    args = parser.parse_args()

    if args.example == "quick":
        example_quick()
    elif args.example == "sp500":
        example_sp500(n_tickers=args.tickers, start=args.start)
    elif args.example == "advanced":
        example_advanced()