"""
example_run.py
==============
End-to-end SmartSignal usage examples.

    python example_run.py --example quick
    python example_run.py --example sp500 --tickers 50 --start 2018-01-01
    python example_run.py --example own_data --data_dir ./my_data/
    python example_run.py --example advanced

Dependencies
    pip install lightgbm scikit-learn pandas numpy matplotlib seaborn yfinance
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


def make_synthetic_universe(n_tickers=30, n_days=1500,
                             start="2019-01-01", seed=42):
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    dfs   = {}
    for i in range(n_tickers):
        t   = f"SYN{i:03d}"
        ret = rng.normal(0.0003, 0.015, n_days)
        c   = 100 * np.exp(np.cumsum(ret))
        dfs[t] = pd.DataFrame({
            "open":   c * (1 + rng.normal(0, 0.005, n_days)),
            "high":   c * (1 + rng.uniform(0.000, 0.012, n_days)),
            "low":    c * (1 - rng.uniform(0.000, 0.012, n_days)),
            "close":  c,
            "volume": rng.integers(500_000, 5_000_000, n_days).astype(float),
        }, index=dates)
    return dfs


# ── Example 1: Quick synthetic demo ───────────────────────────

def example_quick():
    print("\n" + "="*60)
    print("  EXAMPLE 1 - Quick Demo (Synthetic Data)")
    print("="*60)
    from smartsignal.workflow.pipeline import SmartSignalPipeline

    dfs  = make_synthetic_universe(n_tickers=30, n_days=1500)
    pipe = SmartSignalPipeline(
        execution_lag=1,   forward_days=5,
        top_k_features=20, train_years=2,  test_months=3,
        embargo_days=7,    n_long=5,        n_short=5,
        rebalance_freq="W", regime_filter=True, adx_threshold=15.0,
        min_hold_days=2,   transaction_cost=0.001, verbose=True,
    )
    result = pipe.run(dfs=dfs, compute_baselines=True)
    result.print_summary()

    print("\n[Charts] Generating 4-figure report ...")
    result.plot(save_dir="./charts/quick", show=True)
    return result


# ── Example 2: S&P 500 live ───────────────────────────────────

def example_sp500(n_tickers=50, start="2018-01-01"):
    print("\n" + "="*60)
    print("  EXAMPLE 2 - S&P 500 Live Pipeline")
    print("="*60)
    from smartsignal.data.universe     import fetch_sp500_tickers
    from smartsignal.workflow.pipeline import SmartSignalPipeline
    from smartsignal.workflow.helpers  import (
        PipelineConfig, validate_pipeline_config, make_run_id, save_results
    )
    from smartsignal.backtesting.performance import information_coefficient

    try:
        tickers = fetch_sp500_tickers()[:n_tickers]
    except Exception:
        tickers = [
            "AAPL","MSFT","GOOGL","AMZN","NVDA","META","BRK-B","JPM","V","UNH",
            "XOM","LLY","AVGO","MA","HD","CVX","MRK","ABBV","KO","PEP",
            "BAC","COST","PFE","TMO","CSCO","ACN","MCD","DIS","WMT","ABT",
            "NKE","TXN","PM","UPS","CRM","NEE","HON","LIN","DHR","QCOM",
            "BMY","AMGN","LOW","INTC","SBUX","RTX","CAT","GS","IBM","BLK",
        ][:n_tickers]

    print(f"  Using {len(tickers)} tickers: {tickers[:5]} ...")

    cfg = PipelineConfig(
        start_date="2018-01-01", min_history_days=504,
        min_avg_volume=500_000,  min_avg_price=5.0,
        execution_lag=1,         forward_days=5,
        top_k_features=25,       train_years=3,
        test_months=3,           embargo_days=7,
        n_long=10, n_short=10,   rebalance_freq="W",
        regime_filter=True,      adx_threshold=20.0,
        min_hold_days=3,         transaction_cost=0.001,
    )

    pipe = SmartSignalPipeline(**{
        k: v for k, v in cfg.to_dict().items()
        if k in SmartSignalPipeline.__init__.__code__.co_varnames
    })
    result = pipe.run(tickers=tickers, compute_baselines=True)

    run_id = make_run_id(cfg)
    save_results(result, "./runs", run_id=run_id, cfg=cfg)
    result.print_summary()

    ic = information_coefficient(
        result.panel_scored["rank_score"],
        result.panel_scored["fwd_ret"]
    )
    print(f"\n[IC]  Mean={ic['ic_mean']:.4f}  "
          f"ICIR={ic['icir']:.3f}  +ve%={ic['ic_positive_pct']:.1%}")

    print("\n[Charts] Generating 4-figure report ...")
    result.plot(save_dir=f"./charts/sp500_{run_id}", show=True,
                title_suffix=" - S&P 500")
    return result


# ── Example 3: Your own data ──────────────────────────────────

def example_own_data(data_dir="./my_data", start="2015-01-01"):
    """
    HOW TO USE YOUR OWN DATASET
    ===========================
    Option A  Directory of CSV files, one per ticker:
        my_data/AAPL.csv, my_data/MSFT.csv, ...
        Each file needs: date, open, high, low, close, volume columns.
        Run: python example_run.py --example own_data --data_dir ./my_data

    Option B  Single stacked CSV (all tickers in one file):
        date,       ticker, open, high, low,  close, volume
        2018-01-02, AAPL,   170,  172,  169,  172,   25000000
        ...
        Code:
            from smartsignal.data.loader import load_equity_data
            dfs = load_equity_data("./all_stocks.csv", ticker_col="ticker")
            result = pipe.run(dfs=dfs)

    Option C  Pre-loaded DataFrames already in memory:
            dfs = {"AAPL": df_aapl, "MSFT": df_msft}
            result = pipe.run(dfs=dfs)

    Option D  yfinance ticker list:
            result = pipe.run(tickers=["AAPL","MSFT"], start="2018-01-01")
    """
    print("\n" + "="*60)
    print("  EXAMPLE 3 - Your Own Data")
    print("="*60)
    from smartsignal.workflow.pipeline import SmartSignalPipeline

    pipe = SmartSignalPipeline(
        # Adjust to your market/data
        start_date       = start,
        min_history_days = 252,       # >= 1 year per ticker
        min_avg_volume   = 100_000,   # lower for non-US markets
        min_avg_price    = 1.0,       # lower for non-USD prices
        # Features
        execution_lag    = 1,
        forward_days     = 5,
        # Model
        top_k_features   = 25,
        train_years      = 2,
        test_months      = 3,
        embargo_days     = 7,
        # Backtest
        n_long           = 10,
        n_short          = 10,
        rebalance_freq   = "W",
        regime_filter    = True,
        adx_threshold    = 20.0,
        min_hold_days    = 3,
        transaction_cost = 0.001,
        verbose          = True,
    )

    # Single call — point at your data directory
    result = pipe.run(data_dir=data_dir, compute_baselines=True)
    result.print_summary()

    print("\n[Charts] Generating 4-figure report ...")
    result.plot(save_dir="./charts/own_data", show=True)
    return result


# ── Example 4: Advanced (model comparison + grid search) ──────

def example_advanced():
    print("\n" + "="*60)
    print("  EXAMPLE 4 - Advanced: Model Comparison + Grid Search")
    print("="*60)
    from smartsignal.workflow.combined        import ModelComparisonRun, StrategyGridRun
    from smartsignal.workflow.helpers         import PipelineConfig
    from smartsignal.workflow.pipeline        import SmartSignalPipeline, PipelineResult
    from smartsignal.models.panel_trainer     import PanelTrainer
    from smartsignal.models.lambdamart        import LambdaMARTRanker
    from smartsignal.features.equity_features import FEATURE_COLS, FEATURE_CATEGORIES
    from smartsignal.backtesting.baselines    import run_all_baselines
    from smartsignal.backtesting.engine       import run_backtest

    dfs = make_synthetic_universe(n_tickers=40, n_days=1500)
    cfg = PipelineConfig(train_years=2, test_months=3, top_k_features=20)

    # Step 1: model comparison
    print("\n[Step 1] Model comparison across 4 families ...")
    comp  = ModelComparisonRun(dfs=dfs, cfg=cfg,
                               models=["lambdamart","lgbm_classifier",
                                       "random_forest","ridge"])
    table = comp.run(verbose=True)
    print("\n[Model Comparison Table]")
    print(table.to_string())

    # Step 2: LambdaMART scored panel for grid search
    print("\n[Step 2] Pre-computing LambdaMART scored panel ...")
    pipe  = SmartSignalPipeline(train_years=2, test_months=3, verbose=True)
    dfs_f = pipe._stage_data(dfs, None, None)
    panel = pipe._stage_features(dfs_f)
    panel = pipe._stage_labels(panel)

    model   = LambdaMARTRanker(feature_cols=FEATURE_COLS, top_k_features=20)
    trainer = PanelTrainer(model=model, train_years=2, test_months=3,
                           embargo_days=7, forward_days=5, verbose=True)
    panel_scored, training_results = trainer.fit_predict(panel)

    # Step 3: parameter grid search
    print("\n[Step 3] Grid-searching strategy parameters ...")
    grid = StrategyGridRun(
        panel_scored=panel_scored, dfs=dfs_f,
        param_grid={
            "n_long":         [5, 10],
            "n_short":        [5, 10],
            "rebalance_freq": ["W", "ME"],
            "min_hold_days":  [1, 3],
        },
    )
    grid.run(verbose=True)
    best_params = grid.best_params()
    print(f"\n[Grid Search] Best parameters: {best_params}")

    # Step 4: final backtest + visualise
    bt        = run_backtest(panel_scored, dfs_f, verbose=False, **best_params)
    fi_df     = model.feature_importance_df()
    fi_df["category"] = fi_df["feature"].map(FEATURE_CATEGORIES).fillna("other")
    baselines = run_all_baselines(dfs_f, verbose=False)

    result = PipelineResult(
        backtest_result    = bt,
        baselines          = baselines,
        panel_scored       = panel_scored,
        feature_importance = fi_df,
        selected_features  = model.selected_features_,
        training_results   = training_results,
    )
    result.print_summary()

    print("\n[Charts] Generating 4-figure report ...")
    result.plot(save_dir="./charts/advanced", show=True,
                title_suffix=" - Advanced Demo")
    return result


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--example",
                   choices=["quick","sp500","own_data","advanced"],
                   default="quick")
    p.add_argument("--tickers",  type=int, default=50)
    p.add_argument("--start",    default="2018-01-01")
    p.add_argument("--data_dir", default="./my_data")
    args = p.parse_args()

    if   args.example == "quick":    example_quick()
    elif args.example == "sp500":    example_sp500(args.tickers, args.start)
    elif args.example == "own_data": example_own_data(args.data_dir, args.start)
    elif args.example == "advanced": example_advanced()