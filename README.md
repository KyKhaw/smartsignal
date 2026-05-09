# SmartSignal

**An Adaptive Machine Learning Pipeline for Automated Financial Trading Signal Generation**

SmartSignal is a modular, research-grade Python pipeline that takes raw equity OHLCV data and produces interpretable long-short trading signals through a full walk-forward validated machine learning workflow. It is built for U.S. equity markets but is configurable for any daily OHLCV dataset.

The pipeline covers six stages end-to-end: data ingestion and validation → 42-feature equity engineering → cross-sectional quintile label generation → LambdaMART walk-forward training → position construction with regime filtering → backtesting with performance attribution and visualisation.

---

## Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Using Your Own Data](#using-your-own-data)
- [Pipeline Configurations](#pipeline-configurations)
- [Expected Results](#expected-results)
- [Visualisation Report](#visualisation-report)
- [Repository Structure](#repository-structure)
- [Module Overview](#module-overview)
- [References](#references)

---

## Architecture

SmartSignal is organised as six sequential pipeline stages, each backed by a dedicated module:

```
Raw OHLCV Data
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1 · Data Ingestion          smartsignal/data/    │
│  loader · universe · schema · validator                 │
│  CSV / Parquet / yfinance / stacked panel               │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2 · Feature Engineering  smartsignal/features/  │
│  42 equity features across 5 categories:               │
│  Overlap · Momentum · Volatility · Volume · Price      │
│  Execution-lag shift · cross-sectional transforms      │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3 · Label Generation      smartsignal/labels/   │
│  Cross-sectional quintile relevance labels (0–3)       │
│  Forward-return horizon · embargo-safe trimming        │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 4 · Walk-Forward Training  smartsignal/models/  │
│  LambdaMART (LGBMRanker) · expanding window            │
│  Importance-based feature selection on fold 0          │
│  Validation IC-Sharpe tracked per fold                 │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 5 · Backtesting        smartsignal/backtesting/ │
│  Dollar-neutral long-short position construction       │
│  ADX regime filter · minimum-hold filter               │
│  Transaction costs · equal-weight & momentum baselines │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 6 · Reporting            smartsignal/utils/     │
│  4-figure visualisation report · metrics table         │
│  IC analysis · monthly heatmap · risk analytics        │
└─────────────────────────────────────────────────────────┘
```

---

## Installation

### Requirements

- Python 3.9 or later
- The packages listed in `requirements.txt`

```bash
git clone https://github.com/KyKhaw/smartsignal.git
cd smartsignal
pip install -r requirements.txt
```

### Core dependencies

| Package | Purpose |
|---|---|
| `lightgbm` | LambdaMART ranker (LGBMRanker) |
| `scikit-learn` | Preprocessing, sklearn model wrappers |
| `pandas` | Data manipulation throughout |
| `numpy` | Numerical operations |
| `matplotlib` | Visualisation report (4 figures) |
| `seaborn` | Monthly returns heatmap (optional, graceful fallback) |
| `scipy` | Q-Q plot in risk figure (optional, graceful fallback) |
| `yfinance` | Live equity data download |
| `optuna` | Hyperparameter optimisation (optional) |

---

## Quick Start

`example_run.py` exposes four self-contained examples through a single CLI flag.

### Example 1 — Quick (synthetic, offline)

```bash
python example_run.py --example quick
```

Generates a 30-ticker synthetic OHLCV universe (geometric Brownian motion) and runs the full pipeline. No internet connection required. Useful for verifying the installation, testing configuration changes, and understanding the output format before touching real data.

### Example 2 — S&P 500 (live yfinance download)

```bash
python example_run.py --example sp500 --tickers 50 --start 2018-01-01
```

Downloads the specified number of S&P 500 tickers via yfinance, applies liquidity and history filters, then runs the full pipeline. Results and a config JSON are saved to `./runs/<run_id>/`. Requires an internet connection.

Flags:

| Flag | Default | Description |
|---|---|---|
| `--tickers` | `50` | Number of S&P 500 tickers to download |
| `--start` | `2018-01-01` | History start date (YYYY-MM-DD) |

### Example 3 — Your Own Data

```bash
python example_run.py --example own_data --data_dir ./my_data --start 2015-01-01
```

Points the pipeline at a directory of your own CSV or Parquet files. See [Using Your Own Data](#using-your-own-data) for the full set of input options.

### Example 4 — Advanced (model comparison + grid search)

```bash
python example_run.py --example advanced
```

Runs three additional steps on top of the base pipeline:

1. **Model comparison** — trains LambdaMART, LightGBM classifier, Random Forest, and Ridge on the same feature panel and produces a side-by-side Sharpe/return/drawdown table.
2. **Pre-computed scored panel** — reuses the walk-forward predictions to avoid re-training for each grid point.
3. **Parameter grid search** — sweeps `n_long`, `n_short`, `rebalance_freq`, and `min_hold_days` combinations and ranks them by out-of-sample Sharpe.

---

## Using Your Own Data

SmartSignal accepts data in four ways. Once loaded, the rest of the pipeline is identical regardless of source.

### Option A — Directory of per-ticker CSV or Parquet files

Place one file per ticker in a folder. File names become ticker symbols. Each file must contain date, open, high, low, close, and volume columns (column names are auto-detected through a synonym dictionary — `adj_close`, `last_price`, `px`, `Adj Close`, etc. all resolve correctly).

```
my_data/
    AAPL.csv
    MSFT.csv
    TSLA.csv
```

```python
from smartsignal.workflow.pipeline import SmartSignalPipeline

pipe   = SmartSignalPipeline(n_long=10, n_short=10, train_years=3)
result = pipe.run(data_dir="./my_data")
result.print_summary()
result.plot(save_dir="./charts")
```

### Option B — Single stacked CSV (all tickers in one file)

```
date,       ticker, open,  high,  low,   close, volume
2018-01-02, AAPL,   170.1, 172.3, 169.5, 172.0, 25000000
2018-01-02, MSFT,    86.2,  86.9,  85.8,  86.8, 21000000
...
```

```python
from smartsignal.data.loader import load_equity_data

dfs    = load_equity_data("./all_stocks.csv", ticker_col="ticker")
result = pipe.run(dfs=dfs)
```

The `ticker_col` argument is auto-detected if omitted — the loader uses a cardinality heuristic to identify which string column encodes instrument identifiers.

### Option C — Pre-loaded DataFrames in memory

```python
dfs = {
    "AAPL": df_aapl,   # each must have open/high/low/close/volume
    "MSFT": df_msft,
    "TSLA": df_tsla,
}
result = pipe.run(dfs=dfs)
```

### Option D — yfinance ticker list

```python
result = pipe.run(
    tickers=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
    start="2018-01-01",
)
```

### Data requirements

| Requirement | Default threshold | Notes |
|---|---|---|
| Minimum history | 504 trading days (~2 years) | Configurable via `min_history_days` |
| Minimum avg volume | 1 000 000 shares/day | Configurable via `min_avg_volume` |
| Minimum avg price | $5.00 | Penny-stock filter; configurable via `min_avg_price` |
| Required columns | open, high, low, close, volume | Many naming variants auto-detected |
| Maximum missing closes | 5% | Configurable via `max_missing_frac` in `filter_universe()` |

---

## Pipeline Configurations

All pipeline parameters are set in `SmartSignalPipeline.__init__`. Every parameter has a sensible default; you only need to override what matters for your use case.

```python
from smartsignal.workflow.pipeline import SmartSignalPipeline

pipe = SmartSignalPipeline(
    # ── Data ─────────────────────────────────────────────────
    start_date        = "2015-01-01",  # earliest date to include
    end_date          = None,          # None = today
    min_history_days  = 504,           # minimum trading days per ticker
    min_avg_volume    = 1_000_000,     # minimum median daily volume
    min_avg_price     = 5.0,           # minimum median close price

    # ── Features ─────────────────────────────────────────────
    execution_lag     = 1,             # bars to shift features forward
                                       # (prevents look-ahead)
    forward_days      = 5,             # forward-return label horizon (bars)

    # ── Model ─────────────────────────────────────────────────
    top_k_features    = 25,            # features retained after importance
                                       # selection on fold 0
    train_years       = 3,             # initial training window (years)
    test_months       = 3,             # out-of-sample test window per fold
    mode              = "expanding",   # "expanding" or "rolling"
    embargo_days      = 7,             # purge gap between train and test
                                       # (must be >= forward_days)
    ranker_params     = None,          # dict of LGBMRanker overrides

    # ── Backtesting ───────────────────────────────────────────
    n_long            = 10,            # stocks in the long leg
    n_short           = 10,            # stocks in the short leg
    rebalance_freq    = "W",           # "D" daily | "W" weekly | "ME" monthly
    regime_filter     = True,          # suppress signals when ADX < threshold
    adx_threshold     = 20.0,          # ADX cutoff for regime filter
    min_hold_days     = 3,             # minimum holding period per position
    transaction_cost  = 0.001,         # one-way cost (0.001 = 10 bps)
    slippage          = 0.0,           # additional one-way slippage

    verbose           = True,
)
```

### Key trade-offs

| Parameter | Increase | Decrease |
|---|---|---|
| `top_k_features` | More signal coverage, slower | Faster, risk of underfitting |
| `train_years` | More stable models, fewer folds | More folds, adapts faster |
| `embargo_days` | Safer leakage prevention | More data used for training |
| `adx_threshold` | Trade less often (conservative) | Trade more often |
| `min_hold_days` | Lower turnover, higher costs saved | More responsive signals |
| `transaction_cost` | More realistic, lower net return | — |

---

## Expected Results

Results on the S&P 500 universe (50 tickers, 2018–2026, weekly rebalancing, 10 long / 10 short, 10 bps one-way cost):

| Metric | LambdaMART L/S | EW Buy & Hold | CS Momentum |
|---|---|---|---|
| Annualised return | +15.9% | +16.4% | +3.0% |
| Annualised volatility | 17.9% | 20.2% | 35.1% |
| **Sharpe ratio** | **0.92** | 0.86 | 0.26 |
| **Max drawdown** | **−22.4%** | −36.3% | −60.5% |
| Calmar ratio | 0.71 | 0.45 | 0.05 |
| Win rate | 53.9% | 55.2% | 45.0% |
| Longest drawdown (days) | 355 | 393 | 851 |

The L/S strategy is dollar-neutral so its gross return is expected to be similar to (not necessarily higher than) the long-only benchmark. The meaningful advantage is in risk-adjusted terms: meaningfully lower max drawdown (−22% vs −36%), better Sharpe, and a substantially shorter recovery time compared to the buy-and-hold baseline.

> **Note:** Results are sensitive to the universe, date range, and filter settings. Synthetic-data runs will show artificially high Sharpe ratios because returns are i.i.d. by construction.

---

## Visualisation Report

After any pipeline run, call `result.plot()` to generate a full four-figure report. Figures are displayed interactively and optionally saved to disk.

```python
result.plot(
    save_dir  = "./charts",   # directory to save PNGs; None = don't save
    show      = True,         # display interactively
    fmt       = "png",        # "png", "pdf", or "svg"
    dpi       = 150,
)
```

### Figure 1 — Performance Dashboard

- Equity curves for strategy vs baselines
- Long / short leg equity curves
- Underwater (drawdown) chart
- Position count over time (long and short exposure)
- Top-20 feature importance bar chart, coloured by category

### Figure 2 — Signal Quality

- Mean forward return per score quintile
- Directional hit rate per score quintile
- Daily IC and 21-day rolling IC time series
- Walk-forward fold-by-fold validation Sharpe bar chart
- 63-day rolling annualised Sharpe

### Figure 3 — Monthly Returns Heatmap

- Year × month return heatmap for each strategy with annual totals
- Colour-coded green/red cells using a diverging scale centred at zero

### Figure 4 — Risk Analytics

- Return distribution histogram with normal fit overlay and VaR 95% marker
- Q-Q plot vs normal distribution (requires `scipy`; falls back to rolling vol)
- 21-day and 63-day rolling annualised volatility
- Daily portfolio one-way turnover with 21-day rolling mean

---

## Repository Structure

```
smartsignal/
│
├── example_run.py               # Four runnable end-to-end examples
├── CSM_LambdaMART.ipynb         # Preliminary standalone notebook
├── requirements.txt
├── LICENSE
│
└── smartsignal/                 # Main Python package
    │
    ├── __init__.py              # Exposes SmartSignalPipeline
    │
    ├── data/
    │   ├── loader.py            # OHLCV ingestion (CSV/Parquet/yfinance/stacked)
    │   ├── universe.py          # S&P 500 fetch, liquidity filter, calendar align
    │   ├── schema.py            # Column constants, synonym tables, dtype enforcement
    │   └── validator.py         # Structural and leakage-hint validation
    │
    ├── features/
    │   ├── equity_features.py   # 42-feature cross-sectional panel (5 categories)
    │   ├── transforms.py        # Cross-sectional z-score, rank, winsorise
    │   ├── presets.py           # Named feature preset configs (base / heavy)
    │   └── timing.py            # Lookback tracking and warmup utilities
    │
    ├── labels/
    │   ├── generator.py         # Unified generate_labels() entry point
    │   ├── builders.py          # QuintileBuilder, BinaryBuilder, RegressionBuilder
    │   ├── timing.py            # Forward-return computation and horizon trimming
    │   └── workflows.py         # build_ranking_labels(), build_classification_labels()
    │
    ├── models/
    │   ├── base.py              # Abstract BaseModel interface (fit / predict)
    │   ├── types.py             # Shared enums and dataclasses (TrainingResult etc.)
    │   ├── config.py            # Typed hyper-parameter configs per model family
    │   ├── registry.py          # Model factory registry (get_model("lambdamart"))
    │   ├── lambdamart.py        # LambdaMARTRanker with importance-based selection
    │   ├── sklearn_models.py    # LGBMClassifier, RandomForest, Ridge wrappers
    │   ├── advanced_models.py   # EnsembleRanker, StackedRanker
    │   ├── ranking_adapters.py  # ScoreToRankAdapter, PointwiseRankingAdapter
    │   ├── dataset.py           # TabularDataset, SequenceDataset
    │   ├── panel_dataset.py     # PanelDataset (X, y, groups for LGBMRanker)
    │   ├── splits.py            # PurgedTimeSeriesSplit, DateSplitter
    │   ├── splits_panel.py      # PanelWalkForwardSplitter (auto-detects breadth)
    │   ├── trainer.py           # ModelTrainer (single-asset walk-forward)
    │   └── panel_trainer.py     # PanelTrainer (cross-sectional walk-forward)
    │
    ├── validation/
    │   └── walk_forward.py      # WalkForwardSplitter + run_walk_forward()
    │
    ├── backtesting/
    │   ├── engine.py            # run_backtest() orchestrator
    │   ├── portfolio.py         # PortfolioConstructor (positions matrix)
    │   ├── execution.py         # CostModel, ExecutionModel
    │   ├── baselines.py         # Equal-weight buy-and-hold, CS momentum
    │   ├── cross_section.py     # IC, quintile returns, spread decomposition
    │   ├── performance.py       # PerformanceAnalyser, information_coefficient()
    │   ├── visualisation.py     # Standalone backtesting chart functions
    │   └── numba_utils.py       # Rolling Sharpe, drawdown, turnover (Numba-optional)
    │
    ├── utils/
    │   ├── metrics.py           # compute_metrics(), compare_strategies()
    │   ├── plotting.py          # plot_performance() legacy 5-panel chart
    │   ├── report.py            # generate_report() — full 4-figure report
    │   └── compat.py            # Pandas version compatibility (ME/YE aliases)
    │
    └── workflow/
        ├── pipeline.py          # SmartSignalPipeline, PipelineResult
        ├── helpers.py           # PipelineConfig, save/load results, logging setup
        └── combined.py          # ModelComparisonRun, StrategyGridRun
```

---

## Module Overview

### `smartsignal.data`

Handles ingestion from all supported sources. The `EquityLoader` resolves column name variants automatically through a synonym dictionary (so `adj_close`, `Adj Close`, `px`, and `last_price` all map to `close`). `filter_universe()` applies minimum history, volume, and price screens. `align_universe()` forward-fills gaps to a common calendar. `validate_universe()` checks structural integrity and emits leakage-hint warnings for any column whose name matches forward-looking patterns.

### `smartsignal.features`

Computes a 42-feature cross-sectional panel following the five-category framework of Wang & Dong (2025). All computations are strictly backward-looking. The `execution_lag` shift (default 1 bar) forwards all features to simulate realistic execution. The feature registry tracks the minimum lookback required per feature so that warm-up rows are excluded from training automatically.

| Category | Features |
|---|---|
| Overlap / Trend | SMA (5/10/20/50), EMA (12/26), DEMA, WMA, Bollinger Bands width/%, price-to-SMA ratios |
| Momentum | RSI, Stochastic RSI, MACD/signal/histogram, ADX, Williams %R, Aroon oscillator, ROC, momentum, CMO |
| Volatility | ATR, historical volatility (10/20-day), true-range normalised |
| Volume | Volume ratio, Force Index, OBV, A/D line, PVT |
| Price Transform | Body ratio, upper/lower shadow, overnight gap, returns (1/3/5/10-day) |

### `smartsignal.models`

All model families implement the `BaseModel` interface (`fit(panel)` / `predict(panel) → pd.Series`), making them interchangeable in the walk-forward loop. `LambdaMARTRanker` is the primary model: it trains `LGBMRanker` with a `lambdarank` objective, treating each trading day as a query and all stocks as items to be ranked by expected relative return. Feature selection runs once on fold 0 using LightGBM gain importance, retaining the top `top_k_features` for all subsequent folds.

`PanelTrainer` drives the walk-forward loop: it generates chronologically-ordered `(train_panel, test_panel)` pairs via `PanelWalkForwardSplitter`, fits the model on each training slice, and merges test-window predictions back into the panel using integer-position tracking (avoiding the duplicate-DatetimeIndex pitfall of label-based `.loc` assignment).

### `smartsignal.backtesting`

`PortfolioConstructor` converts rank scores into a position matrix, applying an ADX regime filter (suppresses all signals when cross-sectional median ADX falls below `adx_threshold`) and a minimum-hold lock (prevents position reversal for `min_hold_days` bars). `ExecutionModel` computes transaction costs from position changes. `PerformanceAnalyser` provides rolling Sharpe, monthly return tables, and the information coefficient between rank scores and realised returns.

---

## References

1. Wang, J., & Dong, Y. (2025). Combining Interpretable Embedded Multicriteria Feature Cross-Selection Engineering and Machine Learning to Mimic the Brain for Stock Trading Signal Prediction. *Cognitive Computation*, 17(1), Article 7.
2. Daoud, M. B., Hamdi, M., Younes, R., & Oueldoubey, D. (2025). Optimized feature selection based on machine learning models for robust stock market prediction. *International Journal of Innovative Research and Scientific Studies*, 8(3), 5086–5099.
3. Swetha, B., & Arya, K. (2025). Prediction of the Stock Market Using a Hybrid Model Based on Feature Expansion and LSTM-Based Algorithms. *IEEE Access*, 13, 196050–196080.
4. Roostaee, M. R., & Abin, A. A. (2023). Forecasting financial signal for automated trading: An interpretable approach. *Expert Systems with Applications*, 211, Article 118570.
5. Liu, X.-Y. et al. (2024). Dynamic datasets and market environments for financial reinforcement learning. *Machine Learning*, 113(5), 2795–2839.
6. Xu, M., Lan, Z., Tao, Z., Du, J., & Ye, Z. (2023). Deep Reinforcement Learning for Quantitative Trading. arXiv:2312.15730.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
