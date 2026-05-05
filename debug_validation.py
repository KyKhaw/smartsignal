import sys; sys.path.insert(0, '.')
import numpy as np
import pandas as pd

# --- Synthetic data ---
np.random.seed(42)
dates = pd.bdate_range('2019-01-01', periods=3000)
dfs = {}
for i in range(15):
    t = f'SYN{i:03d}'
    ret = np.random.normal(0.0003, 0.015, 3000)
    c   = 100 * np.exp(np.cumsum(ret))
    h   = c * 1.008; l = c * 0.992; o = c * 1.002
    v   = np.random.randint(500_000, 3_000_000, 3000).astype(float)
    dfs[t] = pd.DataFrame({'open':o,'high':h,'low':l,'close':c,'volume':v}, index=dates)

# --- Stage-by-stage smoke test ---
from smartsignal.data.validator import validate_universe
reps = validate_universe(dfs, verbose=False)
print(f'[1] Validator: {sum(r.passed for r in reps.values())}/{len(reps)} passed')

from smartsignal.features.equity_features import build_feature_panel, FEATURE_COLS
panel = build_feature_panel(dfs, execution_lag=1, forward_days=5, verbose=False)
print(f'[2] Features: {len(panel)} rows x {len(FEATURE_COLS)} features')

from smartsignal.labels.workflows import build_ranking_labels
panel = build_ranking_labels(panel, forward_days=5, verbose=False)
print(f'[3] Labels: {panel["relevance"].value_counts().sort_index().to_dict()}')

panel = pd.concat(dfs, names=["ticker"])
panel.index = panel.index.set_names(["ticker", "date"])
panel = panel.swaplevel().sort_index()
dates = panel.index.get_level_values(0)
print("Unique dates:", len(dates.unique()))
print("Total rows:", len(panel))
print("Tickers per date (median):", panel.groupby(level=0).size().median())

from smartsignal.models.lambdamart import LambdaMARTRanker
from smartsignal.models.panel_trainer import PanelTrainer
model   = LambdaMARTRanker(feature_cols=FEATURE_COLS, top_k_features=15)
trainer = PanelTrainer(model=model, train_years=1, test_months=1, embargo_days=5,
                       forward_days=5, verbose=False)
panel_scored, results = trainer.fit_predict(panel)
n_scored = panel_scored['rank_score'].notna().sum()
print(f'[4] Walk-forward: {len(results)} folds, {n_scored} scored rows')

from smartsignal.backtesting.engine import run_backtest
bt = run_backtest(panel_scored, dfs, n_long=3, n_short=3, regime_filter=False,
                  transaction_cost=0.001, verbose=False)
print(f'[5] Backtest: Sharpe={bt.metrics["sharpe"]:.3f}, '
      f'Ann.Ret={bt.metrics["ann_return"]:+.2%}, '
      f'MaxDD={bt.metrics["max_drawdown"]:.2%}')

from smartsignal.backtesting.cross_section import quintile_returns, cross_sectional_ic
qr = quintile_returns(panel_scored)
ic = cross_sectional_ic(panel_scored)
print(f'[6] Cross-section: quintile_returns shape={qr.shape}, mean IC={ic.mean():.4f}')

from smartsignal.backtesting.performance import PerformanceAnalyser, information_coefficient
pa = PerformanceAnalyser(bt.strategy_returns)
monthly = pa.monthly_returns()
ic_stats = information_coefficient(panel_scored['rank_score'], panel_scored['fwd_ret'])
print(f'[7] Performance: monthly table={monthly.shape}, ICIR={ic_stats["icir"]:.3f}')

from smartsignal.utils.metrics import compare_strategies
from smartsignal.backtesting.baselines import run_all_baselines
bl = run_all_baselines(dfs, verbose=False)
tbl = compare_strategies({'Strategy': bt.strategy_returns, **bl})
print(f'[8] Comparison table: {tbl.shape}')

print()
print('ALL INTEGRATION TESTS PASSED')
