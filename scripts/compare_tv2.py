"""Compare Python backtest vs TradingView for the same period."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from scripts.crash_analysis.data_loader import load_all_data
from scripts.strategy_lab.run import prepare_dataframe
from scripts.strategy_lab.strategy import StrategyConfig, generate_signals
from scripts.strategy_lab.backtest import run_backtest
from scripts.strategy_lab.indicators import compute_indicator

# Load best strategy config
with open(ROOT / 'output/strategy_lab/strategies_20260207_124644.json') as f:
    data = json.load(f)
cfg = data[0]['config']
config = StrategyConfig(**cfg)

print("=== STRATEGY CONFIG ===")
print(f"Combine: {config.combine_mode}, threshold: {config.combine_threshold:.4f}")
print(f"Leverage: {config.leverage:.2f}x, SL: {config.stop_loss_pct}%, TP: {config.take_profit_pct}%")
print(f"Risk/trade: {config.risk_per_trade_pct}%, Max hold: {config.max_hold_bars}")
print(f"Exit on flip: {config.exit_on_signal_flip}")
print(f"Vol filter: {config.vol_filter_enabled}, period={config.vol_filter_period}, min_z={config.vol_filter_min_zscore:.3f}")
for ind in config.entry_indicators:
    print(f"  {ind['name']:20s} w={ind['weight']:.3f}  long={ind['long_threshold']:+.4f}  short={ind['short_threshold']:+.4f}")

# Load data the same way as optimizer
raw_data = load_all_data(years=2)
df = prepare_dataframe(raw_data)
df['ts'] = pd.to_datetime(df['timestamp'], unit='ms')
print(f"\nFull data: {len(df)} bars, {df['ts'].iloc[0]} to {df['ts'].iloc[-1]}")

# Generate signals on FULL data (correct expanding normalization)
signals_full = generate_signals(df, config)

# ── FULL HISTORY BACKTEST ──
print("\n" + "="*60)
print("FULL HISTORY BACKTEST (same as optimizer)")
print("="*60)
result_full = run_backtest(df, signals_full, config, initial_balance=10000.0)
print(f"Total Return: {result_full.total_return_pct:.2f}%")
print(f"Total Trades: {result_full.total_trades}")
print(f"Win Rate: {result_full.win_rate:.1f}%")
print(f"Profit Factor: {result_full.profit_factor:.3f}")
print(f"Max DD: {result_full.max_drawdown_pct:.2f}%")
print(f"Avg Monthly: {result_full.avg_monthly_return:.2f}%")
print(f"Monthly returns: {[f'{r:.1f}' for r in result_full.monthly_returns]}")

# ── PERIOD BACKTEST (Dec 2025 - Feb 2026, same as TradingView) ──
mask = (df['ts'] >= '2025-12-01') & (df['ts'] <= '2026-02-07 23:59:59')
df_period = df[mask].copy().reset_index(drop=True)
signals_period = signals_full.loc[df[mask].index].reset_index(drop=True)

print(f"\n{'='*60}")
print(f"PERIOD: Dec 1, 2025 - Feb 7, 2026 ({len(df_period)} bars)")
print(f"{'='*60}")

sig_counts = signals_period['signal'].value_counts().sort_index()
for val, cnt in sig_counts.items():
    label = {1: 'LONG', -1: 'SHORT', 0: 'FLAT'}[val]
    print(f"  {label}: {cnt} bars ({cnt/len(signals_period)*100:.1f}%)")

result = run_backtest(df_period, signals_period, config, initial_balance=10000.0)

print(f"\n{'Metric':<25} {'Python':>12} {'TradingView':>12}")
print(f"{'-'*50}")
print(f"{'Total Return':<25} {result.total_return_pct:>11.2f}% {'9.05%':>12}")
print(f"{'Total Trades':<25} {result.total_trades:>12} {'44':>12}")
print(f"{'Win Rate':<25} {result.win_rate:>11.1f}% {'34.09%':>12}")
print(f"{'Profit Factor':<25} {result.profit_factor:>12.3f} {'1.014':>12}")
print(f"{'Max Drawdown':<25} {result.max_drawdown_pct:>11.2f}% {'25.01%':>12}")
print(f"{'Total Fees':<25} {result.total_fees:>11.2f}$")
print(f"{'Long trades':<25} {result.long_trades:>12}")
print(f"{'Short trades':<25} {result.short_trades:>12}")
print(f"{'Long WR':<25} {result.long_win_rate:>11.1f}%")
print(f"{'Short WR':<25} {result.short_win_rate:>11.1f}%")

# Exit reasons
print(f"\nExit Reasons:")
reasons = {}
for t in result.trades:
    reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
    print(f"  {r}: {c}")

# First 30 trades
print(f"\nFirst 30 trades:")
header = f"{'#':>3} {'Dir':>5} {'Entry$':>10} {'Exit$':>10} {'PnL%':>8} {'Bars':>5} {'Reason':>10} {'Entry Time'}"
print(header)
for i, t in enumerate(result.trades[:30]):
    d = 'LONG' if t.direction == 1 else 'SHORT'
    entry_ts = str(df_period['ts'].iloc[t.entry_bar])[:16] if t.entry_bar < len(df_period) else 'N/A'
    print(f"{i+1:3} {d:>5} {t.entry_price:10.1f} {t.exit_price:10.1f} {t.pnl_pct:8.2f} {t.bars_held:5} {t.exit_reason:>10} {entry_ts}")

# ── NORMALIZATION COMPARISON ──
print(f"\n{'='*60}")
print(f"NORMALIZATION: full history vs period-only (TV sees ~6k bars)")
print(f"{'='*60}")
for ind in config.entry_indicators:
    name = ind['name']
    params = ind['params']
    full_vals = compute_indicator(name, df, params)
    period_from_full = full_vals.loc[df[mask].index].values
    period_only = compute_indicator(name, df_period, params).values
    
    # Last 500 values comparison
    n = min(500, len(period_from_full))
    diff = np.abs(period_from_full[-n:] - period_only[-n:])
    valid = ~np.isnan(diff)
    if valid.any():
        print(f"  {name:20s}: mean_diff={np.mean(diff[valid]):.4f}  max_diff={np.max(diff[valid]):.4f}")
    else:
        print(f"  {name:20s}: all NaN")

# ── SIGNAL COMPARISON: show indicator values at recent entry points ──
print(f"\n{'='*60}")
print(f"INDICATOR VALUES at first 5 trade entries")
print(f"{'='*60}")
for i, t in enumerate(result.trades[:5]):
    bar = t.entry_bar - 1  # signal bar (one before entry)
    if bar < 0 or bar >= len(df_period):
        continue
    d = 'LONG' if t.direction == 1 else 'SHORT'
    ts = str(df_period['ts'].iloc[bar])[:16]
    print(f"\nTrade {i+1}: {d} at {ts}")
    for ind in config.entry_indicators:
        name = ind['name']
        params = ind['params']
        raw = compute_indicator(name, df, params)
        val = raw.loc[df[mask].index].values[bar]
        lt, st = ind['long_threshold'], ind['short_threshold']
        # Vote
        if val < st:
            vote = -1
        elif val > lt:
            vote = 1
        else:
            vote = 0
        print(f"  {name:20s}: val={val:+.4f}  long_thr={lt:+.4f}  short_thr={st:+.4f}  vote={vote:+d}")
