"""Compare Python backtest vs TradingView for the same period."""
import json
import pandas as pd
import numpy as np
from scripts.strategy_lab.strategy import StrategyConfig, generate_signals
from scripts.strategy_lab.backtest import run_backtest

# Load best strategy config
with open('output/strategy_lab/strategies_20260207_124644.json') as f:
    data = json.load(f)
cfg_dict = data[0]['config']  # config is nested under 'config' key
config = StrategyConfig(**{k: v for k, v in cfg_dict.items()})

print("=== STRATEGY CONFIG ===")
print(f"Combine: {config.combine_mode}, threshold: {config.combine_threshold}")
print(f"Leverage: {config.leverage}x, SL: {config.stop_loss_pct}%, TP: {config.take_profit_pct}%")
print(f"Risk/trade: {config.risk_per_trade_pct}%, Max hold: {config.max_hold_bars}")
print(f"Exit on flip: {config.exit_on_signal_flip}")
print(f"Vol filter: {config.vol_filter_enabled}, period={config.vol_filter_period}, min_z={config.vol_filter_min_zscore}")
print(f"Indicators: {len(config.entry_indicators)}")
for ind in config.entry_indicators:
    print(f"  {ind['name']}({ind['params']}) w={ind['weight']:.2f} long={ind['long_threshold']:.3f} short={ind['short_threshold']:.3f}")

# Load data
df = pd.read_parquet('cache/BTCUSDT_futures_15m.parquet')
df = df.sort_values('timestamp').reset_index(drop=True)
df['ts'] = pd.to_datetime(df['timestamp'], unit='ms')

# Generate signals on FULL data (for correct expanding normalization)
signals_full = generate_signals(df, config)

# Filter to Dec 1, 2025 - Feb 7, 2026 (same as TradingView)
mask = (df['ts'] >= '2025-12-01') & (df['ts'] <= '2026-02-07 23:59:59')
df_period = df[mask].copy().reset_index(drop=True)
signals_period = signals_full.loc[df[mask].index].reset_index(drop=True)

print(f'\nPeriod: {df_period["ts"].iloc[0]} to {df_period["ts"].iloc[-1]}')
print(f'Bars: {len(df_period)}')

print(f'\nSignals distribution:')
sig_counts = signals_period['signal'].value_counts().sort_index()
for val, cnt in sig_counts.items():
    label = {1: 'LONG', -1: 'SHORT', 0: 'FLAT'}[val]
    print(f'  {label}: {cnt} bars ({cnt/len(signals_period)*100:.1f}%)')

# Run backtest
result = run_backtest(df_period, signals_period, config, initial_balance=10000.0)

print(f'\n{"="*50}')
print(f'PYTHON BACKTEST  vs  TRADINGVIEW')
print(f'{"="*50}')
print(f'{"Metric":<25} {"Python":>12} {"TV":>12}')
print(f'{"-"*50}')
print(f'{"Total Return":<25} {result.total_return_pct:>11.2f}% {"9.05%":>12}')
print(f'{"Total Trades":<25} {result.total_trades:>12} {"44":>12}')
print(f'{"Win Rate":<25} {result.win_rate:>11.1f}% {"34.09%":>12}')
print(f'{"Profit Factor":<25} {result.profit_factor:>12.3f} {"1.014":>12}')
print(f'{"Max Drawdown":<25} {result.max_drawdown_pct:>11.2f}% {"25.01%":>12}')
print(f'{"Total Fees":<25} {result.total_fees:>11.2f}$')
print(f'{"Long trades":<25} {result.long_trades:>12}')
print(f'{"Short trades":<25} {result.short_trades:>12}')
print(f'{"Long WR":<25} {result.long_win_rate:>11.1f}%')
print(f'{"Short WR":<25} {result.short_win_rate:>11.1f}%')

# Exit reasons
print(f'\nExit Reasons:')
reasons = {}
for t in result.trades:
    reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
    print(f'  {r}: {c}')

# First 25 trades with timestamps
print(f'\nFirst 25 trades:')
print(f'{"#":>3} {"Dir":>5} {"Entry$":>10} {"Exit$":>10} {"PnL%":>8} {"Bars":>5} {"Reason":>10} {"Entry Time"}')
for i, t in enumerate(result.trades[:25]):
    d = 'LONG' if t.direction == 1 else 'SHORT'
    entry_ts = df_period['ts'].iloc[t.entry_bar] if t.entry_bar < len(df_period) else 'N/A'
    print(f'{i+1:3} {d:>5} {t.entry_price:10.1f} {t.exit_price:10.1f} {t.pnl_pct:8.2f} {t.bars_held:5} {t.exit_reason:>10} {entry_ts}')

# Also check: what signals do we generate on the PERIOD data without full history?
# This simulates what TradingView sees (only ~6700 bars from Dec)
print(f'\n{"="*50}')
print(f'SIGNAL ANALYSIS — checking normalization impact')
print(f'{"="*50}')

# What percentage of bars are in a position?
in_trade_bars = sum(t.bars_held for t in result.trades)
print(f'Bars in trade: {in_trade_bars}/{len(df_period)} ({in_trade_bars/len(df_period)*100:.1f}%)')

# Check if expanding normalization on TradingView (starting from chart start) 
# would differ from Python (starting from Feb 2024)
from scripts.strategy_lab.indicators import compute_indicator
# Compute indicators for full data vs period-only
for ind in config.entry_indicators:
    name = ind['name']
    params = ind['params']
    
    # Full history normalization
    full_vals = compute_indicator(name, df, params)
    period_vals_from_full = full_vals.loc[df[mask].index]
    
    # Period-only normalization (what TV might do with limited bars)
    period_only_vals = compute_indicator(name, df_period, params)
    
    # Compare last 100 values
    diff = (period_vals_from_full.values[-100:] - period_only_vals.values[-100:])
    print(f'{name:20s}: mean_diff={np.nanmean(np.abs(diff)):.4f}  max_diff={np.nanmax(np.abs(diff)):.4f}')
