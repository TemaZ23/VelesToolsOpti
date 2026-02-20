"""Quick debug script to test indicator normalization and signal generation."""
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from scripts.crash_analysis.data_loader import load_all_data
from scripts.strategy_lab.run import prepare_dataframe
from scripts.strategy_lab.indicators import compute_indicator
from scripts.strategy_lab.strategy import StrategyConfig, generate_signals
from scripts.strategy_lab.backtest import run_backtest

data = load_all_data(symbol="BTCUSDT", years=2, use_cache=True)
df = prepare_dataframe(data)

# Check normalized indicator outputs
print("\n=== INDICATOR RANGES (after normalization) ===")
for name in ["rsi", "adx", "bb_position", "price_zscore", "obv_slope", "ema_cross", "rsi_signal"]:
    params = {"period": 14, "fast": 9, "slow": 21, "signal": 9, "std_dev": 2.0}
    vals = compute_indicator(name, df, params)
    print(f"  {name:20s}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}")

# Test a simple strategy
print("\n=== TEST STRATEGY ===")
config = StrategyConfig(
    entry_indicators=[
        {"name": "ema_cross", "params": {"fast": 9, "slow": 21}, "weight": 1.5,
         "long_threshold": 0.0, "short_threshold": 0.0},
        {"name": "rsi", "params": {"period": 14}, "weight": 1.0,
         "long_threshold": 0.2, "short_threshold": -0.2},
    ],
    combine_mode="weighted",
    combine_threshold=0.3,
    leverage=10.0,
    stop_loss_pct=2.0,
    take_profit_pct=4.0,
    risk_per_trade_pct=20.0,
    max_hold_bars=96,
    exit_on_signal_flip=True,
)

signals = generate_signals(df, config)
n_long = (signals["signal"] == 1).sum()
n_short = (signals["signal"] == -1).sum()
n_flat = (signals["signal"] == 0).sum()
print(f"  Signals: {n_long} long, {n_short} short, {n_flat} flat")

# Run backtest
print("\n=== BACKTEST ===")
result = run_backtest(df, signals, config)
print(f"  Total Return: {result.total_return_pct:+.2f}%")
print(f"  Avg Monthly:  {result.avg_monthly_return:+.2f}%")
print(f"  Max Drawdown:  {result.max_drawdown_pct:.2f}%")
print(f"  Total Trades:  {result.total_trades}")
print(f"  Win Rate:      {result.win_rate:.1f}%")
print(f"  Sharpe:        {result.sharpe_ratio:.2f}")
print(f"  Profit Factor: {result.profit_factor:.2f}")
