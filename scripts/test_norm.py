"""Quick test: compare old (separate) vs new (concat) signal generation."""
import time, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.crash_analysis.data_loader import load_all_data
from scripts.strategy_lab.run import prepare_dataframe
from scripts.strategy_lab.strategy import StrategyConfig, generate_signals
from scripts.strategy_lab.backtest import run_backtest

raw = load_all_data(years=2)
df = prepare_dataframe(raw)

train_end = 48960
test_end = train_end + 7008
train_df = df.iloc[:train_end].copy().reset_index(drop=True)
test_df = df.iloc[train_end:test_end].copy().reset_index(drop=True)

cfg = StrategyConfig(
    entry_indicators=[
        {"name": "supertrend", "params": {"period": 10, "mult": 3.0}, "weight": 1.0, "long_threshold": -0.5, "short_threshold": 0.5},
        {"name": "rsi", "params": {"period": 14}, "weight": 1.0, "long_threshold": 0.3, "short_threshold": -0.3},
    ],
    combine_mode="vote", combine_threshold=0.3,
    leverage=3.0, stop_loss_pct=3.0, take_profit_pct=6.0,
    trailing_stop_pct=1.0, max_hold_bars=96,
    risk_per_trade_pct=10.0, exit_on_signal_flip=True,
)

# OLD approach
t0 = time.time()
train_sig_old = generate_signals(train_df, cfg)
test_sig_old = generate_signals(test_df, cfg)
t_old = time.time() - t0
print(f"OLD (separate): {t_old:.3f}s, train_sig={len(train_sig_old)}, test_sig={len(test_sig_old)}")

# NEW approach
t0 = time.time()
full_df = pd.concat([train_df, test_df], ignore_index=True)
full_sig = generate_signals(full_df, cfg)
train_sig_new = full_sig.iloc[:len(train_df)].reset_index(drop=True)
test_sig_new = full_sig.iloc[len(train_df):].reset_index(drop=True)
t_new = time.time() - t0
print(f"NEW (concat):   {t_new:.3f}s, full_sig={len(full_sig)}")

# Compare
diff = (test_sig_old["signal"] != test_sig_new["signal"]).sum()
print(f"Signal differences: {diff}/{len(test_sig_old)} ({diff/len(test_sig_old)*100:.1f}%)")

# Backtests
r_old = run_backtest(test_df, test_sig_old, cfg)
r_new = run_backtest(test_df, test_sig_new, cfg)
print(f"\nOLD: return={r_old.total_return_pct:.2f}%, trades={r_old.total_trades}, WR={r_old.win_rate:.1f}%")
print(f"NEW: return={r_new.total_return_pct:.2f}%, trades={r_new.total_trades}, WR={r_new.win_rate:.1f}%")
print(f"\nSlowdown: {t_new/t_old:.1f}x")
