"""
Извлечь фиксированные min/max сырых индикаторов из полного датасета.
Используются для замены expanding normalization в Pine Script.
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.crash_analysis.data_loader import load_all_data
from scripts.strategy_lab.run import prepare_dataframe

# Strategy #2 config (best full-validation)
CONFIG = {
    "adx": {"period": 29},
    "roc": {"period": 47},
    "cci": {"period": 13},
    "momentum": {"period": 49},
}

print("Loading data...")
data = load_all_data(symbol="BTCUSDT", years=5, use_cache=True)
df = prepare_dataframe(data)
print(f"DataFrame: {len(df)} bars")

# ── ADX (raw 0-100) ──
h, l, c = df["high"], df["low"], df["close"]
up_move = h.diff()
dn_move = -l.diff()
plus_dm = up_move.where((up_move > dn_move) & (up_move > 0), 0)
minus_dm = dn_move.where((dn_move > up_move) & (dn_move > 0), 0)
tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
atr = tr.ewm(span=29, adjust=False).mean()
plus_di = 100 * plus_dm.ewm(span=29, adjust=False).mean() / atr.replace(0, np.nan)
minus_di = 100 * minus_dm.ewm(span=29, adjust=False).mean() / atr.replace(0, np.nan)
dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
adx_raw = dx.ewm(span=29, adjust=False).mean().dropna()

# ── ROC (raw) ──
roc_raw = (c.pct_change(47) * 100).dropna()

# ── CCI (raw) ──
tp = (h + l + c) / 3
sma_tp = tp.rolling(13, min_periods=1).mean()
mad = tp.rolling(13, min_periods=1).std() * 0.7979
cci_raw = ((tp - sma_tp) / (0.015 * mad).replace(0, np.nan)).dropna()

# ── Momentum (raw) ──
mom_raw = c.pct_change(49).dropna()

print("\n" + "="*60)
print("FIXED MIN/MAX RANGES FOR PINE SCRIPT")
print("="*60)

for name, series in [("ADX", adx_raw), ("ROC", roc_raw), ("CCI", cci_raw), ("Momentum", mom_raw)]:
    mn = series.min()
    mx = series.max()
    p1 = series.quantile(0.001)
    p99 = series.quantile(0.999)
    print(f"\n{name}:")
    print(f"  Absolute:  min={mn:.6f}  max={mx:.6f}")
    print(f"  0.1%-99.9%: min={p1:.6f}  max={p99:.6f}")
    print(f"  mean={series.mean():.6f}  std={series.std():.6f}")

# Also verify what the expanding normalization gives at key points
from scripts.strategy_lab.indicators import _normalize_to_unit
print("\n" + "="*60)
print("EXPANDING NORM COMPARISON AT KEY POINTS")
print("="*60)

for name, series in [("ADX", adx_raw), ("ROC", roc_raw), ("CCI", cci_raw), ("Momentum", mom_raw)]:
    normed = _normalize_to_unit(series)
    # Show value at bar 5000 (typical TV start) vs bar 50000
    idx_5k = min(5000, len(series)-1)
    idx_50k = min(50000, len(series)-1)
    idx_end = len(series) - 1
    
    raw_5k = series.iloc[idx_5k]
    raw_end = series.iloc[idx_end]
    norm_5k = normed.iloc[idx_5k]
    norm_end = normed.iloc[idx_end]
    
    # Expanding min/max at bar 5000 vs end
    mn_5k = series.iloc[:idx_5k+1].min()
    mx_5k = series.iloc[:idx_5k+1].max()
    mn_all = series.min()
    mx_all = series.max()
    
    print(f"\n{name}:")
    print(f"  At bar 5000: exp_min={mn_5k:.4f}, exp_max={mx_5k:.4f}")
    print(f"  At bar 70k:  exp_min={mn_all:.4f}, exp_max={mx_all:.4f}")
    print(f"  Range ratio: {(mx_5k-mn_5k)/(mx_all-mn_all)*100:.1f}% of full")
