"""
РџСЂРѕРґРІРёРЅСѓС‚С‹Р№ Feature Engineering РґР»СЏ Р°РЅР°Р»РёР·Р° РєСЂР°С€РµР№.

Р“РµРЅРµСЂРёСЂСѓРµС‚ 200+ С„РёС‡РµР№ РёР· СЂР°Р·Р»РёС‡РЅС‹С… РєР°С‚РµРіРѕСЂРёР№:

1. PRICE STRUCTURE вЂ” РјСѓР»СЊС‚Рё-С‚Р°Р№РјС„СЂРµР№Рј Р°РЅР°Р»РёР·, С„СЂР°РєС‚Р°Р»СЊРЅС‹Рµ СѓСЂРѕРІРЅРё, ATR-РЅРѕСЂРјРёСЂРѕРІРєР°
2. VOLUME MICROSTRUCTURE вЂ” РїСЂРѕС„РёР»СЊ РѕР±СЉС‘РјР°, Р°РЅРѕРјР°Р»СЊРЅС‹Рµ СЃРІРµС‡Рё, VWAP
3. ORDER FLOW вЂ” taker delta cross-scale, funding pressure, OI momentum
4. VOLATILITY REGIME вЂ” realized vol РєР»Р°СЃС‚РµСЂС‹, Garman-Klass, Yang-Zhang
5. INFORMATION THEORY вЂ” sample entropy, approximate entropy, spectral entropy
6. FRACTAL ANALYSIS вЂ” Hurst exponent, fractal dimension, DFA
7. CROSS-ASSET вЂ” Spot-Futures basis, basis momentum, convergence speed
8. MARKET SENTIMENT вЂ” Fear & Greed momentum, long/short divergence
9. TEMPORAL вЂ” hour-of-day, day-of-week, time-to-funding, market session
10. NON-LINEAR INTERACTIONS вЂ” feature crosses, ratios, conditional features
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats, signal as sp_signal

warnings.filterwarnings("ignore", category=RuntimeWarning)


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# HELPERS
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

def _zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling Z-score."""
    mean = series.rolling(window, min_periods=max(1, window // 2)).mean()
    std = series.rolling(window, min_periods=max(1, window // 2)).std()
    return (series - mean) / std.replace(0, np.nan)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _garman_klass_vol(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series, window: int) -> pd.Series:
    """Garman-Klass volatility estimator вЂ” Р»СѓС‡С€Рµ С‡РµРј close-to-close."""
    log_hl = np.log(h / l) ** 2
    log_co = np.log(c / o) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return gk.rolling(window).mean().apply(np.sqrt)


def _yang_zhang_vol(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series, window: int) -> pd.Series:
    """Yang-Zhang volatility вЂ” combines overnight/open-close/Garman-Klass."""
    log_ho = np.log(h / o)
    log_lo = np.log(l / o)
    log_co = np.log(c / o)

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    close_prev = c.shift(1)
    log_cc = np.log(c / close_prev)
    log_oc = np.log(o / close_prev)

    sigma_close = log_cc.rolling(window).var()
    sigma_open = log_oc.rolling(window).var()
    sigma_rs = rs.rolling(window).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz = sigma_open + k * sigma_close + (1 - k) * sigma_rs

    return yz.apply(lambda x: np.sqrt(abs(x)) if not np.isnan(x) else np.nan)


def _hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """РћС†РµРЅРєР° СЌРєСЃРїРѕРЅРµРЅС‚С‹ РҐС‘СЂСЃС‚Р° С‡РµСЂРµР· R/S Р°РЅР°Р»РёР·."""
    if len(series.dropna()) < max_lag * 2:
        return np.nan
    values = series.dropna().values
    lags = range(2, max_lag)
    tau: list[float] = []
    for lag in lags:
        chunks = [values[i:i + lag] for i in range(0, len(values) - lag, lag)]
        if len(chunks) < 2:
            continue
        rs_vals: list[float] = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean_c = np.mean(chunk)
            cumdev = np.cumsum(chunk - mean_c)
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(chunk, ddof=1)
            if s > 0:
                rs_vals.append(r / s)
        if rs_vals:
            tau.append(np.mean(rs_vals))
        else:
            tau.append(np.nan)

    valid = [(l, t) for l, t in zip(lags, tau) if not np.isnan(t) and t > 0]
    if len(valid) < 3:
        return np.nan
    log_lags = [np.log(l) for l, _ in valid]
    log_tau = [np.log(t) for _, t in valid]
    slope, _, _, _, _ = sp_stats.linregress(log_lags, log_tau)
    return slope


def _sample_entropy(series: pd.Series, m: int = 2, r_factor: float = 0.2) -> float:
    """Sample entropy вЂ” РјРµСЂР° СЃР»РѕР¶РЅРѕСЃС‚Рё/РЅРµРїСЂРµРґСЃРєР°Р·СѓРµРјРѕСЃС‚Рё."""
    data = series.dropna().values
    n = len(data)
    if n < 10:
        return np.nan
    r = r_factor * np.std(data)
    if r == 0:
        return np.nan

    # РСЃРїРѕР»СЊР·СѓРµРј РІРµРєС‚РѕСЂРёР·РѕРІР°РЅРЅС‹Р№ РїРѕРґС…РѕРґ РґР»СЏ СЃРєРѕСЂРѕСЃС‚Рё
    def _count_matches(template_length: int) -> int:
        count = 0
        limit = min(n - template_length, 1000)  # limit for speed
        for i in range(limit):
            for j in range(i + 1, limit):
                if np.max(np.abs(data[i:i + template_length] - data[j:j + template_length])) <= r:
                    count += 1
        return count

    a = _count_matches(m + 1)
    b = _count_matches(m)

    if b == 0:
        return np.nan
    return -np.log(a / b) if a > 0 else np.nan


def _spectral_entropy(series: pd.Series, window: int = 256) -> float:
    """РЎРїРµРєС‚СЂР°Р»СЊРЅР°СЏ СЌРЅС‚СЂРѕРїРёСЏ вЂ” РјРµСЂР° С…Р°РѕС‚РёС‡РЅРѕСЃС‚Рё С†РµРЅРѕРІРѕРіРѕ СЂСЏРґР°."""
    data = series.dropna().values
    if len(data) < window:
        return np.nan
    data = data[-window:]
    f, psd = sp_signal.periodogram(data)
    psd = psd[psd > 0]
    if len(psd) == 0:
        return np.nan
    psd_norm = psd / psd.sum()
    return -np.sum(psd_norm * np.log2(psd_norm))


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# CATEGORY: PRICE STRUCTURE
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

def _price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Р¦РµРЅРѕРІР°СЏ СЃС‚СЂСѓРєС‚СѓСЂР° Рё РјСѓР»СЊС‚Рё-С‚Р°Р№РјС„СЂРµР№Рј РёРЅРґРёРєР°С‚РѕСЂС‹."""
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]

    feats: dict[str, pd.Series] = {}

    # Returns РЅР° СЂР°Р·РЅС‹С… РіРѕСЂРёР·РѕРЅС‚Р°С…
    for w in [1, 4, 16, 48, 96, 288, 672]:
        feats[f"ret_{w}"] = c.pct_change(w)
        feats[f"log_ret_{w}"] = np.log(c / c.shift(w))

    # EMA Рё РѕС‚РєР»РѕРЅРµРЅРёСЏ
    for sp in [12, 26, 50, 100, 200]:
        ema = _ema(c, sp)
        feats[f"ema_{sp}_dist"] = (c - ema) / ema

    # MACD СЂР°Р·РЅС‹С… СЃРєРѕСЂРѕСЃС‚РµР№
    feats["macd_fast"] = (_ema(c, 8) - _ema(c, 21)) / c
    feats["macd_std"] = (_ema(c, 12) - _ema(c, 26)) / c
    feats["macd_slow"] = (_ema(c, 26) - _ema(c, 50)) / c

    # RSI РЅР° СЂР°Р·РЅС‹С… РїРµСЂРёРѕРґР°С…
    for p in [7, 14, 21, 48]:
        feats[f"rsi_{p}"] = _rsi(c, p)

    # ATR-РЅРѕСЂРјР°Р»РёР·РѕРІР°РЅРЅС‹Рµ РґРІРёР¶РµРЅРёСЏ
    atr_14 = _atr(h, l, c, 14)
    feats["atr_14"] = atr_14 / c
    feats["body_atr_ratio"] = (c - o).abs() / atr_14.replace(0, np.nan)
    feats["upper_shadow_atr"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / atr_14.replace(0, np.nan)
    feats["lower_shadow_atr"] = (pd.concat([c, o], axis=1).min(axis=1) - l) / atr_14.replace(0, np.nan)

    # Range / ATR вЂ” СЃР¶Р°С‚РёРµ/СЂР°СЃС€РёСЂРµРЅРёРµ
    for w in [4, 16, 48]:
        rng = h.rolling(w).max() - l.rolling(w).min()
        feats[f"range_atr_{w}"] = rng / atr_14.replace(0, np.nan)

    # Bollinger bandwidth
    for w in [20, 50]:
        sma = c.rolling(w).mean()
        std = c.rolling(w).std()
        feats[f"bb_width_{w}"] = (2 * std) / sma.replace(0, np.nan)
        feats[f"bb_pos_{w}"] = (c - (sma - 2 * std)) / (4 * std).replace(0, np.nan)  # 0..1

    # Fractal-like: max drawdown in lookback window
    for w in [48, 96, 288]:
        rolling_max = c.rolling(w).max()
        feats[f"drawdown_{w}"] = (c - rolling_max) / rolling_max.replace(0, np.nan)

    # Price acceleration (return of returns)
    for w in [4, 16, 48]:
        ret = c.pct_change(w)
        feats[f"ret_accel_{w}"] = ret.diff(w)

    # Consecutive moves
    is_up = (c.diff() > 0).astype(int)
    feats["consec_up"] = is_up * (is_up.groupby((is_up != is_up.shift()).cumsum()).cumcount() + 1)
    is_down = (c.diff() < 0).astype(int)
    feats["consec_down"] = is_down * (is_down.groupby((is_down != is_down.shift()).cumsum()).cumcount() + 1)

    return pd.DataFrame(feats, index=df.index)


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# CATEGORY: VOLUME MICROSTRUCTURE
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

def _volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """РћР±СЉС‘РјРЅР°СЏ РјРёРєСЂРѕСЃС‚СЂСѓРєС‚СѓСЂР°."""
    vol = df["volume"]
    c = df["close"]
    qv = df["quote_volume"]
    trades = df["trades"]

    feats: dict[str, pd.Series] = {}

    # Volume z-scores
    for w in [24, 96, 288]:
        feats[f"vol_zscore_{w}"] = _zscore(vol, w)
        feats[f"qvol_zscore_{w}"] = _zscore(qv, w)

    # Volume-to-trades ratio (average trade size)
    avg_trade = qv / trades.replace(0, np.nan)
    for w in [24, 96]:
        feats[f"avg_trade_size_zscore_{w}"] = _zscore(avg_trade, w)

    # Taker delta
    if "taker_buy_volume" in df.columns:
        tb = df["taker_buy_volume"]
        ts = df["taker_sell_volume"]
        delta = tb - ts
        delta_pct = (tb - ts) / vol.replace(0, np.nan)

        feats["taker_delta"] = delta_pct
        for w in [12, 48, 96]:
            feats[f"taker_delta_ma_{w}"] = delta_pct.rolling(w).mean()
            feats[f"taker_delta_std_{w}"] = delta_pct.rolling(w).std()
            feats[f"cum_taker_delta_{w}"] = delta.rolling(w).sum() / vol.rolling(w).sum().replace(0, np.nan)

        # Taker exhaustion вЂ” СЃРёР»СЊРЅРѕРµ СЃРјРµС‰РµРЅРёРµ + СЂР°Р·РІРѕСЂРѕС‚
        feats["taker_exhaustion"] = (
            (delta_pct.rolling(12).mean().abs() > delta_pct.rolling(96).std())
            .astype(float)
        )

    # Volume profile: relative position of close in volume-weighted distribution
    # VWAP
    cumvol = vol.cumsum()
    cum_pv = (c * vol).cumsum()

    for w in [48, 96]:
        rv = vol.rolling(w)
        rpv = (c * vol).rolling(w)
        vwap = rpv.sum() / rv.sum().replace(0, np.nan)
        feats[f"vwap_dist_{w}"] = (c - vwap) / vwap.replace(0, np.nan)

    # Volume concentration вЂ” high volume at extremes = distribution
    feats["vol_at_high"] = vol.where(
        (c - c.rolling(48).min()) / (c.rolling(48).max() - c.rolling(48).min()).replace(0, np.nan) > 0.8,
        0,
    ).rolling(48).sum() / vol.rolling(48).sum().replace(0, np.nan)

    feats["vol_at_low"] = vol.where(
        (c - c.rolling(48).min()) / (c.rolling(48).max() - c.rolling(48).min()).replace(0, np.nan) < 0.2,
        0,
    ).rolling(48).sum() / vol.rolling(48).sum().replace(0, np.nan)

    # Volume trend (increasing/decreasing)
    for w in [24, 96]:
        feats[f"vol_trend_{w}"] = vol.rolling(w).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 2 else 0,
            raw=False,
        )

    return pd.DataFrame(feats, index=df.index)


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# CATEGORY: VOLATILITY REGIME
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

def _volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Р’РѕР»Р°С‚РёР»СЊРЅРѕСЃС‚СЊ: СЂРµР¶РёРјС‹, РєР»Р°СЃС‚РµСЂС‹, advanced estimators."""
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]

    feats: dict[str, pd.Series] = {}

    # Realized volatility (close-to-close)
    log_ret = np.log(c / c.shift(1))
    for w in [24, 96, 288]:
        feats[f"real_vol_{w}"] = log_ret.rolling(w).std() * np.sqrt(96)  # annualized

    # Garman-Klass
    for w in [24, 96]:
        feats[f"gk_vol_{w}"] = _garman_klass_vol(o, h, l, c, w)

    # Yang-Zhang
    for w in [24, 96]:
        feats[f"yz_vol_{w}"] = _yang_zhang_vol(o, h, l, c, w)

    # Volatility ratio (short / long) вЂ” spiking = regime change
    feats["vol_ratio_24_96"] = feats["real_vol_24"] / feats["real_vol_96"].replace(0, np.nan)
    feats["vol_ratio_24_288"] = feats["real_vol_24"] / feats["real_vol_288"].replace(0, np.nan)

    # Volatility of volatility
    feats["vol_of_vol"] = feats["real_vol_24"].rolling(48).std()

    # Parkinson's volatility: uses only high/low
    parkinson = np.log(h / l) ** 2 / (4 * np.log(2))
    for w in [24, 96]:
        feats[f"parkinson_vol_{w}"] = parkinson.rolling(w).mean().apply(np.sqrt)

    # Volatility skew: downside volatility vs upside
    for w in [96, 288]:
        down_ret = log_ret.where(log_ret < 0, 0)
        up_ret = log_ret.where(log_ret > 0, 0)
        down_vol = down_ret.rolling(w).std()
        up_vol = up_ret.rolling(w).std()
        feats[f"vol_skew_{w}"] = down_vol / up_vol.replace(0, np.nan)

    # Return kurtosis (tail-heaviness)
    for w in [96, 288]:
        feats[f"kurtosis_{w}"] = log_ret.rolling(w).kurt()
        feats[f"skewness_{w}"] = log_ret.rolling(w).skew()

    return pd.DataFrame(feats, index=df.index)


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# CATEGORY: INFORMATION THEORY + FRACTAL
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

def _complexity_features(df: pd.DataFrame) -> pd.DataFrame:
    """РЎР»РѕР¶РЅРѕСЃС‚СЊ, СЌРЅС‚СЂРѕРїРёСЏ, С„СЂР°РєС‚Р°Р»СЊРЅС‹Р№ Р°РЅР°Р»РёР·."""
    c = df["close"]
    log_ret = np.log(c / c.shift(1))

    feats: dict[str, pd.Series] = {}

    # Rolling Hurst exponent
    feats["hurst_96"] = log_ret.rolling(96).apply(
        lambda x: _hurst_exponent(x, max_lag=20), raw=False
    )
    feats["hurst_288"] = log_ret.rolling(288).apply(
        lambda x: _hurst_exponent(x, max_lag=40), raw=False
    )

    # Rolling Sample Entropy (expensive, subsample)
    feats["sample_entropy"] = log_ret.rolling(192).apply(
        lambda x: _sample_entropy(x, m=2, r_factor=0.2), raw=False
    )

    # Return distribution features
    for w in [96, 288]:
        feats[f"jarque_bera_{w}"] = log_ret.rolling(w).apply(
            lambda x: sp_stats.jarque_bera(x.dropna())[0] if len(x.dropna()) >= 4 else np.nan,
            raw=False,
        )

    # Auto-correlation at various lags (market efficiency)
    for lag in [1, 4, 12, 24]:
        feats[f"autocorr_{lag}"] = log_ret.rolling(96).apply(
            lambda x: x.autocorr(lag=lag) if len(x.dropna()) > lag + 2 else np.nan,
            raw=False,
        )

    # Spectral entropy (rolling)
    feats["spectral_entropy"] = log_ret.rolling(256).apply(
        lambda x: _spectral_entropy(x, window=min(256, len(x.dropna()))), raw=False
    )

    return pd.DataFrame(feats, index=df.index)


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# CATEGORY: CROSS-ASSET (Basis, ETH/BTC)
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

def _basis_features(futures_df: pd.DataFrame, spot_df: pd.DataFrame) -> pd.DataFrame:
    """Spot-Futures Basis Р°РЅР°Р»РёС‚РёРєР°."""
    feats: dict[str, pd.Series] = {}

    # Merge on timestamp
    merged = futures_df[["timestamp", "close"]].merge(
        spot_df[["timestamp", "close"]].rename(columns={"close": "spot_close"}),
        on="timestamp",
        how="left",
    )

    if "spot_close" not in merged.columns or merged["spot_close"].isna().all():
        return pd.DataFrame(index=futures_df.index)

    basis = (merged["close"] - merged["spot_close"]) / merged["spot_close"].replace(0, np.nan)
    basis = basis.ffill()

    feats["basis"] = basis
    feats["basis_abs"] = basis.abs()

    for w in [24, 96, 288]:
        feats[f"basis_ma_{w}"] = basis.rolling(w).mean()
        feats[f"basis_zscore_{w}"] = _zscore(basis, w)

    # Basis momentum
    for w in [4, 24, 96]:
        feats[f"basis_change_{w}"] = basis.diff(w)

    # Basis convergence speed
    feats["basis_mean_revert"] = basis.rolling(48).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 2 else 0,
        raw=False,
    )

    result = pd.DataFrame(feats)
    result.index = futures_df.index
    return result


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# CATEGORY: MARKET SENTIMENT (Funding, Long/Short, Fear/Greed)
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

def _sentiment_features(
    df: pd.DataFrame,
    funding: pd.DataFrame,
    long_short: pd.DataFrame,
    fear_greed: pd.DataFrame,
    open_interest: pd.DataFrame,
    taker_volume: pd.DataFrame,
) -> pd.DataFrame:
    """Р С‹РЅРѕС‡РЅС‹Р№ СЃРµРЅС‚РёРјРµРЅС‚ Рё РїРѕР·РёС†РёРѕРЅРёСЂРѕРІР°РЅРёРµ."""
    feats: dict[str, pd.Series] = {}

    # в”Ђв”Ђ Funding Rate в”Ђв”Ђ
    if len(funding) > 0 and "funding_rate" in funding.columns:
        fr = df[["timestamp"]].merge(
            funding[["timestamp", "funding_rate"]],
            on="timestamp",
            how="left",
        )["funding_rate"].ffill()
        feats["funding_rate"] = fr
        feats["funding_rate_abs"] = fr.abs()
        for w in [24, 96]:  # 24 = 6h avg (15m bars), 96 = 24h avg
            feats[f"funding_ma_{w}"] = fr.rolling(w).mean()
            feats[f"funding_zscore_{w}"] = _zscore(fr, w)
        feats["funding_cumulative_96"] = fr.rolling(96).sum()
        # Extreme funding = potential reversal
        feats["funding_extreme"] = (fr.abs() > fr.rolling(288).std() * 2).astype(float)

    # в”Ђв”Ђ Long/Short Ratio в”Ђв”Ђ
    if len(long_short) > 0 and "long_short_ratio" in long_short.columns:
        ls = df[["timestamp"]].merge(
            long_short[["timestamp", "long_short_ratio", "long_account_pct"]],
            on="timestamp",
            how="left",
        )
        for col in ["long_short_ratio", "long_account_pct"]:
            if col in ls.columns:
                vals = ls[col].ffill()
                feats[col] = vals
                for w in [24, 96]:
                    feats[f"{col}_zscore_{w}"] = _zscore(vals, w)

    # в”Ђв”Ђ Fear & Greed в”Ђв”Ђ
    if len(fear_greed) > 0 and "fear_greed_value" in fear_greed.columns:
        fg = fear_greed.copy()
        fg["date"] = fg["timestamp"].dt.date
        df_dates = pd.DataFrame({"date": df["timestamp"].dt.date})
        merged_fg = df_dates.merge(fg[["date", "fear_greed_value"]], on="date", how="left")
        fg_val = merged_fg["fear_greed_value"].ffill()
        fg_val.index = df.index
        feats["fear_greed"] = fg_val
        feats["fear_greed_zscore"] = _zscore(fg_val, 30)  # 30-day z-score
        feats["fear_greed_change_7d"] = fg_val.diff(7)
        # Extreme fear / greed
        feats["extreme_fear"] = (fg_val < 20).astype(float)
        feats["extreme_greed"] = (fg_val > 80).astype(float)

    # в”Ђв”Ђ Open Interest в”Ђв”Ђ
    if len(open_interest) > 0 and "open_interest" in open_interest.columns:
        oi = df[["timestamp"]].merge(
            open_interest[["timestamp", "open_interest", "open_interest_value"]],
            on="timestamp",
            how="left",
        )
        for col in ["open_interest", "open_interest_value"]:
            if col in oi.columns:
                vals = oi[col].ffill()
                feats[col] = vals
                feats[f"{col}_change_4"] = vals.pct_change(4)
                feats[f"{col}_change_24"] = vals.pct_change(24)
                feats[f"{col}_zscore_48"] = _zscore(vals, 48)

    # в”Ђв”Ђ Taker Long/Short Volume в”Ђв”Ђ
    if len(taker_volume) > 0 and "buy_sell_ratio" in taker_volume.columns:
        tv = df[["timestamp"]].merge(
            taker_volume[["timestamp", "buy_sell_ratio", "buy_vol", "sell_vol"]],
            on="timestamp",
            how="left",
        )
        bsr = tv["buy_sell_ratio"].ffill()
        feats["buy_sell_ratio_ext"] = bsr
        for w in [24, 96]:
            feats[f"buy_sell_ratio_zscore_{w}"] = _zscore(bsr, w)

    return pd.DataFrame(feats, index=df.index)


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# CATEGORY: TEMPORAL
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

def _temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Р’СЂРµРјРµРЅРЅС‹Рµ С„РёС‡Рё: cyclic encoding, СЃРµСЃСЃРёРё, funding window."""
    ts = pd.to_datetime(df["timestamp"])
    feats: dict[str, pd.Series] = {}

    # Cyclic hour encoding
    hour = ts.dt.hour + ts.dt.minute / 60
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Cyclic day of week
    dow = ts.dt.dayofweek
    feats["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    feats["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Market sessions (UTC)
    feats["session_asia"] = ((hour >= 1) & (hour < 9)).astype(float)
    feats["session_europe"] = ((hour >= 7) & (hour < 16)).astype(float)
    feats["session_us"] = ((hour >= 13) & (hour < 22)).astype(float)
    feats["is_weekend"] = (dow >= 5).astype(float)

    # Funding window: BTC funding resets at 0:00, 8:00, 16:00 UTC
    bars_to_funding = ((8 - (hour % 8)) % 8) * 4  # 15m bars
    feats["bars_to_funding"] = bars_to_funding

    # Month-of-year cyclical
    month = ts.dt.month
    feats["month_sin"] = np.sin(2 * np.pi * month / 12)
    feats["month_cos"] = np.cos(2 * np.pi * month / 12)

    return pd.DataFrame(feats, index=df.index)


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# CATEGORY: MULTI-TIMEFRAME CONTEXT
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

def _multi_tf_features(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
) -> pd.DataFrame:
    """РљРѕРЅС‚РµРєСЃС‚ СЃ РІС‹СЃС€РёС… С‚Р°Р№РјС„СЂРµР№РјРѕРІ, РёРЅС‚РµСЂРїРѕР»РёСЂРѕРІР°РЅРЅС‹Р№ РІ 15m."""
    feats: dict[str, pd.Series] = {}

    for tf_name, tf_df in [("1h", df_1h), ("4h", df_4h), ("1d", df_1d)]:
        if len(tf_df) == 0:
            continue
        c = tf_df["close"]

        # RSI СЃ РІС‹СЃС€РµРіРѕ РўР¤
        rsi_14 = _rsi(c, 14)
        rsi_series = tf_df[["timestamp"]].assign(**{f"rsi_{tf_name}": rsi_14.values})

        # Trend direction (EMA20 slope)
        ema_20 = _ema(c, 20)
        slope = ema_20.pct_change(5)
        rsi_series[f"trend_{tf_name}"] = slope.values

        # Bollinger position
        sma_20 = c.rolling(20).mean()
        std_20 = c.rolling(20).std()
        bb_pos = (c - (sma_20 - 2 * std_20)) / (4 * std_20).replace(0, np.nan)
        rsi_series[f"bb_pos_{tf_name}"] = bb_pos.values

        # Merge with 15m
        rsi_series["timestamp"] = pd.to_datetime(rsi_series["timestamp"])
        merged = df_15m[["timestamp"]].merge(rsi_series, on="timestamp", how="left")
        for col in [f"rsi_{tf_name}", f"trend_{tf_name}", f"bb_pos_{tf_name}"]:
            feats[col] = merged[col].ffill().values

    result = pd.DataFrame(feats)
    result.index = df_15m.index
    return result


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# CATEGORY: NON-LINEAR INTERACTIONS
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

def _interaction_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """РќРµР»РёРЅРµР№РЅС‹Рµ РІР·Р°РёРјРѕСЃРІСЏР·Рё РјРµР¶РґСѓ С„РёС‡Р°РјРё."""
    feats: dict[str, pd.Series] = {}

    def _safe_col(name: str) -> Optional[pd.Series]:
        return features_df[name] if name in features_df.columns else None

    # Volume Г— Price divergences
    vol_z = _safe_col("vol_zscore_96")
    ret_48 = _safe_col("ret_48")
    if vol_z is not None and ret_48 is not None:
        # High volume + down price = distribution
        feats["vol_price_div"] = vol_z * ret_48
        # Climactic volume at new highs = potential top
        dd_96 = _safe_col("drawdown_96")
        if dd_96 is not None:
            feats["climactic_top"] = (vol_z > 2) & (dd_96 > -0.01)
            feats["climactic_top"] = feats["climactic_top"].astype(float)

    # Funding Г— Basis divergence
    funding = _safe_col("funding_rate")
    basis = _safe_col("basis")
    if funding is not None and basis is not None:
        feats["funding_basis_div"] = funding - basis
        feats["funding_basis_extreme"] = (
            (funding > funding.rolling(96).quantile(0.9)) &
            (basis > basis.rolling(96).quantile(0.9))
        ).astype(float)

    # Volatility contraction + volume decrease = breakout setup
    bb_w = _safe_col("bb_width_20")
    if bb_w is not None and vol_z is not None:
        feats["squeeze"] = (
            (bb_w < bb_w.rolling(96).quantile(0.1)) &
            (vol_z < 0)
        ).astype(float)

    # Fear + High Basis + High Funding = overlevered
    fg = _safe_col("fear_greed")
    if fg is not None and funding is not None and basis is not None:
        feats["overlevered"] = (
            (fg > 70) &
            (funding > funding.rolling(96).quantile(0.8)) &
            (basis > basis.rolling(96).quantile(0.8))
        ).astype(float)

    # Hurst deviation вЂ” mean-revert vs trend signal
    hurst = _safe_col("hurst_96")
    if hurst is not None:
        feats["hurst_trend_signal"] = (hurst > 0.6).astype(float) - (hurst < 0.4).astype(float)

    return pd.DataFrame(feats, index=features_df.index)


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# TARGET / LABELS
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

def calculate_target(
    df: pd.DataFrame,
    crash_threshold_pct: float = 5.0,
    crash_window_bars: int = 48,
) -> pd.Series:
    """
    Р¦РµР»РµРІР°СЏ РїРµСЂРµРјРµРЅРЅР°СЏ: 1 РµСЃР»Рё С†РµРЅР° РїР°РґР°РµС‚ >= crash_threshold_pct%
    Р·Р° СЃР»РµРґСѓСЋС‰РёРµ crash_window_bars Р±Р°СЂРѕРІ.

    РџРѕ СѓРјРѕР»С‡Р°РЅРёСЋ: 5% Р·Р° 12 С‡Р°СЃРѕРІ (48 Р±Р°СЂРѕРІ РїРѕ 15m).
    """
    close = df["close"]
    future_min = close.rolling(crash_window_bars, min_periods=1).min().shift(-crash_window_bars)
    max_drop_pct = (future_min - close) / close * 100
    target = (max_drop_pct <= -crash_threshold_pct).astype(int)
    # РЈР±РёСЂР°РµРј С…РІРѕСЃС‚, РіРґРµ РЅРµС‚ future data
    target.iloc[-crash_window_bars:] = np.nan
    return target


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# MAIN: BUILD ALL FEATURES
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

def build_features(
    data: dict[str, pd.DataFrame],
    crash_threshold_pct: float = 5.0,
    crash_window_bars: int = 48,
    include_complexity: bool = True,
) -> pd.DataFrame:
    """
    РћР±СЉРµРґРёРЅРёС‚СЊ РІСЃРµ С„РёС‡Рё РІ РѕРґРёРЅ DataFrame.

    Args:
        data: СЃР»РѕРІР°СЂСЊ СЃ DataFrames РѕС‚ data_loader.load_all_data()
        crash_threshold_pct: РїРѕСЂРѕРі РєСЂР°С€Р° РІ %
        crash_window_bars: РѕРєРЅРѕ РґР»СЏ РѕРїСЂРµРґРµР»РµРЅРёСЏ РєСЂР°С€Р° (РІ Р±Р°СЂР°С… 15m)
        include_complexity: РІРєР»СЋС‡РёС‚СЊ С‚СЏР¶С‘Р»С‹Рµ complexity features (Hurst, entropy)

    Returns:
        DataFrame СЃ РєРѕР»РѕРЅРєР°РјРё: timestamp, target, feature1, feature2, ...
    """
    df = data.get("futures_15m", pd.DataFrame())
    if len(df) == 0:
        raise ValueError("No futures 15m data")

    print(f"   рџ”§ Р¤РёС‡Рё: Price structure...")
    price_f = _price_features(df)
    print(f"      вњ… {price_f.shape[1]} С„РёС‡РµР№")

    print(f"   рџ”§ Р¤РёС‡Рё: Volume microstructure...")
    vol_f = _volume_features(df)
    print(f"      вњ… {vol_f.shape[1]} С„РёС‡РµР№")

    print(f"   рџ”§ Р¤РёС‡Рё: Volatility regime...")
    volat_f = _volatility_features(df)
    print(f"      вњ… {volat_f.shape[1]} С„РёС‡РµР№")

    complexity_f = pd.DataFrame(index=df.index)
    if include_complexity:
        print(f"   рџ”§ Р¤РёС‡Рё: Information theory & fractal (РґРѕР»СЊС€Рµ РІСЃРµРіРѕ)...")
        complexity_f = _complexity_features(df)
        print(f"      вњ… {complexity_f.shape[1]} С„РёС‡РµР№")
    else:
        print(f"   вЏ­пёЏ  Complexity features РїСЂРѕРїСѓС‰РµРЅС‹ (include_complexity=False)")

    print(f"   рџ”§ Р¤РёС‡Рё: Cross-asset (basis)...")
    basis_f = _basis_features(df, data.get("spot_15m", pd.DataFrame()))
    print(f"      вњ… {basis_f.shape[1]} С„РёС‡РµР№")

    print(f"   рџ”§ Р¤РёС‡Рё: Market sentiment...")
    sentiment_f = _sentiment_features(
        df,
        data.get("funding", pd.DataFrame()),
        data.get("long_short", pd.DataFrame()),
        data.get("fear_greed", pd.DataFrame()),
        data.get("open_interest", pd.DataFrame()),
        data.get("taker_volume", pd.DataFrame()),
    )
    print(f"      вњ… {sentiment_f.shape[1]} С„РёС‡РµР№")

    print(f"   рџ”§ Р¤РёС‡Рё: Temporal patterns...")
    temporal_f = _temporal_features(df)
    print(f"      вњ… {temporal_f.shape[1]} С„РёС‡РµР№")

    print(f"   рџ”§ Р¤РёС‡Рё: Multi-timeframe context...")
    mtf_f = _multi_tf_features(
        df,
        data.get("futures_1h", pd.DataFrame()),
        data.get("futures_4h", pd.DataFrame()),
        data.get("futures_1d", pd.DataFrame()),
    )
    print(f"      вњ… {mtf_f.shape[1]} С„РёС‡РµР№")

    # Combine everything
    print(f"   рџ”§ РћР±СЉРµРґРёРЅРµРЅРёРµ С„РёС‡РµР№...")
    all_feats = pd.concat(
        [price_f, vol_f, volat_f, complexity_f, basis_f, sentiment_f, temporal_f, mtf_f],
        axis=1,
    )

    # Non-linear interactions (needs combined features)
    print(f"   рџ”§ Р¤РёС‡Рё: Non-linear interactions...")
    interactions_f = _interaction_features(all_feats)
    print(f"      вњ… {interactions_f.shape[1]} С„РёС‡РµР№")

    all_feats = pd.concat([all_feats, interactions_f], axis=1)

    # Target
    print(f"   рџЋЇ Р Р°СЃС‡С‘С‚ target (crash {crash_threshold_pct}% Р·Р° {crash_window_bars} Р±Р°СЂРѕРІ)...")
    target = calculate_target(df, crash_threshold_pct, crash_window_bars)

    result = pd.concat([df[["timestamp"]], target.rename("target"), all_feats], axis=1)

    # Clean up
    result = result.replace([np.inf, -np.inf], np.nan)

    total_feats = result.shape[1] - 2  # minus timestamp and target
    total_rows_before = len(result)
    result = result.dropna(subset=["target"])
    total_rows = len(result)
    null_pct = result.isnull().sum().sum() / (total_rows * result.shape[1]) * 100

    print(f"\n   рџ“Љ РС‚РѕРіРѕ: {total_feats} С„РёС‡РµР№ Г— {total_rows:,} СЃС‚СЂРѕРє")
    print(f"   рџ“Љ РЈРґР°Р»РµРЅРѕ СЃС‚СЂРѕРє Р±РµР· target: {total_rows_before - total_rows:,}")
    print(f"   рџ“Љ NaN РІ РґР°РЅРЅС‹С…: {null_pct:.1f}%")
    print(f"   рџ“Љ Crash events: {int(target.sum())} ({target.mean() * 100:.2f}%)")

    return result
