"""
50+ индикаторов, каждый возвращает серию сигналов.

Категории:
1. Trend — EMA crosses, SuperTrend, ADX, Ichimoku
2. Momentum — RSI, Stochastic, MACD, ROC, Williams %R, CCI
3. Volatility — Bollinger, Keltner, ATR, Donchian
4. Volume — OBV, MFI, VWAP, Volume Profile, Taker Delta
5. Market Structure — Funding, Basis, Fear&Greed, OI
6. Pattern — Engulfing, Doji, Hammer, Divergences
7. Statistical — Z-score, Hurst, Correlation, Mean-reversion

Каждая функция принимает DataFrame с OHLCV и параметры,
возвращает pd.Series числовых значений.
"""

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _sma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=1).mean()


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, p: int) -> pd.Series:
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(p, min_periods=1).mean()


def _rsi(s: pd.Series, p: int) -> pd.Series:
    d = s.diff()
    g = d.where(d > 0, 0.0)
    lo = -d.where(d < 0, 0.0)
    ag = g.ewm(alpha=1 / p, min_periods=p).mean()
    al = lo.ewm(alpha=1 / p, min_periods=p).mean()
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))


def _stoch(h: pd.Series, l: pd.Series, c: pd.Series, k: int, d: int) -> tuple[pd.Series, pd.Series]:
    lowest = l.rolling(k, min_periods=1).min()
    highest = h.rolling(k, min_periods=1).max()
    k_line = 100 * (c - lowest) / (highest - lowest).replace(0, np.nan)
    d_line = k_line.rolling(d, min_periods=1).mean()
    return k_line, d_line


# ═══════════════════════════════════════════════════════════════════════
# TREND INDICATORS
# ═══════════════════════════════════════════════════════════════════════

def ema_cross(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> pd.Series:
    """EMA cross: +1 = golden cross, -1 = death cross."""
    ef = _ema(df["close"], fast)
    es = _ema(df["close"], slow)
    return (ef > es).astype(float) * 2 - 1


def ema_distance(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Расстояние цены от EMA, нормализованное."""
    e = _ema(df["close"], period)
    return (df["close"] - e) / e


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.Series:
    """SuperTrend: +1 = up, -1 = down (vectorized via numpy)."""
    atr = _atr(df["high"], df["low"], df["close"], period)
    hl2 = (df["high"] + df["low"]) / 2
    up = (hl2 - mult * atr).values
    dn = (hl2 + mult * atr).values
    close = df["close"].values
    n = len(close)
    trend = np.ones(n)

    for i in range(1, n):
        if close[i] > dn[i - 1]:
            trend[i] = 1
        elif close[i] < up[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
    return pd.Series(trend, index=df.index)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ADX — сила тренда (0-100)."""
    h, l, c = df["high"], df["low"], df["close"]
    up_move = h.diff()
    dn_move = -l.diff()
    plus_dm = up_move.where((up_move > dn_move) & (up_move > 0), 0)
    minus_dm = dn_move.where((dn_move > up_move) & (dn_move > 0), 0)
    atr_val = _atr(h, l, c, period)
    plus_di = 100 * _ema(plus_dm, period) / atr_val.replace(0, np.nan)
    minus_di = 100 * _ema(minus_dm, period) / atr_val.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _ema(dx, period)


def adx_direction(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """DI+ vs DI- direction: +1 bullish, -1 bearish."""
    h, l, c = df["high"], df["low"], df["close"]
    up_move = h.diff()
    dn_move = -l.diff()
    plus_dm = up_move.where((up_move > dn_move) & (up_move > 0), 0)
    minus_dm = dn_move.where((dn_move > up_move) & (dn_move > 0), 0)
    atr_val = _atr(h, l, c, period)
    plus_di = 100 * _ema(plus_dm, period) / atr_val.replace(0, np.nan)
    minus_di = 100 * _ema(minus_dm, period) / atr_val.replace(0, np.nan)
    return (plus_di > minus_di).astype(float) * 2 - 1


def triple_ema_trend(df: pd.DataFrame, fast: int = 8, mid: int = 21, slow: int = 55) -> pd.Series:
    """Triple EMA: +1 = all aligned up, -1 = all down, 0 mixed."""
    ef = _ema(df["close"], fast)
    em = _ema(df["close"], mid)
    es = _ema(df["close"], slow)
    bullish = ((ef > em) & (em > es)).astype(float)
    bearish = ((ef < em) & (em < es)).astype(float)
    return bullish - bearish


def linear_regression_slope(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Нормализованный наклон линейной регрессии (полностью векторизовано)."""
    c = df["close"]
    # Use rolling covariance with a linear ramp [0..period-1]
    # slope = cov(x, y) / var(x)
    x_mean = (period - 1) / 2.0
    x_var = (period ** 2 - 1) / 12.0  # var of 0..period-1
    # Rolling mean of y and of x·y
    y_mean = c.rolling(period, min_periods=period).mean()
    # x * y — we need sum(x_i * y_i) for rolling window
    # x_i = 0..period-1 from left to right
    # Trick: cumulative weighted sum via reverse weights
    # Instead, use the formula: slope = (12 / (n*(n^2-1))) * sum((2*i - n + 1)* y_i)
    factor = 12.0 / (period * (period**2 - 1))
    # Build symmetric weights [-m, ..., 0, ..., +m] for linear regression
    weights = np.arange(period, dtype=float) - (period - 1) / 2.0
    # Convolve with rolling window
    weighted_sum = c.rolling(period, min_periods=period).apply(
        lambda w: np.dot(weights, w), raw=True
    )
    slope = weighted_sum * factor * period  # un-normalize to match classic formula
    # Normalize by price
    return (slope / c.replace(0, np.nan)).fillna(0)


# ═══════════════════════════════════════════════════════════════════════
# MOMENTUM INDICATORS
# ═══════════════════════════════════════════════════════════════════════

def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI (0-100)."""
    return _rsi(df["close"], period)


def rsi_signal(df: pd.DataFrame, period: int = 14, oversold: float = 30, overbought: float = 70) -> pd.Series:
    """RSI signal: +1 от oversold, -1 от overbought."""
    r = _rsi(df["close"], period)
    sig = pd.Series(0.0, index=df.index)
    sig[r < oversold] = 1.0
    sig[r > overbought] = -1.0
    return sig


def stochastic_signal(df: pd.DataFrame, k: int = 14, d: int = 3, low: float = 20, high: float = 80) -> pd.Series:
    """Stochastic signal: +1 = oversold cross up, -1 = overbought cross down."""
    k_line, d_line = _stoch(df["high"], df["low"], df["close"], k, d)
    sig = pd.Series(0.0, index=df.index)
    sig[(k_line < low) & (k_line > d_line)] = 1.0
    sig[(k_line > high) & (k_line < d_line)] = -1.0
    return sig


def macd_signal(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram normalized."""
    macd_line = _ema(df["close"], fast) - _ema(df["close"], slow)
    sig_line = _ema(macd_line, signal)
    hist = macd_line - sig_line
    return hist / df["close"] * 1000  # normalized


def macd_cross(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD cross signal: +1/-1."""
    macd_line = _ema(df["close"], fast) - _ema(df["close"], slow)
    sig_line = _ema(macd_line, signal)
    return (macd_line > sig_line).astype(float) * 2 - 1


def roc(df: pd.DataFrame, period: int = 12) -> pd.Series:
    """Rate of Change (%)."""
    return df["close"].pct_change(period) * 100


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R (-100 to 0)."""
    highest = df["high"].rolling(period, min_periods=1).max()
    lowest = df["low"].rolling(period, min_periods=1).min()
    return -100 * (highest - df["close"]) / (highest - lowest).replace(0, np.nan)


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index (vectorized MAD)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period, min_periods=1).mean()
    # Vectorized MAD: mean absolute deviation ≈ std * 0.7979
    mad = tp.rolling(period, min_periods=1).std() * 0.7979
    return (tp - sma) / (0.015 * mad).replace(0, np.nan)


def momentum(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Price momentum (pct change)."""
    return df["close"].pct_change(period)


# ═══════════════════════════════════════════════════════════════════════
# VOLATILITY INDICATORS
# ═══════════════════════════════════════════════════════════════════════

def bollinger_position(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """Position within Bollinger Bands (0=lower, 1=upper)."""
    sma = _sma(df["close"], period)
    std = df["close"].rolling(period, min_periods=1).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return (df["close"] - lower) / (upper - lower).replace(0, np.nan)


def bollinger_width(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """Bollinger Band width (squeeze detector)."""
    sma = _sma(df["close"], period)
    std = df["close"].rolling(period, min_periods=1).std()
    return (2 * std_dev * std) / sma.replace(0, np.nan)


def keltner_position(df: pd.DataFrame, period: int = 20, mult: float = 1.5) -> pd.Series:
    """Position within Keltner Channel."""
    ema = _ema(df["close"], period)
    atr = _atr(df["high"], df["low"], df["close"], period)
    upper = ema + mult * atr
    lower = ema - mult * atr
    return (df["close"] - lower) / (upper - lower).replace(0, np.nan)


def atr_normalized(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR / close — volatility relative to price."""
    return _atr(df["high"], df["low"], df["close"], period) / df["close"]


def donchian_position(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Position within Donchian Channel (0-1)."""
    highest = df["high"].rolling(period, min_periods=1).max()
    lowest = df["low"].rolling(period, min_periods=1).min()
    return (df["close"] - lowest) / (highest - lowest).replace(0, np.nan)


def squeeze(df: pd.DataFrame, bb_period: int = 20, kc_period: int = 20, kc_mult: float = 1.5) -> pd.Series:
    """Squeeze: BB inside KC = 1, else 0."""
    sma = _sma(df["close"], bb_period)
    bb_std = df["close"].rolling(bb_period, min_periods=1).std()
    bb_upper = sma + 2 * bb_std
    bb_lower = sma - 2 * bb_std

    ema = _ema(df["close"], kc_period)
    atr = _atr(df["high"], df["low"], df["close"], kc_period)
    kc_upper = ema + kc_mult * atr
    kc_lower = ema - kc_mult * atr

    return ((bb_lower > kc_lower) & (bb_upper < kc_upper)).astype(float)


def realized_volatility(df: pd.DataFrame, period: int = 24) -> pd.Series:
    """Realized vol (annualized)."""
    log_ret = np.log(df["close"] / df["close"].shift(1))
    return log_ret.rolling(period, min_periods=1).std() * np.sqrt(96 * 365)


def atr_percentile(df: pd.DataFrame, atr_period: int = 14, lookback: int = 96) -> pd.Series:
    """ATR percentile rank over lookback — volatility regime detector (0-1)."""
    atr_val = _atr(df["high"], df["low"], df["close"], atr_period)
    # Rolling rank: what fraction of lookback values are <= current
    rolling_min = atr_val.rolling(lookback, min_periods=1).min()
    rolling_max = atr_val.rolling(lookback, min_periods=1).max()
    rng = (rolling_max - rolling_min).replace(0, np.nan)
    return ((atr_val - rolling_min) / rng).fillna(0.5)


# ═══════════════════════════════════════════════════════════════════════
# TREND: ADVANCED
# ═══════════════════════════════════════════════════════════════════════

def ichimoku_signal(df: pd.DataFrame, conv: int = 9, base: int = 26) -> pd.Series:
    """Ichimoku conversion/base cross: +1 bullish, -1 bearish."""
    h, l = df["high"], df["low"]
    conv_line = (h.rolling(conv, min_periods=1).max() + l.rolling(conv, min_periods=1).min()) / 2
    base_line = (h.rolling(base, min_periods=1).max() + l.rolling(base, min_periods=1).min()) / 2
    return (conv_line > base_line).astype(float) * 2 - 1


def ichimoku_cloud_position(df: pd.DataFrame, base: int = 26, span_b: int = 52) -> pd.Series:
    """Price relative to Ichimoku cloud: >0 above, <0 below, normalized."""
    h, l, c = df["high"], df["low"], df["close"]
    conv_line = (h.rolling(9, min_periods=1).max() + l.rolling(9, min_periods=1).min()) / 2
    base_line = (h.rolling(base, min_periods=1).max() + l.rolling(base, min_periods=1).min()) / 2
    span_a = (conv_line + base_line) / 2
    span_b_val = (h.rolling(span_b, min_periods=1).max() + l.rolling(span_b, min_periods=1).min()) / 2
    cloud_top = pd.concat([span_a, span_b_val], axis=1).max(axis=1)
    cloud_bot = pd.concat([span_a, span_b_val], axis=1).min(axis=1)
    cloud_mid = (cloud_top + cloud_bot) / 2
    cloud_width = (cloud_top - cloud_bot).replace(0, np.nan)
    return (c - cloud_mid) / cloud_width


def hull_ma_trend(df: pd.DataFrame, period: int = 16) -> pd.Series:
    """Hull MA direction: +1 rising, -1 falling."""
    half = max(2, period // 2)
    sqrt_p = max(2, int(np.sqrt(period)))
    wma_half = df["close"].rolling(half, min_periods=1).mean()
    wma_full = df["close"].rolling(period, min_periods=1).mean()
    diff = 2 * wma_half - wma_full
    hma = diff.rolling(sqrt_p, min_periods=1).mean()
    return (hma.diff() > 0).astype(float) * 2 - 1


def aroon_oscillator(df: pd.DataFrame, period: int = 25) -> pd.Series:
    """Aroon Oscillator (-100 to +100) — vectorized."""
    # Rolling argmax/argmin for bars since highest/lowest
    high_idx = df["high"].rolling(period + 1, min_periods=1).apply(
        lambda w: np.argmax(w), raw=True
    )
    low_idx = df["low"].rolling(period + 1, min_periods=1).apply(
        lambda w: np.argmin(w), raw=True
    )
    window_len = df["high"].rolling(period + 1, min_periods=1).count()
    aroon_up = high_idx / (window_len - 1).replace(0, 1) * 100
    aroon_dn = low_idx / (window_len - 1).replace(0, 1) * 100
    return (aroon_up - aroon_dn).fillna(0)


def heikin_ashi_trend(df: pd.DataFrame, smooth: int = 5) -> pd.Series:
    """Heikin-Ashi smoothed trend: +1 bullish, -1 bearish."""
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    ha_close = (o + h + l + c) / 4
    ha_open = np.empty_like(ha_close)
    ha_open[0] = o[0]
    for i in range(1, len(ha_open)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
    ha_diff = pd.Series(ha_close - ha_open, index=df.index)
    smoothed = ha_diff.rolling(smooth, min_periods=1).mean()
    return (smoothed > 0).astype(float) * 2 - 1


# ═══════════════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME (higher TF data merged in prepare_dataframe)
# ═══════════════════════════════════════════════════════════════════════

def mtf_ema_trend_1h(df: pd.DataFrame, fast: int = 8, slow: int = 21) -> pd.Series:
    """1h EMA cross trend from merged data (+1/-1)."""
    if "close_1h" not in df.columns:
        return pd.Series(0.0, index=df.index)
    ema_f = _ema(df["close_1h"], fast)
    ema_s = _ema(df["close_1h"], slow)
    return (ema_f > ema_s).astype(float) * 2 - 1


def mtf_ema_trend_4h(df: pd.DataFrame, fast: int = 8, slow: int = 21) -> pd.Series:
    """4h EMA cross trend from merged data (+1/-1)."""
    if "close_4h" not in df.columns:
        return pd.Series(0.0, index=df.index)
    ema_f = _ema(df["close_4h"], fast)
    ema_s = _ema(df["close_4h"], slow)
    return (ema_f > ema_s).astype(float) * 2 - 1


def mtf_rsi_1h(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """1h RSI from merged data."""
    if "close_1h" not in df.columns:
        return pd.Series(50.0, index=df.index)
    return _rsi(df["close_1h"], period)


# ═══════════════════════════════════════════════════════════════════════
# VOLUME INDICATORS
# ═══════════════════════════════════════════════════════════════════════

def obv_slope(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """OBV slope — volume trend (vectorized)."""
    direction = np.sign(df["close"].diff())
    obv = (direction * df["volume"]).cumsum()
    # Slope via difference of EMA (fast proxy for linear regression slope)
    obv_fast = obv.ewm(span=max(2, period // 3), adjust=False).mean()
    obv_slow = obv.ewm(span=period, adjust=False).mean()
    vol_avg = df["volume"].rolling(period, min_periods=1).mean().replace(0, np.nan)
    return (obv_fast - obv_slow) / vol_avg


def volume_zscore(df: pd.DataFrame, period: int = 48) -> pd.Series:
    """Volume z-score."""
    m = df["volume"].rolling(period, min_periods=1).mean()
    s = df["volume"].rolling(period, min_periods=1).std()
    return (df["volume"] - m) / s.replace(0, np.nan)


def vwap_distance(df: pd.DataFrame, period: int = 48) -> pd.Series:
    """Distance from rolling VWAP."""
    vwap = (df["close"] * df["volume"]).rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, np.nan)
    return (df["close"] - vwap) / vwap.replace(0, np.nan)


def taker_delta(df: pd.DataFrame) -> pd.Series:
    """Taker buy/sell delta (if columns exist)."""
    if "taker_buy_volume" in df.columns:
        tb = df["taker_buy_volume"]
        ts = df["volume"] - tb
        return (tb - ts) / df["volume"].replace(0, np.nan)
    return pd.Series(0.0, index=df.index)


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow (-1 to +1)."""
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
    rng = (h - l).replace(0, np.nan)
    mf_mult = ((c - l) - (h - c)) / rng
    mf_vol = mf_mult * v
    return mf_vol.rolling(period, min_periods=1).sum() / v.rolling(period, min_periods=1).sum().replace(0, np.nan)


def vwma_distance(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Distance from VWMA (volume-weighted MA)."""
    vwma = (df["close"] * df["volume"]).rolling(period, min_periods=1).sum() / \
           df["volume"].rolling(period, min_periods=1).sum().replace(0, np.nan)
    return (df["close"] - vwma) / vwma.replace(0, np.nan)


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index (0-100)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
    ratio = pos_mf / neg_mf.replace(0, np.nan)
    return 100 - 100 / (1 + ratio)


# ═══════════════════════════════════════════════════════════════════════
# MARKET STRUCTURE (need extra data merged in)
# ═══════════════════════════════════════════════════════════════════════

def funding_rate_signal(df: pd.DataFrame) -> pd.Series:
    """Funding rate if present."""
    if "funding_rate" in df.columns:
        return df["funding_rate"]
    return pd.Series(0.0, index=df.index)


def basis_signal(df: pd.DataFrame) -> pd.Series:
    """Spot-futures basis if present."""
    if "basis" in df.columns:
        return df["basis"]
    return pd.Series(0.0, index=df.index)


def fear_greed_signal(df: pd.DataFrame) -> pd.Series:
    """Fear & Greed if present."""
    if "fear_greed" in df.columns:
        return df["fear_greed"] / 100.0 * 2 - 1  # normalize to -1..+1
    return pd.Series(0.0, index=df.index)


# ═══════════════════════════════════════════════════════════════════════
# PATTERN / STATISTICAL
# ═══════════════════════════════════════════════════════════════════════

def price_zscore(df: pd.DataFrame, period: int = 48) -> pd.Series:
    """Price z-score (mean reversion signal)."""
    m = df["close"].rolling(period, min_periods=1).mean()
    s = df["close"].rolling(period, min_periods=1).std()
    return (df["close"] - m) / s.replace(0, np.nan)


def consecutive_candles(df: pd.DataFrame) -> pd.Series:
    """Consecutive up/down candles count (positive=up, negative=down)."""
    is_up = (df["close"] > df["open"]).astype(int)
    is_dn = (df["close"] < df["open"]).astype(int)

    up_count = is_up * (is_up.groupby((is_up != is_up.shift()).cumsum()).cumcount() + 1)
    dn_count = is_dn * (is_dn.groupby((is_dn != is_dn.shift()).cumsum()).cumcount() + 1)

    return up_count - dn_count


def body_ratio(df: pd.DataFrame) -> pd.Series:
    """Body / range ratio (doji detector)."""
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return body / rng


def upper_shadow_ratio(df: pd.DataFrame) -> pd.Series:
    """Upper shadow / range."""
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    upper = df["high"] - pd.concat([df["close"], df["open"]], axis=1).max(axis=1)
    return upper / rng


def lower_shadow_ratio(df: pd.DataFrame) -> pd.Series:
    """Lower shadow / range."""
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    lower = pd.concat([df["close"], df["open"]], axis=1).min(axis=1) - df["low"]
    return lower / rng


def price_change_pct(df: pd.DataFrame, period: int = 1) -> pd.Series:
    """Simple % change."""
    return df["close"].pct_change(period)


def drawdown_from_high(df: pd.DataFrame, period: int = 48) -> pd.Series:
    """Current drawdown from rolling high."""
    rolling_max = df["close"].rolling(period, min_periods=1).max()
    return (df["close"] - rolling_max) / rolling_max


def range_position(df: pd.DataFrame, period: int = 48) -> pd.Series:
    """Position within range (0-1)."""
    high = df["high"].rolling(period, min_periods=1).max()
    low = df["low"].rolling(period, min_periods=1).min()
    return (df["close"] - low) / (high - low).replace(0, np.nan)


# ═══════════════════════════════════════════════════════════════════════
# REGISTRY — все индикаторы и их параметры для Optuna
# ═══════════════════════════════════════════════════════════════════════

INDICATOR_REGISTRY: dict[str, dict] = {
    # Trend
    "ema_cross": {"fn": ema_cross, "params": {"fast": (5, 50), "slow": (15, 200)}},
    "ema_distance": {"fn": ema_distance, "params": {"period": (5, 200)}},
    "supertrend": {"fn": supertrend, "params": {"period": (5, 30), "mult": (1.0, 5.0)}},
    "adx": {"fn": adx, "params": {"period": (7, 30)}},
    "adx_direction": {"fn": adx_direction, "params": {"period": (7, 30)}},
    "triple_ema": {"fn": triple_ema_trend, "params": {"fast": (5, 15), "mid": (15, 40), "slow": (30, 100)}},
    "lr_slope": {"fn": linear_regression_slope, "params": {"period": (10, 100)}},
    "ichimoku": {"fn": ichimoku_signal, "params": {"conv": (7, 14), "base": (20, 40)}},
    "ichimoku_cloud": {"fn": ichimoku_cloud_position, "params": {"base": (20, 40), "span_b": (40, 70)}},
    "hull_ma": {"fn": hull_ma_trend, "params": {"period": (9, 50)}},
    "aroon": {"fn": aroon_oscillator, "params": {"period": (14, 50)}},
    "heikin_ashi": {"fn": heikin_ashi_trend, "params": {"smooth": (3, 10)}},

    # Momentum
    "rsi": {"fn": rsi, "params": {"period": (5, 30)}},
    "rsi_signal": {"fn": rsi_signal, "params": {"period": (5, 30), "oversold": (15, 40), "overbought": (60, 85)}},
    "stochastic": {"fn": stochastic_signal, "params": {"k": (5, 21), "d": (2, 7), "low": (10, 30), "high": (70, 90)}},
    "macd_signal": {"fn": macd_signal, "params": {"fast": (8, 20), "slow": (20, 40), "signal": (5, 15)}},
    "macd_cross": {"fn": macd_cross, "params": {"fast": (8, 20), "slow": (20, 40), "signal": (5, 15)}},
    "roc": {"fn": roc, "params": {"period": (5, 50)}},
    "williams_r": {"fn": williams_r, "params": {"period": (7, 28)}},
    "cci": {"fn": cci, "params": {"period": (10, 40)}},
    "momentum": {"fn": momentum, "params": {"period": (5, 50)}},

    # Volatility
    "bb_position": {"fn": bollinger_position, "params": {"period": (10, 50), "std_dev": (1.5, 3.0)}},
    "bb_width": {"fn": bollinger_width, "params": {"period": (10, 50), "std_dev": (1.5, 3.0)}},
    "keltner_pos": {"fn": keltner_position, "params": {"period": (10, 50), "mult": (1.0, 3.0)}},
    "atr_norm": {"fn": atr_normalized, "params": {"period": (7, 30)}},
    "donchian_pos": {"fn": donchian_position, "params": {"period": (10, 50)}},
    "squeeze": {"fn": squeeze, "params": {"bb_period": (10, 30), "kc_period": (10, 30), "kc_mult": (1.0, 2.5)}},
    "real_vol": {"fn": realized_volatility, "params": {"period": (12, 96)}},
    "atr_pctl": {"fn": atr_percentile, "params": {"atr_period": (7, 21), "lookback": (48, 384)}},

    # Volume
    "obv_slope": {"fn": obv_slope, "params": {"period": (10, 50)}},
    "vol_zscore": {"fn": volume_zscore, "params": {"period": (24, 192)}},
    "vwap_dist": {"fn": vwap_distance, "params": {"period": (24, 192)}},
    "taker_delta": {"fn": taker_delta, "params": {}},
    "mfi": {"fn": mfi, "params": {"period": (7, 28)}},
    "cmf": {"fn": cmf, "params": {"period": (10, 40)}},
    "vwma_dist": {"fn": vwma_distance, "params": {"period": (10, 50)}},

    # Market structure
    "funding": {"fn": funding_rate_signal, "params": {}},
    "basis": {"fn": basis_signal, "params": {}},
    "fear_greed": {"fn": fear_greed_signal, "params": {}},

    # Multi-timeframe
    "mtf_ema_1h": {"fn": mtf_ema_trend_1h, "params": {"fast": (5, 15), "slow": (15, 40)}},
    "mtf_ema_4h": {"fn": mtf_ema_trend_4h, "params": {"fast": (5, 15), "slow": (15, 40)}},
    "mtf_rsi_1h": {"fn": mtf_rsi_1h, "params": {"period": (7, 21)}},

    # Pattern / Statistical
    "price_zscore": {"fn": price_zscore, "params": {"period": (24, 192)}},
    "consec_candles": {"fn": consecutive_candles, "params": {}},
    "body_ratio": {"fn": body_ratio, "params": {}},
    "upper_shadow": {"fn": upper_shadow_ratio, "params": {}},
    "lower_shadow": {"fn": lower_shadow_ratio, "params": {}},
    "price_change": {"fn": price_change_pct, "params": {"period": (1, 48)}},
    "drawdown": {"fn": drawdown_from_high, "params": {"period": (24, 192)}},
    "range_pos": {"fn": range_position, "params": {"period": (24, 192)}},
}

# Indicators that already return values in [-1, 1] range (binary signals)
_BINARY_INDICATORS = {
    "ema_cross", "supertrend", "adx_direction", "rsi_signal",
    "stochastic", "macd_signal", "macd_cross", "funding",
    "basis", "fear_greed", "consec_candles",
    "ichimoku", "hull_ma", "heikin_ashi",
    "mtf_ema_1h", "mtf_ema_4h",
}


def _normalize_to_unit(s: pd.Series) -> pd.Series:
    """
    Нормализовать серию в диапазон [-1, 1] используя expanding window
    (без lookahead bias).
    """
    mn = s.expanding(min_periods=50).min()
    mx = s.expanding(min_periods=50).max()
    rng = mx - mn
    mid = (mn + mx) / 2
    half_rng = rng / 2
    # Avoid division by zero
    half_rng = half_rng.replace(0, np.nan)
    result = ((s - mid) / half_rng).clip(-1, 1)
    return result.fillna(0)


# ── FIXED RANGES from full 70,080-bar BTC 15m dataset ──
# Used instead of expanding normalization for TV-reproducible results.
_FIXED_RANGES: dict[str, tuple[float, float]] = {
    "adx":             (5.934,    100.0),
    "roc":             (-14.425,  12.720),
    "cci":             (-273.079, 271.688),
    "momentum":        (-0.14343, 0.12551),
    "ema_distance":    (-100.0,   100.0),
    "lr_slope":        (-100.0,   100.0),
    "williams_r":      (-100.0,   0.0),
    "bb_position":     (-3.0,     3.0),
    "bb_width":        (0.0,      0.3),
    "keltner_pos":     (-3.0,     3.0),
    "atr_norm":        (0.0,      0.1),
    "donchian_pos":    (0.0,      1.0),
    "price_zscore":    (-5.0,     5.0),
    "obv_slope":       (-1e10,    1e10),
    "vol_zscore":      (-3.0,     5.0),
    "vwap_dist":       (-5.0,     5.0),
    "mfi":             (0.0,      100.0),
    "squeeze":         (-1.0,     1.0),
    "body_ratio":      (-1.0,     1.0),
    "upper_shadow":    (0.0,      1.0),
    "lower_shadow":    (0.0,      1.0),
    "price_change":    (-0.2,     0.2),
    "drawdown":        (-0.5,     0.0),
    "range_pos":       (0.0,      1.0),
    "triple_ema":      (-100.0,   100.0),
    "rsi":             (0.0,      100.0),
    "real_vol":        (0.0,      3.0),
    # New indicators
    "ichimoku_cloud":  (-5.0,     5.0),
    "aroon":           (-100.0,   100.0),
    "atr_pctl":        (0.0,      1.0),
    "cmf":             (-1.0,     1.0),
    "vwma_dist":       (-0.05,    0.05),
    "mtf_rsi_1h":      (0.0,      100.0),
}

# Toggle: True = use fixed ranges (TV-reproducible), False = expanding (original)
USE_FIXED_RANGES = True


def _normalize_fixed(s: pd.Series, mn: float, mx: float) -> pd.Series:
    """Normalise using a fixed [mn, mx] range instead of expanding window."""
    mid = (mn + mx) / 2.0
    hr = (mx - mn) / 2.0
    if hr <= 0:
        return pd.Series(0.0, index=s.index)
    return ((s - mid) / hr).clip(-1, 1).fillna(0)


def compute_indicator(name: str, df: pd.DataFrame, params: dict) -> pd.Series:
    """Вычислить индикатор по имени с параметрами. Нормализует в [-1, 1]."""
    info = INDICATOR_REGISTRY[name]
    fn = info["fn"]
    # Filter params to only those the function accepts
    valid_params = {k: v for k, v in params.items() if k in info["params"]}
    result = fn(df, **valid_params)

    # Normalize continuous indicators to [-1, 1]
    if name not in _BINARY_INDICATORS:
        if USE_FIXED_RANGES and name in _FIXED_RANGES:
            fmin, fmax = _FIXED_RANGES[name]
            result = _normalize_fixed(result, fmin, fmax)
        else:
            result = _normalize_to_unit(result)

    return result
