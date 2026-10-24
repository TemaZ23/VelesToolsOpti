"""
Индикаторы в трактовке Veles (режим «На закрытии бара»).

Все серии считаются на собственном таймфрейме (1m → 5m/15m/30m/1h/4h агрегируются
из минуток). Движок берёт значение последнего ЗАКРЫТОГО бара нужного ТФ:
для минуты i (оценка на её закрытии) это бар j = (i + 1) // k - 1, где k — длина ТФ в минутах.

Параметры по умолчанию соответствуют справке Veles:
    RSI 14, CCI 20, CMO 9, Williams %R 14, ATR 14, ADX 14/14, MFI 14,
    BB 20/2, Keltner 20/2 (EMA + ATR), Stochastic 14/1/3, MACD 12/26/9,
    Turtle Zone 200 / InnerAmp 5.6 (лог-шкала, сглаживание Элерса).

Сигнальные («basic») индикаторы для лонга:
    RSI_LEVELS   — RSI пересёк 30 снизу вверх
    CCI_LEVELS   — CCI пересёк -100 снизу вверх
    STOCH_LEVELS — обе линии были ниже 20, K пересекла 20 снизу вверх
    BB / KELTNER — закрытие ниже нижней полосы (обратный сигнал — выше верхней / ниже нижней для выхода)
    TURTLE_ZONE  — закрытие выше верхней внутренней зоны (выход из лонга)
    MACD (ноль)  — пересечение нуля; обратный сигнал для лонга — сверху вниз
Канальные индикаторы хранятся как относительное расстояние до полосы
((close - band) / close), чтобы условие «ниже полосы» было порогом < 0.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

# ═══════════════════════════════════════════════════════════════════════
# АГРЕГАЦИЯ
# ═══════════════════════════════════════════════════════════════════════


@njit(cache=True)
def resample(ohlc: np.ndarray, qvol: np.ndarray, k: int):
    """1m → k-минутные бары. Неполный хвостовой бар отбрасывается."""
    n = ohlc.shape[0] // k
    out = np.empty((n, 4))
    vol = np.empty(n)
    for j in range(n):
        s = j * k
        o = ohlc[s, 0]
        h = ohlc[s, 1]
        lo = ohlc[s, 2]
        v = 0.0
        for t in range(s, s + k):
            if ohlc[t, 1] > h:
                h = ohlc[t, 1]
            if ohlc[t, 2] < lo:
                lo = ohlc[t, 2]
            v += qvol[t]
        out[j, 0] = o
        out[j, 1] = h
        out[j, 2] = lo
        out[j, 3] = ohlc[s + k - 1, 3]
        vol[j] = v
    return out, vol


# ═══════════════════════════════════════════════════════════════════════
# БАЗОВЫЕ СГЛАЖИВАНИЯ
# ═══════════════════════════════════════════════════════════════════════


@njit(cache=True)
def rma(x: np.ndarray, p: int) -> np.ndarray:
    """Wilder RMA (как ta.rma в TradingView): старт — SMA первых p значений."""
    n = x.shape[0]
    out = np.full(n, np.nan)
    if n < p:
        return out
    s = 0.0
    for i in range(p):
        s += x[i]
    acc = s / p
    out[p - 1] = acc
    a = 1.0 / p
    for i in range(p, n):
        acc = acc + a * (x[i] - acc)
        out[i] = acc
    return out


@njit(cache=True)
def ema(x: np.ndarray, p: int) -> np.ndarray:
    n = x.shape[0]
    out = np.full(n, np.nan)
    if n < p:
        return out
    s = 0.0
    for i in range(p):
        s += x[i]
    acc = s / p
    out[p - 1] = acc
    a = 2.0 / (p + 1)
    for i in range(p, n):
        acc = acc + a * (x[i] - acc)
        out[i] = acc
    return out


@njit(cache=True)
def sma(x: np.ndarray, p: int) -> np.ndarray:
    n = x.shape[0]
    out = np.full(n, np.nan)
    s = 0.0
    for i in range(n):
        s += x[i]
        if i >= p:
            s -= x[i - p]
        if i >= p - 1:
            out[i] = s / p
    return out


@njit(cache=True)
def super_smoother(x: np.ndarray, p: int) -> np.ndarray:
    """2-полюсный Super Smoother Элерса."""
    n = x.shape[0]
    out = np.empty(n)
    a1 = math.exp(-math.sqrt(2.0) * math.pi / p)
    b1 = 2.0 * a1 * math.cos(math.sqrt(2.0) * math.pi / p)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1.0 - c2 - c3
    for i in range(n):
        if i < 2:
            out[i] = x[i]
        else:
            out[i] = c1 * (x[i] + x[i - 1]) / 2.0 + c2 * out[i - 1] + c3 * out[i - 2]
    return out


@njit(cache=True)
def true_range(ohlc: np.ndarray) -> np.ndarray:
    n = ohlc.shape[0]
    tr = np.empty(n)
    tr[0] = ohlc[0, 1] - ohlc[0, 2]
    for i in range(1, n):
        pc = ohlc[i - 1, 3]
        h = ohlc[i, 1]
        lo = ohlc[i, 2]
        tr[i] = max(h - lo, abs(h - pc), abs(lo - pc))
    return tr


# ═══════════════════════════════════════════════════════════════════════
# ОСЦИЛЛЯТОРЫ
# ═══════════════════════════════════════════════════════════════════════


@njit(cache=True)
def rsi(close: np.ndarray, p: int = 14) -> np.ndarray:
    n = close.shape[0]
    up = np.zeros(n)
    dn = np.zeros(n)
    for i in range(1, n):
        d = close[i] - close[i - 1]
        if d > 0:
            up[i] = d
        else:
            dn[i] = -d
    ru = rma(up[1:], p)
    rd = rma(dn[1:], p)
    out = np.full(n, np.nan)
    for i in range(n - 1):
        if not np.isnan(ru[i]):
            if rd[i] == 0.0:
                out[i + 1] = 100.0 if ru[i] > 0 else 50.0
            else:
                out[i + 1] = 100.0 - 100.0 / (1.0 + ru[i] / rd[i])
    return out


@njit(cache=True)
def cci(ohlc: np.ndarray, p: int = 20) -> np.ndarray:
    n = ohlc.shape[0]
    tp = (ohlc[:, 1] + ohlc[:, 2] + ohlc[:, 3]) / 3.0
    ma = sma(tp, p)
    out = np.full(n, np.nan)
    for i in range(p - 1, n):
        md = 0.0
        for t in range(i - p + 1, i + 1):
            md += abs(tp[t] - ma[i])
        md /= p
        out[i] = 0.0 if md == 0.0 else (tp[i] - ma[i]) / (0.015 * md)
    return out


@njit(cache=True)
def cmo(close: np.ndarray, p: int = 9) -> np.ndarray:
    n = close.shape[0]
    out = np.full(n, np.nan)
    su = 0.0
    sd = 0.0
    for i in range(1, n):
        d = close[i] - close[i - 1]
        if d > 0:
            su += d
        else:
            sd -= d
        if i > p:
            d0 = close[i - p] - close[i - p - 1]
            if d0 > 0:
                su -= d0
            else:
                sd += d0
        if i >= p:
            tot = su + sd
            out[i] = 0.0 if tot <= 1e-300 else 100.0 * (su - sd) / tot
    return out


@njit(cache=True)
def williams_r(ohlc: np.ndarray, p: int = 14) -> np.ndarray:
    n = ohlc.shape[0]
    out = np.full(n, np.nan)
    for i in range(p - 1, n):
        hh = ohlc[i, 1]
        ll = ohlc[i, 2]
        for t in range(i - p + 1, i + 1):
            if ohlc[t, 1] > hh:
                hh = ohlc[t, 1]
            if ohlc[t, 2] < ll:
                ll = ohlc[t, 2]
        rng = hh - ll
        out[i] = -50.0 if rng == 0.0 else -100.0 * (hh - ohlc[i, 3]) / rng
    return out


@njit(cache=True)
def stochastic(ohlc: np.ndarray, p: int = 14, smooth_k: int = 1, smooth_d: int = 3):
    n = ohlc.shape[0]
    raw = np.full(n, np.nan)
    for i in range(p - 1, n):
        hh = ohlc[i, 1]
        ll = ohlc[i, 2]
        for t in range(i - p + 1, i + 1):
            if ohlc[t, 1] > hh:
                hh = ohlc[t, 1]
            if ohlc[t, 2] < ll:
                ll = ohlc[t, 2]
        rng = hh - ll
        raw[i] = 50.0 if rng == 0.0 else 100.0 * (ohlc[i, 3] - ll) / rng
    k = raw if smooth_k <= 1 else _sma_nan(raw, smooth_k)
    d = _sma_nan(k, smooth_d)
    return k, d


@njit(cache=True)
def _sma_nan(x: np.ndarray, p: int) -> np.ndarray:
    n = x.shape[0]
    out = np.full(n, np.nan)
    for i in range(p - 1, n):
        s = 0.0
        ok = True
        for t in range(i - p + 1, i + 1):
            if np.isnan(x[t]):
                ok = False
                break
            s += x[t]
        if ok:
            out[i] = s / p
    return out


@njit(cache=True)
def mfi(ohlc: np.ndarray, qvol: np.ndarray, p: int = 14) -> np.ndarray:
    """MFI; денежный поток ≈ номинальный объём (quote volume) бара."""
    n = ohlc.shape[0]
    tp = (ohlc[:, 1] + ohlc[:, 2] + ohlc[:, 3]) / 3.0
    pos = np.zeros(n)
    neg = np.zeros(n)
    for i in range(1, n):
        if tp[i] > tp[i - 1]:
            pos[i] = qvol[i]
        elif tp[i] < tp[i - 1]:
            neg[i] = qvol[i]
    out = np.full(n, np.nan)
    sp = 0.0
    sn = 0.0
    for i in range(1, n):
        sp += pos[i]
        sn += neg[i]
        if i > p:
            sp -= pos[i - p]
            sn -= neg[i - p]
        if i >= p:
            tot = sp + sn
            out[i] = 50.0 if tot <= 0.0 else 100.0 * sp / tot
    return out


@njit(cache=True)
def atr(ohlc: np.ndarray, p: int = 14) -> np.ndarray:
    return rma(true_range(ohlc), p)


@njit(cache=True)
def adx(ohlc: np.ndarray, p: int = 14, smoothing: int = 14) -> np.ndarray:
    n = ohlc.shape[0]
    pdm = np.zeros(n)
    ndm = np.zeros(n)
    for i in range(1, n):
        up = ohlc[i, 1] - ohlc[i - 1, 1]
        dn = ohlc[i - 1, 2] - ohlc[i, 2]
        if up > dn and up > 0:
            pdm[i] = up
        if dn > up and dn > 0:
            ndm[i] = dn
    tr = true_range(ohlc)
    str_ = rma(tr[1:], p)
    sp = rma(pdm[1:], p)
    sn = rma(ndm[1:], p)
    dx = np.full(n - 1, np.nan)
    for i in range(n - 1):
        if not np.isnan(str_[i]) and str_[i] > 0:
            pdi = 100.0 * sp[i] / str_[i]
            ndi = 100.0 * sn[i] / str_[i]
            s = pdi + ndi
            dx[i] = 0.0 if s == 0 else 100.0 * abs(pdi - ndi) / s
    out = np.full(n, np.nan)
    # RMA по валидной части DX
    first = -1
    for i in range(n - 1):
        if not np.isnan(dx[i]):
            first = i
            break
    if first >= 0:
        r = rma(dx[first:], smoothing)
        for i in range(r.shape[0]):
            out[first + 1 + i] = r[i]
    return out


@njit(cache=True)
def macd_line(close: np.ndarray, fast: int = 12, slow: int = 26) -> np.ndarray:
    return ema(close, fast) - ema(close, slow)


# ═══════════════════════════════════════════════════════════════════════
# КАНАЛЫ (относительное расстояние до полосы)
# ═══════════════════════════════════════════════════════════════════════


@njit(cache=True)
def bb_lower_dist(close: np.ndarray, p: int = 20, k: float = 2.0) -> np.ndarray:
    n = close.shape[0]
    out = np.full(n, np.nan)
    for i in range(p - 1, n):
        m = 0.0
        for t in range(i - p + 1, i + 1):
            m += close[t]
        m /= p
        v = 0.0
        for t in range(i - p + 1, i + 1):
            v += (close[t] - m) ** 2
        sd = math.sqrt(v / p)
        out[i] = (close[i] - (m - k * sd)) / close[i]
    return out


@njit(cache=True)
def bb_upper_dist(close: np.ndarray, p: int = 20, k: float = 2.0) -> np.ndarray:
    n = close.shape[0]
    out = np.full(n, np.nan)
    for i in range(p - 1, n):
        m = 0.0
        for t in range(i - p + 1, i + 1):
            m += close[t]
        m /= p
        v = 0.0
        for t in range(i - p + 1, i + 1):
            v += (close[t] - m) ** 2
        sd = math.sqrt(v / p)
        out[i] = (close[i] - (m + k * sd)) / close[i]
    return out


@njit(cache=True)
def keltner_lower_dist(ohlc: np.ndarray, p: int = 20, k: float = 2.0) -> np.ndarray:
    close = ohlc[:, 3]
    mid = ema(close, p)
    a = atr(ohlc, p)
    return (close - (mid - k * a)) / close


@njit(cache=True)
def keltner_upper_dist(ohlc: np.ndarray, p: int = 20, k: float = 2.0) -> np.ndarray:
    close = ohlc[:, 3]
    mid = ema(close, p)
    a = atr(ohlc, p)
    return (close - (mid + k * a)) / close


@njit(cache=True)
def turtle_bands(ohlc: np.ndarray, p: int, inner_amp: float):
    n = ohlc.shape[0]
    lh = np.log(ohlc[:, 1])
    ll = np.log(ohlc[:, 2])
    lc = np.log(ohlc[:, 3])
    src = (lh + ll + lc) / 3.0
    tr = np.empty(n)
    tr[0] = lh[0] - ll[0]
    for i in range(1, n):
        tr[i] = max(lh[i], lc[i - 1]) - min(ll[i], lc[i - 1])
    return super_smoother(src, p), super_smoother(tr, p)


@njit(cache=True)
def turtle_lower_dist(ohlc: np.ndarray, p: int = 200, inner_amp: float = 5.6) -> np.ndarray:
    """Turtle Zone, нижняя внутренняя зона: (close - lower) / close; < 0 — закрытие ниже зоны."""
    n = ohlc.shape[0]
    mean, rng = turtle_bands(ohlc, p, inner_amp)
    out = np.full(n, np.nan)
    for i in range(p, n):
        lower = math.exp(mean[i] - inner_amp * rng[i])
        out[i] = (ohlc[i, 3] - lower) / ohlc[i, 3]
    return out


@njit(cache=True)
def turtle_upper_dist(ohlc: np.ndarray, p: int = 200, inner_amp: float = 5.6) -> np.ndarray:
    """
    Turtle Zone (верхняя внутренняя зона), аппроксимация оригинала:
    лог-шкала, среднее — Super Smoother(hlc3, p), ширина — Super Smoother(TR, p) * InnerAmp.
    """
    n = ohlc.shape[0]
    lh = np.log(ohlc[:, 1])
    ll = np.log(ohlc[:, 2])
    lc = np.log(ohlc[:, 3])
    src = (lh + ll + lc) / 3.0
    tr = np.empty(n)
    tr[0] = lh[0] - ll[0]
    for i in range(1, n):
        tr[i] = max(lh[i], lc[i - 1]) - min(ll[i], lc[i - 1])
    mean = super_smoother(src, p)
    rng = super_smoother(tr, p)
    out = np.full(n, np.nan)
    for i in range(p, n):
        upper = math.exp(mean[i] + inner_amp * rng[i])
        out[i] = (ohlc[i, 3] - upper) / ohlc[i, 3]
    return out


# ═══════════════════════════════════════════════════════════════════════
# СИГНАЛЫ-СОБЫТИЯ (1.0 — сигнал на этом баре)
# ═══════════════════════════════════════════════════════════════════════


@njit(cache=True)
def cross_up_event(x: np.ndarray, level: float) -> np.ndarray:
    n = x.shape[0]
    out = np.zeros(n)
    for i in range(1, n):
        if x[i - 1] < level and x[i] >= level:
            out[i] = 1.0
    return out


@njit(cache=True)
def cross_down_event(x: np.ndarray, level: float) -> np.ndarray:
    n = x.shape[0]
    out = np.zeros(n)
    for i in range(1, n):
        if x[i - 1] >= level and x[i] < level:
            out[i] = 1.0
    return out


@njit(cache=True)
def stoch_levels_short_event(k: np.ndarray, d: np.ndarray, level: float = 80.0) -> np.ndarray:
    """Обе линии были выше уровня, затем K пересекла уровень сверху вниз."""
    n = k.shape[0]
    out = np.zeros(n)
    for i in range(1, n):
        if k[i - 1] > level and d[i - 1] > level and k[i] <= level:
            out[i] = 1.0
    return out


@njit(cache=True)
def stoch_levels_long_event(k: np.ndarray, d: np.ndarray, level: float = 20.0) -> np.ndarray:
    """Обе линии были ниже уровня, затем K пересекла уровень снизу вверх."""
    n = k.shape[0]
    out = np.zeros(n)
    for i in range(1, n):
        if k[i - 1] < level and d[i - 1] < level and k[i] >= level:
            out[i] = 1.0
    return out


# ═══════════════════════════════════════════════════════════════════════
# МЕНЮ СЕРИЙ ДЛЯ ДВИЖКА
# ═══════════════════════════════════════════════════════════════════════

TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}

# Имя серии → (таймфрейм, функция-построитель над (ohlc, qvol) этого ТФ)
SERIES_MENU: dict[str, str] = {
    # 1m сигналы
    "RSI_LEVELS@1m": "1m",
    "CCI_LEVELS@1m": "1m",
    "STOCH_LEVELS@1m": "1m",
    "BBL@1m": "1m",
    "BBU@1m": "1m",
    "RSI@1m": "1m",
    "CCI@1m": "1m",
    "RSI_LEVELS_DN@1m": "1m",
    "CCI_LEVELS_DN@1m": "1m",
    "STOCH_LEVELS_DN@1m": "1m",
    # 5m
    "ATRP@5m": "5m",
    "MFI@5m": "5m",
    "BBL@5m": "5m",
    "KCL@5m": "5m",
    "TZU@5m": "5m",
    "RSI@5m": "5m",
    "CCI@5m": "5m",
    "CMO@5m": "5m",
    "WR@5m": "5m",
    "VOL@5m": "5m",
    "KCL4@5m": "5m",
    "BBL3@5m": "5m",
    "KCU@5m": "5m",
    "BBU@5m": "5m",
    "TZL@5m": "5m",
    "KCU4@5m": "5m",
    "BBU3@5m": "5m",
    # 15m
    "ADX@15m": "15m",
    "ATRP@15m": "15m",
    "RSI@15m": "15m",
    "CMO@15m": "15m",
    "MFI@15m": "15m",
    "KCL@15m": "15m",
    "TZU@15m": "15m",
    "TZL@15m": "15m",
    # 30m
    "CMO@30m": "30m",
    "WR@30m": "30m",
    "KCL@30m": "30m",
    "CCI@30m": "30m",
    "RSI@30m": "30m",
    "ADX@30m": "30m",
    "BBL@30m": "30m",
    "MFI@30m": "30m",
    "KCU@30m": "30m",
    # 1h
    "CCI@1h": "1h",
    "VOL@1h": "1h",
    "RSI@1h": "1h",
    "ADX@1h": "1h",
    "CMO@1h": "1h",
    "TZU@1h": "1h",
    # 4h
    "MACD_ZERO_DOWN@4h": "4h",
    "MACD_ZERO_UP@4h": "4h",
    "MACD@4h": "4h",
    "RSI@4h": "4h",
    "CCI@4h": "4h",
}

SERIES_NAMES: list[str] = list(SERIES_MENU.keys())
SERIES_INDEX: dict[str, int] = {name: i for i, name in enumerate(SERIES_NAMES)}


def _build_one(kind: str, ohlc: np.ndarray, qvol: np.ndarray, tz_amp: float) -> np.ndarray:
    close = ohlc[:, 3]
    if kind == "RSI_LEVELS":
        return cross_up_event(rsi(close, 14), 30.0)
    if kind == "CCI_LEVELS":
        return cross_up_event(cci(ohlc, 20), -100.0)
    if kind == "STOCH_LEVELS":
        k, d = stochastic(ohlc, 14, 1, 3)
        return stoch_levels_long_event(k, d, 20.0)
    if kind == "BBL":
        return bb_lower_dist(close, 20, 2.0)
    if kind == "BBU":
        return bb_upper_dist(close, 20, 2.0)
    if kind == "ATRP":
        return atr(ohlc, 14) / close * 100.0
    if kind == "MFI":
        return mfi(ohlc, qvol, 14)
    if kind == "KCL":
        return keltner_lower_dist(ohlc, 20, 2.0)
    if kind == "RSI_LEVELS_DN":
        return cross_down_event(rsi(close, 14), 70.0)
    if kind == "CCI_LEVELS_DN":
        return cross_down_event(cci(ohlc, 20), 100.0)
    if kind == "STOCH_LEVELS_DN":
        k, d = stochastic(ohlc, 14, 1, 3)
        return stoch_levels_short_event(k, d, 80.0)
    if kind == "KCU":
        return keltner_upper_dist(ohlc, 20, 2.0)
    if kind == "KCU4":
        return keltner_upper_dist(ohlc, 20, 4.0)
    if kind == "BBU3":
        return bb_upper_dist(close, 20, 3.0)
    if kind == "TZL":
        return turtle_lower_dist(ohlc, 200, tz_amp)
    if kind == "KCL4":
        return keltner_lower_dist(ohlc, 20, 4.0)
    if kind == "BBL3":
        return bb_lower_dist(close, 20, 3.0)
    if kind == "TZU":
        return turtle_upper_dist(ohlc, 200, tz_amp)
    if kind == "RSI":
        return rsi(close, 14)
    if kind == "CCI":
        return cci(ohlc, 20)
    if kind == "CMO":
        return cmo(close, 9)
    if kind == "WR":
        return williams_r(ohlc, 14)
    if kind == "VOL":
        return qvol.copy()
    if kind == "ADX":
        return adx(ohlc, 14, 14)
    if kind == "MACD":
        return macd_line(close, 12, 26)
    if kind == "MACD_ZERO_DOWN":
        return cross_down_event(macd_line(close, 12, 26), 0.0)
    if kind == "MACD_ZERO_UP":
        return cross_up_event(macd_line(close, 12, 26), 0.0)
    raise KeyError(kind)


def build_series(ohlc_1m: np.ndarray, qvol_1m: np.ndarray, tz_amp: float = 5.6) -> dict[str, np.ndarray]:
    """Считает все серии меню. Возвращает имя → float32 массив на собственном ТФ."""
    by_tf: dict[str, tuple[np.ndarray, np.ndarray]] = {"1m": (ohlc_1m, qvol_1m)}
    for tf, k in TF_MINUTES.items():
        if k > 1:
            by_tf[tf] = resample(ohlc_1m, qvol_1m, k)
    out: dict[str, np.ndarray] = {}
    for name, tf in SERIES_MENU.items():
        kind = name.split("@")[0]
        ohlc, qvol = by_tf[tf]
        vals = _build_one(kind, ohlc, qvol, tz_amp)
        out[name] = vals.astype(np.float32)
    return out
