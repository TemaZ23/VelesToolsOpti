"""
Рыночный режим → множитель блокировки по дням (для страницы «Динамическая блокировка» Veles Tools).

Сигнал считается по ВЧЕРАШНЕМУ дневному закрытию BTCUSDT (без заглядывания вперёд):
    1.0, если close > SMA(N); иначе low_mult.
"""

from __future__ import annotations

import numpy as np

from .data import load_symbol
from .portfolio import GLOBAL_T0_MS

DAY_MS = 86_400_000


def btc_daily_close() -> tuple[np.ndarray, np.ndarray]:
    d = load_symbol("BTCUSDT")
    ts, close = d["ts"], d["ohlc"][:, 3]
    day = (ts // DAY_MS).astype(np.int64)
    last = np.concatenate([day[1:] != day[:-1], [True]])
    return day[last], close[last]


def sma_regime(n: int = 200, low_mult: float = 0.0, days_total: int = 3000) -> np.ndarray:
    """Множитель блокировки на каждый день от GLOBAL_T0."""
    days, close = btc_daily_close()
    sma = np.convolve(close, np.ones(n) / n, mode="full")[: len(close)]
    sma[: n - 1] = np.nan
    bull = close > sma  # на закрытии дня d
    g0 = GLOBAL_T0_MS // DAY_MS
    out = np.ones(days_total)
    for i in range(len(days)):
        gd = int(days[i] - g0) + 1  # действует со следующего дня
        if 0 <= gd < days_total and not np.isnan(sma[i]):
            out[gd] = 1.0 if bull[i] else low_mult
    return out


def drawdown_regime(window: int = 30, dd_limit: float = 20.0, low_mult: float = 0.3, days_total: int = 3000) -> np.ndarray:
    """Если BTC ниже максимума за window дней более чем на dd_limit% — блокировка × low_mult."""
    days, close = btc_daily_close()
    out = np.ones(days_total)
    g0 = GLOBAL_T0_MS // DAY_MS
    for i in range(len(days)):
        lo = max(0, i - window + 1)
        peak = close[lo : i + 1].max()
        gd = int(days[i] - g0) + 1
        if 0 <= gd < days_total:
            out[gd] = low_mult if close[i] < peak * (1 - dd_limit / 100) else 1.0
    return out


def schedule_from_spec(spec: dict | None, days_total: int = 3000) -> np.ndarray | None:
    """{"kind": "dd", "window": 30, "dd": 20, "low": 0.3} | {"kind": "sma", "n": 200, "low": 0} | None."""
    if not spec:
        return None
    if spec["kind"] == "dd":
        return drawdown_regime(spec["window"], spec["dd"], spec["low"], days_total)
    if spec["kind"] == "sma":
        return sma_regime(spec["n"], spec["low"], days_total)
    raise ValueError(spec)


def describe_spec(spec: dict | None) -> str:
    if not spec:
        return "нет"
    if spec["kind"] == "dd":
        return (
            f"если BTC закрыл день ниже максимума за {spec['window']} дн. более чем на {spec['dd']:g}% — "
            f"блокировка × {spec['low']:g} (до восстановления)"
        )
    return f"если BTC закрыл день ниже SMA{spec['n']} — блокировка × {spec['low']:g}"
