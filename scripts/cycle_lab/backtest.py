"""
«Циклы»: лонг с плечом в растущем режиме, шорт в падающем, обязательный стоп.

Дневные свечи спота Binance (scripts/cycle_lab/data.py), торговля как на фьючерсе USDT-M:
вход и выход по закрытию дня (сигнал считается по закрытию), комиссия тейкера + проскальзывание,
фандинг из истории Binance (до 2020 — 0.01% за 8 ч), кросс-маржа на всю сумму «рукава»,
ликвидация при падении маржи до поддерживающей (0.5%).

Стоп проверяется по минимуму/максимуму дня: если цена дошла до стопа — выход по цене стопа
с проскальзыванием (если день открылся уже за стопом — по открытию). Порядок экстремумов
внутри дня неизвестен, поэтому стоп считается сработавшим всегда, когда цена его касалась.

    PYTHONPATH=. python3 -m scripts.cycle_lab.backtest
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

from .data import load, load_funding

OUT = Path(__file__).resolve().parents[2] / "output" / "cycle_lab"
COINS = ["BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "LINK", "SOL", "AVAX"]
FEE = 0.0005  # тейкер Binance USDT-M
SLIP = 0.0005  # проскальзывание на вход/выход по рынку
STOP_SLIP = 0.002  # стоп-маркет в движении исполняется хуже
MM = 0.005  # поддерживающая маржа
DEFAULT_FUNDING = 0.0003  # 0.01% × 3 раза в сутки
DAY = 86_400_000

IS = ("2017-08-01", "2023-01-01")
OOS = ("2023-01-01", "2026-10-01")

# Вершины и дна BTC по закрытию (scripts/cycle_lab — поиск экстремумов в известных окнах).
ORACLE_TURNS = [
    ("2017-12-16", -1),
    ("2018-12-15", 1),
    ("2021-11-08", -1),
    ("2022-11-21", 1),
    ("2025-10-06", -1),
    ("2026-06-30", 1),
]


@dataclass
class Market:
    coin: str
    ts: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    funding: np.ndarray  # дневная ставка, доля номинала (+ платят лонги)


def market(coin: str) -> Market:
    d = load(f"{coin}USDT")
    ts = d["ts"]
    o, h, l, c = (d["ohlc"][:, k].astype(np.float64) for k in range(4))
    funding = np.full(len(ts), DEFAULT_FUNDING)
    f = load_funding(f"{coin}USDT")
    if f is not None:
        idx = np.searchsorted(ts, f["ts"])
        ok = (idx < len(ts)) & (ts[np.minimum(idx, len(ts) - 1)] == f["ts"])
        funding[idx[ok]] = f["rate"][ok]
    return Market(coin, ts, o, h, l, c, funding)


def day_of(s: str) -> int:
    return int(np.datetime64(s, "ms").astype(np.int64))


# ---------------------------------------------------------------- режимы (сигнал по закрытию дня)


def sma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    cs = np.cumsum(np.insert(x, 0, 0.0))
    out[n - 1 :] = (cs[n:] - cs[:-n]) / n
    return out


def ema(x: np.ndarray, n: int) -> np.ndarray:
    out = np.empty(len(x))
    a = 2.0 / (n + 1)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    out[: n - 1] = np.nan
    return out


def _sign(cond: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.where(valid, np.where(cond, 1, -1), 0).astype(np.int8)


def regime_sma200(m: Market) -> np.ndarray:
    s = sma(m.c, 200)
    return _sign(m.c > s, ~np.isnan(s))


def regime_cross(m: Market, fast: int = 50, slow: int = 200) -> np.ndarray:
    f, s = sma(m.c, fast), sma(m.c, slow)
    return _sign(f > s, ~np.isnan(s))


def regime_ema(m: Market, fast: int = 20, slow: int = 50) -> np.ndarray:
    f, s = ema(m.c, fast), ema(m.c, slow)
    return _sign(f > s, ~np.isnan(s))


def regime_tsmom(m: Market, n: int = 90) -> np.ndarray:
    out = np.zeros(len(m.c), dtype=np.int8)
    out[n:] = np.where(m.c[n:] > m.c[:-n], 1, -1)
    return out


def regime_donchian(m: Market, n: int = 55) -> np.ndarray:
    """Пробой n-дневного максимума — лонг, минимума — шорт; между пробоями режим держится."""
    out = np.zeros(len(m.c), dtype=np.int8)
    state = 0
    for i in range(n, len(m.c)):
        hi = m.h[i - n : i].max()
        lo = m.l[i - n : i].min()
        if m.c[i] > hi:
            state = 1
        elif m.c[i] < lo:
            state = -1
        out[i] = state
    return out


def regime_oracle(m: Market) -> np.ndarray:
    """Идеальное знание вершин и дна цикла BTC задним числом — верхняя граница идеи «торговать циклы»."""
    out = np.ones(len(m.c), dtype=np.int8)
    for date, state in ORACLE_TURNS:
        out[m.ts > day_of(date)] = state
    return out


REGIMES = {
    "sma200": regime_sma200,
    "cross50_200": regime_cross,
    "ema20_50": regime_ema,
    "tsmom90": regime_tsmom,
    "donchian55": regime_donchian,
    "oracle": regime_oracle,
}


# ---------------------------------------------------------------- исполнение


@njit(cache=True)
def _simulate(o, h, l, c, funding, sig, start, end, lev, stop, trailing, reentry, allow_short):
    """
    sig[i] — режим по закрытию дня i, позиция открывается по закрытию i и живёт с дня i+1.
    stop — доля от цены входа (или от лучшей цены при trailing); reentry: 0 — после стопа ждать
    смены режима, 1 — входить снова по закрытию дня, если режим тот же.
    Возвращает дневной капитал (1.0 на старте) и счётчики.
    """
    n = len(c)
    eq = np.ones(n)
    equity = 1.0
    pos = 0
    qty = 0.0
    entry = 0.0
    best = 0.0
    liq = 0.0
    blocked = 0  # режим, в котором сработал стоп (reentry == 0)
    trades = 0
    wins = 0
    stops = 0
    liqs = 0
    fees = 0.0
    fund = 0.0
    trade_eq = 0.0
    dead = False
    for i in range(start, end):
        if dead:
            eq[i] = 0.0
            continue
        if pos != 0:
            # стоп и ликвидация по экстремуму дня
            if trailing:
                sp = best * (1 - stop) if pos > 0 else best * (1 + stop)
            else:
                sp = entry * (1 - stop) if pos > 0 else entry * (1 + stop)
            hit_stop = (l[i] <= sp) if pos > 0 else (h[i] >= sp)
            stop_first = (sp > liq) if pos > 0 else (sp < liq)
            hit_liq = (l[i] <= liq) if pos > 0 else (h[i] >= liq)
            if hit_stop and stop_first:
                px = min(o[i], sp) * (1 - STOP_SLIP) if pos > 0 else max(o[i], sp) * (1 + STOP_SLIP)
                pnl = qty * pos * (px - entry)
                fee = qty * px * FEE
                equity += pnl - fee
                fees += fee
                trades += 1
                stops += 1
                if equity > trade_eq:
                    wins += 1
                blocked = pos
                pos = 0
                qty = 0.0
            elif hit_liq:
                equity = 0.0
                liqs += 1
                trades += 1
                pos = 0
                qty = 0.0
                dead = True
                eq[i] = 0.0
                continue
            if pos != 0:
                if pos > 0:
                    best = max(best, h[i])
                else:
                    best = min(best, l[i])
                f = funding[i] * qty * c[i] * pos
                equity -= f
                fund += f
        # решение по закрытию дня
        want = sig[i]
        if want < 0 and not allow_short:
            want = 0
        if reentry == 0 and blocked != 0:
            if want == blocked:
                want = 0
            else:
                blocked = 0
        mark = equity + (qty * pos * (c[i] - entry) if pos != 0 else 0.0)
        if want != pos:
            if pos != 0:
                px = c[i] * (1 - SLIP) if pos > 0 else c[i] * (1 + SLIP)
                pnl = qty * pos * (px - entry)
                fee = qty * px * FEE
                equity += pnl - fee
                fees += fee
                trades += 1
                if equity > trade_eq:
                    wins += 1
                pos = 0
                qty = 0.0
            if want != 0 and equity > 0:
                px = c[i] * (1 + SLIP) if want > 0 else c[i] * (1 - SLIP)
                qty = equity * lev / px
                fee = qty * px * FEE
                trade_eq = equity
                equity -= fee
                fees += fee
                pos = want
                entry = px
                best = px
                liq = px * (1 - 1 / lev) / (1 - MM) if pos > 0 else px * (1 + 1 / lev) / (1 + MM)
            mark = equity
        eq[i] = mark if mark > 0 else 0.0
        if eq[i] <= 1e-4:
            dead = True
    return eq, trades, wins, stops, liqs, fees, fund


def run(m: Market, sig: np.ndarray, window: tuple[str, str], lev: float, stop: float, trailing: bool = False,
        reentry: int = 1, allow_short: bool = True) -> dict:  # fmt: skip
    start = int(np.searchsorted(m.ts, day_of(window[0])))
    end = int(np.searchsorted(m.ts, day_of(window[1])))
    start = max(start, int(np.argmax(sig != 0)))  # первый день, когда режим определён
    if end - start < 60:
        return None
    eq, trades, wins, stops, liqs, fees, fund = _simulate(
        m.o, m.h, m.l, m.c, m.funding, sig, start, end, float(lev), float(stop), bool(trailing), int(reentry),
        bool(allow_short),
    )  # fmt: skip
    return stats(m.ts[start:end], eq[start:end], trades, wins, stops, liqs, fees, fund)


def monthly(ts: np.ndarray, eq: np.ndarray) -> np.ndarray:
    months = ts.astype("datetime64[ms]").astype("datetime64[M]")
    last = np.r_[months[1:] != months[:-1], True]
    ends = np.r_[1.0, eq[last]]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(ends[:-1] > 0, ends[1:] / ends[:-1] - 1, 0.0)
    return r


def stats(ts, eq, trades, wins, stops, liqs, fees, fund) -> dict:
    peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    dd = float(np.max(1 - eq / peak)) * 100
    mr = monthly(ts, eq)
    years = (ts[-1] - ts[0]) / DAY / 365.25
    final = float(eq[-1])
    cagr = (final ** (1 / years) - 1) * 100 if final > 0 else -100.0
    geo_month = (final ** (1 / max(len(mr), 1)) - 1) * 100 if final > 0 else -100.0
    ruin_day = int(np.argmax(eq <= 1e-4)) if np.any(eq <= 1e-4) else -1
    return {
        "final_x": round(final, 4),
        "cagr_pct": round(cagr, 1),
        "geo_month_pct": round(geo_month, 2),
        "avg_month_pct": round(float(mr.mean()) * 100, 2),
        "max_dd_pct": round(dd, 1),
        "worst_month_pct": round(float(mr.min()) * 100, 1),
        "neg_months_pct": round(float((mr < 0).mean()) * 100, 0),
        "trades": int(trades),
        "win_pct": round(wins / trades * 100, 0) if trades else 0.0,
        "stops": int(stops),
        "liquidated": bool(liqs),
        "ruined_at": str(ts[ruin_day].astype("datetime64[ms]").astype("datetime64[D]")) if ruin_day >= 0 else None,
        "fees_x": round(float(fees), 3),
        "funding_x": round(float(fund), 3),
        "years": round(years, 2),
    }
