"""
flow: общие функции семейства «поток ордеров, всплески объёма, пампы и обвалы».

Соглашения (часовая панель, ts — ОТКРЫТИЕ бара):
- сигнал по бару t (его close, объём, taker-buy) -> вход по open[t+1];
- выход «по времени через h часов» = close[t+h] (≈ open[t+h+1]);
- фандинг: panel['funding'][b] — сумма ставок, выплаченных внутри бара b (выплата в момент открытия бара
  попадает в этот бар). Позиция, открытая по open[t+1] и закрытая по close[te], платит/получает выплаты баров
  t+1..te. В событийном исследовании (fwd_event) выплата в момент входа (открытие бара t+1) включается всегда;
  в портфельном симуляторе (flow_sim) — только если она против позиции. Лонг платит rate*notional, шорт получает;
- издержки: cost_per_side(ADV) на каждую сделку (taker + проскальзывание), ADV — за 30 прошлых полных суток;
- если монета перестаёт торговаться при открытой позиции — выход по последнему close
  (пессимистичный вариант — дополнительно −50% номинала позиции).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from scripts.edge_lab.common import DAY, IS, OOS, cost_per_side, metrics, ms, panel, per_year
from scripts.edge_lab.flow_data import load_adv, load_trades

try:
    from scripts.edge_lab.xsec_data import crypto_mask
except Exception:  # noqa: BLE001
    def crypto_mask(symbols):  # type: ignore[no-redef]
        return np.ones(len(symbols), bool)


def load_all(with_trades: bool = False) -> dict:
    p = panel("1h")
    p["adv"] = load_adv()
    if with_trades:
        p["trades"] = load_trades()
    p["crypto"] = crypto_mask(p["symbols"])
    return p


def cost_matrix(adv: np.ndarray, maker: bool = False) -> np.ndarray:
    """cost_per_side по матрице ADV (float32); NaN ADV -> самый дорогой уровень."""
    a = np.nan_to_num(adv, nan=0.0)
    out = np.full(a.shape, 0.004, dtype=np.float32)
    out[a >= 2e6] = 0.002
    out[a >= 1e7] = 0.001
    out[a >= 5e7] = 0.0005
    out[a >= 2e8] = 0.0002
    out += np.float32(0.0005 if not maker else 0.0)
    if maker:
        out[:] = 0.0002
    return out


@njit(cache=True)
def past_median_at(x, rows, cols, n, min_valid, w=1):
    """Медиана x[t-w-n+1 .. t-w, j] (только валидные > 0) для пар (rows[k], cols[k]) — n баров,
    закончившихся ДО окна события из w баров (t-w+1..t)."""
    out = np.full(len(rows), np.nan)
    buf = np.empty(n)
    for k in range(len(rows)):
        t, j = rows[k], cols[k]
        m = 0
        for s in range(max(0, t - w - n + 1), t - w + 1):
            v = x[s, j]
            if v == v and v > 0:
                buf[m] = v
                m += 1
        if m >= min_valid:
            out[k] = np.median(buf[:m])
    return out


@njit(cache=True)
def age_hours(close):
    T, N = close.shape
    age = np.zeros((T, N), np.int32)
    for j in range(N):
        a = 0
        for t in range(T):
            if close[t, j] == close[t, j]:
                a += 1
            age[t, j] = a
    return age


@njit(cache=True)
def fwd_event(open_, close, funding, rows, cols, hs):
    """Для событий (t, j): вход open[t+1], выход close[t+h] (или последний close, если монета пропала).
    Возвращает ret[k, ih] (доля), fund[k, ih] (сумма ставок за удержание), dead[k, ih] (выход по пропаже)."""
    K, H = len(rows), len(hs)
    T = open_.shape[0]
    ret = np.full((K, H), np.nan)
    fund = np.zeros((K, H))
    dead = np.zeros((K, H), np.bool_)
    for k in range(K):
        t, j = rows[k], cols[k]
        if t + 1 >= T:
            continue
        p0 = open_[t + 1, j]
        if not (p0 == p0) or p0 <= 0:
            continue
        hmax = hs[H - 1]
        last = p0
        f = 0.0
        ih = 0
        died = False
        for s in range(1, hmax + 1):
            tt = t + s
            if tt >= T:
                died = True
                break
            c = close[tt, j]
            if not (c == c):
                died = True
                break
            last = c
            f += funding[tt, j]
            while ih < H and hs[ih] == s:
                ret[k, ih] = last / p0 - 1.0
                fund[k, ih] = f
                ih += 1
        while ih < H:
            if t + 1 >= T:
                break
            ret[k, ih] = last / p0 - 1.0
            fund[k, ih] = f
            dead[k, ih] = died
            ih += 1
    return ret, fund, dead


def day_clustered_t(x: np.ndarray, day: np.ndarray) -> tuple[float, float, int]:
    """Среднее и t-статистика с кластеризацией по суткам (суточные суммы отклонений)."""
    ok = np.isfinite(x)
    x, day = x[ok], day[ok]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), n
    m = x.mean()
    u, inv = np.unique(day, return_inverse=True)
    s = np.bincount(inv, weights=x - m)
    se = np.sqrt(np.sum(s**2)) / n
    return float(m), float(m / se) if se > 0 else float("nan"), n


def split_masks(ts_ev: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "IS": (ts_ev >= ms(IS[0])) & (ts_ev < ms(IS[1])),
        "OOS": (ts_ev >= ms(OOS[0])) & (ts_ev < ms(OOS[1])),
    }


def daily_from_hourly(ts: np.ndarray, eq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Суточные доходности из часового ряда капитала (eq после бара t). ts — открытие часовых баров."""
    days = ts // DAY
    last = np.r_[days[1:] != days[:-1], True]
    deq = eq[last]
    dts = days[last] * DAY
    r = np.r_[deq[0] - 1.0, deq[1:] / deq[:-1] - 1.0]
    return dts.astype(np.int64), r


def summarize(dts: np.ndarray, dret: np.ndarray, trades: int | None = None) -> dict:
    out = {}
    for name, (a, b) in (("IS", IS), ("OOS", OOS), ("ALL", ("2020-01-01", "2026-10-01"))):
        m = (dts >= ms(a)) & (dts < ms(b))
        if m.sum() > 30:
            out[name] = metrics(dts[m], dret[m])
    out["per_year"] = per_year(dts, dret)
    out["half_years"] = half_years(dts, dret)
    if trades is not None:
        out["trades"] = int(trades)
    return out


def half_years(dts: np.ndarray, dret: np.ndarray) -> dict[str, float]:
    d = dts.astype("datetime64[ms]")
    y = d.astype("datetime64[Y]").astype(int) + 1970
    mth = d.astype("datetime64[M]").astype(int) % 12
    key = np.array([f"{a}H{1 if b < 6 else 2}" for a, b in zip(y, mth)])
    out = {}
    for k in np.unique(key):
        r = dret[key == k]
        out[str(k)] = round((float(np.prod(1 + r)) - 1) * 100, 2)
    return out


__all__ = [
    "load_all", "cost_matrix", "past_median_at", "age_hours", "fwd_event", "day_clustered_t", "split_masks",
    "daily_from_hourly", "summarize", "half_years", "cost_per_side",
]  # fmt: skip
