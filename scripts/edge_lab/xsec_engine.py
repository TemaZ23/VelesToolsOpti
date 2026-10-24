"""
xsec: движок кросс-секционного бэктеста (numba) и метрики.

Соглашения:
  W[t, i]  — целевые веса (доля капитала, + лонг, − шорт), рассчитанные ТОЛЬКО по данным бара t
             (закрытие t). Исполняются по open[t+1] панели исполнения и держатся бар t+1.
  Доходность позиции за бар s: open[s+1]/open[s]−1; если у монеты нет open[s+1] (делистинг/пропуск) —
             выход по close[s] (+ издержки выхода), в пессимистичном варианте −50% этой позиции.
  Фандинг:   fund_hold[s, i] — сумма ставок, которые платит лонг за удержание в баре s; P&L = −w·f.
  Издержки:  |Δw| · cost_per_side(ADV на момент сигнала) на каждом ребалансе.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from scripts.edge_lab.common import DAY, IS, OOS, cost_per_side, metrics, ms, per_year


# ----------------------------------------------------------------------------- portfolio construction
@njit(cache=True)
def ls_weights(S, valid, side_w, q, min_n, mode, kmax):
    """Кросс-секционный портфель по скору S (высокий скор → лонг).
    mode 0: лонг top-q / шорт bottom-q; mode 1: шорт top-q / лонг остальные;
    mode 2: только шорт top-q (лонг-нога пустая, хеджируется отдельно); mode 3: только лонг top-q.
    side_w — сырые веса внутри ноги (1 — равные, 1/σ — по волатильности). Каждая нога = 0.5 капитала.
    kmax > 0 ограничивает число монет в ноге."""
    T, N = S.shape
    W = np.zeros((T, N))
    idx = np.empty(N, np.int64)
    sc = np.empty(N)
    for t in range(T):
        n = 0
        for i in range(N):
            if valid[t, i] and np.isfinite(S[t, i]) and np.isfinite(side_w[t, i]) and side_w[t, i] > 0:
                idx[n] = i
                sc[n] = S[t, i]
                n += 1
        if n < min_n:
            continue
        order = np.argsort(sc[:n])
        k = int(q * n)
        if k < 1:
            k = 1
        if kmax > 0 and k > kmax:
            k = kmax
        if mode == 0 or mode == 3:
            s = 0.0
            for j in range(n - k, n):
                s += side_w[t, idx[order[j]]]
            for j in range(n - k, n):
                W[t, idx[order[j]]] = 0.5 * side_w[t, idx[order[j]]] / s
        if mode == 0:
            s = 0.0
            for j in range(k):
                s += side_w[t, idx[order[j]]]
            for j in range(k):
                W[t, idx[order[j]]] = -0.5 * side_w[t, idx[order[j]]] / s
        if mode == 1 or mode == 2:
            s = 0.0
            for j in range(n - k, n):
                s += side_w[t, idx[order[j]]]
            for j in range(n - k, n):
                W[t, idx[order[j]]] = -0.5 * side_w[t, idx[order[j]]] / s
        if mode == 1:
            s = 0.0
            for j in range(n - k):
                s += side_w[t, idx[order[j]]]
            if s > 0:
                for j in range(n - k):
                    W[t, idx[order[j]]] = 0.5 * side_w[t, idx[order[j]]] / s
    return W


def overlap(W: np.ndarray, H: int) -> np.ndarray:
    """Перекрывающиеся портфели (Jegadeesh–Titman): среднее портфелей, сформированных за последние H баров."""
    if H <= 1:
        return W
    cs = np.cumsum(np.vstack([np.zeros((1, W.shape[1])), W]), axis=0)
    out = np.empty_like(W)
    out[H - 1 :] = (cs[H:] - cs[:-H]) / H
    for t in range(H - 1):
        out[t] = cs[t + 1] / H
    return out


def beta_hedge(W: np.ndarray, beta: np.ndarray, j_btc: int, renorm: bool = True) -> np.ndarray:
    """Добавляет позицию в BTC, обнуляющую суммарную бету портфеля к BTC (бета — по прошлым данным)."""
    b = np.where(np.isfinite(beta), np.clip(beta, 0.0, 3.0), 1.0)
    W = W.copy()
    net_beta = (W * b).sum(axis=1)
    W[:, j_btc] -= net_beta
    if renorm:
        g = np.abs(W).sum(axis=1)
        W = np.where(g[:, None] > 0, W / np.where(g > 0, g, 1)[:, None], 0.0)
    return W


# ----------------------------------------------------------------------------- backtest
@njit(cache=True)
def _bt(W, O, C, F, CPS, pess, band, tmask):
    T, N = W.shape
    w = np.zeros(N)  # weights after drift at the start of bar s (before trading)
    R = np.zeros(T)
    Rg = np.zeros(T)  # gross of costs (price + funding)
    Rp = np.zeros(T)  # price P&L
    Rf = np.zeros(T)  # funding P&L
    Rc = np.zeros(T)  # costs
    TO = np.zeros(T)  # one-way turnover sum |dw|
    G = np.zeros(T)  # gross exposure
    NL = np.zeros(T)
    NS = np.zeros(T)
    ENT = np.zeros(T)  # position entries (0 -> non-zero, or sign flip)
    DEAD = np.zeros(T)  # forced exits
    for s in range(1, T):
        cost = 0.0
        to = 0.0
        tgt = np.zeros(N)
        for i in range(N):
            ti = W[s - 1, i] if tmask[s] else w[i]
            if not np.isfinite(O[s, i]):
                ti = 0.0
            if band > 0.0 and ti != 0.0 and w[i] != 0.0 and abs(ti - w[i]) < band:
                ti = w[i]
            d = abs(ti - w[i])
            if d > 0.0:
                c = CPS[s - 1, i]
                if not np.isfinite(c):
                    c = 0.004 + 0.0005
                cost += d * c
                to += d
                if ti != 0.0 and (w[i] == 0.0 or ti * w[i] < 0.0):
                    ENT[s] += 1.0
            tgt[i] = ti
        pr = 0.0
        fr = 0.0
        pen = 0.0
        g = 0.0
        r = np.zeros(N)
        dead = np.zeros(N, np.bool_)
        for i in range(N):
            ti = tgt[i]
            if ti == 0.0:
                continue
            g += abs(ti)
            if ti > 0:
                NL[s] += 1.0
            else:
                NS[s] += 1.0
            if s + 1 < T and np.isfinite(O[s + 1, i]):
                r[i] = O[s + 1, i] / O[s, i] - 1.0
            else:
                r[i] = C[s, i] / O[s, i] - 1.0
                if s + 1 < T:
                    dead[i] = True
                    DEAD[s] += 1.0
                    c = CPS[s - 1, i]
                    if not np.isfinite(c):
                        c = 0.0045
                    cost += abs(ti) * c
                    to += abs(ti)
                    pen += pess * abs(ti)
            pr += ti * r[i]
            f = F[s, i]
            if np.isfinite(f):
                fr -= ti * f
        tot = pr + fr - cost - pen
        R[s] = tot
        Rg[s] = pr + fr
        Rp[s] = pr
        Rf[s] = fr
        Rc[s] = cost + pen
        TO[s] = to
        G[s] = g
        den = 1.0 + tot
        if den <= 1e-9:
            den = 1e-9
        for i in range(N):
            if tgt[i] == 0.0 or dead[i]:
                w[i] = 0.0
            else:
                w[i] = tgt[i] * (1.0 + r[i]) / den
    return R, Rg, Rp, Rf, Rc, TO, G, NL, NS, ENT, DEAD


def backtest(W, ex: dict, pess: float = 0.0, band: float = 0.0, tmask=None) -> dict:
    """ex: панель исполнения с ключами open, close, fund_hold, cps (издержки на сторону по ADV), ts.
    tmask[s] = False — в баре s не торгуем (веса дрейфуют), кроме принудительных выходов."""
    if tmask is None:
        tmask = np.ones(len(W), dtype=np.bool_)
    out = _bt(np.ascontiguousarray(W, dtype=np.float64), ex["open"], ex["close"], ex["fund_hold"], ex["cps"],
              float(pess), float(band), np.asarray(tmask, dtype=np.bool_))  # fmt: skip
    keys = ("ret", "gross", "price", "fund", "cost", "turnover", "exposure", "n_long", "n_short", "entries", "dead")
    return dict(zip(keys, out))


# ----------------------------------------------------------------------------- metrics
def to_daily(ts: np.ndarray, ret: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Сворачивает бары в сутки (UTC) по компаундингу."""
    d = ts // DAY
    u, inv = np.unique(d, return_inverse=True)
    lr = np.log1p(np.maximum(np.nan_to_num(ret), -0.999999))
    out = np.zeros(len(u))
    np.add.at(out, inv, lr)
    return u * DAY, np.expm1(out)


def tstat(x: np.ndarray) -> float:
    x = np.nan_to_num(x)
    s = x.std(ddof=1)
    return float(x.mean() / s * np.sqrt(len(x))) if s > 0 else 0.0


def window_stats(ts, bt: dict, window, bars_per_day: float = 1.0) -> dict:
    i0, i1 = int(np.searchsorted(ts, ms(window[0]))), int(np.searchsorted(ts, ms(window[1])))
    t, r = ts[i0:i1], bt["ret"][i0:i1]
    dts, dr = to_daily(t, r)
    m = metrics(dts, dr, int(bt["entries"][i0:i1].sum()))
    g = bt["gross"][i0:i1]
    dts_g, dg = to_daily(t, g)
    mg = metrics(dts_g, dg)
    expo = bt["exposure"][i0:i1]
    act = expo > 0
    to = bt["turnover"][i0:i1]
    m.update(
        gross_cagr_pct=mg["cagr_pct"],
        gross_sharpe=mg["sharpe"],
        tstat=round(tstat(dr), 2),
        ann_ret_pct=round(float(np.mean(dr)) * 365 * 100, 2),
        ann_gross_pct=round(float(np.mean(dg)) * 365 * 100, 2),
        ann_price_pct=round(float(np.sum(bt["price"][i0:i1])) / max(len(dr), 1) * 365 * 100, 2),
        ann_fund_pct=round(float(np.sum(bt["fund"][i0:i1])) / max(len(dr), 1) * 365 * 100, 2),
        ann_cost_pct=round(float(np.sum(bt["cost"][i0:i1])) / max(len(dr), 1) * 365 * 100, 2),
        turnover_per_day=round(float(np.sum(to)) / max(len(dr), 1), 4),
        avg_exposure=round(float(np.mean(expo[act])) if act.any() else 0.0, 3),
        active_share=round(float(np.mean(act)), 3),
        avg_hold_days=round(float(2 * np.mean(expo[act]) / max(np.mean(to[act]), 1e-12) / bars_per_day) if act.any() else 0.0, 2),
        avg_positions=round(float(np.mean((bt["n_long"] + bt["n_short"])[i0:i1][act])) if act.any() else 0.0, 1),
        forced_exits=int(bt["dead"][i0:i1].sum()),
    )  # fmt: skip
    return m


def half_years(ts, ret, start="2024-01-01", end="2026-10-01") -> dict:
    out = {}
    y0, y1 = int(start[:4]), int(end[:4])
    for y in range(y0, y1 + 1):
        for h, (a, b) in enumerate((("01-01", "07-01"), ("07-01", "01-01"))):
            s = f"{y}-{a}"
            e = f"{y + 1}-{b}" if h == 1 else f"{y}-{b}"
            if ms(s) >= ms(end):
                continue
            i0, i1 = np.searchsorted(ts, ms(s)), np.searchsorted(ts, min(ms(e), ms(end)))
            if i1 > i0:
                out[f"{y}H{h + 1}"] = round((float(np.prod(1 + np.nan_to_num(ret[i0:i1]))) - 1) * 100, 2)
    return out


def summarize(ts, bt: dict, bars_per_day: float = 1.0) -> dict:
    dts, dr = to_daily(ts, bt["ret"])
    return {
        "IS": window_stats(ts, bt, IS, bars_per_day),
        "OOS": window_stats(ts, bt, OOS, bars_per_day),
        "per_year": per_year(dts, dr),
        "oos_half_years": half_years(dts, dr),
    }


@njit(cache=True)
def fill_gaps(O, C, max_gap):
    """Короткие пропуски данных внутри жизни монеты (≤ max_gap баров, напр. дыры в архиве Binance) —
    позиция не закрывается: open=close=последнее close, доходность за пропуск 0, скачок цены — на первом
    баре после пропуска. Длинные пропуски (делистинг/релистинг) остаются NaN → принудительный выход."""
    T, N = O.shape
    O2 = O.copy()
    C2 = C.copy()
    for i in range(N):
        last = -1
        for t in range(T):
            if np.isfinite(C[t, i]) and np.isfinite(O[t, i]):
                if last >= 0 and t - last - 1 > 0 and t - last - 1 <= max_gap:
                    for g in range(last + 1, t):
                        O2[g, i] = C[last, i]
                        C2[g, i] = C[last, i]
                last = t
    return O2, C2


def make_exec(d: dict, adv: np.ndarray, max_gap: int = 7) -> dict:
    """Панель исполнения: open/close/fund_hold из d (короткие дыры заполнены), издержки по ADV на момент сигнала."""
    O2, C2 = fill_gaps(d["open"], d["close"], max_gap)
    return {"ts": d["ts"], "open": O2, "close": C2, "fund_hold": d["fund_hold"],
            "cps": cost_per_side(np.nan_to_num(adv))}  # fmt: skip


def capacity(W: np.ndarray, adv: np.ndarray, ts: np.ndarray, window=OOS) -> dict:
    """Ёмкость: макс. капитал, при котором каждая позиция ≤ 1% среднего часового объёма монеты.
    Возвращает медиану и 10-й перцентиль по дням окна, а также номинал сделки в самой «узкой» монете."""
    i0, i1 = int(np.searchsorted(ts, ms(window[0]))), int(np.searchsorted(ts, ms(window[1])))
    Wa = np.abs(W[i0:i1])
    hv = np.nan_to_num(adv[i0:i1]) / 24.0
    with np.errstate(divide="ignore", invalid="ignore"):
        cap = np.where(Wa > 1e-6, 0.01 * hv / Wa, np.inf)
    eq = cap.min(axis=1)
    eq = eq[np.isfinite(eq)]
    if len(eq) == 0:
        return {}
    # min hourly volume among held coins
    hvm = np.where(Wa > 1e-6, hv, np.inf).min(axis=1)
    hvm = hvm[np.isfinite(hvm)]
    return {
        "equity_cap_median_usd": float(np.median(eq)),
        "equity_cap_p10_usd": float(np.percentile(eq, 10)),
        "max_trade_notional_usd_median": float(np.median(0.01 * hvm)),
        "max_trade_notional_usd_p10": float(np.percentile(0.01 * hvm, 10)),
        "max_weight_median": float(np.median(Wa.max(axis=1)[Wa.max(axis=1) > 0])) if (Wa.max(axis=1) > 0).any() else 0.0,
    }  # fmt: skip
