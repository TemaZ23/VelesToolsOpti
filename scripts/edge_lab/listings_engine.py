"""
listings: движок (numba) — сделки «шорт нового листинга» по часовым окнам событий и портфель
с K слотами на глобальной часовой сетке.

Соглашения (часы — открытия баров, относительный индекс rel от бара листинга L):
- вход по open[rel=X] (если бара нет — первый живой бар в течение 24 ч, иначе сделки нет);
- стоп (шорт): если high[t] >= stop → выход по max(open[t], stop)·(1+STOP_EXTRA) — гэп через стоп
  исполняется по open; при одновременном касании стопа и тейка считается стоп;
- тейк (шорт): лимитная покупка pe·(1−TP), исполняется только если low[t] < цены лимита
  (цена прошла СКВОЗЬ лимит), по цене лимита, комиссия мейкера;
- выход по времени: open[X+H]; если бара нет — первый живой open за 24 ч;
- монета перестала торговаться (>= 24 ч без сделок / конец сегмента): выход по последнему close
  (xt=3), пессимистичный вариант — дополнительно −50% номинала позиции;
- фандинг: расчёт в момент открытия часа; вход «по open» происходит сразу ПОСЛЕ расчёта, поэтому
  ставка бара входа не учитывается, ставки баров (вход, выход] — учитываются (вкл. бар выхода);
  P&L = −q·rate·price (шорт получает при rate>0, платит при rate<0);
- издержки входа/выхода: TAKER + max(проскальзывание по ADV прошлых баров, пол по возрасту листинга).
"""

from __future__ import annotations

import heapq

import numpy as np
from numba import njit

TAKER = 0.0005
MAKER = 0.0002
STOP_EXTRA = 0.003  # дополнительное проскальзывание стоп-маркет ордера


@njit(cache=True)
def _slip(adv):
    if adv >= 2e8:
        return 0.0002
    if adv >= 5e7:
        return 0.0005
    if adv >= 1e7:
        return 0.001
    if adv >= 2e6:
        return 0.002
    return 0.004


@njit(cache=True)
def _age_floor(rel):
    if rel < 6:
        return 0.005
    if rel < 24:
        return 0.003
    if rel < 72:
        return 0.0015
    return 0.0


@njit(cache=True)
def _adv(qv, i, t):
    a = t - 720
    if a < 0:
        a = 0
    s = 0.0
    n = 0
    for k in range(a, t):
        v = qv[i, k]
        if v == v:
            s += v
            n += 1
    if n == 0:
        return 0.0
    return s / n * 24.0


@njit(cache=True)
def sim_trades(o, h, l, c, qv, f, Lg, NH, X, H, S, TP, cost_mult, direction):
    """direction −1 — шорт (основной случай), +1 — лонг (для симметрии / контроля).
    Возвращает массивы по событиям: ok, er, xr, pe, px, xt, ce, cx, ret (чистая доходность сделки
    в долях начального номинала, вкл. фандинг и издержки), fund (фандинг в долях номинала), adv_e."""
    E, W = o.shape
    ok = np.zeros(E, np.bool_)
    er = np.zeros(E, np.int64)
    xr = np.zeros(E, np.int64)
    pe = np.zeros(E)
    px = np.zeros(E)
    xt = np.zeros(E, np.int64)
    ce = np.zeros(E)
    cx = np.zeros(E)
    ret = np.zeros(E)
    fund = np.zeros(E)
    adv_e = np.zeros(E)
    for i in range(E):
        e = X
        while e < X + 24 and e < W and not (o[i, e] == o[i, e]):
            e += 1
        if e >= X + 24 or e >= W or Lg[i] + e >= NH:
            continue
        p0 = o[i, e]
        a0 = _adv(qv, i, e)
        c0 = TAKER + cost_mult * max(_slip(a0), _age_floor(e))
        if direction < 0:
            stop = p0 * (1 + S) if S > 0 else 1e300
            tpp = p0 * (1 - TP) if TP > 0 else -1.0
        else:
            stop = p0 * (1 - S) if S > 0 else -1.0
            tpp = p0 * (1 + TP) if TP > 0 else 1e300
        last_valid = e
        t = e
        nanrun = 0
        typ = -1
        xp = 0.0
        xi = e
        fsum = 0.0
        lastp = p0
        while True:
            if t >= e + H:
                u = t
                while u < t + 24 and u < W and Lg[i] + u < NH and not (o[i, u] == o[i, u]):
                    u += 1
                if u < W and Lg[i] + u < NH and o[i, u] == o[i, u] and u < t + 24:
                    for k in range(t, u + 1):
                        fsum += f[i, k] * (o[i, u] if k == u else lastp)
                    xp = o[i, u]
                    xi = u
                    typ = 0
                elif Lg[i] + u >= NH:
                    xp = c[i, last_valid]
                    xi = last_valid
                    typ = 4
                else:
                    xp = c[i, last_valid]
                    xi = last_valid
                    typ = 3
                break
            if Lg[i] + t >= NH:
                xp = c[i, last_valid]
                xi = last_valid
                typ = 4
                break
            if t >= W:
                xp = c[i, last_valid]
                xi = last_valid
                typ = 3
                break
            if not (o[i, t] == o[i, t]):
                nanrun += 1
                if nanrun >= 24:
                    xp = c[i, last_valid]
                    xi = last_valid
                    typ = 3
                    break
                t += 1
                continue
            nanrun = 0
            if t > e:
                fsum += f[i, t] * o[i, t]
            if direction < 0:
                if h[i, t] >= stop:
                    xp = max(o[i, t], stop) * (1 + STOP_EXTRA * cost_mult)
                    typ = 1
                elif l[i, t] < tpp:
                    xp = tpp
                    typ = 2
            else:
                if l[i, t] <= stop:
                    xp = min(o[i, t], stop) * (1 - STOP_EXTRA * cost_mult)
                    typ = 1
                elif h[i, t] > tpp:
                    xp = tpp
                    typ = 2
            if typ >= 0:
                xi = t
                break
            last_valid = t
            lastp = c[i, t]
            t += 1
        a1 = _adv(qv, i, xi)
        if typ == 2:
            c1 = MAKER
        else:
            c1 = TAKER + cost_mult * max(_slip(a1), _age_floor(xi))
        ok[i] = True
        er[i] = e
        xr[i] = xi
        pe[i] = p0
        px[i] = xp
        xt[i] = typ
        ce[i] = c0
        cx[i] = c1
        fund[i] = -direction * fsum / p0
        ret[i] = direction * (xp - p0) / p0 + fund[i] - c0 - c1 * xp / p0
        adv_e[i] = a0
    return ok, er, xr, pe, px, xt, ce, cx, ret, fund, adv_e


def accept_slots(ge: np.ndarray, gx: np.ndarray, K: int) -> np.ndarray:
    """Слоты: сделка принимается, если в момент входа открыто < K позиций (по времени, без капитала).
    Слот освобождается только если выход был строго раньше часа входа."""
    order = np.lexsort((np.arange(len(ge)), ge))
    acc = np.zeros(len(ge), bool)
    heap: list[int] = []
    for j in order:
        while heap and heap[0] < ge[j]:
            heapq.heappop(heap)
        if len(heap) < K:
            acc[j] = True
            heapq.heappush(heap, int(gx[j]))
    return acc


@njit(cache=True)
def portfolio(order, ev, Lg, er, xr, pe, px, xt, ce, cx, o, c, f, btc_o, btc_c, btc_f, NH, K, hedge, cb, pess, direction):
    """Капитал на часовой сетке. order — индексы принятых сделок, отсортированные по часу входа.
    Позиция = equity/K на момент входа (без ребалансировки). hedge=1 — лонг BTC тем же номиналом
    (для шорта), закрывается вместе со сделкой. Возвращает equity по часам и разбивку P&L."""
    n = len(order)
    eqh = np.empty(NH)
    eq = 1.0
    act = np.empty(K + 1, np.int64)
    q = np.zeros(K + 1)
    last = np.zeros(K + 1)
    qb = np.zeros(K + 1)
    lastb = np.zeros(K + 1)
    na = 0
    p = 0
    pnl_price = 0.0
    pnl_fund = 0.0
    pnl_cost = 0.0
    pnl_hedge = 0.0
    gross_sum = 0.0
    traded = 0.0
    n_exec = 0
    for g in range(NH):
        while p < n and Lg[ev[order[p]]] + er[order[p]] == g:
            j = order[p]
            if na < K:
                n_exec += 1
                notional = eq / K
                act[na] = j
                q[na] = direction * notional / pe[j]
                last[na] = pe[j]
                cst = ce[j] * notional
                eq -= cst
                pnl_cost -= cst
                traded += notional
                if hedge == 1:
                    qb[na] = -direction * notional / btc_o[g]
                    lastb[na] = btc_o[g]
                    cst = cb * notional
                    eq -= cst
                    pnl_cost -= cst
                    traded += notional
                else:
                    qb[na] = 0.0
                na += 1
            p += 1
        k = 0
        while k < na:
            j = act[k]
            i = ev[j]
            rel = g - Lg[i]
            if rel > er[j]:
                sp = o[i, rel] if o[i, rel] == o[i, rel] else last[k]
                d = -q[k] * f[i, rel] * sp
                eq += d
                pnl_fund += d
                if hedge == 1:
                    d = -qb[k] * btc_f[g] * btc_o[g]
                    eq += d
                    pnl_fund += d
            if rel == xr[j]:
                if xt[j] >= 3:
                    xp = c[i, rel]
                else:
                    xp = px[j]
                d = q[k] * (xp - last[k])
                eq += d
                pnl_price += d
                if pess == 1 and xt[j] == 3:
                    d = 0.5 * abs(q[k]) * xp
                    eq -= d
                    pnl_price -= d
                cst = cx[j] * abs(q[k]) * xp
                eq -= cst
                pnl_cost -= cst
                traded += abs(q[k]) * xp
                if hedge == 1:
                    bp = btc_o[g] if xt[j] == 0 else btc_c[g]
                    d = qb[k] * (bp - lastb[k])
                    eq += d
                    pnl_hedge += d
                    cst = cb * abs(qb[k]) * bp
                    eq -= cst
                    pnl_cost -= cst
                    traded += abs(qb[k]) * bp
                na -= 1
                act[k] = act[na]
                q[k] = q[na]
                last[k] = last[na]
                qb[k] = qb[na]
                lastb[k] = lastb[na]
                continue
            cc = c[i, rel]
            if cc == cc:
                d = q[k] * (cc - last[k])
                eq += d
                pnl_price += d
                last[k] = cc
            if hedge == 1:
                d = qb[k] * (btc_c[g] - lastb[k])
                eq += d
                pnl_hedge += d
                lastb[k] = btc_c[g]
            k += 1
        gs = 0.0
        for k in range(na):
            gs += abs(q[k]) * last[k]
        if eq > 0:
            gross_sum += gs / eq
        if eq <= 0:
            eq = 0.0
        eqh[g] = eq
    return eqh, pnl_price, pnl_fund, pnl_cost, pnl_hedge, gross_sum / NH, traded, n_exec
