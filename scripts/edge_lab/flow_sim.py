"""
flow: событийный портфельный симулятор на часовых барах (numba).

Сигнал (t, j) — по закрытию бара t; вход по open[t+1] тейкером (cost[t, j] на сторону: taker + проскальзывание).
Позиции равного веса: номинал = w * капитал (mark-to-market на close[t]); не больше K одновременно; монета,
которая уже в портфеле, повторно не открывается. Приоритет сигналов одного бара — по порядку во входном массиве.

Выходы (проверяются на каждом баре начиная с бара входа t+1; ПЕССИМИСТИЧНО: стоп раньше тейка):
- стоп (stop > 0): для шорта уровень entry*(1+stop). Если open бара уже за уровнем (гэп) — исполнение по open,
  иначе по уровню; если high/low бара задел уровень — стоп сработал. Исполнение рыночное: taker + проскальзывание,
  проскальзывание стопа умножается на stop_slip_mult;
- тейк (tp > 0): лимитный ордер, исполняется только если цена прошла ЗА уровень (low < уровень для шорта);
  maker 0.02% при tp_maker=True, иначе taker+проскальзывание. Цена исполнения — уровень (даже при гэпе);
- по времени: close бара t_entry + hold - 1 (позиция держится hold часов), taker + проскальзывание;
- ликвидация шорта (изолированная маржа 1x): если цена выросла на liq_move (90%) от входа — потеря 100% номинала;
- монета перестала торговаться (close NaN): выход по последнему известному close, taker; дополнительно
  delist_pen * номинал убытка (пессимистичный вариант).
Фандинг: на каждом баре удержания pnl -= dir * funding[t,j] * qty * close[t]; выплата в баре входа (она совпадает
с моментом входа по open) учитывается только если она против позиции (консервативно); бар выхода — учитывается.
"""

from __future__ import annotations

import numpy as np
from numba import njit

MAKER = 0.0002
TAKER = 0.0005


@njit(cache=True)
def simulate(O, Hh, L, C, F, COST, sig_t, sig_j, direction, hold, stop, tp, tp_maker, w, K,
             delist_pen, stop_slip_mult, cooldown, t_start, t_end, liq_move=0.9):  # fmt: skip
    T, N = C.shape
    eq_curve = np.ones(T)
    cash = 1.0
    # позиции
    pj = np.full(K, -1, np.int64)
    pt0 = np.zeros(K, np.int64)
    pent = np.zeros(K)
    pqty = np.zeros(K)
    plast = np.zeros(K)
    pnot = np.zeros(K)
    held = np.zeros(N, np.bool_)
    next_ok = np.zeros(N, np.int64)  # кулдаун: следующая разрешённая сделка по монете
    # журнал сделок
    maxtr = len(sig_t) + 1
    tr_j = np.zeros(maxtr, np.int64)
    tr_t0 = np.zeros(maxtr, np.int64)
    tr_t1 = np.zeros(maxtr, np.int64)
    tr_ret = np.zeros(maxtr)  # ценовая доходность позиции (со знаком направления)
    tr_fund = np.zeros(maxtr)  # фандинг в долях номинала входа
    tr_cost = np.zeros(maxtr)  # издержки (вход+выход) в долях номинала входа
    tr_reason = np.zeros(maxtr, np.int64)  # 0 время, 1 стоп, 2 тейк, 3 делистинг, 4 конец, 5 ликвидация
    tr_not = np.zeros(maxtr)
    tr_fund_acc = np.zeros(K)
    tr_cost_acc = np.zeros(K)
    p_tr = np.full(K, -1, np.int64)
    ntr = 0
    k_sig = 0
    nsig = len(sig_t)
    # пропустить сигналы до начала
    while k_sig < nsig and sig_t[k_sig] < t_start - 1:
        k_sig += 1
    gross_sum = 0.0
    nbars = 0
    for t in range(t_start, t_end):
        # 1) обработка открытых позиций на баре t
        for s in range(K):
            j = pj[s]
            if j < 0:
                continue
            if pt0[s] > t:
                continue
            c = C[t, j]
            exit_px = -1.0
            reason = -1
            exit_cost = 0.0
            if not (c == c):
                # монета пропала: выход по последнему close
                exit_px = plast[s]
                reason = 3
                exit_cost = COST[t - 1, j] if t > 0 else COST[t, j]
                if not (exit_cost == exit_cost):
                    exit_cost = 0.0045
            else:
                o, hi, lo = O[t, j], Hh[t, j], L[t, j]
                if not (o == o):
                    o = c
                if not (hi == hi):
                    hi = max(o, c)
                if not (lo == lo):
                    lo = min(o, c)
                # фандинг бара
                f = F[t, j]
                if f == f and f != 0.0:
                    fp = -direction * f * pqty[s] * c
                    # выплата в баре входа приходится на момент открытия бара = момент входа: консервативно
                    # учитываем её, только если она против нас
                    if t > pt0[s] or fp < 0.0:
                        cash += fp
                        tr_fund_acc[s] += fp
                cst = COST[t, j]
                slip = cst - TAKER
                if stop > 0.0:
                    if direction < 0:
                        sp = pent[s] * (1.0 + stop)
                        if t > pt0[s] and o >= sp:
                            exit_px = o
                            reason = 1
                        elif hi >= sp:
                            exit_px = sp
                            reason = 1
                    else:
                        sp = pent[s] * (1.0 - stop)
                        if t > pt0[s] and o <= sp:
                            exit_px = o
                            reason = 1
                        elif lo <= sp:
                            exit_px = sp
                            reason = 1
                    if reason == 1:
                        exit_cost = TAKER + slip * stop_slip_mult
                # ликвидация шорта на изолированной марже 1x: рост на liq_move -> потеря всей маржи (= номинала)
                if reason < 0 and direction < 0 and liq_move > 0.0:
                    lp = pent[s] * (1.0 + liq_move)
                    if hi >= lp or (t > pt0[s] and o >= lp):
                        exit_px = pent[s] * 2.0  # убыток = 100% номинала
                        reason = 5
                        exit_cost = 0.0
                if reason < 0 and tp > 0.0:
                    if direction < 0:
                        tpp = pent[s] * (1.0 - tp)
                        if lo < tpp:
                            exit_px = tpp
                            reason = 2
                    else:
                        tpp = pent[s] * (1.0 + tp)
                        if hi > tpp:
                            exit_px = tpp
                            reason = 2
                    if reason == 2:
                        exit_cost = MAKER if tp_maker else cst
                if reason < 0 and t >= pt0[s] + hold - 1:
                    exit_px = c
                    reason = 0
                    exit_cost = cst
                if reason < 0 and t == t_end - 1:
                    exit_px = c
                    reason = 4
                    exit_cost = cst
                plast[s] = c
            if reason >= 0:
                pnl = direction * pqty[s] * (exit_px - pent[s])
                if direction < 0 and pnl < -pnot[s]:
                    pnl = -pnot[s]  # изолированная маржа 1x: убыток шорта не больше маржи
                notional_exit = pqty[s] * exit_px
                cst_val = exit_cost * notional_exit
                cash += pnl - cst_val
                if reason == 3 and delist_pen > 0.0:
                    cash -= delist_pen * pnot[s]
                    pnl -= delist_pen * pnot[s]
                k = p_tr[s]
                tr_t1[k] = t
                tr_ret[k] = pnl / pnot[s]
                tr_fund[k] = tr_fund_acc[s] / pnot[s]
                tr_cost[k] = (tr_cost_acc[s] + cst_val) / pnot[s]
                tr_reason[k] = reason
                held[j] = False
                next_ok[j] = t + 1 + cooldown
                pj[s] = -1
        # 2) капитал по close[t]
        unreal = 0.0
        gross = 0.0
        for s in range(K):
            j = pj[s]
            if j >= 0 and pt0[s] <= t:
                c = C[t, j]
                if c == c:
                    unreal += direction * pqty[s] * (c - pent[s])
                    gross += pqty[s] * c
                else:
                    unreal += direction * pqty[s] * (plast[s] - pent[s])
        eq = cash + unreal
        eq_curve[t] = eq
        if eq > 0:
            gross_sum += gross / eq
        nbars += 1
        if eq <= 0:
            for tt in range(t, t_end):
                eq_curve[tt] = 0.0
            break
        # 3) новые входы по сигналам бара t -> open[t+1]
        if t + 1 >= t_end - 1:
            continue
        while k_sig < nsig and sig_t[k_sig] < t:
            k_sig += 1
        while k_sig < nsig and sig_t[k_sig] == t:
            j = sig_j[k_sig]
            k_sig += 1
            if held[j] or t + 1 < next_ok[j]:
                continue
            slot = -1
            for s in range(K):
                if pj[s] < 0:
                    slot = s
                    break
            if slot < 0:
                continue
            px = O[t + 1, j]
            if not (px == px) or px <= 0:
                continue
            cst = COST[t, j]
            notional = w * eq
            pj[slot] = j
            pt0[slot] = t + 1
            pent[slot] = px
            pqty[slot] = notional / px
            plast[slot] = px
            pnot[slot] = notional
            held[j] = True
            cash -= cst * notional
            tr_cost_acc[slot] = cst * notional
            tr_fund_acc[slot] = 0.0
            tr_j[ntr] = j
            tr_t0[ntr] = t + 1
            tr_not[ntr] = notional
            p_tr[slot] = ntr
            ntr += 1
    avg_gross = gross_sum / max(nbars, 1)
    return (eq_curve, tr_j[:ntr], tr_t0[:ntr], tr_t1[:ntr], tr_ret[:ntr], tr_fund[:ntr], tr_cost[:ntr],
            tr_reason[:ntr], avg_gross)  # fmt: skip
