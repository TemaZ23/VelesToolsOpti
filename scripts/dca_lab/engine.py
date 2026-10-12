"""
Numba-движок портфеля DCA-ботов Veles.

Симуляция синхронная по минутам для всех монет и всех стратегий сразу — это нужно,
чтобы честно учитывать блокировки, динамическую блокировку, общую кросс-маржу
(лонг- и шорт-боты в режиме хеджирования) и ликвидацию.

Порядок событий внутри минуты для бота в позиции:
    1) стоп по цене / по сигналу,
    2) тейк по цене (лимитный) / по сигналу (рыночный, на закрытии минуты),
    3) следующий усредняющий ордер: при сигнале — по рынку на закрытии минуты,
       если цена прошла минимальный отступ от цены входа; без условий — лимитный.
Для бота без позиции: проверка условий входа → лимитный первый ордер с отступом
(или рыночный при отступе 0) → ожидание исполнения или подтяжка.

Для шорта всё зеркально: ордера выше цены входа, тейк ниже средней, «худшая» цена минуты — high.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# Индексы общей статистики
S_FINAL_EQ = 0
S_REALIZED = 1
S_MAX_DD = 2
S_MIN_EQ_LOW = 3
S_LIQUIDATED = 4
S_LIQ_MINUTE = 5
S_DEALS = 6
S_TP = 7
S_SL = 8
S_MAX_CONC = 9
S_MAX_NOTIONAL = 10
S_MPU = 11
S_FEES = 12
S_FUNDING = 13
S_FULL_GRID = 14
S_PULLUPS = 15
S_AVG_DEAL_MIN = 16
S_SUM_WIN = 17
S_SUM_LOSS = 18
S_MAX_DD_LOW = 19
S_FORCED = 20
S_OPEN_UPNL = 21
S_OPEN_DEALS_END = 22
S_MAX_DD_ABS = 23  # max(пик эквити - эквити по худшим ценам) в USDT
N_STATS = 24

# Статистика по стратегиям
SS_REALIZED = 0
SS_DEALS = 1
SS_TP = 2
SS_SL = 3
SS_MAX_CONC = 4
SS_FULL_GRID = 5
N_SSTATS = 6

R_TP = 0
R_SL = 1
R_FORCED = 2

# fparams стратегии
F_PULL_UP = 0
F_TP_MIN = 1
F_TP_PCT = 2
F_SL_INDENT = 3
F_SL_PCT = 4
F_SHARE = 5
F_LEV = 6
N_F = 7
# iparams стратегии
I_N_ORDERS = 0
I_TP_MODE = 1
I_SL_MODE = 2
I_BLOCK = 3
I_DYN = 4
I_DIR = 5  # +1 лонг, -1 шорт
N_I = 6
# gparams портфеля
G_DEPOSIT = 0
G_MAKER = 1
G_TAKER = 2
G_FUNDING = 3
G_MMR = 4
G_COMPOUND = 5
N_G = 6


@njit(cache=True, inline="always")
def _eval_group(g, c, i, ser, ser_off, ser_len, ser_tf, cser, cop, cthr):
    for m in range(cser.shape[1]):
        s = cser[g, m]
        if s < 0:
            return True
        k = ser_tf[s]
        j = (i + 1) // k - 1
        if j < 0 or j >= ser_len[c, s]:
            return False
        v = ser[ser_off[c, s] + j]
        if cop[g, m] == 0:
            if not (v < cthr[g, m]):
                return False
        else:
            if not (v > cthr[g, m]):
                return False
    return True


@njit(cache=True, nogil=True)
def simulate(
    px,  # float32[N, 3]: high, low, close
    coin_off,  # int64[C]
    coin_len,  # int64[C]
    coin_start,  # int64[C] глобальная минута первой свечи монеты
    masks,  # uint8[S, C]
    ser,  # float32[...]
    ser_off,  # int64[C, K]
    ser_len,  # int64[C, K]
    ser_tf,  # int64[K]
    cser,  # int32[S, G, M]
    cop,  # int8[S, G, M]
    cthr,  # float32[S, G, M]
    indent,  # float64[S, MAX_ORDERS]
    volume,  # float64[S, MAX_ORDERS]
    fparams,  # float64[S, N_F]
    iparams,  # int64[S, N_I]
    gparams,  # float64[N_G]
    t_from,
    t_to,
    max_deals,
):
    n_coins = coin_off.shape[0]
    n_strat = cser.shape[0]
    deposit = gparams[G_DEPOSIT]
    maker = gparams[G_MAKER]
    taker = gparams[G_TAKER]
    funding = gparams[G_FUNDING]
    mmr = gparams[G_MMR]
    compound = gparams[G_COMPOUND] > 0.5
    g_tp = cser.shape[1] - 2
    g_sl = cser.shape[1] - 1
    n_ord_max = indent.shape[1]

    # есть ли у ордера условия (иначе — лимитный)
    has_cond = np.zeros((n_strat, n_ord_max), dtype=np.bool_)
    for s in range(n_strat):
        for o in range(1, iparams[s, I_N_ORDERS]):
            has_cond[s, o] = cser[s, o, 0] >= 0

    shape = (n_strat, n_coins)
    state = np.zeros(shape, dtype=np.int8)  # 0 — ждёт сигнал, 1 — висит 1-й ордер, 2 — в позиции
    nf = np.zeros(shape, dtype=np.int64)
    qty = np.zeros(shape)
    cost = np.zeros(shape)
    base_px = np.zeros(shape)
    sig_px = np.zeros(shape)
    lim_px = np.zeros(shape)
    bot_dep = np.zeros(shape)
    t_open = np.zeros(shape, dtype=np.int64)
    tagged = np.zeros(shape, dtype=np.bool_)
    max_not = np.zeros(shape)
    deal_fees = np.zeros(shape)
    deal_mae = np.zeros(shape)

    active = np.zeros(n_strat, dtype=np.int64)
    dyn_reduce = np.zeros(n_strat, dtype=np.int64)
    sstats = np.zeros((n_strat, N_SSTATS))

    n_days = (t_to - t_from + 1439) // 1440
    daily = np.zeros((n_days, 4))  # equity close, equity min (худшие цены), open deals, notional
    coin_pnl = np.zeros(shape)
    stats = np.zeros(N_STATS)

    nd = max(max_deals, 1)
    d_strat = np.zeros(nd, dtype=np.int32)
    d_coin = np.zeros(nd, dtype=np.int32)
    d_t0 = np.zeros(nd, dtype=np.int64)
    d_t1 = np.zeros(nd, dtype=np.int64)
    d_nf = np.zeros(nd, dtype=np.int32)
    d_pnl = np.zeros(nd)
    d_reason = np.zeros(nd, dtype=np.int8)
    d_maxnot = np.zeros(nd)
    d_mae = np.zeros(nd)
    n_rec = 0

    realized = 0.0
    fees_total = 0.0
    funding_total = 0.0
    peak = deposit
    max_dd = 0.0
    max_dd_low = 0.0
    max_dd_abs = 0.0
    min_eq_low = deposit
    mpu = 0.0
    max_conc = 0
    max_notional = 0.0
    n_deals = 0
    n_tp = 0
    n_sl = 0
    n_full = 0
    n_pull = 0
    n_forced = 0
    sum_win = 0.0
    sum_loss = 0.0
    deal_minutes = 0.0
    liquidated = 0
    liq_minute = -1
    day_min = deposit

    for t in range(t_from, t_to):
        upnl = 0.0
        upnl_low = 0.0
        notional = 0.0
        is_funding = (t % 480) == 0
        for s in range(n_strat):
            pull_up = fparams[s, F_PULL_UP]
            tp_min = fparams[s, F_TP_MIN]
            tp_pct = fparams[s, F_TP_PCT]
            sl_indent = fparams[s, F_SL_INDENT]
            sl_pct = fparams[s, F_SL_PCT]
            share = fparams[s, F_SHARE]
            lev = fparams[s, F_LEV]
            n_orders = iparams[s, I_N_ORDERS]
            tp_mode = iparams[s, I_TP_MODE]
            sl_mode = iparams[s, I_SL_MODE]
            block = iparams[s, I_BLOCK]
            dyn_order = iparams[s, I_DYN]
            dr = float(iparams[s, I_DIR])  # +1 / -1
            long = dr > 0
            cs = cser[s]
            co = cop[s]
            ct = cthr[s]
            for c in range(n_coins):
                if masks[s, c] == 0:
                    continue
                i = t - coin_start[c]
                if i < 0 or i >= coin_len[c]:
                    continue
                row = coin_off[c] + i
                hi = px[row, 0]
                lo = px[row, 1]
                cl = px[row, 2]
                # благоприятная / неблагоприятная цена минуты для направления
                fav = hi if long else lo
                adv = lo if long else hi
                st = state[s, c]
                last_minute = i == coin_len[c] - 1

                if st == 0:
                    if last_minute:
                        continue
                    if active[s] < block - dyn_reduce[s]:
                        if _eval_group(0, c, i, ser, ser_off, ser_len, ser_tf, cs, co, ct):
                            if compound:
                                bot_dep[s, c] = (deposit + realized) * share
                            else:
                                bot_dep[s, c] = deposit * share
                            active[s] += 1
                            t_open[s, c] = t
                            deal_fees[s, c] = 0.0
                            deal_mae[s, c] = 0.0
                            if indent[s, 0] <= 0.0:
                                notion = bot_dep[s, c] * lev * volume[s, 0]
                                qty[s, c] = notion / cl
                                cost[s, c] = notion
                                fee = notion * taker
                                realized -= fee
                                fees_total += fee
                                deal_fees[s, c] += fee
                                base_px[s, c] = cl
                                nf[s, c] = 1
                                state[s, c] = 2
                                max_not[s, c] = notion
                                if dyn_order == 1:
                                    dyn_reduce[s] += 1
                                    tagged[s, c] = True
                            else:
                                sig_px[s, c] = cl
                                lim_px[s, c] = cl * (1.0 - dr * indent[s, 0])
                                state[s, c] = 1
                    continue

                if st == 1:
                    filled = (lo <= lim_px[s, c]) if long else (hi >= lim_px[s, c])
                    if filled:
                        p = lim_px[s, c]
                        notion = bot_dep[s, c] * lev * volume[s, 0]
                        qty[s, c] = notion / p
                        cost[s, c] = notion
                        fee = notion * maker
                        realized -= fee
                        fees_total += fee
                        deal_fees[s, c] += fee
                        base_px[s, c] = p
                        nf[s, c] = 1
                        state[s, c] = 2
                        max_not[s, c] = notion
                        if dyn_order == 1:
                            dyn_reduce[s] += 1
                            tagged[s, c] = True
                        upnl += dr * (qty[s, c] * cl - cost[s, c])
                        upnl_low += dr * (qty[s, c] * adv - cost[s, c])
                        notional += qty[s, c] * cl
                        continue
                    if long:
                        pulled = fav >= sig_px[s, c] * (1.0 + pull_up)
                    else:
                        pulled = fav <= sig_px[s, c] * (1.0 - pull_up)
                    if pulled or last_minute:
                        state[s, c] = 0
                        active[s] -= 1
                        n_pull += 1
                    continue

                # st == 2 — в позиции
                avg = cost[s, c] / qty[s, c]
                exit_px = -1.0
                exit_fee = taker
                reason = -1
                if last_minute:
                    exit_px = cl
                    reason = R_FORCED
                if reason < 0 and sl_mode == 2:
                    lvl = avg * (1.0 - dr * sl_pct)
                    if (adv <= lvl) if long else (adv >= lvl):
                        exit_px = lvl
                        reason = R_SL
                if reason < 0 and sl_mode == 1:
                    lvl = avg * (1.0 - dr * sl_indent)
                    if (cl < lvl) if long else (cl > lvl):
                        if _eval_group(g_sl, c, i, ser, ser_off, ser_len, ser_tf, cs, co, ct):
                            exit_px = cl
                            reason = R_SL
                if reason < 0 and (tp_mode == 1 or tp_mode == 2):
                    lvl = avg * (1.0 + dr * tp_pct)
                    if (fav >= lvl) if long else (fav <= lvl):
                        exit_px = lvl
                        exit_fee = maker
                        reason = R_TP
                if reason < 0 and (tp_mode == 0 or tp_mode == 2):
                    lvl = avg * (1.0 + dr * tp_min)
                    if (cl >= lvl) if long else (cl <= lvl):
                        if _eval_group(g_tp, c, i, ser, ser_off, ser_len, ser_tf, cs, co, ct):
                            exit_px = cl
                            reason = R_TP

                if reason >= 0:
                    proceeds = qty[s, c] * exit_px
                    fee = proceeds * exit_fee
                    pnl = dr * (proceeds - cost[s, c]) - fee
                    realized += pnl
                    fees_total += fee
                    deal_pnl = pnl - deal_fees[s, c]
                    coin_pnl[s, c] += deal_pnl
                    sstats[s, SS_REALIZED] += deal_pnl
                    sstats[s, SS_DEALS] += 1
                    n_deals += 1
                    deal_minutes += t - t_open[s, c]
                    if deal_pnl >= 0:
                        sum_win += deal_pnl
                    else:
                        sum_loss -= deal_pnl
                    if reason == R_TP:
                        n_tp += 1
                        sstats[s, SS_TP] += 1
                    elif reason == R_SL:
                        n_sl += 1
                        sstats[s, SS_SL] += 1
                    else:
                        n_forced += 1
                    if nf[s, c] >= n_orders:
                        n_full += 1
                        sstats[s, SS_FULL_GRID] += 1
                    if max_deals > 0 and n_rec < max_deals:
                        d_strat[n_rec] = s
                        d_coin[n_rec] = c
                        d_t0[n_rec] = t_open[s, c]
                        d_t1[n_rec] = t
                        d_nf[n_rec] = nf[s, c]
                        d_pnl[n_rec] = deal_pnl
                        d_reason[n_rec] = reason
                        d_maxnot[n_rec] = max_not[s, c]
                        d_mae[n_rec] = deal_mae[s, c]
                        n_rec += 1
                    qty[s, c] = 0.0
                    cost[s, c] = 0.0
                    nf[s, c] = 0
                    state[s, c] = 0
                    active[s] -= 1
                    if tagged[s, c]:
                        dyn_reduce[s] -= 1
                        tagged[s, c] = False
                    continue

                # усреднение
                o = nf[s, c]
                if o < n_orders:
                    lvl = base_px[s, c] * (1.0 - dr * indent[s, o])
                    buy_px = -1.0
                    buy_fee = taker
                    if has_cond[s, o]:
                        if (cl <= lvl) if long else (cl >= lvl):
                            if _eval_group(o, c, i, ser, ser_off, ser_len, ser_tf, cs, co, ct):
                                buy_px = cl
                    else:
                        if (adv <= lvl) if long else (adv >= lvl):
                            buy_px = lvl
                            buy_fee = maker
                    if buy_px > 0:
                        notion = bot_dep[s, c] * lev * volume[s, o]
                        qty[s, c] += notion / buy_px
                        cost[s, c] += notion
                        fee = notion * buy_fee
                        realized -= fee
                        fees_total += fee
                        deal_fees[s, c] += fee
                        nf[s, c] = o + 1
                        if dyn_order > 0 and nf[s, c] == dyn_order and not tagged[s, c]:
                            dyn_reduce[s] += 1
                            tagged[s, c] = True

                if is_funding:
                    f = dr * qty[s, c] * cl * funding  # лонг платит, шорт получает
                    realized -= f
                    funding_total += f
                    deal_fees[s, c] += f
                q = qty[s, c]
                upnl += dr * (q * cl - cost[s, c])
                ul = dr * (q * adv - cost[s, c])
                upnl_low += ul
                if ul < deal_mae[s, c]:
                    deal_mae[s, c] = ul
                v = q * cl
                notional += v
                if v > max_not[s, c]:
                    max_not[s, c] = v

        # ── портфельные метрики минуты ──
        eq = deposit + realized + upnl
        eq_low = deposit + realized + upnl_low
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd
        ddl = (peak - eq_low) / peak
        if ddl > max_dd_low:
            max_dd_low = ddl
        if peak - eq_low > max_dd_abs:
            max_dd_abs = peak - eq_low
        if eq_low < min_eq_low:
            min_eq_low = eq_low
        if -upnl_low > mpu:
            mpu = -upnl_low
        conc = 0
        for s in range(n_strat):
            conc += active[s]
            if active[s] > sstats[s, SS_MAX_CONC]:
                sstats[s, SS_MAX_CONC] = active[s]
        if conc > max_conc:
            max_conc = conc
        if notional > max_notional:
            max_notional = notional
        d = (t - t_from) // 1440
        if eq_low < day_min:
            day_min = eq_low
        if (t - t_from) % 1440 == 1439 or t == t_to - 1:
            daily[d, 0] = eq
            daily[d, 1] = day_min
            daily[d, 2] = conc
            daily[d, 3] = notional
            day_min = eq
        if eq_low <= mmr * notional:
            # ликвидация кросс-маржи: всё закрывается
            liquidated = 1
            liq_minute = t
            realized = -deposit
            for s in range(n_strat):
                for c in range(n_coins):
                    qty[s, c] = 0.0
                    cost[s, c] = 0.0
            for dd_i in range(d, n_days):
                daily[dd_i, 0] = 0.0
                daily[dd_i, 1] = 0.0
            break

    upnl_end = 0.0
    open_end = 0
    if liquidated == 0:
        for s in range(n_strat):
            dr = float(iparams[s, I_DIR])
            for c in range(n_coins):
                if qty[s, c] > 0:
                    i = min(t_to - 1 - coin_start[c], coin_len[c] - 1)
                    if i >= 0:
                        cl = px[coin_off[c] + i, 2]
                        upnl_end += dr * (qty[s, c] * cl - cost[s, c])
                        open_end += 1

    stats[S_FINAL_EQ] = deposit + realized + upnl_end if liquidated == 0 else 0.0
    stats[S_REALIZED] = realized
    stats[S_MAX_DD] = max_dd if liquidated == 0 else 1.0
    stats[S_MIN_EQ_LOW] = min_eq_low if liquidated == 0 else 0.0
    stats[S_LIQUIDATED] = liquidated
    stats[S_LIQ_MINUTE] = liq_minute
    stats[S_DEALS] = n_deals
    stats[S_TP] = n_tp
    stats[S_SL] = n_sl
    stats[S_MAX_CONC] = max_conc
    stats[S_MAX_NOTIONAL] = max_notional
    stats[S_MPU] = mpu
    stats[S_FEES] = fees_total
    stats[S_FUNDING] = funding_total
    stats[S_FULL_GRID] = n_full
    stats[S_PULLUPS] = n_pull
    stats[S_AVG_DEAL_MIN] = deal_minutes / n_deals if n_deals > 0 else 0.0
    stats[S_SUM_WIN] = sum_win
    stats[S_SUM_LOSS] = sum_loss
    stats[S_MAX_DD_LOW] = max_dd_low if liquidated == 0 else 1.0
    stats[S_FORCED] = n_forced
    stats[S_OPEN_UPNL] = upnl_end
    stats[S_OPEN_DEALS_END] = open_end
    stats[S_MAX_DD_ABS] = max_dd_abs if liquidated == 0 else deposit

    return (
        stats,
        daily,
        coin_pnl,
        sstats,
        d_strat[:n_rec],
        d_coin[:n_rec],
        d_t0[:n_rec],
        d_t1[:n_rec],
        d_nf[:n_rec],
        d_pnl[:n_rec],
        d_reason[:n_rec],
        d_maxnot[:n_rec],
        d_mae[:n_rec],
    )


@njit(cache=True, parallel=True)
def simulate_batch(
    px,
    coin_off,
    coin_len,
    coin_start,
    masks,  # uint8[B, S, C]
    ser,
    ser_off,
    ser_len,
    ser_tf,
    cser,  # [B, S, G, M]
    cop,
    cthr,
    indent,  # [B, S, MAX_ORDERS]
    volume,
    fparams,  # [B, S, N_F]
    iparams,  # [B, S, N_I]
    gparams,  # [B, N_G]
    t_from,
    t_to,
):
    b_count = cser.shape[0]
    n_days = (t_to - t_from + 1439) // 1440
    out_stats = np.zeros((b_count, N_STATS))
    out_daily = np.zeros((b_count, n_days, 4))
    out_sstats = np.zeros((b_count, cser.shape[1], N_SSTATS))
    for b in prange(b_count):
        res = simulate(
            px,
            coin_off,
            coin_len,
            coin_start,
            masks[b],
            ser,
            ser_off,
            ser_len,
            ser_tf,
            cser[b],
            cop[b],
            cthr[b],
            indent[b],
            volume[b],
            fparams[b],
            iparams[b],
            gparams[b],
            t_from,
            t_to,
            0,
        )
        out_stats[b] = res[0]
        out_daily[b] = res[1]
        out_sstats[b] = res[3]
    return out_stats, out_daily, out_sstats
