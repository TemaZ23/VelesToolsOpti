"""
(a) Дельта-нейтральный cash-and-carry: лонг спот + шорт перп (равное количество монет) на монетах
с высоким ПРОШЛЫМ фандингом. Обратная сторона (шорт спота) не используется — нужен маржинальный заём.

    PYTHONPATH=. python3 -m scripts.edge_lab.funding_carry

Модель счёта:
- спот куплен полностью (кошелёк Spot), шорт перпа в кросс-марже кошелька USDT-M с маржой m * номинал;
  капитал = (1 + m) * номинал (APR считается на ВЕСЬ капитал, свободный кэш ничего не приносит);
- каждый день по open: если «виртуальная» маржа монеты < band_lo или > band_hi номинала — перебалансировка
  (продаём часть спота и откупаем столько же перпа или наоборот), издержки на объём сделок обеих ног;
- ликвидация (консервативно): дневные high ВСЕХ перпов одновременно; если кошелёк USDT-M с учётом
  худших цен <= mm * номинал — теряем весь кошелёк USDT-M, спот продаём по close дня;
- издержки 4 ноги: спот 0.1% (VIP0) + проскальзывание по объёму спота, перп cost_per_side(adv перпа);
- фандинг — fund_hold (выплаты в (open_k, open_k+1]), зачисляется в кошелёк USDT-M;
- «multi-asset» вариант для BTC/ETH: монета лежит в кошельке USDT-M как залог (Multi-Assets Mode),
  капитал = 1.05 * номинал, ликвидации нет (залог и шорт компенсируют друг друга) — оптимистичная граница;
  номинал возвращается к капитал/1.05 при уходе цены на ~15% (перебалансировка с издержками).
"""

from __future__ import annotations

import json
import time

import numba as nb
import numpy as np

from scripts.edge_lab.common import liquid_mask, per_year, rolling_mean_past
from scripts.edge_lab.funding import OUTD, SPOT_FEE, half_years, prep, save_json, slip_of, stats, summarize_configs


@nb.njit(cache=True)
def run_carry(select, rebal, SO, SC, FO, FC, FH, fhold, cs, cf, m, mm, band_lo, band_hi, kslots, liq_on):
    T, N = SO.shape
    q = np.zeros(N)
    M = np.zeros(N)
    F0 = np.zeros(N)
    cash = 1.0
    out = np.zeros((T, 9))  # ret, basis, fund, cost, turnover, gross(спот/капитал), liq events, n_pos, liq pnl
    trades = 0
    liqs = 0
    for t in range(T - 1):
        k = t + 1
        c = 0.0
        turn = 0.0
        eq = cash
        for i in range(N):
            if q[i] > 0:
                eq += q[i] * SO[k, i] + M[i] - q[i] * (FO[k, i] - F0[i])
        if eq <= 0:
            break
        if rebal[t]:
            for i in range(N):
                want = select[t, i] and not np.isnan(SO[k, i]) and not np.isnan(FO[k, i])
                if q[i] > 0 and not want:
                    vs = q[i] * SO[k, i]
                    vf = q[i] * FO[k, i]
                    c += vs * cs[t, i] + vf * cf[t, i]
                    turn += vs + vf
                    cash += vs + M[i] - q[i] * (FO[k, i] - F0[i])
                    q[i] = 0.0
                    M[i] = 0.0
            nsel = 0
            for i in range(N):
                if q[i] > 0:
                    nsel += 1
            for i in range(N):
                want = select[t, i] and not np.isnan(SO[k, i]) and not np.isnan(FO[k, i])
                if want and q[i] == 0.0 and nsel < kslots:
                    alloc = min(eq / kslots, cash)
                    if alloc <= 1e-9:
                        continue
                    notional = alloc / (1 + m)
                    q[i] = notional / SO[k, i]
                    M[i] = m * q[i] * FO[k, i]
                    F0[i] = FO[k, i]
                    cash -= q[i] * SO[k, i] + M[i]
                    c += q[i] * SO[k, i] * cs[t, i] + q[i] * FO[k, i] * cf[t, i]
                    turn += q[i] * (SO[k, i] + FO[k, i])
                    trades += 1
                    nsel += 1
        # контроль маржи по монете (перевод спот -> USDT-M через сокращение обеих ног и обратно)
        for i in range(N):
            if q[i] <= 0:
                continue
            wallet = M[i] - q[i] * (FO[k, i] - F0[i])
            ratio = wallet / (q[i] * FO[k, i])
            if ratio < band_lo or ratio > band_hi:
                E = q[i] * SO[k, i] + wallet
                q_new = E / (SO[k, i] + m * FO[k, i])
                dq = abs(q_new - q[i])
                c += dq * SO[k, i] * cs[t, i] + dq * FO[k, i] * cf[t, i]
                turn += dq * (SO[k, i] + FO[k, i])
                q[i] = q_new
                M[i] = m * q_new * FO[k, i]
                F0[i] = FO[k, i]
        # ликвидация всего кошелька USDT-M (кросс-маржа) по одновременным дневным high
        liquidated = False
        if liq_on:
            worst = 0.0
            maint = 0.0
            npos0 = 0
            for i in range(N):
                if q[i] <= 0:
                    continue
                npos0 += 1
                hi = FH[k, i] if not np.isnan(FH[k, i]) else FO[k, i]
                worst += M[i] - q[i] * (FO[k, i] - F0[i]) - q[i] * (hi - FO[k, i])
                maint += mm * q[i] * hi
            liquidated = npos0 > 0 and worst <= maint
        pr = 0.0
        fr = 0.0
        lp = 0.0
        gross = 0.0
        npos = 0
        nliq = 0
        for i in range(N):
            if q[i] <= 0:
                continue
            npos += 1
            gross += q[i] * SO[k, i]
            wallet = M[i] - q[i] * (FO[k, i] - F0[i])
            if liquidated:
                s_exit = SC[k, i] if not np.isnan(SC[k, i]) else SO[k, i]
                lp += q[i] * (s_exit - SO[k, i]) - wallet
                c += q[i] * s_exit * cs[t, i]
                turn += q[i] * s_exit
                cash += q[i] * s_exit
                q[i] = 0.0
                M[i] = 0.0
                nliq += 1
                continue
            s_next = SO[k + 1, i] if k + 1 < T else np.nan
            f_next = FO[k + 1, i] if k + 1 < T else np.nan
            fund = q[i] * FO[k, i] * fhold[k, i]
            fr += fund
            if np.isnan(s_next) or np.isnan(f_next):  # нога перестала торговаться — закрываем обе по close[k]
                s_exit = SC[k, i] if not np.isnan(SC[k, i]) else SO[k, i]
                f_exit = FC[k, i] if not np.isnan(FC[k, i]) else FO[k, i]
                pr += q[i] * (s_exit - SO[k, i]) - q[i] * (f_exit - FO[k, i])
                c += q[i] * s_exit * cs[t, i] + q[i] * f_exit * cf[t, i]
                turn += q[i] * (s_exit + f_exit)
                cash += q[i] * s_exit + wallet - q[i] * (f_exit - FO[k, i]) + fund
                q[i] = 0.0
                M[i] = 0.0
                continue
            pr += q[i] * (s_next - SO[k, i]) - q[i] * (f_next - FO[k, i])
            M[i] += fund
        if liquidated:
            liqs += 1
        cash -= c
        eq_new = cash
        for i in range(N):
            if q[i] > 0:
                eq_new += q[i] * SO[k + 1, i] + M[i] - q[i] * (FO[k + 1, i] - F0[i])
        out[k, 0] = eq_new / eq - 1
        out[k, 1] = pr / eq
        out[k, 2] = fr / eq
        out[k, 3] = c / eq
        out[k, 4] = turn / eq
        out[k, 5] = gross / eq
        out[k, 6] = nliq
        out[k, 7] = npos
        out[k, 8] = lp / eq
    return out, trades, liqs


def carry_inputs(d: dict):
    aux = d["aux"]
    s_ok = ~np.isnan(aux["spot_open"]) & (np.nan_to_num(aux["spot_qvol"]) > 0)  # без «зомби»-баров спота
    SO = np.where(s_ok, aux["spot_open"], np.nan)
    SC = np.where(s_ok, aux["spot_close"], np.nan)
    sadv = np.nan_to_num(rolling_mean_past(np.where(s_ok, aux["spot_qvol"], np.nan), 30))
    cs = SPOT_FEE + slip_of(sadv)
    with np.errstate(invalid="ignore", divide="ignore"):
        basis = d["C"] / SC - 1
    sane = np.isfinite(basis) & (np.abs(basis) < 0.05)  # защита от неверного маппинга спот/перп
    s_age = np.cumsum(s_ok, axis=0)
    return SO, SC, sadv, cs, d["cost"], sane, s_age, basis


def carry_select(sig, elig, K, entry, exit_):
    """Держим до K монет: вход при годовом трейлинг-фандинге >= entry (лучшие по рангу), выход при < exit_."""
    T, N = sig.shape
    sel = np.zeros((T, N), dtype=np.bool_)
    cur = np.zeros(N, dtype=np.bool_)
    ann = sig * 365
    for t in range(T):
        e = elig[t] & np.isfinite(ann[t])
        keep = cur & e & (ann[t] >= exit_)
        cand = np.where(e & ~keep & (ann[t] >= entry))[0]
        nfree = K - int(keep.sum())
        new = np.zeros(N, dtype=np.bool_)
        if nfree > 0 and len(cand):
            new[cand[np.argsort(-ann[t, cand], kind="stable")][:nfree]] = True
        cur = keep | new
        sel[t] = cur
    return sel


def main() -> None:
    t0 = time.time()
    d = prep()
    ts, T, syms = d["ts"], len(d["ts"]), d["syms"]
    N = len(syms)
    SO, SC, sadv, cs, cf, sane, s_age, basis = carry_inputs(d)
    FO, FC, fhold = d["O"], d["C"], d["fhold"]
    # Ликвидации Binance считаются по mark price: дневной high перпа (last) содержит «фитили» (BTC 2021-07-26
    # +36%, ETH 2020-03-13 +200%), которых не было в mark. Худшая цена дня = open перпа * (mark high / mark open);
    # если mark нет — high последней цены (консервативно).
    aux = d["aux"]
    with np.errstate(invalid="ignore", divide="ignore"):
        mk = aux["mark_high"] / aux["mark_open"]
    FH = np.where(np.isfinite(mk) & (mk > 0), FO * mk, d["H"])
    mm = 0.02

    def go(sel, rebal, K, m, cs_=cs, cf_=cf, liq_on=True, band=None):
        lo, hi = band if band else (0.75 * m, 2 * m)
        out, tr, lq = run_carry(sel, rebal, SO, SC, FO, FC, FH, fhold, cs_, cf_, m, mm, lo, hi, K, liq_on)
        st = stats(ts, out, tr, {"liquidation_days": int(lq)})
        st["avg_npos"] = round(float(out[:, 7].mean()), 2)
        yrs = len(ts) / 365.25
        st["full"]["ann_liq_pct"] = round(float(out[:, 8].sum() / yrs * 100), 2)
        return st, out

    res = {"assumptions": dict(spot_fee=SPOT_FEE, spot_slip="лестница common.cost_per_side по ADV спота",
                               perp_cost="common.cost_per_side(ADV перпа)", maint=mm, band="[0.75m, 2m]",
                               capital="(1+m)*номинал",
                               liquidation="кросс-маржа, одновременные дневные high mark price (или last, если mark нет)",
                               fund="fund_hold (выплаты (open_k, open_k+1])")}  # fmt: skip
    jb, je = syms.index("BTCUSDT"), syms.index("ETHUSDT")
    # --- статичный carry BTC / ETH
    static = {}
    for name, js in (("BTC", [jb]), ("ETH", [je]), ("BTC+ETH", [jb, je])):
        sel = np.zeros((T, N), dtype=np.bool_)
        sel[:, js] = True
        rebal = np.ones(T, dtype=bool)
        for mname, m, liq_on, band in (("m1/3", 1 / 3, True, None), ("m1/2", 0.5, True, None),
                                       ("multiasset", 0.05, False, (-0.10, 0.20))):  # fmt: skip
            st, out = go(sel, rebal, len(js), m, liq_on=liq_on, band=band)
            st["per_year"] = per_year(ts, out[:, 0])
            st["oos_half_years"] = half_years(ts, out[:, 0])
            static[f"{name}|{mname}"] = st
            np.savez(OUTD / f"carry_static_{name.replace('+', '_')}_{mname.replace('/', '')}.npz", ts=ts, ret=out[:, 0])
    res["static"] = static
    for k, v in static.items():
        print("static", k, "IS", v["is"]["cagr_pct"], v["is"]["sharpe"], "OOS", v["oos"]["cagr_pct"],
              v["oos"]["sharpe"], v["oos"]["max_dd_pct"], "liq", v["liquidation_days"], v["per_year"], flush=True)  # fmt: skip
    # --- ротация по трейлинг-фандингу (только монеты со спотом на Binance)
    rows = []
    elig_by_A = {}
    for A in (1e7, 5e7):
        el_p = liquid_mask(d["p"], min_age_bars=30, min_adv_usd=A)
        elig_by_A[A] = el_p & sane & (s_age >= 30) & (sadv >= A / 2)
    sel_cache = {}
    for L in (3, 7, 14, 30):
        sig = rolling_mean_past(d["fsig"], L)
        for A in (1e7, 5e7):
            for K in (1, 3, 5, 10):
                for entry in (0.1, 0.2, 0.4):
                    for xf in (0.0, 0.5):
                        sel = carry_select(sig, elig_by_A[A], K, entry, entry * xf)
                        sel_cache[(L, A, K, entry, xf)] = sel
                        for R in (1, 7):
                            rebal = (np.arange(T) % R) == 0
                            for m in (1 / 3, 0.5, 1.0):
                                st, _ = go(sel, rebal, K, m)
                                st["cfg"] = dict(L=L, A=A, K=K, entry=entry, exit=entry * xf, R=R, m=round(m, 3))
                                rows.append(st)
        print("carry L", L, len(rows), f"{time.time() - t0:.0f}s", flush=True)
    res["summary_all"] = summarize_configs(rows)
    res["summary_by_m"] = {str(mv): summarize_configs([r for r in rows if r["cfg"]["m"] == round(mv, 3)])
                           for mv in (1 / 3, 0.5, 1.0)}  # fmt: skip
    best = max(rows, key=lambda r: r["is"]["sharpe"])
    cfg = best["cfg"]
    xf = 0.0 if cfg["exit"] == 0 else 0.5
    sel = sel_cache[(cfg["L"], cfg["A"], cfg["K"], cfg["entry"], xf)]
    rebal = (np.arange(T) % cfg["R"]) == 0
    st, out = go(sel, rebal, cfg["K"], cfg["m"])
    st["per_year"] = per_year(ts, out[:, 0])
    st["oos_half_years"] = half_years(ts, out[:, 0])
    st["cfg"] = cfg
    st_mk, _ = go(sel, rebal, cfg["K"], cfg["m"], cs_=np.full_like(cs, SPOT_FEE), cf_=np.full_like(cf, 0.0002))
    st["maker_variant"] = {"is": st_mk["is"], "oos": st_mk["oos"]}
    capv = [0.01 * min(d["adv"][t, i], sadv[t, i]) / 24 for t in range(T) for i in np.where(sel[t])[0]]
    st["capacity_usd_per_leg_median"] = float(np.median(capv)) if capv else None
    st["capacity_usd_per_leg_p10"] = float(np.percentile(capv, 10)) if capv else None
    keys = ("L", "A", "K", "entry", "exit", "R", "m")
    st["neighbors"] = [{"cfg": r["cfg"], "is_sh": r["is"]["sharpe"], "oos_sh": r["oos"]["sharpe"],
                        "oos_cagr": r["oos"]["cagr_pct"]} for r in rows
                       if sum(r["cfg"][k] != cfg[k] for k in keys) == 1]  # fmt: skip
    res["best_is"] = st
    bc = max(rows, key=lambda r: r["is"]["cagr_pct"])
    res["best_is_cagr_ref"] = {"cfg": bc["cfg"], "is": bc["is"], "oos": bc["oos"]}
    # лучший по IS среди конфигов, у которых IS CAGR выше статичного BTC+ETH (m=1/3) — справочно
    ref = static["BTC+ETH|m1/3"]["is"]["cagr_pct"]
    beat = [r for r in rows if r["is"]["cagr_pct"] > ref]
    if beat:
        bb = max(beat, key=lambda r: r["is"]["sharpe"])
        res["best_is_beating_static"] = {"cfg": bb["cfg"], "is": bb["is"], "oos": bb["oos"], "n": len(beat),
                                         "summary": summarize_configs(beat)}  # fmt: skip
    res["top15_is"] = [{"cfg": r["cfg"], "is": r["is"], "oos": r["oos"], "liq": r["liquidation_days"]}
                       for r in sorted(rows, key=lambda r: -r["is"]["sharpe"])[:15]]  # fmt: skip
    res["configs"] = rows
    bsel = basis[sel & np.isfinite(basis)]
    res["basis_selected_pct"] = np.round(np.percentile(bsel * 100, [1, 5, 50, 95, 99]), 3).tolist() if len(bsel) else None
    with np.errstate(invalid="ignore", divide="ignore"):
        jump = FH / FO - 1
    js = jump[sel & np.isfinite(jump)]
    res["intraday_up_jump_selected"] = {f">{int(x * 100)}%": round(float(np.mean(js > x)), 4) for x in (0.1, 0.2, 0.3)}
    np.savez(OUTD / "carry_best_daily.npz", ts=ts, ret=out[:, 0], basis=out[:, 1], fund=out[:, 2], cost=out[:, 3],
             liq=out[:, 8])  # fmt: skip
    save_json("part_carry.json", res)
    print(json.dumps({"summary": res["summary_all"], "by_m": res["summary_by_m"], "best_cfg": cfg, "is": st["is"],
                      "oos": st["oos"], "per_year": st["per_year"], "halves": st["oos_half_years"],
                      "liq_days": st["liquidation_days"], "maker_oos": st["maker_variant"]["oos"],
                      "beating_static": res.get("best_is_beating_static", {}).get("cfg"),
                      "beating_static_oos": res.get("best_is_beating_static", {}).get("oos"),
                      "basis": res["basis_selected_pct"], "jumps": res["intraday_up_jump_selected"]},
                     indent=1, default=float))  # fmt: skip


if __name__ == "__main__":
    main()
