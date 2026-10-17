"""
Проверки устойчивости лучших кандидатов семейства funding (дописано после прерванного прогона).

    PYTHONPATH=. python3 -m scripts.edge_lab.funding_robust carry   # статичный carry BTC+ETH: ±25% параметров,
                                                                   # издержки, маржа, вкл/выкл по фандингу,
                                                                   # независимая проверка движка
    PYTHONPATH=. python3 -m scripts.edge_lab.funding_robust alt     # (c) лонг альтов при экстремально
                                                                   # отрицательном фандинге: концентрация P&L,
                                                                   # издержки ×1.5, без выплаты 00:00

Всё пишется в output/edge_lab/funding/part_robust_<cmd>.json; ничего из прежних частей не меняется.
"""

from __future__ import annotations

import json
import sys
import time

import numba as nb
import numpy as np

from scripts.edge_lab.common import DAY, IS, OOS, liquid_mask, load_funding, metrics, per_year, rolling_mean_past, window_idx
from scripts.edge_lab.funding import (OUTD, SPOT_FEE, half_years, prep, run_book, save_json, stats, summarize_configs,
                                      timing_weights, trailing_pct)  # fmt: skip
from scripts.edge_lab.funding_carry import carry_inputs, carry_select, run_carry


def brief(st: dict) -> dict:
    keys = ("cagr_pct", "sharpe", "max_dd_pct", "worst_month_pct", "pos_months_pct", "ann_fund_pct", "ann_price_pct",
            "ann_cost_pct")  # fmt: skip
    return {p: {k: st[p].get(k) for k in keys} for p in ("is", "oos") if p in st}


# --------------------------------------------------------------------------------------------- carry
def run_carry_checks(d: dict) -> None:
    t0 = time.time()
    ts, T, syms = d["ts"], len(d["ts"]), d["syms"]
    N = len(syms)
    SO, SC, sadv, cs, cf, sane, s_age, basis = carry_inputs(d)
    FO, FC, fhold = d["O"], d["C"], d["fhold"]
    aux = d["aux"]
    with np.errstate(invalid="ignore", divide="ignore"):
        mk = aux["mark_high"] / aux["mark_open"]
    # в архиве markPriceKlines есть пропуски (BTC 2021-07-26 — день фитиля last-цены до 48k): при пропуске берём
    # спотовый high (mark ≈ индекс ≈ спот), и только если нет спота — high последней цены перпа
    with np.errstate(invalid="ignore", divide="ignore"):
        sk = aux["spot_high"] / aux["spot_open"]
    FH_mark = np.where(np.isfinite(mk) & (mk > 0), FO * mk, np.where(np.isfinite(sk) & (sk > 0), FO * sk, d["H"]))
    FH_last = d["H"]
    res_missing = {s: int(np.sum(~np.isfinite(mk[:, syms.index(s)]) & d["alive"][:, syms.index(s)]))
                   for s in ("BTCUSDT", "ETHUSDT")}  # fmt: skip
    jb, je = syms.index("BTCUSDT"), syms.index("ETHUSDT")
    sel = np.zeros((T, N), dtype=np.bool_)
    sel[:, [jb, je]] = True
    always = np.ones(T, dtype=bool)

    def go(sel_, K, m, lo_f=0.75, hi_f=2.0, cmul=1.0, spot_fee=SPOT_FEE, mm=0.02, FH=FH_mark, liq_on=True,
           band=None, rebal=always):  # fmt: skip
        cs_ = (cs - SPOT_FEE + spot_fee) * cmul
        cf_ = cf * cmul
        lo, hi = band if band else (lo_f * m, hi_f * m)
        out, tr, lq = run_carry(sel_, rebal, SO, SC, FO, FC, FH, fhold, cs_, cf_, m, mm, lo, hi, K, liq_on)
        st = stats(ts, out, tr, {"liquidation_days": int(lq)})
        st["oos_half_years"] = half_years(ts, out[:, 0])
        st["per_year"] = per_year(ts, out[:, 0])
        st["band_rebalances"] = int(np.sum(out[:, 4] > 1e-9)) - 1  # дни с оборотом, кроме входа
        return st, out

    res = {"mark_missing_days": res_missing}
    print("mark missing days", res_missing, flush=True)
    base, out_base = go(sel, 2, 1 / 3)
    res["base"] = {"cfg": "BTC+ETH 50/50, m=1/3, band [0.75m, 2m], costs x1, spot fee 0.1%, mm 2%, liq by mark high (missing mark -> spot high)",
                   **brief(base), "per_year": base["per_year"], "oos_half_years": base["oos_half_years"],
                   "liq": base["liquidation_days"], "band_rebalances": base["band_rebalances"]}  # fmt: skip
    # --- ±25% и стресс-варианты
    grid = []
    variants = [
        ("m=0.25", dict(m=0.25)), ("m=0.3125", dict(m=0.3125)), ("m=0.375", dict(m=0.375)), ("m=0.417", dict(m=5 / 12)),
        ("m=0.5", dict(m=0.5)), ("m=0.625", dict(m=0.625)), ("m=1.0", dict(m=1.0)),
        ("m=0.5 band_lo x0.75", dict(m=0.5, lo_f=0.5625)), ("m=0.5 band_lo x1.25", dict(m=0.5, lo_f=0.9375)),
        ("m=0.5 band_hi x0.75", dict(m=0.5, hi_f=1.5)), ("m=0.5 band_hi x1.25", dict(m=0.5, hi_f=2.5)),
        ("m=0.5 costs x2", dict(m=0.5, cmul=2.0)), ("m=0.5 liq by LAST high", dict(m=0.5, FH=FH_last)),
        ("m=0.375 liq by LAST high", dict(m=0.375, FH=FH_last)),
        ("band_lo x0.75", dict(lo_f=0.5625)), ("band_lo x1.25", dict(lo_f=0.9375)),
        ("band_hi x0.75", dict(hi_f=1.5)), ("band_hi x1.25", dict(hi_f=2.5)),
        ("costs x1.5", dict(cmul=1.5)), ("costs x2", dict(cmul=2.0)), ("spot fee 0.075% (BNB)", dict(spot_fee=0.00075)),
        ("mm 4%", dict(mm=0.04)), ("liq by LAST-price high (фитили)", dict(FH=FH_last)),
        ("m=1/3 liq by LAST high", dict(FH=FH_last, m=1 / 3)), ("m=0.25 liq by LAST high", dict(FH=FH_last, m=0.25)),
        ("multiasset (залог в USDT-M)", dict(m=0.05, liq_on=False, band=(-0.10, 0.20))),
        ("multiasset band x1.25", dict(m=0.05, liq_on=False, band=(-0.125, 0.25))),
        ("multiasset band x0.75", dict(m=0.05, liq_on=False, band=(-0.075, 0.15))),
        ("weekly margin check", dict(rebal=(np.arange(T) % 7) == 0)),
    ]  # fmt: skip
    for name, kw in variants:
        m = kw.pop("m", 1 / 3)
        st, _ = go(sel, 2, m, **kw)
        grid.append({"variant": name, **brief(st), "liq": st["liquidation_days"], "per_year": st["per_year"],
                     "oos_half_years": st["oos_half_years"], "band_rebalances": st["band_rebalances"]})  # fmt: skip
        print(f"{name:34s} IS {st['is']['cagr_pct']:6.2f}% sh {st['is']['sharpe']:5.2f} | OOS {st['oos']['cagr_pct']:5.2f}% "
              f"sh {st['oos']['sharpe']:5.2f} dd {st['oos']['max_dd_pct']:5.2f} liq {st['liquidation_days']} "
              f"halves {list(st['oos_half_years'].values())}", flush=True)  # fmt: skip
    res["variants"] = grid
    # --- выбранный кандидат: m = 0.5 (минимальная маржа, при которой вся окрестность ±25% по m — 0.375..0.625 —
    # прошла IS без ликвидаций; m=1/3 отвергнута: у m=0.25 две ликвидации в IS)
    bst, bout = go(sel, 2, 0.5)
    oa, ob = window_idx(ts, OOS)
    trade_days = int(np.sum(bout[:, 4] > 1e-9))
    best = {"cfg": dict(coins="BTC+ETH 50/50", m=0.5, band=[0.375, 1.0], spot_fee=SPOT_FEE, mm=0.02),
            "is": bst["is"], "oos": bst["oos"], "full": bst["full"], "per_year": bst["per_year"],
            "oos_half_years": bst["oos_half_years"], "liq": bst["liquidation_days"], "trades_entries": bst["trades"],
            "days_with_trades": trade_days, "band_rebalances": bst["band_rebalances"]}  # fmt: skip
    res["best"] = best
    print("BEST m=0.5", brief(bst), bst["per_year"], bst["oos_half_years"], "rebal", bst["band_rebalances"], flush=True)
    a_, b_ = window_idx(ts, ("2020-01-01", "2026-10-01"))
    np.savez(OUTD / "robust_carry_m05_daily.npz", ts=ts[a_:b_], ret=bout[a_:b_, 0], basis=bout[a_:b_, 1],
             fund=bout[a_:b_, 2], cost=bout[a_:b_, 3])  # fmt: skip
    # --- BTC-only m=1/3: когда была ликвидация (почему кросс-маржа на 2 монеты важна)
    selb = np.zeros((T, N), dtype=np.bool_)
    selb[:, jb] = True
    stb, outb = go(selb, 1, 1 / 3)
    liq_days = [str(np.datetime64(int(ts[k]), "ms").astype("datetime64[D]")) for k in np.where(outb[:, 6] > 0)[0]]
    res["btc_only_m13_liquidations"] = {"days": liq_days, "liq_pnl_pct": [round(float(outb[k, 8]) * 100, 2)
                                                                         for k in np.where(outb[:, 6] > 0)[0]]}  # fmt: skip
    print("BTC-only m=1/3 liquidations:", res["btc_only_m13_liquidations"], flush=True)
    # --- максимальное внутридневное движение против шорта (mark high / open) — запас до ликвидации
    up = {}
    for s, j in (("BTC", jb), ("ETH", je)):
        x = mk[:, j]
        x = x[np.isfinite(x)]
        top = np.argsort(-np.where(np.isfinite(mk[:, j]), mk[:, j], 0))[:3]
        up[s] = {"max_mark_high_over_open_pct": round(float(x.max() - 1) * 100, 2),
                 "top_days": [str(np.datetime64(int(ts[k]), "ms").astype("datetime64[D]")) for k in top]}  # fmt: skip
    res["intraday_adverse_moves"] = up
    # --- вкл/выкл по трейлинг-фандингу (BTC/ETH), выбор по IS
    elig = np.zeros((T, N), dtype=np.bool_)
    elig[:, [jb, je]] = True
    rows = []
    for L in (3, 7, 14, 30):
        sig = rolling_mean_past(d["fsig"], L)
        for entry in (0.0, 0.03, 0.05, 0.08, 0.12):
            for xf in (0.0, 0.5, 1.0):
                if entry == 0.0 and xf > 0:
                    continue
                s_ = carry_select(sig, elig, 2, entry, entry * xf)
                st, _ = go(s_, 2, 1 / 3)
                rows.append({"cfg": dict(L=L, entry=entry, exit=round(entry * xf, 3)), "is": st["is"], "oos": st["oos"],
                             "trades": st["trades"], "liq": st["liquidation_days"], "per_year": st["per_year"],
                             "oos_half_years": st["oos_half_years"]})  # fmt: skip
    best = max(rows, key=lambda r: r["is"]["sharpe"])
    res["onoff"] = {"summary": summarize_configs(rows), "best_is": best,
                    "always_on_is_sharpe": base["is"]["sharpe"], "always_on_oos_sharpe": base["oos"]["sharpe"],
                    "configs": rows}  # fmt: skip
    print("on/off best IS", best["cfg"], best["is"]["cagr_pct"], best["is"]["sharpe"], "OOS", best["oos"]["cagr_pct"],
          best["oos"]["sharpe"], "| summary", res["onoff"]["summary"], flush=True)  # fmt: skip
    # --- независимая грубая проверка движка: доход ≈ (номинал/капитал) * (фандинг + изменение базиса) − издержки
    a, b = window_idx(ts, ("2020-01-01", "2026-10-01"))
    w = 0.5 / (1 + 1 / 3)  # номинал на монету / капитал
    simple = np.zeros(T)
    for j in (jb, je):
        with np.errstate(invalid="ignore", divide="ignore"):
            r_s = np.r_[SO[1:, j] / SO[:-1, j] - 1, 0.0]
            r_f = np.r_[FO[1:, j] / FO[:-1, j] - 1, 0.0]
        simple += w * (np.nan_to_num(r_s - r_f) + fhold[:, j])
    simple[1] -= 2 * w * (cs[0, jb] + cf[0, jb])  # вход (у движка — завышенные издержки первого дня)
    eng_cum = float(np.prod(1 + out_base[a:b, 0]) - 1)
    simple_cum = float(np.prod(1 + simple[a:b]) - 1)
    res["engine_check"] = {"engine_cum_pct": round(eng_cum * 100, 2), "simple_cum_pct": round(simple_cum * 100, 2),
                           "corr_daily": round(float(np.corrcoef(out_base[a:b, 0], simple[a:b])[0, 1]), 3),
                           "note": "simple: фиксированные доли без перебалансировки маржи и без её издержек"}  # fmt: skip
    print("engine check", res["engine_check"], flush=True)
    # --- скользящая годовая доходность (180 дней) — тренд выдыхания премии
    r = out_base[:, 0]
    roll = {}
    for k in range(180, T, 90):
        roll[str(np.datetime64(int(ts[k]), "ms").astype("datetime64[D]"))] = round(float((np.prod(1 + r[k - 180:k]) ** (365 / 180) - 1) * 100), 2)
    res["rolling_180d_apr_pct"] = roll
    # --- «обратный carry» для держателя монеты: заменить спот на лонг перпа, когда фандинг < 0 (идеальный тайминг —
    # верхняя граница), сколько это даёт в год на номинал
    rev = {}
    for s, j in (("BTC", jb), ("ETH", je)):
        f = load_funding(s + "USDT")
        yrs = f["ts"].astype("datetime64[ms]").astype("datetime64[Y]").astype(int) + 1970
        rev[s] = {str(y): round(float(-np.minimum(f["rate"][yrs == y], 0).sum()) * 100, 2) for y in np.unique(yrs)}
    res["reverse_carry_upper_bound_pct_per_year"] = rev
    print("reverse carry upper bound (perfect foresight, before 4 legs of costs):", rev, flush=True)
    # --- ёмкость: 1% среднего часового объёма (минимум из спота и перпа) по OOS
    oa, ob = window_idx(ts, OOS)
    cap = {}
    for s, j in (("BTC", jb), ("ETH", je)):
        v = 0.01 * np.minimum(d["adv"][oa:ob, j], sadv[oa:ob, j]) / 24
        cap[s] = {"median_usd": round(float(np.median(v))), "p5_usd": round(float(np.percentile(v, 5)))}
    res["capacity_per_leg"] = cap
    np.savez(OUTD / "robust_carry_base_daily.npz", ts=ts, ret=out_base[:, 0], basis=out_base[:, 1], fund=out_base[:, 2],
             cost=out_base[:, 3])  # fmt: skip
    save_json("part_robust_carry.json", res)
    print(f"done {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------------------------- alt
@nb.njit(cache=True)
def run_book_pc(target, rebal, ret, fhold, cost, tradable, stop):
    """Как funding.run_book (pess=False), но копит вклад каждой монеты: цена, фандинг, издержки (доли капитала)."""
    T, N = ret.shape
    w = np.zeros(N)
    pc = np.zeros((N, 3))
    daily = np.zeros(T)
    for t in range(T - 1):
        k = t + 1
        cvec = np.zeros(N)
        if rebal[t]:
            for i in range(N):
                tw = target[t, i]
                if np.isnan(tw) or not tradable[k, i]:
                    tw = 0.0
                dw = tw - w[i]
                if dw != 0.0:
                    cvec[i] += abs(dw) * cost[t, i]
                    w[i] = tw
        R = 0.0
        for i in range(N):
            if w[i] == 0.0 and cvec[i] == 0.0:
                continue
            r = ret[k, i]
            p_ = w[i] * r
            f_ = -w[i] * fhold[k, i]
            if w[i] != 0.0 and stop[k, i]:
                cvec[i] += abs(w[i] * (1 + r)) * cost[t, i]
            pc[i, 0] += p_
            pc[i, 1] += f_
            pc[i, 2] -= cvec[i]
            R += p_ + f_ - cvec[i]
        daily[k] = R
        for i in range(N):
            if w[i] == 0.0:
                continue
            if stop[k, i]:
                w[i] = 0.0
            else:
                w[i] = w[i] * (1 + ret[k, i]) / (1 + R) if 1 + R > 1e-9 else 0.0
    return daily, pc


def run_alt_checks(d: dict) -> None:
    t0 = time.time()
    ts, T, syms = d["ts"], len(d["ts"]), d["syms"]
    N = len(syms)
    jb, je = syms.index("BTCUSDT"), syms.index("ETHUSDT")
    alive = d["alive"]
    elig_alt = liquid_mask(d["p"], min_age_bars=60, min_adv_usd=2e7)
    elig_alt[:, [jb, je]] = False
    tm = json.loads((OUTD / "part_timing.json").read_text())
    cfg = tm["best_is_by_asset"]["ALT"]["cfg"]
    L, W = cfg["L"], cfg["W"]
    sig = rolling_mean_past(d["fsig"], L)
    pcts = {(L, W): trailing_pct(sig, W, alive)}
    w, _ = timing_weights(d, {L: sig}, pcts, elig_alt, cfg, jb, je)
    rebal = np.r_[True, np.any(w[1:] != w[:-1], axis=1)]
    res = {"cfg": cfg}
    daily, pc = run_book_pc(w, rebal, d["ret"], d["fhold"], d["cost"], alive, d["stop"])
    out, tr = run_book(w, rebal, d["ret"], d["fhold"], d["cost"], alive, d["stop"], d["true_del"], False)
    res["engine_match_max_abs_diff"] = float(np.max(np.abs(daily - out[:, 0])))
    # концентрация P&L по монетам в OOS: пересчёт по OOS-отрезку
    oa, ob = window_idx(ts, OOS)
    ia, ib = window_idx(ts, IS)
    for name, (a, b) in (("is", (ia, ib)), ("oos", (oa, ob))):
        tsel = np.zeros(T, dtype=bool)
        tsel[a:b] = True
        wsub = np.where(tsel[:, None], w, 0.0)
        rb = np.r_[True, np.any(wsub[1:] != wsub[:-1], axis=1)]
        dd, pcs = run_book_pc(wsub, rb, d["ret"], d["fhold"], d["cost"], alive, d["stop"])
        tot = pcs.sum(1)
        order = np.argsort(-tot)
        total = float(tot.sum())
        res[f"{name}_concentration"] = {
            "total_simple_sum_pct": round(total * 100, 1),
            "n_coins_traded": int(np.sum(np.abs(pcs).sum(1) > 0)),
            "top5": [{"sym": syms[i], "pnl_pct": round(float(tot[i]) * 100, 1), "price_pct": round(float(pcs[i, 0]) * 100, 1),
                      "fund_pct": round(float(pcs[i, 1]) * 100, 1)} for i in order[:5]],
            "worst5": [{"sym": syms[i], "pnl_pct": round(float(tot[i]) * 100, 1)} for i in order[::-1][:5]],
            "share_top5": round(float(tot[order[:5]].sum() / total), 3) if total != 0 else None,
            "sum_without_top5_pct": round(float(total - tot[order[:5]].sum()) * 100, 1),
            "sum_without_top10_pct": round(float(total - tot[order[:10]].sum()) * 100, 1),
        }  # fmt: skip
        print(name, json.dumps(res[f"{name}_concentration"], ensure_ascii=False), flush=True)
    # консервативно: позиция не получает выплату 00:00 (выход в первые секунды после неё нереалистичен)
    s00 = np.zeros((T, N))
    t0d = int(ts[0])
    for j, s in enumerate(syms):
        f = load_funding(s)
        if not len(f["ts"]):
            continue
        at0 = (f["ts"] % DAY) < 60_000
        b = (f["ts"][at0] - t0d) // DAY
        ok = (b >= 0) & (b < T)
        np.add.at(s00[:, j], b[ok].astype(np.int64), f["rate"][at0][ok])
    fhold_ex = d["fhold"] - np.where(alive, np.vstack([s00[1:], np.zeros((1, N))]), 0.0)
    variants = {}
    for name, fh, cm in (("base", d["fhold"], 1.0), ("costs x1.5", d["fhold"], 1.5), ("costs x2", d["fhold"], 2.0),
                         ("no 00:00 settlement", fhold_ex, 1.0), ("no 00:00 + costs x1.5", fhold_ex, 1.5)):  # fmt: skip
        o, trd = run_book(w, rebal, d["ret"], fh, d["cost"] * cm, alive, d["stop"], d["true_del"], False)
        op, _ = run_book(w, rebal, d["ret"], fh, d["cost"] * cm, alive, d["stop"], d["true_del"], True)
        st = stats(ts, o, trd)
        variants[name] = {**brief(st), "oos_half_years": half_years(ts, o[:, 0]),
                          "oos_pess_delist": {k: stats(ts, op, trd)["oos"][k] for k in ("cagr_pct", "sharpe", "max_dd_pct")}}  # fmt: skip
        print(f"{name:24s} IS {st['is']['cagr_pct']:7.2f} sh {st['is']['sharpe']:5.2f} | OOS {st['oos']['cagr_pct']:7.2f} "
              f"sh {st['oos']['sharpe']:5.2f} dd {st['oos']['max_dd_pct']:5.1f} pessOOS sh "
              f"{variants[name]['oos_pess_delist']['sharpe']}", flush=True)  # fmt: skip
    res["variants"] = variants
    # соседи ±: x (0.98/0.95), L (5/10 нет в сетке -> 3), W (365), H (3) — из part_timing
    keys = ("kind", "L", "W", "x", "H", "dir", "side")
    nb_rows = [r for r in tm["configs"] if r["cfg"]["asset"] == "ALT" and sum(r["cfg"][k] != cfg[k] for k in keys) == 1]
    res["neighbors"] = [{"cfg": r["cfg"], "is_sh": r["is"]["sharpe"], "oos_sh": r["oos"]["sharpe"],
                         "oos_cagr": r["oos"]["cagr_pct"], "oos_dd": r["oos"]["max_dd_pct"]} for r in nb_rows]  # fmt: skip
    for r in res["neighbors"]:
        print("nb", r)
    # доля дней с позицией и средняя экспозиция
    gross = np.abs(w).sum(1)
    res["exposure"] = {"days_with_position_share_oos": round(float(np.mean(gross[oa:ob] > 0)), 3),
                       "avg_gross_when_active_oos": round(float(gross[oa:ob][gross[oa:ob] > 0].mean()), 3)}  # fmt: skip
    save_json("part_robust_alt.json", res)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "carry"
    dd = prep()
    {"carry": run_carry_checks, "alt": run_alt_checks}[cmd](dd)
