"""
Семейство «listings»: эффекты нового листинга и жизненного цикла USDT-M перпетуалов Binance.

    PYTHONPATH=. python3 -m scripts.edge_lab.listings_data      # окна событий (один раз)
    PYTHONPATH=. python3 -m scripts.edge_lab.listings  a        # (a) event study
    PYTHONPATH=. python3 -m scripts.edge_lab.listings  b        # (b) сетка стратегий + контроль + робастность
    PYTHONPATH=. python3 -m scripts.edge_lab.listings_c         # (c) пред-делистинг и фактор возраста
    PYTHONPATH=. python3 -m scripts.edge_lab.listings  report   # results.json + best_daily.npz

Выход: output/edge_lab/listings/.
"""

from __future__ import annotations

import itertools
import json
import sys
import time

import numpy as np

from scripts.edge_lab.common import DAY, HOUR, IS, OOS, OUT, CACHE, metrics, ms, per_year
from scripts.edge_lab.listings_engine import TAKER, accept_slots, portfolio, sim_trades

ODIR = OUT / "listings"
EVF = CACHE / "listings" / "events.npz"
T0 = ms("2020-01-01")
NH = (ms("2026-10-01") - T0) // HOUR
NDAYS = NH // 24
DTS = T0 + np.arange(NDAYS, dtype=np.int64) * DAY
BTC_COST = TAKER + 0.0002  # ADV BTC >> $200 млн


def load(kind: str = "ev") -> dict:
    z = np.load(EVF, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    return d


def year_of(Lg: np.ndarray) -> np.ndarray:
    return (T0 + Lg * HOUR).astype("datetime64[ms]").astype("datetime64[Y]").astype(int) + 1970


def ffill_rows(a: np.ndarray) -> np.ndarray:
    idx = np.where(~np.isnan(a), np.arange(a.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return np.take_along_axis(a, idx, axis=1)


def _stats(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "mean": round(float(np.mean(x)) * 100, 2),
        "median": round(float(np.median(x)) * 100, 2),
        "p10": round(float(np.percentile(x, 10)) * 100, 1),
        "p90": round(float(np.percentile(x, 90)) * 100, 1),
        "neg_pct": round(float(np.mean(x < 0)) * 100, 1),
    }


def _month_t(x: np.ndarray, Lg: np.ndarray) -> float:
    """t-статистика среднего с кластеризацией по месяцу листинга (события одного месяца коррелированы)."""
    m = np.isfinite(x)
    mon = ((T0 + Lg[m] * HOUR).astype("datetime64[ms]").astype("datetime64[M]")).astype(int)
    u, inv = np.unique(mon, return_inverse=True)
    if len(u) < 3:
        return float("nan")
    xm = np.bincount(inv, weights=x[m]) / np.bincount(inv)
    return round(float(np.mean(xm) / (np.std(xm, ddof=1) / np.sqrt(len(xm)))), 2)


# ============================================================================ (a) event study
def event_study() -> dict:
    d = load()
    o, h, l, c, f = (d[f"ev_{k}"].astype(float) for k in ("o", "h", "l", "c", "f"))
    qv = d["ev_qv"].astype(float)
    Lg, cat = d["ev_L"], d["ev_cat"]
    bo = d["btc_o"]
    yr = year_of(Lg)
    cf = ffill_rows(c)
    po = np.where(np.isnan(o), cf, o)
    P0 = o[:, 1]  # open часа «листинг + 1 ч»
    res = {"n_events": {str(k): int(v) for k, v in zip(*np.unique(cat, return_counts=True))}}
    # первый час: доходность и диапазон
    crypto = cat == "crypto"
    res["listing_hour"] = {
        "ret_first_hour": _stats(c[crypto, 0] / o[crypto, 0] - 1),
        "range_first_hour": _stats(h[crypto, 0] / l[crypto, 0] - 1),
        "range_second_hour": _stats(h[crypto, 1] / l[crypto, 1] - 1),
        "P0_vs_first_print": _stats(P0[crypto] / o[crypto, 0] - 1),
        "qvol_first_hour_musd_median": round(float(np.nanmedian(qv[crypto, 0])) / 1e6, 2),
    }
    hz = [1, 3, 7, 14, 30, 60, 90]
    tabs = {}
    for group in ("crypto", "tradfi"):
        gm = cat == group
        rows = {}
        for dd in hz:
            e = 1 + 24 * dd
            valid = gm & np.isfinite(P0) & (Lg + e < NH)
            pe = np.where(np.isnan(o[:, e]), cf[:, e - 1], o[:, e])
            r = pe / P0 - 1
            br = bo[np.minimum(Lg + e, NH - 1)] / bo[Lg + 1] - 1
            fund_short = np.nansum(f[:, 2 : e + 1] * po[:, 2 : e + 1], axis=1) / P0  # что получает шорт (−: платит)
            mae = np.nanmax(h[:, 1:e], axis=1) / P0 - 1
            short_pnl = -r + fund_short
            row = {
                "ret": _stats(r[valid]),
                "ret_minus_btc": _stats((r - br)[valid]),
                "short_funding": _stats(fund_short[valid]),
                "short_pnl_gross": _stats(short_pnl[valid]),
                "short_pnl_t_month_cluster": _month_t(short_pnl[valid], Lg[valid]),
                "mae_short": _stats(mae[valid]),
                "mae_share_gt": {str(k): round(float(np.mean(mae[valid] > k)) * 100, 1) for k in (0.25, 0.5, 1.0, 2.0)},
                "dead_by_horizon_pct": round(float(np.mean(np.isnan(o[valid, e]))) * 100, 1),
            }
            if group == "crypto":
                row["by_year"] = {}
                for y in range(2020, 2027):
                    my = valid & (yr == y)
                    if my.sum() >= 5:
                        row["by_year"][str(y)] = {
                            "n": int(my.sum()),
                            "ret_mean": round(float(np.mean(r[my])) * 100, 1),
                            "ret_median": round(float(np.median(r[my])) * 100, 1),
                            "ex_btc_median": round(float(np.median((r - br)[my])) * 100, 1),
                            "short_funding_mean": round(float(np.mean(fund_short[my])) * 100, 2),
                            "short_pnl_mean": round(float(np.mean(short_pnl[my])) * 100, 1),
                            "mae_p90": round(float(np.percentile(mae[my], 90)) * 100, 0),
                        }
            rows[f"{dd}d"] = row
        tabs[group] = rows
    res["horizons"] = tabs
    # усреднённый путь цены (медиана и среднее по крипто-событиям), шаг — сутки
    path = {}
    for dd in range(0, 91, 1):
        e = 1 + 24 * dd
        pe = np.where(np.isnan(o[:, e]), cf[:, e - 1], o[:, e]) if dd > 0 else P0
        r = pe / P0 - 1
        valid = crypto & (Lg + e < NH)
        path[dd] = (round(float(np.median(r[valid])) * 100, 2), round(float(np.mean(r[valid])) * 100, 2))
    res["path_median_mean_pct"] = path
    # крупнейшие сквизы
    e = 1 + 24 * 30
    mae30 = np.nanmax(h[:, 1:e], axis=1) / P0 - 1
    top = np.argsort(-np.where(crypto, mae30, -1))[:15]
    res["top_squeezes_30d"] = [
        {"sym": str(d["ev_sym"][i]), "date": str((T0 + Lg[i] * HOUR).astype("datetime64[ms]"))[:10], "max_up_pct": round(float(mae30[i]) * 100)}
        for i in top
    ]  # fmt: skip
    return res


# ============================================================================ (b) стратегия
GRID = {
    "X": [1, 4, 12, 24, 72, 168],  # часов после бара листинга (вход по open бара L+X)
    "H": [3, 7, 14, 30, 60],  # дней удержания
    "S": [0.0, 0.15, 0.3, 0.5, 1.0],  # стоп (доля выше входа), 0 — без стопа
    "TP": [0.0, 0.25, 0.5],  # тейк (доля ниже входа), 0 — нет
    "K": [10, 20],
    "hedge": [0, 1],
}


def _daily(eqh: np.ndarray) -> np.ndarray:
    de = eqh[23::24][:NDAYS]
    prev = np.r_[1.0, de[:-1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(prev > 0, de / prev - 1, 0.0)
    return r


def _half_years(ret: np.ndarray) -> dict:
    out = {}
    dt = DTS.astype("datetime64[ms]")
    y = dt.astype("datetime64[Y]").astype(int) + 1970
    m = dt.astype("datetime64[M]").astype(int) % 12
    for yy in range(2020, 2027):
        for hh in (0, 1):
            sel = (y == yy) & ((m >= 6) == bool(hh))
            if sel.sum() > 20:
                out[f"{yy}H{hh + 1}"] = round((float(np.prod(1 + ret[sel])) - 1) * 100, 1)
    return out


def run_config(d, tr, K, hedge, pess=0, sel=None, keep=False, prefix="ev"):
    """tr — результат sim_trades; sel — маска допустимых событий."""
    ok, er, xr, pe, px, xt, ce, cx, ret, fund, adv_e = tr
    Lg = d["ev_L"]
    m = ok.copy()
    if sel is not None:
        m &= sel
    idx = np.where(m)[0]
    ge, gx = Lg[idx] + er[idx], Lg[idx] + xr[idx]
    acc = accept_slots(ge, gx, K)
    idx = idx[acc]
    order = idx[np.lexsort((idx, (Lg + er)[idx]))]
    o, c, f = d[f"{prefix}_o"], d[f"{prefix}_c"], d[f"{prefix}_f"]
    eqh, pp, pf, pc, ph, gross, traded, n_exec = portfolio(
        order.astype(np.int64), np.arange(len(Lg), dtype=np.int64), Lg, er, xr, pe, px, xt, ce, cx, o, c, f,
        d["btc_o"], d["btc_c"], d["btc_f"], NH, K, hedge, BTC_COST, pess, -1)  # fmt: skip
    r = _daily(eqh)
    a0, a1 = np.searchsorted(DTS, ms(IS[0])), np.searchsorted(DTS, ms(IS[1]))
    b0, b1 = np.searchsorted(DTS, ms(OOS[0])), np.searchsorted(DTS, ms(OOS[1]))
    ge_o = (Lg + er)[order]
    n_is = int(np.sum((ge_o >= (ms(IS[0]) - T0) // HOUR) & (ge_o < (ms(IS[1]) - T0) // HOUR)))
    n_oos = int(np.sum(ge_o >= (ms(OOS[0]) - T0) // HOUR))
    out = {
        "is": metrics(DTS[a0:a1], r[a0:a1], n_is),
        "oos": metrics(DTS[b0:b1], r[b0:b1], n_oos),
        "full": metrics(DTS, r, len(order)),
        "n_exec": int(n_exec),
        "pnl_price": round(pp, 4), "pnl_funding": round(pf, 4), "pnl_cost": round(pc, 4), "pnl_hedge": round(ph, 4),
        "avg_gross_short": round(gross, 3),
    }  # fmt: skip
    if keep:
        out["_ret"] = r
        out["_order"] = order
        out["_eqh"] = eqh
        out["_traded"] = traded
    return out


def grid_search() -> dict:
    t_start = time.time()
    d = load()
    for k in ("o", "h", "l", "c", "qv", "f"):
        d[f"ev_{k}"] = d[f"ev_{k}"].astype(np.float64)
    sel = d["ev_cat"] == "crypto"
    Lg = d["ev_L"].astype(np.int64)
    d["ev_L"] = Lg
    rows = []
    trade_cache = {}
    for X, H, S, TP in itertools.product(GRID["X"], GRID["H"], GRID["S"], GRID["TP"]):
        if TP > 0 and S == 0.0 and False:
            continue
        tr = sim_trades(d["ev_o"], d["ev_h"], d["ev_l"], d["ev_c"], d["ev_qv"], d["ev_f"], Lg, NH, X, H * 24, S, TP, 1.0, -1)
        trade_cache[(X, H, S, TP)] = tr
        ok, ret = tr[0] & sel, tr[8]
        tstats = {"n": int(ok.sum()), "mean_ret": round(float(np.mean(ret[ok])) * 100, 2), "median_ret": round(float(np.median(ret[ok])) * 100, 2),
                  "win": round(float(np.mean(ret[ok] > 0)) * 100, 1), "stop_share": round(float(np.mean(tr[5][ok] == 1)) * 100, 1)}  # fmt: skip
        for K, hedge in itertools.product(GRID["K"], GRID["hedge"]):
            res = run_config(d, tr, K, hedge, sel=sel)
            rows.append({"X": X, "H": H, "S": S, "TP": TP, "K": K, "hedge": hedge, "trades_all_events": tstats, **res})
    print(f"сетка: {len(rows)} конфигураций за {time.time() - t_start:.0f} с", flush=True)
    return {"rows": rows}


def summarize_grid(rows: list[dict]) -> dict:
    is_sh = np.array([r["is"]["sharpe"] for r in rows])
    is_dd = np.array([r["is"]["max_dd_pct"] for r in rows])
    oos_sh = np.array([r["oos"]["sharpe"] for r in rows])
    oos_cagr = np.array([r["oos"]["cagr_pct"] for r in rows])
    best = int(np.lexsort((is_dd, -is_sh))[0])  # максимум IS Sharpe, при равенстве — меньшая IS просадка
    corr = float(np.corrcoef(is_sh, oos_sh)[0, 1])
    # ранговая корреляция
    rk = lambda x: np.argsort(np.argsort(x))  # noqa: E731
    rcorr = float(np.corrcoef(rk(is_sh), rk(oos_sh))[0, 1])
    top10 = np.argsort(-is_sh)[:10]
    return {
        "n_configs": len(rows),
        "oos_sharpe_median": round(float(np.median(oos_sh)), 2),
        "oos_sharpe_share_positive": round(float(np.mean(oos_sh > 0)) * 100, 1),
        "oos_cagr_share_positive": round(float(np.mean(oos_cagr > 0)) * 100, 1),
        "oos_sharpe_p10_p90": [round(float(np.percentile(oos_sh, 10)), 2), round(float(np.percentile(oos_sh, 90)), 2)],
        "is_sharpe_median": round(float(np.median(is_sh)), 2),
        "is_oos_sharpe_corr": round(corr, 2),
        "is_oos_sharpe_rank_corr": round(rcorr, 2),
        "best_is_index": best,
        "top10_by_is": [{k: rows[i][k] for k in ("X", "H", "S", "TP", "K", "hedge")} | {"is_sh": rows[i]["is"]["sharpe"], "oos_sh": rows[i]["oos"]["sharpe"], "oos_cagr": rows[i]["oos"]["cagr_pct"]} for i in top10],
    }  # fmt: skip


def _prep() -> dict:
    d = load()
    for k in ("o", "h", "l", "c", "qv", "f"):
        d[f"ev_{k}"] = d[f"ev_{k}"].astype(np.float64)
    d["ev_L"] = d["ev_L"].astype(np.int64)
    return d


def _trades(d, X, H, S, TP, cost_mult=1.0, prefix="ev"):
    return sim_trades(d[f"{prefix}_o"], d[f"{prefix}_h"], d[f"{prefix}_l"], d[f"{prefix}_c"], d[f"{prefix}_qv"], d[f"{prefix}_f"],
                      d["ev_L"], NH, int(X), int(round(H * 24)), float(S), float(TP), cost_mult, -1)  # fmt: skip


def _short(res: dict) -> dict:
    keys = ("cagr_pct", "sharpe", "max_dd_pct", "worst_month_pct", "pos_months_pct", "trades")
    return {"is": {k: res["is"].get(k) for k in keys}, "oos": {k: res["oos"].get(k) for k in keys}}


def robust() -> dict:
    t_start = time.time()
    grid = json.loads((ODIR / "part_b_grid.json").read_text())
    rows, summ = grid["rows"], grid["summary"]
    b = rows[summ["best_is_index"]]
    X, H, S, TP, K, hedge = (b[k] for k in ("X", "H", "S", "TP", "K", "hedge"))
    print("лучший по IS:", dict(X=X, H=H, S=S, TP=TP, K=K, hedge=hedge), flush=True)
    d = _prep()
    crypto = d["ev_cat"] == "crypto"
    out = {"best_cfg": dict(X=X, H=H, S=S, TP=TP, K=K, hedge=hedge)}
    tr = _trades(d, X, H, S, TP)
    base = run_config(d, tr, K, hedge, sel=crypto, keep=True)
    # --- +-25% по каждому параметру (все комбинации)
    pert = []
    for mx, mh, ms_, mt in itertools.product((0.75, 1.0, 1.25), repeat=4):
        t2 = _trades(d, max(1, round(X * mx)), H * mh, S * ms_, TP * mt)
        r2 = run_config(d, t2, K, hedge, sel=crypto)
        pert.append({"X": max(1, round(X * mx)), "H": round(H * mh, 2), "S": round(S * ms_, 4), "TP": round(TP * mt, 4), **_short(r2)})
    osh = np.array([p_["oos"]["sharpe"] for p_ in pert])
    ish = np.array([p_["is"]["sharpe"] for p_ in pert])
    out["perturb_25pct"] = {
        "n": len(pert), "oos_sharpe_median": round(float(np.median(osh)), 2), "oos_sharpe_min": round(float(osh.min()), 2),
        "oos_sharpe_share_ge1": round(float(np.mean(osh >= 1)) * 100, 1), "oos_sharpe_share_pos": round(float(np.mean(osh > 0)) * 100, 1),
        "is_sharpe_median": round(float(np.median(ish)), 2), "rows": pert,
    }  # fmt: skip
    print("±25%:", {k: v for k, v in out["perturb_25pct"].items() if k != "rows"}, flush=True)
    # --- варианты
    var = {}
    var["base"] = _short(base)
    var["pessimistic_delist"] = _short(run_config(d, tr, K, hedge, pess=1, sel=crypto))
    var["cost_x2"] = _short(run_config(d, _trades(d, X, H, S, TP, cost_mult=2.0), K, hedge, sel=crypto))
    var["cost_x3"] = _short(run_config(d, _trades(d, X, H, S, TP, cost_mult=3.0), K, hedge, sel=crypto))
    var["no_hedge" if hedge else "btc_hedge"] = _short(run_config(d, tr, K, 1 - hedge, sel=crypto))
    for k2 in (5, 10, 20, 40):
        var[f"K{k2}"] = _short(run_config(d, tr, k2, hedge, sel=crypto))
    var["no_relist"] = _short(run_config(d, tr, K, hedge, sel=crypto & ~d["ev_relist"]))
    var["with_renames"] = _short(run_config(d, tr, K, hedge, sel=np.isin(d["ev_cat"], ["crypto", "rename"])))
    var["tradfi_only"] = _short(run_config(d, tr, K, hedge, sel=d["ev_cat"] == "tradfi"))
    for x2 in (336, 720):  # тот же шорт, но через 14 / 30 дней после листинга
        var[f"late_entry_X{x2}"] = _short(run_config(d, _trades(d, x2, H, S, TP), K, hedge, sel=crypto))
    out["variants"] = var
    for k2, v in var.items():
        print(f"  {k2:22s} IS sh {v['is']['sharpe']:5} cagr {v['is']['cagr_pct']:6} | OOS sh {v['oos']['sharpe']:5} cagr {v['oos']['cagr_pct']:6} dd {v['oos']['max_dd_pct']}", flush=True)
    # --- контроль: те же даты входа и правила, случайная зрелая монета (3 зерна)
    z = np.load(EVF, allow_pickle=False)
    for kind, title in (("ct", "control_random_mature_coin"), ("cv", "control_high_vol_coin_age180")):
        ctrl = []
        for sd in range(z[f"{kind}_o"].shape[0]):
            dc = {"ev_L": d["ev_L"], "btc_o": d["btc_o"], "btc_c": d["btc_c"], "btc_f": d["btc_f"]}
            for k in ("o", "h", "l", "c", "qv", "f"):
                dc[f"cs_{k}"] = z[f"{kind}_{k}"][sd].astype(np.float64)
            has = (z[f"{kind}_sym"][sd] != "") & crypto
            t3 = _trades(dc, X, H, S, TP, prefix="cs")
            r3 = run_config(dc, t3, K, hedge, sel=has, prefix="cs")
            okm = t3[0] & has
            ctrl.append({"seed": sd, **_short(r3), "trade_mean_ret_pct": round(float(np.mean(t3[8][okm])) * 100, 2),
                         "trade_median_ret_pct": round(float(np.median(t3[8][okm])) * 100, 2), "stop_share": round(float(np.mean(t3[5][okm] == 1)) * 100, 1)})  # fmt: skip
            print(f"  {title} seed {sd}: IS sh {ctrl[-1]['is']['sharpe']} OOS sh {ctrl[-1]['oos']['sharpe']} trade mean {ctrl[-1]['trade_mean_ret_pct']} stop {ctrl[-1]['stop_share']}", flush=True)
        out[title] = ctrl
    # --- детали лучшего
    ok, er, xr, pe, px, xt, ce, cx, ret, fund, adv_e = tr
    order = base["_order"]
    r = base["_ret"]
    qv = d["ev_qv"]
    cap = []
    for j in order:
        a = max(0, er[j] - 24)
        pre = np.nanmean(qv[j, a : er[j]]) if er[j] > a else np.nan
        hold = np.nanmean(qv[j, er[j] : xr[j] + 1])
        cap.append(0.01 * np.nanmin([pre, hold]))
    cap = np.array(cap)
    yrs = NDAYS / 365.25
    eqh = base["_eqh"]
    det = {
        "per_year": per_year(DTS, r),
        "half_years": _half_years(r),
        "trades": int(len(order)),
        "trade_mean_ret_pct": round(float(np.mean(ret[order])) * 100, 2),
        "trade_median_ret_pct": round(float(np.median(ret[order])) * 100, 2),
        "trade_win_pct": round(float(np.mean(ret[order] > 0)) * 100, 1),
        "exit_types_pct": {n_: round(float(np.mean(xt[order] == k_)) * 100, 1) for k_, n_ in enumerate(("time", "stop", "take", "delist", "end"))},
        "avg_hold_hours": round(float(np.mean(xr[order] - er[order])), 1),
        "trade_funding_mean_pct": round(float(np.mean(fund[order])) * 100, 2),
        "pnl_split": {k: base[k] for k in ("pnl_price", "pnl_funding", "pnl_cost", "pnl_hedge")},
        "avg_gross_short": base["avg_gross_short"],
        "turnover_x_per_year": round(float(base["_traded"] / np.mean(eqh) / yrs), 1),
        "capacity_per_trade_usd_median": round(float(np.nanmedian(cap))),
        "capacity_per_trade_usd_p10": round(float(np.nanpercentile(cap, 10))),
        "capacity_account_usd_median": round(float(np.nanmedian(cap)) * K),
        "capacity_account_usd_p10": round(float(np.nanpercentile(cap, 10)) * K),
        "max_concurrent_hint": "см. avg_gross_short·K",
    }  # fmt: skip
    # среднее число одновременно открытых позиций
    ge, gx = (d["ev_L"] + er)[order], (d["ev_L"] + xr)[order]
    cnt = np.zeros(NH + 1)
    np.add.at(cnt, ge, 1)
    np.add.at(cnt, np.minimum(gx + 1, NH), -1)
    conc = np.cumsum(cnt)[:NH]
    det["concurrent_mean"] = round(float(conc.mean()), 2)
    det["concurrent_max"] = int(conc.max())
    oos0 = (ms(OOS[0]) - T0) // HOUR
    det["concurrent_mean_oos"] = round(float(conc[oos0:].mean()), 2)
    out["best_details"] = det
    np.savez(ODIR / "best_daily.npz", ts=DTS, ret=r)
    print("детали:", json.dumps({k: v for k, v in det.items()}, ensure_ascii=False), flush=True)
    print(f"готово за {time.time() - t_start:.0f} с")
    return out


def report() -> dict:
    """Собирает results.json и проверяет лучший кандидат: кластерная t-статистика сделок OOS,
    скользящая доходность, «недавний» период 2025-07..2026-09 по всем конфигурациям сетки."""
    a = json.loads((ODIR / "part_a.json").read_text())
    g = json.loads((ODIR / "part_b_grid.json").read_text())
    rb = json.loads((ODIR / "part_robust.json").read_text())
    pc = json.loads((ODIR / "part_c.json").read_text())
    cfg = rb["best_cfg"]
    d = _prep()
    crypto = d["ev_cat"] == "crypto"
    tr = _trades(d, cfg["X"], cfg["H"], cfg["S"], cfg["TP"])
    base = run_config(d, tr, cfg["K"], cfg["hedge"], sel=crypto, keep=True)
    order, r = base["_order"], base["_ret"]
    Lg, er, ret = d["ev_L"], tr[1], tr[8]
    oos0 = (ms(OOS[0]) - T0) // HOUR
    rec0 = (ms("2025-07-01") - T0) // HOUR
    ge = Lg[order] + er[order]
    chk = {}
    for name, msk in (("IS", ge < oos0), ("OOS", ge >= oos0), ("OOS_2024-01..2025-06", (ge >= oos0) & (ge < rec0)), ("recent_2025-07..2026-09", ge >= rec0)):
        x = ret[order][msk]
        chk[name] = {"trades": int(len(x)), "mean_pct": round(float(np.mean(x)) * 100, 2), "t_month_cluster": _month_t(x, ge[msk] - er[order][msk])}
    # «недавний» период по всем конфигурациям сетки (без пересчёта портфелей: K=20, хедж)
    a_ = np.searchsorted(DTS, ms("2024-01-01"))
    b_ = np.searchsorted(DTS, ms("2025-07-01"))
    rec = []
    for X, H, S, TP in itertools.product(GRID["X"], GRID["H"], GRID["S"], GRID["TP"]):
        rr = run_config(d, _trades(d, X, H, S, TP), 20, 1, sel=crypto, keep=True)["_ret"]
        rec.append((metrics(DTS[a_:b_], rr[a_:b_])["sharpe"], metrics(DTS[b_:], rr[b_:])["sharpe"]))
    rec = np.array(rec)
    m_b = metrics(DTS[b_:], r[b_:])
    chk["grid_K20_hedged_450cfg"] = {
        "oos_early_2024-01..2025-06_median_sharpe": round(float(np.median(rec[:, 0])), 2),
        "oos_early_share_pos": round(float(np.mean(rec[:, 0] > 0)) * 100, 1),
        "recent_2025-07..2026-09_median_sharpe": round(float(np.median(rec[:, 1])), 2),
        "recent_share_pos": round(float(np.mean(rec[:, 1] > 0)) * 100, 1),
    }
    chk["best_recent_2025-07..2026-09"] = {k: m_b[k] for k in ("cagr_pct", "sharpe", "max_dd_pct", "worst_month_pct")}
    # скользящая 12-месячная доходность лучшего (на конец каждого квартала)
    eq = np.cumprod(1 + r)
    roll = {}
    for i in range(365, len(DTS), 91):
        roll[str(DTS[i].astype("datetime64[ms]"))[:10]] = round((float(eq[i] / eq[i - 365]) - 1) * 100, 1)
    chk["rolling_12m_ret_pct"] = roll
    rows_compact = [{k: row[k] for k in ("X", "H", "S", "TP", "K", "hedge", "is", "oos", "pnl_price", "pnl_funding", "pnl_cost", "pnl_hedge")} for row in g["rows"]]
    best_row = g["rows"][g["summary"]["best_is_index"]]
    res = {
        "family": "listings",
        "a_event_study": a,
        "b_grid_summary": g["summary"],
        "b_grid_param_space": GRID,
        "b_best_row": best_row,
        "b_robustness": rb,
        "b_checks": chk,
        "c_lifecycle": pc,
        "b_grid_rows": rows_compact,
        "verdict": {
            "edge": "weak",
            "why": "формально критерии yes выполнены на всём OOS (Sharpe 1.20, 4/6 полугодий > 0, ±25% медиана 1.21), "
                   "но эффект исчез в 2025-07..2026-09: лучший кандидат Sharpe −0.31, медиана сетки −0.15 (37% > 0), "
                   "средняя сделка +0.01% (t=0.6); понижено до weak",
        },
        "exposure_note": "best_daily.npz: каждая сделка = equity/20 шорт новой монеты + equity/20 лонг BTCUSDT-перп (хедж); "
                         "в среднем открыто ~1.1 позиции (OOS ~1.4, макс 8) => средний gross ≈ 0.05 шорт + 0.05 лонг, макс 0.4+0.4",
    }  # fmt: skip
    (ODIR / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    np.savez(ODIR / "best_daily.npz", ts=DTS, ret=r)
    print(json.dumps(chk, ensure_ascii=False, indent=1))
    return res


def main() -> None:
    ODIR.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "a"
    if what == "a":
        res = event_study()
        (ODIR / "part_a.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        h = res["horizons"]["crypto"]
        for k, v in h.items():
            print(k, "ret", v["ret"], "| short_pnl", v["short_pnl_gross"], "t", v["short_pnl_t_month_cluster"], "| fund", v["short_funding"]["mean"], "| mae", v["mae_share_gt"])
    elif what == "b":
        res = grid_search()
        summ = summarize_grid(res["rows"])
        (ODIR / "part_b_grid.json").write_text(json.dumps({"summary": summ, "rows": res["rows"]}, ensure_ascii=False))
        print(json.dumps(summ, indent=1, ensure_ascii=False))
    elif what == "r":
        res = robust()
        (ODIR / "part_robust.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    elif what == "report":
        report()


if __name__ == "__main__":
    main()
