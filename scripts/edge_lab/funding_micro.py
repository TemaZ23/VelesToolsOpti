"""
(d) Микроструктура выплат фандинга: доходности в часы до/после выплаты (00/08/16 UTC, у части монет 4ч/1ч)
у монет с большим |фандингом| («уклонение от выплаты»: лонги закрываются перед выплатой и перезаходят после).

    PYTHONPATH=. python3 -m scripts.edge_lab.funding_micro

Данные — часовые свечи перпов (load_hourly) и точные времена выплат (load_funding); часы с нулевым
объёмом («зомби»-бары после делистинга) исключены. Для каждой выплаты s: часовые бары b-3..b+3, где b0 —
бар, открывающийся в s (сразу после выплаты), b-1 — последний час перед выплатой.
Отбор события — по ПРЕДЫДУЩЕЙ ставке f_prev (известна заранее); P&L «держим через выплату» получает
фактическую текущую ставку f_cur с правильным знаком. ADV — 30 суток часового объёма ДО бара b-1.

ВНИМАНИЕ: стратегии здесь исполняются по open часового бара b0 = первой сделке после выплаты. Минутные данные
(funding_m1.py) показывают, что основное движение (~ размер ставки) проходит в первую минуту после выплаты,
поэтому эти результаты НЕДОСТИЖИМЫ для розничного бота; реалистичная проверка — funding_m1.py.
"""

from __future__ import annotations

import json
import time

import numpy as np

from scripts.edge_lab.common import (CACHE, HOUR, IS, OOS, cost_per_side, load_funding, load_hourly, metrics, ms,
                                     per_year, universe)  # fmt: skip
from scripts.edge_lab.funding import OUTD, half_years, save_json, summarize_configs

REL = (-3, -2, -1, 0, 1, 2, 3)  # бары относительно выплаты


def collect() -> dict[str, np.ndarray]:
    syms = sorted(universe())
    recs = []
    for j, s in enumerate(syms):
        h = load_hourly(s)
        f = load_funding(s)
        if len(f["ts"]) < 3 or len(h["ts"]) < 800:
            continue
        ts = h["ts"]
        ok = h["qvol"] > 0
        o, c, qv = h["ohlc"][:, 0], h["ohlc"][:, 3], np.where(ok, h["qvol"], 0.0)
        # непрерывная шкала часов
        t0 = ts[0]
        idx = ((ts - t0) // HOUR).astype(np.int64)
        L = idx[-1] + 1
        O = np.full(L, np.nan)
        C = np.full(L, np.nan)
        V = np.zeros(L)
        O[idx[ok]] = o[ok]
        C[idx[ok]] = c[ok]
        V[idx] = qv
        cv = np.r_[0.0, np.cumsum(V)]
        fts, rate = f["ts"], f["rate"]
        sh = np.round(fts / HOUR).astype(np.int64) * HOUR  # выплата, округлённая до часа
        b0 = (sh - t0) // HOUR
        prev = np.r_[np.nan, rate[:-1]]
        interval = np.r_[np.nan, np.diff(sh) / HOUR]
        good = (b0 - 720 - 3 >= 0) & (b0 + 4 < L) & np.isfinite(prev)
        b0, sh, rate, prev, interval = b0[good], sh[good], rate[good], prev[good], interval[good]
        if not len(b0):
            continue
        adv = (cv[b0 - 1] - cv[b0 - 1 - 720]) / 30.0  # сутки, только прошлое
        R = np.full((len(b0), len(REL)), np.nan)
        for k, rr in enumerate(REL):
            bb = b0 + rr
            with np.errstate(invalid="ignore", divide="ignore"):
                R[:, k] = C[bb] / O[bb] - 1
        # цена через выплату: open[b-1] -> open[b0] (выход сразу после выплаты)
        with np.errstate(invalid="ignore", divide="ignore"):
            r_through = O[b0] / O[b0 - 1] - 1
            r_pre3 = O[b0] / O[b0 - 3] - 1
        recs.append(np.column_stack([sh, np.full(len(b0), j), prev, rate, interval, adv, R, r_through, r_pre3]))
        if j % 150 == 0:
            print(j, len(syms), flush=True)
    a = np.concatenate(recs)
    cols = ["ts", "sym", "f_prev", "f_cur", "interval_h", "adv"] + [f"r{k:+d}" for k in REL] + ["r_through", "r_pre3"]
    return {c: a[:, i] for i, c in enumerate(cols)}


def cluster_t(ts, x):
    """t-статистика с кластеризацией по времени выплаты (усредняем внутри одного момента)."""
    m = np.isfinite(x)
    ts, x = ts[m], x[m]
    if len(x) < 30:
        return np.nan, np.nan, len(x)
    u, inv = np.unique(ts, return_inverse=True)
    s = np.bincount(inv, x) / np.bincount(inv)
    return float(np.mean(x)), float(np.mean(s) / (np.std(s) / np.sqrt(len(s)))) if np.std(s) > 0 else np.nan, len(x)


def load_records() -> dict[str, np.ndarray]:
    path = CACHE / "funding" / "derived" / "micro_records.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    D = collect()
    np.savez(path, **D)
    return D


def main() -> None:
    t0 = time.time()
    D = load_records()
    print("records", len(D["ts"]), f"{time.time() - t0:.0f}s", flush=True)
    ts = D["ts"]
    # рыночная поправка: среднее по всем ликвидным монетам на тот же момент выплаты
    liq = D["adv"] >= 1e7
    adj = {}
    for key in [f"r{k:+d}" for k in REL] + ["r_through"]:
        x = np.where(liq, D[key], np.nan)
        u, inv = np.unique(ts, return_inverse=True)
        okx = np.isfinite(x)
        ssum = np.bincount(inv[okx], x[okx], minlength=len(u))
        scnt = np.bincount(inv[okx], minlength=len(u))
        mean = np.where(scnt > 0, ssum / np.maximum(scnt, 1), 0.0)
        adj[key] = D[key] - mean[inv]
    is_m = ts < ms(IS[1])
    buckets = [(-1, -0.002), (-0.002, -0.001), (-0.001, -0.0003), (-0.0003, 0.00005), (0.00005, 0.00011),
               (0.00011, 0.0003), (0.0003, 0.001), (0.001, 0.002), (0.002, 1)]  # fmt: skip
    table = []
    for A in (1e7, 5e7):
        for (lo, hi) in buckets:
            m = (D["adv"] >= A) & (D["f_prev"] > lo) & (D["f_prev"] <= hi)
            row = {"A": A, "f_prev": [lo, hi], "n": int(m.sum())}
            for per, pm in (("is", is_m), ("oos", ~is_m)):
                mm_ = m & pm
                for key in ("r-1", "r+0", "r+1", "r_through"):
                    mu, tt, n = cluster_t(ts[mm_], D[key][mm_])
                    mua, tta, _ = cluster_t(ts[mm_], adj[key][mm_])
                    row[f"{per}_{key}_bp"] = round(mu * 1e4, 2) if np.isfinite(mu) else None
                    row[f"{per}_{key}_t"] = round(tt, 2) if np.isfinite(tt) else None
                    row[f"{per}_{key}_adj_bp"] = round(mua * 1e4, 2) if np.isfinite(mua) else None
                    row[f"{per}_{key}_adj_t"] = round(tta, 2) if np.isfinite(tta) else None
                row[f"{per}_fcur_bp"] = round(float(np.nanmean(D["f_cur"][mm_])) * 1e4, 2) if mm_.any() else None
            table.append(row)
    for r in table:
        print(r["A"], r["f_prev"], r["n"], "IS r-1", r["is_r-1_bp"], r["is_r-1_adj_t"], "r0", r["is_r+0_bp"],
              r["is_r+0_adj_t"], "thr", r["is_r_through_bp"], "fcur", r["is_fcur_bp"], "| OOS r-1", r["oos_r-1_bp"],
              "r0", r["oos_r+0_bp"], "thr", r["oos_r_through_bp"], "fcur", r["oos_fcur_bp"], flush=True)  # fmt: skip
    # --- стратегии
    days = (ts // 86_400_000).astype(np.int64)
    d0 = int(ms("2020-01-01") // 86_400_000)
    d1 = int(ms("2026-10-01") // 86_400_000)
    dts = np.arange(d0, d1, dtype=np.int64) * 86_400_000
    rows = []
    for A in (1e7, 5e7, 2e8):
        for thr in (0.0005, 0.001, 0.002, 0.005):
            for sign in ("pos", "neg", "both"):
                for kind in ("through", "post1", "post3", "through_adj"):
                    for maker in (False, True):
                        sel = (D["adv"] >= A)
                        if sign == "pos":
                            sel &= D["f_prev"] >= thr
                        elif sign == "neg":
                            sel &= D["f_prev"] <= -thr
                        else:
                            sel &= np.abs(D["f_prev"]) >= thr
                        sgn = -np.sign(D["f_prev"])  # «через выплату»: против знака ставки (получаем фандинг)
                        if kind == "through":
                            gross = sgn * D["r_through"] - sgn * D["f_cur"]  # лонг (sgn=+1) платит f_cur, шорт получает
                        elif kind == "through_adj":
                            gross = sgn * adj["r_through"] - sgn * D["f_cur"]
                        elif kind == "post1":
                            gross = -sgn * D["r+0"]  # после выплаты — в сторону ставки (перезаход толпы)
                        else:
                            gross = -sgn * (D["r+0"] + D["r+1"] + D["r+2"])
                        cps = 0.0002 if maker else cost_per_side(D["adv"])
                        net = gross - 2 * cps
                        net = np.where(sel & np.isfinite(net), net, np.nan)
                        # размер позиции 10% капитала на событие, максимум 10 событий за выплату
                        okk = np.isfinite(net)
                        if okk.sum() < 30:
                            continue
                        e_ts, e_net, e_f = ts[okk], net[okk], np.abs(D["f_prev"][okk])
                        order = np.lexsort((-e_f, e_ts))
                        e_ts, e_net = e_ts[order], e_net[order]
                        first = np.r_[True, e_ts[1:] != e_ts[:-1]]
                        grp = np.cumsum(first) - 1
                        start = np.where(first)[0]
                        rank = np.arange(len(e_ts)) - start[grp]
                        keep = rank < 10
                        e_ts, e_net = e_ts[keep], e_net[keep]
                        di = (e_ts // 86_400_000).astype(np.int64) - d0
                        inr = (di >= 0) & (di < len(dts))
                        dr = np.bincount(di[inr], 0.1 * e_net[inr], minlength=len(dts))[: len(dts)]
                        st = {}
                        for name, win in (("is", IS), ("oos", OOS)):
                            a, b = np.searchsorted(dts, ms(win[0])), np.searchsorted(dts, ms(win[1]))
                            ev = (e_ts >= ms(win[0])) & (e_ts < ms(win[1]))
                            st[name] = metrics(dts[a:b], dr[a:b], int(ev.sum()))
                            st[name]["mean_event_bp"] = round(float(np.mean(e_net[ev])) * 1e4, 2) if ev.any() else None
                        st["cfg"] = dict(A=A, thr=thr, sign=sign, kind=kind, maker=maker)
                        rows.append(st)
    taker = [r for r in rows if not r["cfg"]["maker"]]
    res = {"n_records": int(len(ts)), "bucket_table": table, "summary_taker": summarize_configs(taker),
           "summary_maker": summarize_configs([r for r in rows if r["cfg"]["maker"]]),
           "note": "maker-вариант предполагает исполнение лимитками на обеих ногах по цене open (оптимистично)"}
    best = max(taker, key=lambda r: r["is"]["sharpe"])
    res["best_is_taker"] = best
    res["top10_is_taker"] = sorted(taker, key=lambda r: -r["is"]["sharpe"])[:10]
    res["configs"] = rows
    save_json("part_micro.json", res)
    print(json.dumps({k: res[k] for k in ("summary_taker", "summary_maker")}, indent=1))
    for r in res["top10_is_taker"]:
        print(r["cfg"], "IS", r["is"]["sharpe"], r["is"]["cagr_pct"], r["is"]["mean_event_bp"], r["is"]["trades"],
              "OOS", r["oos"]["sharpe"], r["oos"]["cagr_pct"], r["oos"]["mean_event_bp"], r["oos"]["trades"])  # fmt: skip
    mk = sorted([r for r in rows if r["cfg"]["maker"]], key=lambda r: -r["is"]["sharpe"])[:10]
    for r in mk:
        print("maker", r["cfg"], "IS", r["is"]["sharpe"], r["is"]["mean_event_bp"], "OOS", r["oos"]["sharpe"],
              r["oos"]["mean_event_bp"], r["oos"]["trades"])  # fmt: skip
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
