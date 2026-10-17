"""
flow (d): «каскад ликвидаций» — рыночный обвал за 1–4 ч с всплеском объёма -> покупка BTC/ETH/корзины,
выход через 1–72 ч; и обратная гипотеза — резкий рост рынка (шорт-сквиз) -> шорт.

    PYTHONPATH=. python3 -m scripts.edge_lab.flow_cascade       # -> output/edge_lab/flow/part_d.json

Сигнал на баре t (только данные ≤ t):
  src=BTC: r_w = close_BTC[t]/close_BTC[t-w] - 1; vr = средний объём BTC за окно / медиана за 168 ч до окна;
  src=ALT: r_w = накопленная за w часов равновзвешенная доходность ликвидных альтов (ADV ≥ 20 млн, возраст ≥ 30 сут,
           без BTC; состав по данным на t-1); vr — по суммарному объёму всех перпов.
  Обвал: r_w ≤ −X и vr ≥ V -> лонг инструмента по open[t+1]; сквиз: r_w ≥ +X -> шорт.
Инструменты: BTC, ETH, BTC+ETH (по 1/2), TOP10 (10 крупнейших по ADV на момент t, по 1/10). Валовая экспозиция 1x.
Выход: через H часов, опционально стоп (рыночный, пессимистично) и тейк (лимит, maker). Пока позиция открыта,
новые сигналы игнорируются. Издержки taker + проскальзывание на обе стороны, фандинг по факту.
"""

from __future__ import annotations

import itertools
import json
import sys
import time

import numpy as np

from scripts.edge_lab.common import DAY, IS, OOS, OUT, metrics, ms
from scripts.edge_lab.flow_common import (
    age_hours, cost_matrix, daily_from_hourly, day_clustered_t, fwd_event, load_all, past_median_at, summarize,
)  # fmt: skip
from scripts.edge_lab.flow_sim import simulate

ODIR = OUT / "flow"
HS_EV = np.array([1, 2, 4, 8, 12, 24, 48, 72], dtype=np.int64)


def market_series(p: dict) -> dict:
    c = p["close"]
    T, N = c.shape
    syms = list(p["symbols"])
    bi, ei = syms.index("BTCUSDT"), syms.index("ETHUSDT")
    age = age_hours(c)
    adv = np.nan_to_num(p["adv"])
    elig = p["crypto"][None, :] & (age >= 720) & (adv >= 2e7)
    elig[:, bi] = False
    r1 = np.full(c.shape, np.nan, np.float32)
    r1[1:] = c[1:] / c[:-1] - 1
    e_prev = np.zeros_like(elig)
    e_prev[1:] = elig[:-1]
    rr = np.where(e_prev & np.isfinite(r1), np.clip(r1, -0.9, 3.0), np.nan)
    with np.errstate(invalid="ignore"):
        alt_r = np.nanmean(rr, axis=1)
    n_alt = np.isfinite(rr).sum(axis=1)
    alt_r = np.where(n_alt >= 5, alt_r, 0.0)
    alt_lvl = np.cumprod(1 + np.nan_to_num(alt_r))
    btc = c[:, bi].astype(float)
    mvol = np.nansum(p["qvol"], axis=1).astype(float)
    # top-10 по ADV на момент t (крипто, возраст ≥ 30 сут), включая BTC и ETH
    e10 = p["crypto"][None, :] & (age >= 720) & np.isfinite(c)
    a10 = np.where(e10, adv, -1)
    top10 = np.argsort(-a10, axis=1)[:, :10]
    return {"btc": btc, "alt": alt_lvl, "btc_vol": p["qvol"][:, bi].astype(float), "mkt_vol": mvol, "bi": bi, "ei": ei,
            "top10": top10.astype(np.int64), "n_alt": n_alt}  # fmt: skip


def series_vr(vol: np.ndarray, w: int) -> np.ndarray:
    """mean(vol[t-w+1..t]) / median(vol[t-w-167..t-w]) для каждого t."""
    T = len(vol)
    x = vol.reshape(-1, 1).astype(float)
    rows = np.arange(T, dtype=np.int64)
    med = past_median_at(x, rows, np.zeros(T, np.int64), 168, 100, w)
    cs = np.r_[0.0, np.cumsum(np.nan_to_num(vol))]
    win = np.full(T, np.nan)
    win[w - 1 :] = (cs[w:] - cs[:-w]) / w
    return win / med


def trigger_mask(ms_: dict, src: str, w: int, X: float, V: float, sign: int, vr_cache: dict) -> np.ndarray:
    lvl = ms_["btc"] if src == "BTC" else ms_["alt"]
    T = len(lvl)
    rw = np.full(T, np.nan)
    rw[w:] = lvl[w:] / lvl[:-w] - 1
    key = (src, w)
    if key not in vr_cache:
        vr_cache[key] = series_vr(ms_["btc_vol"] if src == "BTC" else ms_["mkt_vol"], w)
    vr = vr_cache[key]
    m = (sign * rw >= X) & np.isfinite(rw)
    if V > 0:
        m &= np.nan_to_num(vr) >= V
    if src == "ALT":
        m &= ms_["n_alt"] >= 5
    return m


def inst_signals(mask: np.ndarray, inst: str, ms_: dict) -> tuple[np.ndarray, np.ndarray]:
    t = np.nonzero(mask)[0]
    if inst == "BTC":
        return t.astype(np.int64), np.full(len(t), ms_["bi"], np.int64)
    if inst == "ETH":
        return t.astype(np.int64), np.full(len(t), ms_["ei"], np.int64)
    if inst == "BTC+ETH":
        tt = np.repeat(t, 2)
        jj = np.tile([ms_["bi"], ms_["ei"]], len(t))
        return tt.astype(np.int64), jj.astype(np.int64)
    tt = np.repeat(t, 10)
    jj = ms_["top10"][t].reshape(-1)
    return tt.astype(np.int64), jj.astype(np.int64)


INST_K = {"BTC": 1, "ETH": 1, "BTC+ETH": 2, "TOP10": 10}
GRID = dict(src=["BTC", "ALT"], w=[1, 2, 4], X=None, V=[0, 2, 3], inst=["BTC", "ETH", "BTC+ETH", "TOP10"],
            H=[1, 2, 4, 8, 12, 24, 48, 72], stop=[0.0, 0.05], tp=[0.0, 0.03])  # fmt: skip
XS = {"BTC": [0.02, 0.03, 0.04, 0.05, 0.07], "ALT": [0.03, 0.05, 0.07, 0.10, 0.15]}


def run_cfg(p, COST, ms_, cfg, sign, vr_cache, full=False):
    mask = trigger_mask(ms_, cfg["src"], cfg["w"], cfg["X"], cfg["V"], sign, vr_cache)
    st, sj = inst_signals(mask, cfg["inst"], ms_)
    K = INST_K[cfg["inst"]]
    direction = 1 if sign < 0 else -1  # обвал -> лонг; сквиз -> шорт
    T = len(p["ts"])
    out = simulate(p["open"], p["high"], p["low"], p["close"], p["funding"], COST, st, sj, direction, cfg["H"],
                   cfg["stop"], cfg["tp"], True, 1.0 / K, K, 0.0, 2.0, 0, 0, T)  # fmt: skip
    eq, tj, t0, t1, tret, tfund, tcost, treason, avg_gross = out
    dts, dret = daily_from_hourly(p["ts"], eq)
    tts = p["ts"][t0] if len(t0) else np.empty(0, np.int64)
    res = {}
    for name, (a, b) in (("IS", IS), ("OOS", OOS)):
        m = (dts >= ms(a)) & (dts < ms(b))
        res[name] = metrics(dts[m], dret[m])
        tm = (tts >= ms(a)) & (tts < ms(b))
        res[name]["trades"] = int(tm.sum())
        if tm.sum():
            res[name]["avg_trade_net_pct"] = round(float(np.mean(tret[tm] + tfund[tm] - tcost[tm])) * 100, 3)
    res["avg_gross"] = round(float(avg_gross), 3)
    if full:
        return res, {"eq": eq, "dts": dts, "dret": dret, "tj": tj, "t0": t0, "t1": t1, "tret": tret, "tfund": tfund,
                     "tcost": tcost, "treason": treason}  # fmt: skip
    return res


def event_study(p, ms_, vr_cache) -> list[dict]:
    """Средняя доходность BTC/ETH/TOP10-EW после рыночных обвалов/сквизов (первое событие за 24 ч)."""
    out = []
    o, c, f = p["open"], p["close"], p["funding"]
    T = len(p["ts"])
    rows_all = np.arange(T, dtype=np.int64)
    base = {}
    for nm, j in (("BTC", ms_["bi"]), ("ETH", ms_["ei"])):
        r, _, _ = fwd_event(o, c, f, rows_all, np.full(T, j, np.int64), HS_EV)
        base[nm] = r
    for src in ("BTC", "ALT"):
        for w in (1, 4):
            for sign in (-1, 1):
                for X in XS[src]:
                    for V in (0, 3):
                        m = trigger_mask(ms_, src, w, X, V, sign, vr_cache)
                        t = np.nonzero(m)[0]
                        keep = np.r_[True, np.diff(t) > 24] if len(t) else np.zeros(0, bool)
                        t = t[keep]
                        t = t[t < T - 73]
                        for inst in ("BTC", "ETH"):
                            rr = base[inst][t]
                            unc = base[inst]
                            for per, (a, bnd) in (("IS", IS), ("OOS", OOS)):
                                pm = (p["ts"][t] >= ms(a)) & (p["ts"][t] < ms(bnd))
                                um = (p["ts"] >= ms(a)) & (p["ts"] < ms(bnd))
                                row = {"src": src, "w": w, "sign": sign, "X": X, "V": V, "inst": inst, "period": per,
                                       "n": int(pm.sum())}  # fmt: skip
                                if pm.sum() >= 3:
                                    day = p["ts"][t][pm] // DAY
                                    for ih, h in enumerate(HS_EV):
                                        mu, tt, _ = day_clustered_t(rr[pm, ih], day)
                                        row[f"h{h}"] = {"mean_pct": round(mu * 100, 3), "t": round(tt, 2),
                                                        "excess_pct": round((mu - float(np.nanmean(unc[um, ih]))) * 100, 3),
                                                        "hit": round(float(np.nanmean(rr[pm, ih] > 0)), 3)}  # fmt: skip
                                out.append(row)
    return out


def run() -> None:
    t0 = time.time()
    p = load_all()
    COST = cost_matrix(p["adv"])
    ms_ = market_series(p)
    vr_cache: dict = {}
    print("market series", round(time.time() - t0, 1), flush=True)
    res = {"desc": __doc__, "event_study": event_study(p, ms_, vr_cache)}
    print("event study", round(time.time() - t0, 1), flush=True)
    for hyp, sign in (("cascade_bounce", -1), ("squeeze_fade", 1)):
        rows = []
        for src in GRID["src"]:
            for vals in itertools.product(GRID["w"], XS[src], GRID["V"], GRID["inst"], GRID["H"], GRID["stop"], GRID["tp"]):
                cfg = dict(zip(["w", "X", "V", "inst", "H", "stop", "tp"], vals))
                cfg["src"] = src
                r = run_cfg(p, COST, ms_, cfg, sign, vr_cache)
                rows.append({"cfg": cfg, **r})
        res[hyp] = rows
        print(hyp, len(rows), round(time.time() - t0, 1), flush=True)
    # базовые линии: buy&hold BTC, ETH, TOP10
    bl = {}
    for inst in ("BTC", "ETH"):
        j = ms_["bi"] if inst == "BTC" else ms_["ei"]
        c = p["close"][:, j].astype(float)
        ok = np.isfinite(c)
        eq = np.where(ok, c / c[ok][0], np.nan)
        eq = np.nan_to_num(eq, nan=1.0)
        dts, dret = daily_from_hourly(p["ts"], eq)
        bl[inst] = summarize(dts, dret)
    res["buy_hold"] = bl
    ODIR.mkdir(parents=True, exist_ok=True)
    (ODIR / "part_d.json").write_text(json.dumps(res, default=float))
    show(res)


def show(res: dict | None = None) -> None:
    if res is None:
        res = json.loads((ODIR / "part_d.json").read_text())
    print("EVENT STUDY (first event per 24h): mean% [t] (excess vs unconditional) at h=4,24,72")
    for r in res["event_study"]:
        if r["n"] < 3 or r["V"] != 0:
            continue
        s = " ".join(f"{r[f'h{h}']['mean_pct']:+.2f}[{r[f'h{h}']['t']:+.1f}]({r[f'h{h}']['excess_pct']:+.2f})" for h in (4, 24, 72))
        print(f"{r['src']} w={r['w']} {'drop' if r['sign'] < 0 else 'rise'} X={r['X']} V={r['V']} {r['inst']} {r['period']} n={r['n']:4d} {s}")
    for hyp in ("cascade_bounce", "squeeze_fade"):
        rows = res[hyp]
        isr = np.array([r["IS"]["sharpe"] for r in rows])
        oos = np.array([r["OOS"]["sharpe"] for r in rows])
        print(f"\n== {hyp}: {len(rows)} configs; IS SR median {np.median(isr):.2f}; OOS SR median {np.median(oos):.2f}, "
              f"share OOS>0 {np.mean(oos > 0):.2f}; corr {np.corrcoef(isr, oos)[0, 1]:.2f}")  # fmt: skip
        top = sorted([r for r in rows if r["IS"].get("trades", 0) >= 20], key=lambda r: -r["IS"]["sharpe"])[:15]
        for r in top:
            print(" ", r["cfg"], "| IS", {k: r["IS"].get(k) for k in ("sharpe", "cagr_pct", "max_dd_pct", "trades")},
                  "| OOS", {k: r["OOS"].get(k) for k in ("sharpe", "cagr_pct", "max_dd_pct", "trades")})  # fmt: skip
    print("\nbuy&hold", {k: {kk: v.get(kk) for kk in ("sharpe", "cagr_pct", "max_dd_pct")} for k, v in res["buy_hold"]["BTC"].items() if k in ("IS", "OOS")})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "show":
        show()
    else:
        run()
