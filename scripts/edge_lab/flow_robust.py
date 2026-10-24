"""
flow: детальный разбор и проверки устойчивости для IS-лучших конфигураций (c) и (d).

    PYTHONPATH=. python3 -m scripts.edge_lab.flow_robust <hyp> [<hyp> ...]   # -> output/edge_lab/flow/robust_<hyp>.json

Для конфигурации, выбранной по максимуму IS Sharpe (≥ 30 сделок в IS для (c), ≥ 20 для (d)):
- полный прогон 2020–2026: метрики IS/OOS/ALL, по годам, по полугодиям, сделки, оборот, среднее время удержания,
  доля P&L от цены / фандинга / издержек, причины выхода, ёмкость (1% среднечасового объёма на входе);
- пессимистичный делистинг (−50% номинала), стоп с проскальзыванием ×1 и ×4;
- ±25% по каждому числовому параметру (по одному);
- случайные сигналы с тем же числом сделок по месяцам и теми же правилами выхода (60 прогонов).
Сохраняет суточные доходности в robust_<hyp>_daily.npz.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from scripts.edge_lab.common import DAY, HOUR, IS, OOS, OUT, metrics, ms
from scripts.edge_lab.flow_common import age_hours, cost_matrix, daily_from_hourly, load_all, summarize
from scripts.edge_lab.flow_sim import simulate

ODIR = OUT / "flow"


def pick_best(rows, min_tr):
    ok = [r for r in rows if r["IS"].get("trades", 0) >= min_tr]
    return max(ok, key=lambda r: r["IS"]["sharpe"]) if ok else None


def sim_full(p, COST, st, sj, direction, H, stop, tp, K, delist_pen=0.0, stop_slip_mult=2.0):
    T = len(p["ts"])
    out = simulate(p["open"], p["high"], p["low"], p["close"], p["funding"], COST, st, sj, direction, int(H),
                   float(stop), float(tp), True, 1.0 / K, int(K), float(delist_pen), float(stop_slip_mult), 0, 0, T)  # fmt: skip
    eq, tj, t0, t1, tret, tfund, tcost, treason, avg_gross = out
    dts, dret = daily_from_hourly(p["ts"], eq)
    return {"eq": eq, "dts": dts, "dret": dret, "tj": tj, "t0": t0, "t1": t1, "tret": tret, "tfund": tfund,
            "tcost": tcost, "treason": treason, "avg_gross": avg_gross, "K": K}  # fmt: skip


def quick(full, p):
    tts = p["ts"][full["t0"]] if len(full["t0"]) else np.empty(0, np.int64)
    out = {}
    for name, (a, b) in (("IS", IS), ("OOS", OOS)):
        m = (full["dts"] >= ms(a)) & (full["dts"] < ms(b))
        mm = metrics(full["dts"][m], full["dret"][m])
        out[name] = {k: mm[k] for k in ("sharpe", "cagr_pct", "max_dd_pct", "avg_month_pct", "worst_month_pct", "pos_months_pct")}
        out[name]["trades"] = int(((tts >= ms(a)) & (tts < ms(b))).sum())
    return out


def details(full, p, ADV):
    d = summarize(full["dts"], full["dret"], trades=len(full["t0"]))
    n = len(full["t0"])
    if n == 0:
        return d
    w = 1.0 / full["K"]
    hold_h = (full["t1"] - full["t0"] + 1).astype(float)
    tts = p["ts"][full["t0"]]
    per = {}
    for name, (a, b) in (("IS", IS), ("OOS", OOS)):
        m = (tts >= ms(a)) & (tts < ms(b))
        if m.sum() == 0:
            continue
        net = full["tret"][m] + full["tfund"][m] - full["tcost"][m]
        per[name] = {
            "trades": int(m.sum()),
            "win_rate": round(float(np.mean(net > 0)), 3),
            "avg_trade_net_pct": round(float(np.mean(net)) * 100, 3),
            "median_trade_net_pct": round(float(np.median(net)) * 100, 3),
            "avg_price_pct": round(float(np.mean(full["tret"][m])) * 100, 3),
            "avg_funding_pct": round(float(np.mean(full["tfund"][m])) * 100, 3),
            "avg_cost_pct": round(float(np.mean(full["tcost"][m])) * 100, 3),
            "sum_price_pnl_x_w": round(float(np.sum(full["tret"][m])) * w, 4),
            "sum_funding_pnl_x_w": round(float(np.sum(full["tfund"][m])) * w, 4),
            "sum_cost_x_w": round(float(np.sum(full["tcost"][m])) * w, 4),
            "avg_hold_h": round(float(np.mean(hold_h[m])), 2),
            "exit_reasons": {k: int(v) for k, v in zip(["time", "stop", "tp", "delist", "end", "liq"], np.bincount(full["treason"][m], minlength=6))},
            "best_trade_pct": round(float(np.max(net)) * 100, 2),
            "worst_trade_pct": round(float(np.min(net)) * 100, 2),
        }  # fmt: skip
        years = (p["ts"][-1] - p["ts"][0]) / DAY / 365.25
    d["trades_by_period"] = per
    d["avg_gross_exposure"] = round(float(full["avg_gross"]), 3)
    # оборот: (вход+выход) * w на сделку, в среднем за год, в долях капитала
    yrs = (full["dts"][-1] - full["dts"][0]) / DAY / 365.25
    d["turnover_per_year_x_equity"] = round(float(2 * n * w / yrs), 2)
    cap = 0.01 * np.nan_to_num(ADV[full["t0"] - 1, full["tj"]]) / 24.0
    d["capacity_per_trade_usd"] = {"median": round(float(np.median(cap)), 0), "p10": round(float(np.percentile(cap, 10)), 0),
                                   "p25": round(float(np.percentile(cap, 25)), 0)}  # fmt: skip
    d["capacity_account_usd_p25"] = round(float(np.percentile(cap, 25)) / w, 0)
    return d


def random_baseline(p, COST, st, sj, eligible_fn, direction, H, stop, tp, K, n_draws=60, seed=0):
    """Те же месяцы и число сигналов, случайные (t, j) среди допустимых монет."""
    rng = np.random.default_rng(seed)
    T = len(p["ts"])
    months = p["ts"].astype("datetime64[ms]").astype("datetime64[M]").astype(np.int64)
    mstart = {}
    for m in np.unique(months):
        idx = np.nonzero(months == m)[0]
        mstart[m] = (idx[0], idx[-1])
    sig_m = months[st]
    out = []
    for k in range(n_draws):
        tt = np.empty(len(st), np.int64)
        jj = np.empty(len(st), np.int64)
        for i, m in enumerate(sig_m):
            a, b = mstart[m]
            for _ in range(50):
                t = int(rng.integers(a, min(b, T - 2) + 1))
                cand = eligible_fn(t)
                if len(cand):
                    tt[i] = t
                    jj[i] = int(rng.choice(cand))
                    break
            else:
                tt[i], jj[i] = st[i], sj[i]
        o = np.argsort(tt, kind="stable")
        f = sim_full(p, COST, tt[o], jj[o], direction, H, stop, tp, K)
        q = quick(f, p)
        out.append({"IS": q["IS"]["sharpe"], "OOS": q["OOS"]["sharpe"], "IS_cagr": q["IS"]["cagr_pct"], "OOS_cagr": q["OOS"]["cagr_pct"]})
    a_is = np.array([r["IS"] for r in out])
    a_oos = np.array([r["OOS"] for r in out])
    return {"n": n_draws, "IS_sharpe_median": round(float(np.median(a_is)), 2), "IS_sharpe_p90": round(float(np.percentile(a_is, 90)), 2),
            "OOS_sharpe_median": round(float(np.median(a_oos)), 2), "OOS_sharpe_p90": round(float(np.percentile(a_oos, 90)), 2),
            "draws": out}  # fmt: skip


def perturb(cfg: dict, keys: list[str]) -> list[tuple[str, dict]]:
    out = []
    for k in keys:
        v = cfg[k]
        if not isinstance(v, (int, float)) or v == 0:
            continue
        for f in (0.75, 1.25):
            c = dict(cfg)
            nv = v * f
            if k in ("H", "K", "w"):
                nv = max(1, int(round(nv)))
                if nv == v:
                    nv = v + (1 if f > 1 else -1)
                    if nv < 1:
                        continue
            c[k] = nv
            out.append((f"{k}x{f}", c))
    return out


def robust_c(hyp: str, p, COST, ADV) -> dict:
    from scripts.edge_lab.flow_pumpfade import HYP, prep, signals

    rows = json.loads((ODIR / "part_c.json").read_text())[hyp]
    best = pick_best(rows, 30)
    cfg = best["cfg"]
    side, direction = HYP[hyp]
    ev = prep(p)
    age = age_hours(p["close"])
    st, sj = signals(ev, side, cfg["X"], cfg["V"], cfg["min_adv"], cfg["confirm"])
    full = sim_full(p, COST, st, sj, direction, cfg["H"], cfg["stop"], cfg["tp"], cfg["K"])
    res = {"hyp": hyp, "cfg": cfg, "grid_IS": best["IS"], "grid_OOS": best["OOS"], "details": details(full, p, ADV)}
    np.savez(ODIR / f"robust_{hyp}_daily.npz", ts=full["dts"], ret=full["dret"])
    v = {}
    for name, kw in (("delist_pen_50", {"delist_pen": 0.5}), ("stop_slip_x1", {"stop_slip_mult": 1.0}), ("stop_slip_x4", {"stop_slip_mult": 4.0})):
        f = sim_full(p, COST, st, sj, direction, cfg["H"], cfg["stop"], cfg["tp"], cfg["K"], **kw)
        v[name] = quick(f, p)
    res["variants"] = v
    pert = {}
    for name, c in perturb(cfg, ["X", "V", "H", "stop", "tp", "K"]):
        st2, sj2 = signals(ev, side, c["X"], c["V"], c["min_adv"], c["confirm"])
        f = sim_full(p, COST, st2, sj2, direction, c["H"], c["stop"], c["tp"], c["K"])
        pert[name] = {"cfg": c, **quick(f, p)}
    res["perturb"] = pert
    crypto = p["crypto"]
    adv = p["adv"]
    close = p["close"]

    def elig(t):
        return np.nonzero(crypto & (age[t] >= 169) & np.isfinite(close[t]) & (np.nan_to_num(adv[t]) >= cfg["min_adv"]))[0]

    res["random_baseline"] = random_baseline(p, COST, st, sj, elig, direction, cfg["H"], cfg["stop"], cfg["tp"], cfg["K"])
    return res


def robust_d(hyp: str, p, COST, ADV) -> dict:
    from scripts.edge_lab.flow_cascade import INST_K, inst_signals, market_series, trigger_mask

    rows = json.loads((ODIR / "part_d.json").read_text())[hyp]
    best = pick_best(rows, 20)
    cfg = best["cfg"]
    sign = -1 if hyp == "cascade_bounce" else 1
    direction = 1 if sign < 0 else -1
    ms_ = market_series(p)
    vrc: dict = {}
    mask = trigger_mask(ms_, cfg["src"], cfg["w"], cfg["X"], cfg["V"], sign, vrc)
    st, sj = inst_signals(mask, cfg["inst"], ms_)
    K = INST_K[cfg["inst"]]
    full = sim_full(p, COST, st, sj, direction, cfg["H"], cfg["stop"], cfg["tp"], K)
    res = {"hyp": hyp, "cfg": cfg, "grid_IS": best["IS"], "grid_OOS": best["OOS"], "details": details(full, p, ADV)}
    np.savez(ODIR / f"robust_{hyp}_daily.npz", ts=full["dts"], ret=full["dret"])
    pert = {}
    for name, c in perturb(cfg, ["X", "V", "H", "stop", "tp", "w"]):
        if c["w"] < 1:
            continue
        m2 = trigger_mask(ms_, c["src"], c["w"], c["X"], c["V"], sign, vrc)
        st2, sj2 = inst_signals(m2, c["inst"], ms_)
        f = sim_full(p, COST, st2, sj2, direction, c["H"], c["stop"], c["tp"], K)
        pert[name] = {"cfg": c, **quick(f, p)}
    res["perturb"] = pert
    # случайные моменты входа (тот же инструмент, то же число сигналов по месяцам)
    rng = np.random.default_rng(1)
    T = len(p["ts"])
    months = p["ts"].astype("datetime64[ms]").astype("datetime64[M]").astype(np.int64)
    tsig = np.nonzero(mask)[0]
    draws = []
    for k in range(60):
        tt = []
        for m in months[tsig]:
            idx = np.nonzero(months == m)[0]
            tt.append(int(rng.choice(idx[idx < T - 2])))
        mk = np.zeros(T, bool)
        mk[np.array(tt, np.int64)] = True
        st2, sj2 = inst_signals(mk, cfg["inst"], ms_)
        f = sim_full(p, COST, st2, sj2, direction, cfg["H"], cfg["stop"], cfg["tp"], K)
        q = quick(f, p)
        draws.append({"IS": q["IS"]["sharpe"], "OOS": q["OOS"]["sharpe"]})
    a_is = np.array([d["IS"] for d in draws])
    a_oos = np.array([d["OOS"] for d in draws])
    res["random_baseline"] = {"n": 60, "IS_sharpe_median": round(float(np.median(a_is)), 2), "IS_sharpe_p90": round(float(np.percentile(a_is, 90)), 2),
                              "OOS_sharpe_median": round(float(np.median(a_oos)), 2), "OOS_sharpe_p90": round(float(np.percentile(a_oos, 90)), 2)}  # fmt: skip
    return res


def main(hyps: list[str]) -> None:
    t0 = time.time()
    p = load_all()
    COST = cost_matrix(p["adv"])
    for hyp in hyps:
        if hyp in ("fade_pump", "bounce_dump", "follow_pump", "follow_dump"):
            r = robust_c(hyp, p, COST, p["adv"])
        else:
            r = robust_d(hyp, p, COST, p["adv"])
        (ODIR / f"robust_{hyp}.json").write_text(json.dumps(r, default=float, indent=1))
        d = r["details"]
        print(hyp, r["cfg"], round(time.time() - t0, 1))
        print("  IS", d.get("IS"), "\n  OOS", d.get("OOS"))
        print("  per_year", d["per_year"], "\n  half", d["half_years"])
        print("  trades", d.get("trades_by_period"))
        print("  variants", r.get("variants"))
        print("  perturb", {k: (v["IS"]["sharpe"], v["OOS"]["sharpe"]) for k, v in r["perturb"].items()})
        print("  random", {k: v for k, v in r["random_baseline"].items() if k != "draws"})
        print("  capacity", d.get("capacity_per_trade_usd"), d.get("capacity_account_usd_p25"), "turnover", d.get("turnover_per_year_x_equity"))


if __name__ == "__main__":
    main(sys.argv[1:])
