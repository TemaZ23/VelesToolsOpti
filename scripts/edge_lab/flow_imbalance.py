"""
flow (a): дисбаланс агрессивных покупок (taker_buy_qvol/qvol) и шоки числа сделок как предикторы доходности
следующих 1h / 4h / 24h — кросс-секционно и по отдельным монетам.

    PYTHONPATH=. python3 -m scripts.edge_lab.flow_imbalance        # -> output/edge_lab/flow/part_a.json

Частоты: 1h, 4h, 1d (бары агрегируются из часовой панели, 4h/1d начинаются в 00:00 UTC), плюс часовой сигнал
с горизонтом 4h и 24h (ребаланс раз в 4/24 ч по сигналу ПОСЛЕДНЕГО часа).
Сигналы на баре b (только данные ≤ b; L — окно истории монеты: 7 суток):
  tbr_dm   = tbr - mean(tbr за L прошлых баров), tbr = taker_buy_qvol/qvol;
  tbr_z    = tbr_dm / std(tbr за L прошлых баров);
  netflow  = (2*taker_buy - qvol) / ADV_30d;
  trd_shock = log(trades) - mean(log trades за L прошлых баров);
  size_shock = log(qvol/trades) - mean(... за L прошлых баров)  (средний размер сделки);
  vol_shock = log(qvol) - mean(log qvol за L);
  div      = rank(tbr_z) - rank(ret бара) (поток «против» цены: поглощение).
Кросс-секционный тест: IC Спирмена с доходностью open[b+1] -> close[b+1]; портфель L/S (верхние/нижние q по
сигналу, 0.5 капитала на ногу), издержки = |Δw| * cost_per_side(ADV), фандинг по факту.
Временные ряды по 10 ликвидным монетам: позиция ±1/10 капитала, когда |tbr_z| > k (или другой сигнал), держим 1 бар.
"""

from __future__ import annotations

import json
import time

import numpy as np

from scripts.edge_lab.common import DAY, IS, OOS, OUT, metrics, ms, per_year
from scripts.edge_lab.flow_common import age_hours, cost_matrix, half_years, load_all

ODIR = OUT / "flow"
TS_COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT", "LTCUSDT", "SOLUSDT", "AVAXUSDT"]


def agg(p: dict, k: int) -> dict:
    """Агрегирует часовую панель в бары по k часов (начиная с первого часа панели = 00:00 UTC)."""
    T, N = p["close"].shape
    nb = T // k
    sl = slice(0, nb * k)

    def rs(a):
        return a[sl].reshape(nb, k, a.shape[1])

    with np.errstate(invalid="ignore"):
        o = rs(p["open"])
        c = rs(p["close"])
        alive = ~np.isnan(c)
        anyv = alive.any(axis=1)
        # open — первый валидный, close — последний валидный
        fi = np.argmax(alive, axis=1)
        li = k - 1 - np.argmax(alive[:, ::-1, :], axis=1)
        ii = np.arange(nb)[:, None]
        jj = np.arange(N)[None, :]
        out = {
            "ts": p["ts"][sl][::k],
            "open": np.where(anyv, o[ii, fi, jj], np.nan).astype(np.float32),
            "close": np.where(anyv, c[ii, li, jj], np.nan).astype(np.float32),
            "qvol": np.where(anyv, np.nansum(rs(p["qvol"]), axis=1), np.nan).astype(np.float32),
            "tb": np.where(anyv, np.nansum(rs(p["taker_buy_qvol"]), axis=1), np.nan).astype(np.float32),
            "trades": np.where(anyv, np.nansum(rs(p["trades"]), axis=1), np.nan).astype(np.float32),
            "funding": np.nansum(rs(p["funding"]), axis=1).astype(np.float32),
            "adv": p["adv"][sl][k - 1 :: k],  # ADV, известный на конец бара
            "hours": alive.sum(axis=1).astype(np.int8),
        }
    return out


def roll_past(x: np.ndarray, L: int, chunk: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Среднее и std x по L ПРОШЛЫМ барам (без текущего), только валидные; нужно ≥ L/2 валидных.
    Считается блоками столбцов (экономия памяти)."""
    T, N = x.shape
    mean = np.full((T, N), np.nan, np.float32)
    sd = np.full((T, N), np.nan, np.float32)
    hi = np.arange(T)
    lo = np.maximum(hi - L, 0)
    for c0 in range(0, N, chunk):
        xc = x[:, c0 : c0 + chunk]
        v = np.nan_to_num(xc).astype(np.float64)
        cnt = (~np.isnan(xc)).astype(np.float64)
        z = np.zeros((1, xc.shape[1]))
        cs = np.cumsum(np.vstack([z, v]), axis=0)
        cs2 = np.cumsum(np.vstack([z, v * v]), axis=0)
        cc = np.cumsum(np.vstack([z, cnt]), axis=0)
        s = cs[hi] - cs[lo]
        s2 = cs2[hi] - cs2[lo]
        n = cc[hi] - cc[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            m = s / n
            var = s2 / n - m * m
            okm = n >= max(2, L // 2)
            mean[:, c0 : c0 + chunk] = np.where(okm, m, np.nan)
            sd[:, c0 : c0 + chunk] = np.where(okm, np.sqrt(np.maximum(var, 0)), np.nan)
    return mean, sd


def rank_rows(x: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Ранги по строкам в [0,1] (NaN остаются NaN); блоками строк."""
    T, N = x.shape
    out = np.full((T, N), np.nan, np.float32)
    ar = np.arange(N, dtype=np.int32)
    for r0 in range(0, T, chunk):
        xc = x[r0 : r0 + chunk]
        nanm = np.isnan(xc)
        y = np.where(nanm, np.inf, xc)
        order = np.argsort(y, axis=1, kind="stable").astype(np.int32)
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.broadcast_to(ar, order.shape), axis=1)
        n = (~nanm).sum(axis=1, keepdims=True)
        r = (ranks / np.maximum(n - 1, 1)).astype(np.float32)
        r[nanm] = np.nan
        out[r0 : r0 + chunk] = r
    return out


def spearman_rows(a: np.ndarray, b: np.ndarray, chunk: int = 4096) -> np.ndarray:
    T = a.shape[0]
    ic = np.full(T, np.nan)
    for r0 in range(0, T, chunk):
        ac, bc = a[r0 : r0 + chunk], b[r0 : r0 + chunk]
        m = np.isfinite(ac) & np.isfinite(bc)
        ra = rank_rows(np.where(m, ac, np.nan))
        rb = rank_rows(np.where(m, bc, np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            ra = ra - np.nanmean(ra, axis=1, keepdims=True)
            rb = rb - np.nanmean(rb, axis=1, keepdims=True)
            num = np.nansum(ra * rb, axis=1)
            den = np.sqrt(np.nansum(ra * ra, axis=1) * np.nansum(rb * rb, axis=1))
            icc = num / den
        icc[m.sum(axis=1) < 10] = np.nan
        ic[r0 : r0 + chunk] = icc
    return ic


def nw_t(x: np.ndarray, lags: int = 5) -> float:
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return float("nan")
    u = x - x.mean()
    g0 = np.dot(u, u) / n
    s = g0
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * np.dot(u[L:], u[:-L]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else float("nan")


def make_signals(b: dict, L: int) -> dict:
    qv, tb, tr, c, o = b["qvol"], b["tb"], b["trades"], b["close"], b["open"]
    with np.errstate(invalid="ignore", divide="ignore"):
        tbr = np.where(qv > 0, tb / qv, np.nan).astype(np.float32)
        m, sd = roll_past(tbr, L)
        tbr_dm = tbr - m
        tbr_z = np.where(sd > 1e-6, tbr_dm / sd, np.nan).astype(np.float32)
        netflow = ((2 * tb - qv) / np.where(b["adv"] > 0, b["adv"], np.nan)).astype(np.float32)
        ltr = np.where(tr > 0, np.log(tr), np.nan).astype(np.float32)
        m2, _ = roll_past(ltr, L)
        trd = (ltr - m2).astype(np.float32)
        lsz = np.where((tr > 0) & (qv > 0), np.log(qv / tr), np.nan).astype(np.float32)
        m3, _ = roll_past(lsz, L)
        size = (lsz - m3).astype(np.float32)
        lqv = np.where(qv > 0, np.log(qv), np.nan).astype(np.float32)
        m4, _ = roll_past(lqv, L)
        vol = (lqv - m4).astype(np.float32)
        ret = (c / o - 1).astype(np.float32)
    return {"tbr_dm": tbr_dm, "tbr_z": tbr_z, "netflow": netflow, "trd_shock": trd, "size_shock": size,
            "vol_shock": vol, "ret": ret}  # fmt: skip


def ls_portfolio(sig, fwd, fund, cost, elig, q, step_bars=1):
    """L/S: на баре b (кратном step_bars) берём верх/низ q сигнала среди elig; держим до следующего ребаланса.
    fwd[b] — доходность удержания после сигнала b (open[b+1] -> close[b+step]); fund[b] — фандинг за это время.
    Возвращает ряд доходностей капитала на ребаланс (gross, net), оборот."""
    idx = np.arange(0, sig.shape[0], step_bars)
    sig, fwd, fund, cost, elig = sig[idx], fwd[idx], fund[idx], cost[idx], elig[idx]
    T, N = sig.shape
    s = np.where(elig & np.isfinite(sig) & np.isfinite(fwd), sig, np.nan)
    r = rank_rows(s)
    del s
    n = np.isfinite(r).sum(axis=1)
    longm = r >= 1 - q
    shortm = r <= q
    nl = longm.sum(axis=1)
    ns = shortm.sum(axis=1)
    okb = (n >= 20) & (nl > 0) & (ns > 0)
    w = np.zeros((T, N), np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        w += np.where(longm, 0.5 / np.maximum(nl, 1)[:, None], 0).astype(np.float32)
        w -= np.where(shortm, 0.5 / np.maximum(ns, 1)[:, None], 0).astype(np.float32)
    w[~okb] = 0
    del r, longm, shortm
    gross = (w * np.nan_to_num(fwd)).sum(axis=1) - (w * np.nan_to_num(fund)).sum(axis=1)
    dw = np.abs(np.diff(np.vstack([np.zeros((1, N), np.float32), w]), axis=0))
    cst = (dw * np.nan_to_num(cost, nan=0.0045)).sum(axis=1)
    return gross, gross - cst, dw.sum(axis=1), idx


def period_stats(ts_b, rets, step_ms):
    out = {}
    for name, (a, bnd) in (("IS", IS), ("OOS", OOS)):
        m = (ts_b >= ms(a)) & (ts_b < ms(bnd))
        if m.sum() < 30:
            continue
        rr = rets[m]
        mm = metrics(ts_b[m], rr)
        out[name] = {k: mm[k] for k in ("sharpe", "cagr_pct", "max_dd_pct", "avg_month_pct", "worst_month_pct", "pos_months_pct")}
        out[name]["mean_bp"] = round(float(np.mean(rr)) * 1e4, 3)
    return out


def run() -> dict:
    t0 = time.time()
    p = load_all(with_trades=True)
    del p["high"], p["low"]
    crypto = p["crypto"]
    res = {"desc": __doc__, "xs": [], "ic": [], "ts": [], "buckets": []}
    ODIR.mkdir(parents=True, exist_ok=True)
    syms = list(p["symbols"])
    for fname, k, L, steps in (("1h", 1, 168, (1, 4, 24)), ("4h", 4, 42, (1,)), ("1d", 24, 30, (1,))):
        b = agg(p, k) if k > 1 else {"ts": p["ts"], "open": p["open"], "close": p["close"], "qvol": p["qvol"],
                                      "tb": p["taker_buy_qvol"], "trades": p["trades"], "funding": p["funding"],
                                      "adv": p["adv"], "hours": p["hours"]}  # fmt: skip
        T, N = b["close"].shape
        age = age_hours(b["close"])
        alive = np.isfinite(b["close"]) & np.isfinite(b["open"])
        sigs = make_signals(b, L)
        cost = cost_matrix(b["adv"])
        print(fname, "signals", round(time.time() - t0, 1), flush=True)
        for step in steps:
            # fwd: вход open[b+1], выход close[b+step]
            o1 = np.full((T, N), np.nan, np.float32)
            o1[:-1] = b["open"][1:]
            cS = np.full((T, N), np.nan, np.float32)
            cS[: T - step] = b["close"][step:]
            with np.errstate(invalid="ignore", divide="ignore"):
                fwd = cS / o1 - 1
            fu = np.zeros((T, N), np.float32)
            fcum = np.vstack([np.zeros((1, N)), np.cumsum(b["funding"].astype(np.float64), axis=0)])
            # фандинг баров b+1..b+step
            hi = np.minimum(np.arange(T) + step + 1, T)
            lo = np.minimum(np.arange(T) + 1, T)
            fu[:] = (fcum[hi] - fcum[lo]).astype(np.float32)
            tag = fname if step == 1 else f"1h->{step}h"
            for uni_name, min_adv in (("adv5M", 5e6), ("adv50M", 5e7)):
                elig = alive & crypto[None, :] & (age >= max(30, 720 // k)) & (np.nan_to_num(b["adv"]) >= min_adv)
                for sname in ("tbr_dm", "tbr_z", "netflow", "trd_shock", "size_shock", "vol_shock", "div"):
                    if sname == "div":
                        sg = rank_rows(np.where(elig, sigs["tbr_z"], np.nan)) - rank_rows(np.where(elig, sigs["ret"], np.nan))
                    else:
                        sg = sigs[sname]
                    sgm = np.where(elig, sg, np.nan)
                    idx = np.arange(0, T, step)
                    ic = spearman_rows(sgm[idx], fwd[idx])
                    tsb = b["ts"][idx]
                    icrow = {"freq": tag, "uni": uni_name, "signal": sname}
                    for per, (a, bb) in (("IS", IS), ("OOS", OOS)):
                        mm = (tsb >= ms(a)) & (tsb < ms(bb))
                        icrow[per] = {"ic_mean": round(float(np.nanmean(ic[mm])), 4), "t_nw": round(nw_t(ic[mm]), 2),
                                      "n": int(np.isfinite(ic[mm]).sum())}  # fmt: skip
                    res["ic"].append(icrow)
                    for q in (0.1, 0.2):
                        gross, net, turn, idx2 = ls_portfolio(sgm, fwd, fu, cost, elig, q, step)
                        tsb2 = b["ts"][idx2]
                        for direction in (1, -1):
                            row = {"freq": tag, "uni": uni_name, "signal": sname, "q": q, "dir": direction,
                                   "turnover_per_rebal": round(float(np.mean(turn)), 3)}  # fmt: skip
                            row["gross"] = period_stats(tsb2, direction * gross, step * k * 3600000)
                            row["net"] = period_stats(tsb2, direction * gross - (gross - net), step * k * 3600000)
                            res["xs"].append(row)
                print(fname, step, uni_name, round(time.time() - t0, 1), flush=True)
            # бакеты tbr_z / trd_shock (пул монет ADV>=50M): средняя fwd доходность
            elig = alive & crypto[None, :] & (age >= max(30, 720 // k)) & (np.nan_to_num(b["adv"]) >= 5e7)
            for sname in ("tbr_z", "trd_shock", "netflow"):
                sg = sigs[sname]
                edges = [-np.inf, -2, -1, -0.5, 0.5, 1, 2, np.inf] if sname != "netflow" else [-np.inf, -0.05, -0.02, -0.005, 0.005, 0.02, 0.05, np.inf]
                for per, (a, bb) in (("IS", IS), ("OOS", OOS)):
                    tm = (b["ts"] >= ms(a)) & (b["ts"] < ms(bb))
                    rowsel = np.zeros(T, bool)
                    rowsel[np.arange(0, T, step)] = True
                    mm = elig & tm[:, None] & rowsel[:, None] & np.isfinite(sg) & np.isfinite(fwd)
                    x, y = sg[mm], fwd[mm]
                    br = {"freq": tag, "signal": sname, "period": per, "bins": []}
                    for lo_, hi_ in zip(edges[:-1], edges[1:]):
                        s_ = (x >= lo_) & (x < hi_)
                        br["bins"].append({"lo": float(lo_), "hi": float(hi_), "n": int(s_.sum()),
                                           "mean_bp": round(float(np.mean(y[s_])) * 1e4, 2) if s_.sum() else None})  # fmt: skip
                    res["buckets"].append(br)
            # временные ряды по 10 монетам
            if step == 1 or fname == "1h":
                cj = [syms.index(s) for s in TS_COINS]
                for sname in ("tbr_z", "netflow", "trd_shock", "size_shock"):
                    for thr in ((1.0, 2.0) if sname != "netflow" else (0.01, 0.03)):
                        sg = sigs[sname][:, cj]
                        pos = np.where(sg > thr, 1.0, np.where(sg < -thr, -1.0, 0.0)) * (1 / len(cj))
                        pos = np.where(alive[:, cj], pos, 0)
                        idx = np.arange(0, T, step)
                        pw = pos[idx]
                        fw = np.nan_to_num(fwd[idx][:, cj])
                        fuu = np.nan_to_num(fu[idx][:, cj])
                        cc = np.nan_to_num(cost[idx][:, cj], nan=0.0045)
                        gross = (pw * fw).sum(1) - (pw * fuu).sum(1)
                        dw = np.abs(np.diff(np.vstack([np.zeros((1, len(cj))), pw]), axis=0))
                        net = gross - (dw * cc).sum(1)
                        tsb = b["ts"][idx]
                        for direction in (1, -1):
                            row = {"freq": tag, "signal": sname, "thr": thr, "dir": direction,
                                   "active_share": round(float(np.mean(np.abs(pw) > 0)), 3),
                                   "turnover_per_rebal": round(float(np.mean(dw.sum(1))), 4)}  # fmt: skip
                            row["gross"] = period_stats(tsb, direction * gross, 0)
                            row["net"] = period_stats(tsb, direction * gross - (gross - net), 0)
                            res["ts"].append(row)
            del fwd, fu, o1, cS
        del sigs, b, cost, age, alive
    (ODIR / "part_a.json").write_text(json.dumps(res, default=float))
    print("done", round(time.time() - t0, 1))
    return res


def show(res: dict | None = None) -> None:
    if res is None:
        res = json.loads((ODIR / "part_a.json").read_text())
    print("IC (mean, NW t) IS | OOS")
    for r in res["ic"]:
        print(f"{r['freq']:7s} {r['uni']:7s} {r['signal']:10s} IS {r['IS']['ic_mean']:+.4f} ({r['IS']['t_nw']:+.1f}) | OOS {r['OOS']['ic_mean']:+.4f} ({r['OOS']['t_nw']:+.1f})")
    print("\nXS L/S: top by IS net Sharpe")
    rows = sorted(res["xs"], key=lambda r: -r["net"].get("IS", {}).get("sharpe", -99))
    for r in rows[:25]:
        print(f"{r['freq']:7s} {r['uni']:7s} {r['signal']:10s} q={r['q']} d={r['dir']:+d} turn={r['turnover_per_rebal']:.2f} "
              f"gross IS {r['gross']['IS']['sharpe']:+.2f} OOS {r['gross']['OOS']['sharpe']:+.2f} | net IS {r['net']['IS']['sharpe']:+.2f} "
              f"({r['net']['IS']['cagr_pct']:+.1f}%) OOS {r['net']['OOS']['sharpe']:+.2f} ({r['net']['OOS']['cagr_pct']:+.1f}%)")  # fmt: skip
    print("\nXS L/S: top by IS GROSS Sharpe")
    rows = sorted(res["xs"], key=lambda r: -r["gross"].get("IS", {}).get("sharpe", -99))
    for r in rows[:15]:
        print(f"{r['freq']:7s} {r['uni']:7s} {r['signal']:10s} q={r['q']} d={r['dir']:+d} turn={r['turnover_per_rebal']:.2f} "
              f"gross IS {r['gross']['IS']['sharpe']:+.2f} ({r['gross']['IS']['mean_bp']:+.1f}bp) OOS {r['gross']['OOS']['sharpe']:+.2f} ({r['gross']['OOS']['mean_bp']:+.1f}bp) | net IS {r['net']['IS']['sharpe']:+.2f} OOS {r['net']['OOS']['sharpe']:+.2f}")  # fmt: skip
    print("\nTS (10 coins): top by IS net Sharpe")
    rows = sorted(res["ts"], key=lambda r: -r["net"].get("IS", {}).get("sharpe", -99))
    for r in rows[:15]:
        print(f"{r['freq']:7s} {r['signal']:10s} thr={r['thr']} d={r['dir']:+d} act={r['active_share']} turn={r['turnover_per_rebal']} "
              f"gross IS {r['gross']['IS']['sharpe']:+.2f} OOS {r['gross']['OOS']['sharpe']:+.2f} | net IS {r['net']['IS']['sharpe']:+.2f} OOS {r['net']['OOS']['sharpe']:+.2f}")  # fmt: skip
    print("\nBuckets (pooled ADV>=50M), mean fwd bp")
    for r in res["buckets"]:
        print(r["freq"], r["signal"], r["period"], [(b["n"], b["mean_bp"]) for b in r["bins"]])


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "show":
        show()
    else:
        show(run())
