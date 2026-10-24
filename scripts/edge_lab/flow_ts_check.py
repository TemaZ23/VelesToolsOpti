"""
flow (a): проверка устойчивости IS-лучшей временной (по монете) конфигурации «шок числа сделок»
(1h-сигнал, удержание 24 ч, порог 1.0, лонг при всплеске) на 10 крупных монетах.

    PYTHONPATH=. python3 -m scripts.edge_lab.flow_ts_check     # -> output/edge_lab/flow/ts_check.json

Сигнал на часовом баре b: trd_shock = log(trades[b]) - mean(log trades за 168 прошлых часов).
Ребаланс раз в 24 ч на баре b ≡ offset (mod 24): позиция +1/10 капитала в монете, если trd_shock > thr,
−1/10 если < −thr (dir=+1), иначе 0; вход open[b+1], выход close[b+24]; издержки |Δw|·cost_per_side(ADV),
фандинг по факту. Проверяем: все 24 часа ребаланса (offset), порог ±25%, только лонг / только шорт,
вклад монет, по годам, «десезонированный» шок (относительно того же часа суток за 7 прошлых дней).
"""

from __future__ import annotations

import json

import numpy as np

from scripts.edge_lab.common import IS, OOS, OUT, metrics, ms, per_year
from scripts.edge_lab.flow_common import cost_matrix, half_years
from scripts.edge_lab.flow_data import load_adv, load_trades
from scripts.edge_lab.flow_imbalance import TS_COINS, roll_past
from scripts.edge_lab.common import load_hourly, load_funding, universe, HOUR

ODIR = OUT / "flow"


def load_coins():
    uni = universe()
    syms = sorted(uni)
    t0 = min(ms(v["first"]) for v in uni.values()) // HOUR * HOUR
    t1 = max(ms(v["last"]) for v in uni.values()) // HOUR * HOUR
    ts = np.arange(t0, t1 + HOUR, HOUR, dtype=np.int64)
    T, n = len(ts), len(TS_COINS)
    o = np.full((T, n), np.nan)
    c = np.full((T, n), np.nan)
    tr = np.full((T, n), np.nan)
    f = np.zeros((T, n))
    for k, s in enumerate(TS_COINS):
        d = load_hourly(s)
        b = (d["ts"] - t0) // HOUR
        o[b, k] = d["ohlc"][:, 0]
        c[b, k] = d["ohlc"][:, 3]
        tr[b, k] = d["trades"]
        fd = load_funding(s)
        fb = (fd["ts"] - t0) // HOUR
        ok = (fb >= 0) & (fb < T)
        np.add.at(f[:, k], fb[ok], fd["rate"][ok])
    cols = [syms.index(s) for s in TS_COINS]
    adv = load_adv()[:, cols].astype(float)
    # сверка с панелью: trades
    trp = load_trades()[:, cols]
    assert np.allclose(np.nan_to_num(trp), np.nan_to_num(tr.astype(np.float32)), rtol=1e-5), "trades mismatch"
    return ts, o, c, tr, f, adv


def run_ts(ts, o, c, sig, f, cost, thr, direction, offset, hold=24, side="both"):
    T, n = c.shape
    idx = np.arange(offset, T - hold - 1, hold)
    s = sig[idx]
    pos = np.where(s > thr, 1.0, np.where(s < -thr, -1.0, 0.0)) * direction
    if side == "long":
        pos = np.maximum(pos, 0)
    elif side == "short":
        pos = np.minimum(pos, 0)
    alive = np.isfinite(c[idx]) & np.isfinite(o[np.minimum(idx + 1, T - 1)])
    pos = np.where(alive, pos, 0.0) / n
    with np.errstate(invalid="ignore", divide="ignore"):
        fwd = c[idx + hold] / o[idx + 1] - 1
    fwd = np.nan_to_num(fwd)
    fc = np.vstack([np.zeros((1, n)), np.cumsum(f, axis=0)])
    fu = fc[idx + hold + 1] - fc[idx + 1]
    gross_c = pos * fwd - pos * fu
    dw = np.abs(np.diff(np.vstack([np.zeros((1, n)), pos]), axis=0))
    net_c = gross_c - dw * np.nan_to_num(cost[idx], nan=0.0045)
    return ts[idx], net_c.sum(1), gross_c.sum(1), net_c, pos


def stats(tsb, r):
    out = {}
    for name, (a, b) in (("IS", IS), ("OOS", OOS)):
        m = (tsb >= ms(a)) & (tsb < ms(b))
        mm = metrics(tsb[m], r[m])
        out[name] = {k: mm[k] for k in ("sharpe", "cagr_pct", "max_dd_pct", "avg_month_pct", "worst_month_pct", "pos_months_pct")}
    return out


def main() -> None:
    ts, o, c, tr, f, adv = load_coins()
    cost = cost_matrix(adv.astype(np.float32)).astype(float)
    ltr = np.where(tr > 0, np.log(tr), np.nan).astype(np.float32)
    m, _ = roll_past(ltr, 168)
    sig = (ltr - m).astype(float)
    # десезонированный: относительно того же часа суток за 7 прошлых дней
    T, n = ltr.shape
    ds = np.full((T, n), np.nan)
    acc = np.zeros((T, n))
    cnt = np.zeros((T, n))
    for k in range(1, 8):
        sh = np.full((T, n), np.nan)
        sh[24 * k :] = ltr[: -24 * k]
        acc += np.nan_to_num(sh)
        cnt += np.isfinite(sh)
    with np.errstate(invalid="ignore", divide="ignore"):
        ds = ltr - np.where(cnt >= 4, acc / cnt, np.nan)
    res = {"desc": __doc__}
    tsb, net, gross, net_c, pos = run_ts(ts, o, c, sig, f, cost, 1.0, 1, 0)
    res["base"] = stats(tsb, net)
    res["base"]["per_year"] = per_year(tsb, net)
    res["base"]["half_years"] = half_years(tsb, net)
    res["base"]["avg_gross_exposure"] = round(float(np.mean(np.abs(pos).sum(1))), 4)
    res["base"]["active_days_share"] = round(float(np.mean(np.abs(pos).sum(1) > 0)), 3)
    res["base"]["per_coin_net_sum"] = {s: round(float(net_c[:, k].sum()), 4) for k, s in enumerate(TS_COINS)}
    act = np.abs(pos) > 0
    starts = act & ~np.vstack([np.zeros((1, n), bool), act[:-1]])
    res["base"]["trades"] = int(starts.sum())
    res["base"]["trades_IS"] = int(starts[(tsb >= ms(IS[0])) & (tsb < ms(IS[1]))].sum())
    res["base"]["trades_OOS"] = int(starts[(tsb >= ms(OOS[0])) & (tsb < ms(OOS[1]))].sum())
    res["base"]["avg_hold_h"] = round(float(act.sum() / max(starts.sum(), 1) * 24), 1)
    dw = np.abs(np.diff(np.vstack([np.zeros((1, n)), pos]), axis=0)).sum(1)
    res["base"]["turnover_per_year_x_equity"] = round(float(dw.sum() / ((tsb[-1] - tsb[0]) / 86400000 / 365.25)), 2)
    res["base"]["gross_vs_net_sum"] = {"gross": round(float(gross.sum()), 4), "net": round(float(net.sum()), 4)}
    fc_ = np.vstack([np.zeros((1, n)), np.cumsum(f, axis=0)])
    idx_ = np.arange(0, len(ts) - 25, 24)
    fu_ = (fc_[idx_ + 25] - fc_[idx_ + 1]) * pos
    res["base"]["funding_sum"] = round(float(-fu_.sum()), 4)
    np.savez(ODIR / "tradecount_ts_daily.npz", ts=(tsb // 86400000) * 86400000, ret=net)
    res["offsets"] = {}
    for off in range(24):
        tsb, net, *_ = run_ts(ts, o, c, sig, f, cost, 1.0, 1, off)
        res["offsets"][off] = stats(tsb, net)
    res["thr"] = {}
    for thr in (0.75, 1.25, 1.5):
        tsb, net, *_ = run_ts(ts, o, c, sig, f, cost, thr, 1, 0)
        res["thr"][thr] = stats(tsb, net)
    res["side"] = {}
    for side in ("long", "short"):
        tsb, net, *_ = run_ts(ts, o, c, sig, f, cost, 1.0, 1, 0, side=side)
        res["side"][side] = stats(tsb, net)
    res["hold"] = {}
    for hold in (12, 18, 30, 48):
        tsb, net, *_ = run_ts(ts, o, c, sig, f, cost, 1.0, 1, 0, hold=hold)
        res["hold"][hold] = stats(tsb, net)
    res["deseason"] = {}
    for thr in (0.75, 1.0, 1.25):
        sh = []
        for off in (0, 6, 12, 18):
            tsb, net, *_ = run_ts(ts, o, c, ds, f, cost, thr, 1, off)
            sh.append(stats(tsb, net))
        res["deseason"][thr] = sh
    (ODIR / "ts_check.json").write_text(json.dumps(res, indent=1, default=float))
    b = res["base"]
    print("base", b["IS"], b["OOS"], b["per_year"], "exp", b["avg_gross_exposure"], "active", b["active_days_share"])
    print("per coin", b["per_coin_net_sum"])
    oi = np.array([v["IS"]["sharpe"] for v in res["offsets"].values()])
    oo = np.array([v["OOS"]["sharpe"] for v in res["offsets"].values()])
    print("offsets IS", np.round(oi, 2), "\n   OOS", np.round(oo, 2), "median IS %.2f OOS %.2f" % (np.median(oi), np.median(oo)))
    print("thr", {k: (v["IS"]["sharpe"], v["OOS"]["sharpe"]) for k, v in res["thr"].items()})
    print("side", {k: (v["IS"]["sharpe"], v["OOS"]["sharpe"]) for k, v in res["side"].items()})
    print("hold", {k: (v["IS"]["sharpe"], v["OOS"]["sharpe"]) for k, v in res["hold"].items()})
    print("deseason", {k: [(x["IS"]["sharpe"], x["OOS"]["sharpe"]) for x in v] for k, v in res["deseason"].items()})


if __name__ == "__main__":
    main()
