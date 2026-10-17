"""
Семейство funding — неэффективности, связанные с фандингом USDT-M перпов Binance.

    PYTHONPATH=. python3 -m scripts.edge_lab.funding_data spot && ... funding_data build   # данные
    PYTHONPATH=. python3 -m scripts.edge_lab.funding xs       # (b) кросс-секционный фактор фандинга
    PYTHONPATH=. python3 -m scripts.edge_lab.funding timing   # (c) экстремумы фандинга как сигнал
    PYTHONPATH=. python3 -m scripts.edge_lab.funding carry    # (a) cash-and-carry спот/перп
    PYTHONPATH=. python3 -m scripts.edge_lab.funding_micro    # (d) микроструктура выплат (часовые)
    PYTHONPATH=. python3 -m scripts.edge_lab.funding_m1 fetch|study      # (d) минутки вокруг выплат
    PYTHONPATH=. python3 -m scripts.edge_lab.funding_tick fetch|study    # (d) сделки aggTrades ±120 с
    PYTHONPATH=. python3 -m scripts.edge_lab.funding control              # (c) контроль: реверсия цены
    PYTHONPATH=. python3 -m scripts.edge_lab.funding_robust carry|alt    # устойчивость кандидатов
    PYTHONPATH=. python3 -m scripts.edge_lab.funding_report               # results.json / best_daily.npz

Тайминг (без заглядывания): решение после закрытия дневного бара t по данным <= t, сделка по open[t+1].
Позиция держится на баре k = open[k] -> open[k+1]; цена: ret[k] = open[k+1]/open[k]-1 (если монета
перестала торговаться — выход по close[k]); фандинг: fund_hold[k] = выплаты в (open_k, open_{k+1}]
(см. funding_data.py), знак: лонг платит положительную ставку. Сигнал по фандингу — panel funding[<=t].
"""

from __future__ import annotations

import json
import sys
import time

import numba as nb
import numpy as np

from scripts.edge_lab.common import (IS, OOS, OUT, adv_usd, cost_per_side, liquid_mask, metrics, panel, per_year,
                                     rolling_mean_past, window_idx)  # fmt: skip
from scripts.edge_lab.funding_data import load_aux

OUTD = OUT / "funding"
SPOT_FEE = 0.001  # спот Binance VIP0 тейкер/мейкер 0.1% (0.075% с оплатой в BNB)


# ----------------------------------------------------------------------------------------------- data
def slip_of(adv: np.ndarray) -> np.ndarray:
    return cost_per_side(adv) - 0.0005  # та же лестница проскальзывания, что и в common.cost_per_side


def prep() -> dict:
    p = dict(panel("1d"))
    aux = load_aux()
    # В архиве klines после делистинга идут «зомби»-бары: цена заморожена, объём 0. Считаем монету
    # торгующейся только при qvol > 0 (иначе делистинги ~150 символов были бы невидимы).
    alive = ~np.isnan(p["open"]) & ~np.isnan(p["close"]) & (np.nan_to_num(p["qvol"]) > 0)
    for k in ("open", "high", "low", "close", "qvol", "taker_buy_qvol"):
        p[k] = np.where(alive, p[k], np.nan)
    p["funding"] = np.where(alive, p["funding"], 0.0)
    O, C, H = p["open"], p["close"], p["high"]
    T, N = O.shape
    nxt = np.vstack([O[1:], np.full((1, N), np.nan)])
    stop = alive & np.isnan(nxt)
    stop[-1] = False  # конец выборки — не делистинг
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = np.where(stop, C / O - 1, nxt / O - 1)
    ret[-1] = C[-1] / O[-1] - 1
    ret[~alive] = 0.0
    ret = np.nan_to_num(ret)
    # настоящий делистинг: последний торговый бар символа, если торги закончились раньше чем за 7 дней
    # до конца выборки (флаг universe.json ловит только 31 из ~150 делистингов)
    true_del = np.zeros((T, N), dtype=np.bool_)
    for j in range(N):
        idx = np.where(alive[:, j])[0]
        if len(idx) and idx[-1] < T - 8:
            true_del[idx[-1], j] = True
    adv = np.nan_to_num(adv_usd(p))
    d = dict(p=p, aux=aux, ts=p["ts"], syms=list(p["symbols"]), O=O, C=C, H=H, alive=alive, stop=stop,
             true_del=true_del, ret=ret, adv=adv, cost=cost_per_side(adv), fsig=p["funding"].astype(float),
             fhold=np.where(alive, aux["fund_hold"], 0.0))  # fmt: skip
    return d


# --------------------------------------------------------------------------------------------- engine
@nb.njit(cache=True)
def run_book(target, rebal, ret, fhold, cost, tradable, stop, true_del, pess):
    """Портфель весов (доли капитала). target[t] — цель после бара t, исполняется по open[t+1], если rebal[t].
    Возвращает дневные ряды (по барам k удержания): ret, price, fund, cost, turnover, gross, trades."""
    T, N = ret.shape
    w = np.zeros(N)
    out = np.zeros((T, 6))
    trades = 0
    for t in range(T - 1):
        k = t + 1
        c = 0.0
        turn = 0.0
        if rebal[t]:
            for i in range(N):
                tw = target[t, i]
                if np.isnan(tw) or not tradable[k, i]:
                    tw = 0.0
                dw = tw - w[i]
                if dw != 0.0:
                    if w[i] == 0.0 and tw != 0.0:
                        trades += 1
                    turn += abs(dw)
                    c += abs(dw) * cost[t, i]
                    w[i] = tw
        pr = 0.0
        fr = 0.0
        gross = 0.0
        for i in range(N):
            if w[i] == 0.0:
                continue
            r = ret[k, i]
            if pess and true_del[k, i]:
                r = -0.5 if w[i] > 0 else 0.5  # пессимистично: −50% к позиции (для шорта — рост на 50%)
            gross += abs(w[i])
            pr += w[i] * r
            fr -= w[i] * fhold[k, i]
            if stop[k, i]:  # вынужденный выход по последнему close
                c += abs(w[i] * (1 + r)) * cost[t, i]
                turn += abs(w[i] * (1 + r))
        R = pr + fr - c
        for i in range(N):
            if w[i] == 0.0:
                continue
            if stop[k, i]:
                w[i] = 0.0
            else:
                w[i] = w[i] * (1 + ret[k, i]) / (1 + R) if 1 + R > 1e-9 else 0.0
        out[k, 0] = R
        out[k, 1] = pr
        out[k, 2] = fr
        out[k, 3] = c
        out[k, 4] = turn
        out[k, 5] = gross
    return out, trades


def stats(ts, out, trades, extra=None) -> dict:
    """IS/OOS/полные метрики + разложение P&L."""
    res = {}
    for name, win in (("is", IS), ("oos", OOS), ("full", ("2020-01-01", "2026-10-01"))):
        a, b = window_idx(ts, win)
        o = out[a:b]
        m = metrics(ts[a:b], o[:, 0])
        yrs = (b - a) / 365.25
        m["ann_price_pct"] = round(float(o[:, 1].sum() / yrs * 100), 2)
        m["ann_fund_pct"] = round(float(o[:, 2].sum() / yrs * 100), 2)
        m["ann_cost_pct"] = round(float(o[:, 3].sum() / yrs * 100), 2)
        m["turnover_per_day"] = round(float(o[:, 4].mean()), 4)
        m["avg_gross"] = round(float(o[:, 5].mean()), 3)
        m["hold_days"] = round(float(2 * o[:, 5].mean() / o[:, 4].mean()), 1) if o[:, 4].mean() > 0 else None
        res[name] = m
    res["trades"] = int(trades)
    if extra:
        res.update(extra)
    return res


def half_years(ts, r) -> dict:
    a, b = window_idx(ts, OOS)
    t, x = ts[a:b], r[a:b]
    d = t.astype("datetime64[ms]").astype("datetime64[M]").astype(int)
    yr, mo = d // 12 + 1970, d % 12
    out = {}
    for y in np.unique(yr):
        for h in (0, 1):
            m = (yr == y) & ((mo >= 6) == bool(h))
            if m.sum() > 20:
                out[f"{y}H{h + 1}"] = round((float(np.prod(1 + x[m])) - 1) * 100, 2)
    return out


def summarize_configs(rows: list[dict]) -> dict:
    s = np.array([r["oos"]["sharpe"] for r in rows])
    c = np.array([r["oos"]["cagr_pct"] for r in rows])
    i = np.array([r["is"]["sharpe"] for r in rows])
    return {"count": len(rows), "median_oos_sharpe": round(float(np.median(s)), 3),
            "share_oos_positive": round(float(np.mean(c > 0)), 3), "median_is_sharpe": round(float(np.median(i)), 3),
            "corr_is_oos_sharpe": round(float(np.corrcoef(i, s)[0, 1]), 3) if len(rows) > 2 else None}  # fmt: skip


def save_json(name: str, obj) -> None:
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / name).write_text(json.dumps(obj, indent=1, default=float))


# ------------------------------------------------------------------------------------- (b) XS factor
def xs_weights(sig, elig, q, nmin, direction, vol=None):
    """Доллар-нейтрально: шорт верхнего квантиля сигнала, лонг нижнего (direction=+1), брутто 1."""
    T, N = sig.shape
    w = np.full((T, N), np.nan)
    for t in range(T):
        m = elig[t] & np.isfinite(sig[t])
        n = int(m.sum())
        if n < 10:
            w[t] = 0.0
            continue
        k = max(nmin, int(np.floor(q * n)))
        k = min(k, n // 2)
        idx = np.where(m)[0]
        order = idx[np.argsort(sig[t, idx], kind="stable")]
        lo, hi = order[:k], order[-k:]
        row = np.zeros(N)
        if vol is None:
            row[lo], row[hi] = 0.5 / k, -0.5 / k
        else:
            iv_lo, iv_hi = 1 / vol[t, lo], 1 / vol[t, hi]
            row[lo], row[hi] = 0.5 * iv_lo / iv_lo.sum(), -0.5 * iv_hi / iv_hi.sum()
        w[t] = row * direction
    return w


def run_xs(d: dict) -> None:
    ts, T = d["ts"], len(d["ts"])
    tradable = d["alive"]
    rv = np.abs(d["ret"])
    vol30 = np.where(rolling_mean_past(np.where(d["alive"], rv, np.nan), 30) > 0,
                     rolling_mean_past(np.where(d["alive"], rv, np.nan), 30), np.nan)  # fmt: skip
    vol30 = np.where(np.isfinite(vol30), vol30, 0.05)
    # ret[t] — доходность бара t (open t -> open t+1) известна только после open t+1: в vol30 на t
    # используем |ret| до t-1 (сдвиг), чтобы не было заглядывания
    vol30 = np.vstack([np.full((1, vol30.shape[1]), 0.05), vol30[:-1]])
    rows = []
    elig_cache = {}
    t0 = time.time()
    for A in (5e6, 2e7, 1e8):
        elig_cache[A] = liquid_mask(d["p"], min_age_bars=30, min_adv_usd=A)
    for L in (1, 3, 7, 14, 30):
        sig = rolling_mean_past(d["fsig"], L)
        for A in (5e6, 2e7, 1e8):
            elig = elig_cache[A]
            for q in (0.1, 0.2, 0.33):
                for wt in ("eq", "iv"):
                    for direction in (1, -1):
                        w = xs_weights(sig, elig, q, 3, direction, vol30 if wt == "iv" else None)
                        for R in (1, 3, 7):
                            rebal = (np.arange(T) % R) == 0
                            out, tr = run_book(w, rebal, d["ret"], d["fhold"], d["cost"], tradable, d["stop"],
                                               d["true_del"], False)  # fmt: skip
                            st = stats(ts, out, tr)
                            st["cfg"] = dict(L=L, A=A, q=q, wt=wt, dir=direction, R=R)
                            rows.append(st)
        print("xs L", L, len(rows), f"{time.time() - t0:.0f}s", flush=True)
    summ_all = summarize_configs(rows)
    summ = {str(dr): summarize_configs([r for r in rows if r["cfg"]["dir"] == dr]) for dr in (1, -1)}
    best = max(rows, key=lambda r: r["is"]["sharpe"])
    cfg = best["cfg"]
    sig = rolling_mean_past(d["fsig"], cfg["L"])
    w = xs_weights(sig, elig_cache[cfg["A"]], cfg["q"], 3, cfg["dir"], vol30 if cfg["wt"] == "iv" else None)
    rebal = (np.arange(T) % cfg["R"]) == 0
    out, tr = run_book(w, rebal, d["ret"], d["fhold"], d["cost"], tradable, d["stop"], d["true_del"], False)
    outp, _ = run_book(w, rebal, d["ret"], d["fhold"], d["cost"], tradable, d["stop"], d["true_del"], True)
    best_full = stats(ts, out, tr)
    best_full["pessimistic_delist"] = stats(ts, outp, tr)["oos"]
    best_full["per_year"] = per_year(ts, out[:, 0])
    best_full["oos_half_years"] = half_years(ts, out[:, 0])
    best_full["cfg"] = cfg
    # соседи ±: L, q, R, A
    nb_rows = [r for r in rows if r["cfg"]["dir"] == cfg["dir"] and r["cfg"]["wt"] == cfg["wt"] and
               sum(r["cfg"][k] != cfg[k] for k in ("L", "A", "q", "R")) == 1]  # fmt: skip
    best_full["neighbors"] = [{"cfg": r["cfg"], "is_sh": r["is"]["sharpe"], "oos_sh": r["oos"]["sharpe"],
                               "oos_cagr": r["oos"]["cagr_pct"]} for r in nb_rows]  # fmt: skip
    np.savez(OUTD / "xs_best_daily.npz", ts=ts, ret=out[:, 0], price=out[:, 1], fund=out[:, 2], cost=out[:, 3])
    top_is = sorted(rows, key=lambda r: -r["is"]["sharpe"])[:15]
    save_json("part_xs.json", {"summary_all": summ_all, "summary_by_dir": summ, "best_is": best_full,
                               "top15_is": top_is, "configs": rows})  # fmt: skip
    print(json.dumps({"summary": summ_all, "by_dir": summ, "best_cfg": cfg, "is": best_full["is"],
                      "oos": best_full["oos"], "per_year": best_full["per_year"]}, indent=1, default=float))  # fmt: skip


# ------------------------------------------------------------------------------- (c) timing signals
@nb.njit(cache=True)
def trailing_pct(x, W, valid):
    """Перцентиль x[t] среди x[t-W+1..t] (только прошлое и текущее), NaN если мало истории."""
    T, N = x.shape
    out = np.full((T, N), np.nan)
    for j in range(N):
        for t in range(T):
            if not valid[t, j] or np.isnan(x[t, j]):
                continue
            a = max(0, t - W + 1)
            cnt = 0
            below = 0
            for s in range(a, t + 1):
                if valid[s, j] and not np.isnan(x[s, j]):
                    cnt += 1
                    if x[s, j] < x[t, j]:
                        below += 1
                    elif x[s, j] == x[t, j] and s != t:
                        below += 0.5
            if cnt >= min(W, 60):
                out[t, j] = below / (cnt - 1) if cnt > 1 else np.nan
    return out


@nb.njit(cache=True)
def hold_state(trig, H):
    """trig[t,j] in {-1,0,1}: позиция держится H баров после последнего срабатывания."""
    T, N = trig.shape
    st = np.zeros((T, N))
    for j in range(N):
        left = 0
        cur = 0.0
        for t in range(T):
            if trig[t, j] != 0:
                cur = trig[t, j]
                left = H
            if left > 0:
                st[t, j] = cur
                left -= 1
            else:
                cur = 0.0
    return st


def timing_weights(d, sigs, pcts, elig_alt, cfg, jb, je):
    """Веса стратегии «экстремум фандинга»: dir=+1 — контртренд (экстремально высокий фандинг -> шорт,
    низкий -> лонг), dir=-1 — по тренду. Сигнал по закрытию бара t, позиция держится H баров."""
    alive = d["alive"]
    kind, L, W, x = cfg["kind"], cfg["L"], cfg["W"], cfg["x"]
    if kind == "pct":
        s = pcts[(L, W)]
        up, dn = s >= x, s <= 1 - x
    else:
        s = sigs[L]
        up, dn = s >= x, s <= -x / 2  # база фандинга ≈ +0.03%/сутки, отрицательный порог вдвое меньше
    up = np.nan_to_num(up).astype(bool)
    dn = np.nan_to_num(dn).astype(bool)
    trig = (dn.astype(float) - up.astype(float)) * cfg["dir"]
    if cfg["side"] == "long":
        trig = np.where(trig > 0, trig, 0.0)
    elif cfg["side"] == "short":
        trig = np.where(trig < 0, trig, 0.0)
    if cfg["asset"] == "ALT":
        trig = np.where(elig_alt, trig, 0.0)
        stt = np.where(alive, hold_state(trig, cfg["H"]), 0.0)
        n_act = np.abs(stt).sum(1, keepdims=True)
        return stt / np.maximum(n_act, 10), n_act
    j = jb if cfg["asset"] == "BTC" else je
    tj = np.zeros_like(trig)
    tj[:, j] = trig[:, j]
    return hold_state(tj, cfg["H"]), None


def run_timing(d: dict) -> None:
    ts, T = d["ts"], len(d["ts"])
    syms = d["syms"]
    jb, je = syms.index("BTCUSDT"), syms.index("ETHUSDT")
    elig_alt = liquid_mask(d["p"], min_age_bars=60, min_adv_usd=2e7)
    elig_alt[:, [jb, je]] = False
    alive = d["alive"]
    rows = []
    t0 = time.time()
    sigs = {L: rolling_mean_past(d["fsig"], L) for L in (1, 3, 7)}
    pcts = {}
    for L in (1, 3, 7):
        for W in (90, 365):
            pcts[(L, W)] = trailing_pct(sigs[L], W, alive)
    print("pct done", f"{time.time() - t0:.0f}s", flush=True)
    defs = []
    for L in (1, 3, 7):
        for W in (90, 365):
            for hi in (0.9, 0.95, 0.99):
                defs.append(("pct", L, W, hi))
        for thr in (0.0005, 0.001, 0.002):  # средний дневной фандинг (сумма ставок за сутки)
            defs.append(("abs", L, 0, thr))
    for asset in ("BTC", "ETH", "ALT"):
        for (kind, L, W, x) in defs:
            for H in (1, 3, 7):
                for direction in (1, -1):
                    for side in ("both", "long", "short"):
                        cfg = dict(asset=asset, kind=kind, L=L, W=W, x=x, H=H, dir=direction, side=side)
                        w, _ = timing_weights(d, sigs, pcts, elig_alt, cfg, jb, je)
                        rebal = np.r_[True, np.any(w[1:] != w[:-1], axis=1)]
                        out, tr = run_book(w, rebal, d["ret"], d["fhold"], d["cost"], alive, d["stop"],
                                           d["true_del"], False)  # fmt: skip
                        st = stats(ts, out, tr)
                        st["cfg"] = cfg
                        rows.append(st)
        print("timing", asset, len(rows), f"{time.time() - t0:.0f}s", flush=True)
    res = {"summary_all": summarize_configs(rows)}
    best_by = {}
    for asset in ("BTC", "ETH", "ALT"):
        ra = [r for r in rows if r["cfg"]["asset"] == asset and r["trades"] >= 10]
        res[f"summary_{asset}"] = summarize_configs(ra)
        for dr in (1, -1):
            res[f"summary_{asset}_dir{dr}"] = summarize_configs([r for r in ra if r["cfg"]["dir"] == dr])
        b = max(ra, key=lambda r: r["is"]["sharpe"])
        best_by[asset] = {"cfg": b["cfg"], "is": b["is"], "oos": b["oos"], "trades": b["trades"]}
    # базовые линии: buy&hold BTC/ETH перп (с фандингом)
    for asset, j in (("BTC", jb), ("ETH", je)):
        w = np.zeros((T, len(syms)))
        w[:, j] = 1.0
        rebal = np.ones(T, dtype=bool)  # ежедневно к весу 1 (издержки только на дрейф)
        out, tr = run_book(w, rebal, d["ret"], d["fhold"], d["cost"], alive, d["stop"], d["true_del"], False)
        res[f"baseline_hold_{asset}"] = stats(ts, out, tr)
        res[f"baseline_hold_{asset}"]["per_year"] = per_year(ts, out[:, 0])
    # базовая линия для лучшего ALT-конфига: случайные входы с тем же числом позиций по дням (200 перестановок)
    ba = best_by["ALT"]["cfg"]
    w_best, _ = timing_weights(d, sigs, pcts, elig_alt, ba, jb, je)
    rebal = np.r_[True, np.any(w_best[1:] != w_best[:-1], axis=1)]
    out, tr = run_book(w_best, rebal, d["ret"], d["fhold"], d["cost"], alive, d["stop"], d["true_del"], False)
    outp, _ = run_book(w_best, rebal, d["ret"], d["fhold"], d["cost"], alive, d["stop"], d["true_del"], True)
    best_by["ALT"]["pessimistic_delist_oos"] = stats(ts, outp, tr)["oos"]
    best_by["ALT"]["per_year"] = per_year(ts, out[:, 0])
    best_by["ALT"]["oos_half_years"] = half_years(ts, out[:, 0])
    np.savez(OUTD / "timing_alt_best_daily.npz", ts=ts, ret=out[:, 0])
    rng = np.random.default_rng(0)
    rnd = []
    nact = (w_best != 0).sum(1)
    side = np.sign(w_best.sum(1))
    H = ba["H"]
    for it in range(100):
        wr = np.zeros_like(w_best)
        for t in range(0, T, H):
            k = int(nact[t])
            if k == 0:
                continue
            cand = np.where(elig_alt[t])[0]
            if len(cand) == 0:
                continue
            pick = rng.choice(cand, size=min(k, len(cand)), replace=False)
            wr[t:t + H, pick] = w_best[t, w_best[t] != 0].mean() if side[t] != 0 else 0.0
        wr = np.where(alive, wr, 0.0)
        rb = np.r_[True, np.any(wr[1:] != wr[:-1], axis=1)]
        o, _ = run_book(wr, rb, d["ret"], d["fhold"], d["cost"], alive, d["stop"], d["true_del"], False)
        st = stats(ts, o, 0)
        rnd.append((st["is"]["sharpe"], st["oos"]["sharpe"], st["oos"]["cagr_pct"]))
    rnd = np.array(rnd)
    best_by["ALT"]["random_entry_baseline"] = {
        "n": len(rnd), "is_sharpe_median": float(np.median(rnd[:, 0])), "oos_sharpe_median": float(np.median(rnd[:, 1])),
        "oos_sharpe_p95": float(np.percentile(rnd[:, 1], 95)), "oos_cagr_median": float(np.median(rnd[:, 2])),
        "share_random_oos_sharpe_ge_best": float(np.mean(rnd[:, 1] >= best_by["ALT"]["oos"]["sharpe"]))}  # fmt: skip
    res["best_is_by_asset"] = best_by
    res["configs"] = rows
    save_json("part_timing.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "configs"}, indent=1, default=float)[:6000])


def run_control(d: dict) -> None:
    """Контроль для лучшего ALT-сигнала (c): тот же отбор, но вместо фандинга — доходность цены за L дней
    (перцентиль в собственной истории). Если контроль не хуже, фандинг ничего не добавляет к реверсии цены."""
    ts, T = d["ts"], len(d["ts"])
    syms = d["syms"]
    jb, je = syms.index("BTCUSDT"), syms.index("ETHUSDT")
    elig_alt = liquid_mask(d["p"], min_age_bars=60, min_adv_usd=2e7)
    elig_alt[:, [jb, je]] = False
    alive = d["alive"]
    res = json.loads((OUTD / "part_timing.json").read_text())
    cfg = res["best_is_by_asset"]["ALT"]["cfg"]
    L, W = cfg["L"], cfg["W"]
    C = d["C"]
    # доходность close[t]/close[t-L] (известна на закрытии t)
    pr = np.full_like(C, np.nan)
    pr[L:] = C[L:] / C[:-L] - 1
    sigs = {L: rolling_mean_past(d["fsig"], L)}
    out_rows = {}
    for name, x in (("funding", sigs[L]), ("price_return", pr)):
        pc = {(L, W): trailing_pct(x, W, alive)}
        w, _ = timing_weights(d, {L: x}, pc, elig_alt, cfg, jb, je)
        rebal = np.r_[True, np.any(w[1:] != w[:-1], axis=1)]
        out, tr = run_book(w, rebal, d["ret"], d["fhold"], d["cost"], alive, d["stop"], d["true_del"], False)
        st = stats(ts, out, tr)
        st["per_year"] = per_year(ts, out[:, 0])
        out_rows[name] = (st, out[:, 0], w)
    a, b = window_idx(ts, OOS)
    r1, r2 = out_rows["funding"][1], out_rows["price_return"][1]
    w1, w2 = out_rows["funding"][2] != 0, out_rows["price_return"][2] != 0
    ctrl = {k: {"is": v[0]["is"], "oos": v[0]["oos"], "per_year": v[0]["per_year"], "trades": v[0]["trades"]}
            for k, v in out_rows.items()}  # fmt: skip
    ctrl["corr_daily_oos"] = round(float(np.corrcoef(r1[a:b], r2[a:b])[0, 1]), 3)
    ctrl["position_overlap"] = round(float((w1 & w2).sum() / max(w1.sum(), 1)), 3)
    res["control_price_reversal"] = ctrl
    save_json("part_timing.json", res)
    for k in ("funding", "price_return"):
        print(k, "IS", ctrl[k]["is"]["sharpe"], ctrl[k]["is"]["cagr_pct"], "OOS", ctrl[k]["oos"]["sharpe"],
              ctrl[k]["oos"]["cagr_pct"], ctrl[k]["oos"]["max_dd_pct"], ctrl[k]["per_year"])  # fmt: skip
    print("corr", ctrl["corr_daily_oos"], "overlap", ctrl["position_overlap"])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "xs"
    t0 = time.time()
    OUTD.mkdir(parents=True, exist_ok=True)
    d = prep()
    print("prep", f"{time.time() - t0:.0f}s", flush=True)
    if cmd == "xs":
        run_xs(d)
    elif cmd == "timing":
        run_timing(d)
    elif cmd == "control":
        run_control(d)
