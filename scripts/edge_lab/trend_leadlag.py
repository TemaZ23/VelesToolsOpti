"""
Семейство «trend», часть (b): лид-лаг BTC → альты на часовых барах.

    PYTHONPATH=. python3 -m scripts.edge_lab.trend_leadlag      # part_b.json

Вопрос: предсказывает ли доходность BTC за последние h = 1–3 часа доходность альтов в следующий час
(запаздывающая реакция)? Сначала статистика (регрессии на всём срезе и Fama–MacBeth по «отставанию»
альта от BTC), затем торгуемые версии с издержками тейкера.

Тайминг: сигнал по close часа t, вход по open[t+1], выход по close[t+H] (≈ open[t+H+1]).
Вселенная на t: крипто-перпы, кроме BTC, возраст ≥ 30 дней, top-K по ADV за 30 дней (по t вкл.).
Издержки: 2 × cost_per_side(ADV) на номинал за круг (вход + выход), для хеджа — ещё круг по BTC.
Фандинг: удержание 1–4 часа, выплата учитывается, только если она против позиции (консервативно).
"""

from __future__ import annotations

import json
import time

import numpy as np

from scripts.edge_lab.common import CACHE, DAY, HOUR, IS, OOS, OUT, cost_per_side, metrics, ms, per_year, window_idx
from scripts.edge_lab.trend_engine import EXCLUDE

ODIR = OUT / "trend"
ODIR.mkdir(parents=True, exist_ok=True)


def load() -> dict:
    """Часовая панель загружается один раз; оставляем только монеты, хоть раз входившие в top-60 по ADV."""
    t0 = time.time()
    z = np.load(CACHE / "panel_1h.npz", allow_pickle=False)
    ts = z["ts"]
    syms = z["symbols"]
    qv = z["qvol"]
    keep = np.array([s not in EXCLUDE for s in syms])
    # 30-дневный ADV (720 часов, включая t), по прошлым данным
    q = np.nan_to_num(qv.astype(np.float64))
    cnt = (~np.isnan(qv)).astype(np.float64)
    cs = np.cumsum(q, axis=0)
    cc = np.cumsum(cnt, axis=0)
    n = 720
    s = cs.copy()
    s[n:] -= cs[:-n]
    k = cc.copy()
    k[n:] -= cc[:-n]
    del cs, cc, q
    with np.errstate(invalid="ignore", divide="ignore"):
        adv = np.where(k > 0, s / k * 24.0, 0.0)  # $/сутки
    del s
    alive = ~np.isnan(qv)
    age = np.cumsum(alive, axis=0, dtype=np.int32)
    ok = alive & (age >= 720) & keep[None, :]
    score = np.where(ok, adv, -1.0)
    top = np.argpartition(-score, 60, axis=1)[:, :60]
    ever = np.zeros(len(syms), bool)
    ever[np.unique(top)] = True
    ever &= keep
    jb = int(np.where(syms == "BTCUSDT")[0][0])
    ever[jb] = True
    cols = np.where(ever)[0]
    out = {
        "ts": ts, "symbols": syms[cols], "adv": adv[:, cols].astype(np.float32), "age": age[:, cols],
        "alive": alive[:, cols], "open": z["open"][:, cols], "close": z["close"][:, cols],
        "funding": z["funding"][:, cols],
    }  # fmt: skip
    z.close()
    print(f"loaded 1h panel, kept {len(cols)} symbols, {time.time() - t0:.0f}s", flush=True)
    return out


def topk(p: dict, K: int) -> np.ndarray:
    jb = int(np.where(p["symbols"] == "BTCUSDT")[0][0])
    ok = p["alive"] & (p["age"] >= 720)
    ok[:, jb] = False
    score = np.where(ok, p["adv"], -1.0)
    T, N = score.shape
    idx = np.argpartition(-score, K - 1, axis=1)[:, :K]
    m = np.zeros((T, N), bool)
    m[np.repeat(np.arange(T), K), idx.ravel()] = True
    return m & ok


def fwd_ret(o: np.ndarray, c: np.ndarray, H: int) -> np.ndarray:
    """Доходность от open[t+1] до close[t+H] (позиция на часах t+1..t+H). Если монета перестала
    торговаться — выход по последнему доступному close в окне."""
    T, N = o.shape
    entry = np.full((T, N), np.nan, np.float64)
    entry[:-1] = o[1:]
    last = np.full((T, N), np.nan, np.float64)
    for k in range(1, H + 1):
        ck = np.full((T, N), np.nan, np.float64)
        ck[:-k] = c[k:]
        last = np.where(np.isnan(ck), last, ck)
    with np.errstate(invalid="ignore", divide="ignore"):
        return last / entry - 1


def past_ret(c: np.ndarray, h: int) -> np.ndarray:
    out = np.full(c.shape, np.nan, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[h:] = c[h:] / c[:-h] - 1
    return out


def rolling_beta(r: np.ndarray, rb: np.ndarray, n: int = 720) -> np.ndarray:
    """β альта к BTC по прошлым n часам (включая t). NaN-часы пропускаются."""
    m = ~np.isnan(r) & ~np.isnan(rb)[:, None]
    x = np.where(m, rb[:, None], 0.0)
    y = np.where(m, r, 0.0)

    def rs(a):
        cs = np.cumsum(a, axis=0)
        cs[n:] = cs[n:] - cs[:-n].copy()
        return cs

    sn = rs(m.astype(np.float64))
    sx, sy, sxx, sxy = rs(x), rs(y), rs(x * x), rs(x * y)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy / sn - sx / sn * sy / sn
        var = sxx / sn - (sx / sn) ** 2
        b = np.where(sn >= 240, cov / var, np.nan)
    return np.clip(b, 0.0, 3.0)


def ols(y: np.ndarray, x: np.ndarray, nw_lag: int = 6) -> dict:
    """OLS y = a + b·x с t-статистикой Ньюи–Уэста."""
    ok = ~np.isnan(y) & ~np.isnan(x)
    y, x = y[ok], x[ok]
    X = np.c_[np.ones_like(x), x]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, nw_lag + 1):
        w = 1 - L / (nw_lag + 1)
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    r2 = 1 - e.var() / y.var() if y.var() > 0 else 0.0
    return {"b": float(beta[1]), "t": float(beta[1] / np.sqrt(V[1, 1])), "r2_pct": round(float(r2) * 100, 3), "n": int(len(y))}


def tstat_mean(x: np.ndarray, nw_lag: int = 6) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return float("nan"), float("nan")
    m = x.mean()
    e = x - m
    v = (e * e).mean()
    for L in range(1, nw_lag + 1):
        v += 2 * (1 - L / (nw_lag + 1)) * (e[L:] * e[:-L]).mean()
    return float(m), float(m / np.sqrt(v / len(x)))


def stats_part(p: dict) -> dict:
    """Предсказуемость: регрессия средней доходности альтов в следующий час на доходность BTC за h часов,
    и Fama–MacBeth по отставанию альта gap_i = β_i·B_h − A_h,i."""
    ts = p["ts"]
    o, c = p["open"], p["close"]
    jb = int(np.where(p["symbols"] == "BTCUSDT")[0][0])
    uni = topk(p, 50)
    y1 = fwd_ret(o, c, 1)  # open[t+1] -> close[t+1]
    yb = y1[:, jb].copy()
    r1 = past_ret(c, 1)
    beta = rolling_beta(r1, r1[:, jb])
    out = {}
    wins = {"IS": IS, "OOS": OOS, **{str(y): (f"{y}-01-01", f"{y + 1}-01-01") for y in range(2020, 2027)}}
    ybar = np.where(uni, y1, np.nan)
    with np.errstate(invalid="ignore"):
        ybar_m = np.nanmean(ybar, axis=1)
        hedged = np.nanmean(np.where(uni, y1 - beta * yb[:, None], np.nan), axis=1)
    for h in (1, 2, 3):
        B = past_ret(c[:, jb : jb + 1], h)[:, 0]
        A = past_ret(c, h)
        gap = beta * B[:, None] - A
        gap = np.where(uni, gap, np.nan)
        # Fama–MacBeth: кросс-секционный наклон y_{i,t+1} на gap_{i,t} в каждом часе
        g = gap - np.nanmean(gap, axis=1, keepdims=True)
        yy = np.where(uni, y1 - beta * yb[:, None], np.nan)  # β-хеджированная доходность
        yy = yy - np.nanmean(yy, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            slope = np.nansum(g * yy, axis=1) / np.nansum(g * g, axis=1)
        own = past_ret(c, h)  # собственный моментум альтов (для сравнения)
        own_bar = np.nanmean(np.where(uni, own, np.nan), axis=1)
        for wn, win in wins.items():
            a, b = window_idx(ts, win)
            sl = slice(a, b - 1)
            key = f"h{h}_{wn}"
            m_fm, t_fm = tstat_mean(slope[sl])
            out[key] = {
                "alts_next_on_btc": ols(ybar_m[sl], B[sl]),
                "alts_hedged_next_on_btc": ols(hedged[sl], B[sl]),
                "btc_next_on_btc": ols(yb[sl], B[sl]),
                "alts_next_on_alts_own": ols(ybar_m[sl], own_bar[sl]),
                "fm_gap_slope": {"mean": m_fm, "t": t_fm},
            }
        print("h", h, {k: (round(v["alts_next_on_btc"]["b"], 4), round(v["alts_next_on_btc"]["t"], 1),
                          round(v["fm_gap_slope"]["t"], 1)) for k, v in out.items() if k.startswith(f"h{h}_")}, flush=True)  # fmt: skip
    # условные средние: следующий час корзины альтов после сильного часа BTC
    B1 = past_ret(c[:, jb : jb + 1], 1)[:, 0]
    cond = {}
    for wn, win in (("IS", IS), ("OOS", OOS)):
        a, b = window_idx(ts, win)
        sl = slice(a, b - 1)
        rows = []
        for lo, hi in ((-1, -0.02), (-0.02, -0.01), (-0.01, -0.005), (-0.005, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 1)):
            mk = (B1[sl] > lo) & (B1[sl] <= hi)
            yv = ybar_m[sl][mk]
            hv = hedged[sl][mk]
            rows.append({"btc_1h_bin": [lo, hi], "n": int(mk.sum()),
                         "alts_next_bp": round(float(np.nanmean(yv)) * 1e4, 2) if mk.any() else None,
                         "alts_hedged_next_bp": round(float(np.nanmean(hv)) * 1e4, 2) if mk.any() else None})  # fmt: skip
        cond[wn] = rows
    out["conditional_bins"] = cond
    return out


def _nonoverlap(fire: np.ndarray, H: int) -> np.ndarray:
    out = []
    busy = -1
    for t in fire:
        if t > busy:
            out.append(t)
            busy = t + H
    return np.array(out, dtype=np.int64)


def strat_part(p: dict) -> list[dict]:
    """Торгуемые версии. Неперекрывающиеся сделки: пока позиция открыта, новые сигналы игнорируются.
    Варианты: basket — равновзвешенная корзина top-K альтов в сторону движения BTC; lag5 — 5 альтов,
    сильнее всех отставших от β·BTC; lag5_hedged — то же с хеджем β·BTC (рыночно-нейтрально)."""
    ts = p["ts"]
    o, c = p["open"], p["close"]
    jb = int(np.where(p["symbols"] == "BTCUSDT")[0][0])
    r1 = past_ret(c, 1)
    beta = rolling_beta(r1, r1[:, jb])
    del r1
    cost = cost_per_side(np.maximum(p["adv"].astype(np.float64), 1.0))
    fund = np.nan_to_num(p["funding"]).astype(np.float64)
    T = len(ts)
    dayidx = ((ts - ts[0]) // DAY).astype(np.int64)
    nd = dayidx[-1] + 1
    dts = ts[0] + np.arange(nd) * DAY
    rows = []
    unis = {K: topk(p, K) for K in (20, 50)}
    ar = np.arange(T)[:, None]
    for H in (1, 2, 4):
        F = fwd_ret(o, c, H)
        fb = np.nan_to_num(F[:, jb])
        fw = np.zeros_like(fund)  # фандинг внутри окна удержания (часы t+1..t+H)
        for k in range(1, H + 1):
            fw[:-k] += fund[k:]
        for h in (1, 2, 3):
            B = np.nan_to_num(past_ret(c[:, jb : jb + 1], h)[:, 0])
            sgn = np.sign(B)
            A = past_ret(c, h)
            gap = beta * B[:, None] - A
            for K, uni in unis.items():
                m = uni & ~np.isnan(F)
                nm = m.sum(1)
                # корзина: на каждый час t — средняя доходность / издержки / фандинг в сторону sgn
                with np.errstate(invalid="ignore", divide="ignore"):
                    bk_g = sgn * np.where(m, F, 0.0).sum(1) / np.maximum(nm, 1)
                    bk_c = np.where(m, 2 * cost, 0.0).sum(1) / np.maximum(nm, 1)
                    bk_f = np.where(m, np.maximum(sgn[:, None] * fw, 0.0), 0.0).sum(1) / np.maximum(nm, 1)
                # 5 самых отставших (в сторону движения BTC)
                g = np.where(m, sgn[:, None] * gap, -np.inf)
                g = np.where(np.isnan(g), -np.inf, g)
                top = np.argpartition(-g, 4, axis=1)[:, :5]
                okt = np.isfinite(g[ar, top])
                n5 = np.maximum(okt.sum(1), 1)
                Ft = np.where(okt, F[ar, top], 0.0)
                l5_g = sgn * Ft.sum(1) / n5
                l5_c = np.where(okt, 2 * cost[ar, top], 0.0).sum(1) / n5
                l5_f = np.where(okt, np.maximum(sgn[:, None] * fw[ar, top], 0.0), 0.0).sum(1) / n5
                bh = sgn * np.where(okt, np.nan_to_num(beta[ar, top], nan=1.0), 0.0).sum(1) / n5
                hd_g = l5_g - bh * fb
                hd_c = l5_c + np.abs(bh) * 2 * cost[:, jb]
                hd_f = l5_f + np.maximum(-bh * fw[:, jb], 0.0)
                variants = {"basket": (bk_g, bk_c, bk_f), "lag5": (l5_g, l5_c, l5_f), "lag5_hedged": (hd_g, hd_c, hd_f)}
                for x in (0.005, 0.01, 0.015, 0.02, 0.03):
                    fire = np.where((np.abs(B) > x) & (nm > 0))[0]
                    fire = fire[fire < T - H - 1]
                    fire = _nonoverlap(fire, H)
                    for var, (vg, vc, vf) in variants.items():
                        gross = vg[fire]
                        net = gross - vc[fire] - vf[fire]
                        pnl_g = np.zeros(nd)
                        pnl_n = np.zeros(nd)
                        np.add.at(pnl_g, dayidx[fire + 1], gross)
                        np.add.at(pnl_n, dayidx[fire + 1], net)
                        row = {"H": H, "h": h, "K": K, "x": x, "var": var, "trades": int(len(fire))}
                        for wn, win in (("is", IS), ("oos", OOS)):
                            a, b = window_idx(dts, win)
                            mn = metrics(dts[a:b], pnl_n[a:b], None)
                            mg = metrics(dts[a:b], pnl_g[a:b], None)
                            ta, tb = window_idx(ts, win)
                            sel = (fire >= ta) & (fire < tb)
                            mn["trades"] = int(sel.sum())
                            mn["avg_trade_gross_bp"] = round(float(gross[sel].mean()) * 1e4, 2) if sel.any() else None
                            mn["avg_trade_net_bp"] = round(float(net[sel].mean()) * 1e4, 2) if sel.any() else None
                            row[wn] = {**mn, "sharpe_gross": mg["sharpe"], "cagr_gross_pct": mg["cagr_pct"]}
                        rows.append(row)
            print(f"H{H} h{h} done", flush=True)
        del F
    return rows


def main() -> None:
    t0 = time.time()
    p = load()
    st = stats_part(p)
    print(f"stats {time.time() - t0:.0f}s", flush=True)
    rows = strat_part(p)
    print(f"strats {time.time() - t0:.0f}s", flush=True)
    best = max(rows, key=lambda r: r["is"]["sharpe"])
    sh = np.array([r["oos"]["sharpe"] for r in rows])
    shg = np.array([r["oos"]["sharpe_gross"] for r in rows])
    summary = {"count": len(rows), "median_oos_sharpe": float(np.median(sh)), "share_oos_pos": float(np.mean(sh > 0)),
               "median_oos_sharpe_gross": float(np.median(shg)), "share_oos_gross_pos": float(np.mean(shg > 0)),
               "best_is": best}  # fmt: skip
    (ODIR / "part_b.json").write_text(json.dumps({"stats": st, "summary": summary, "configs": rows}, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
