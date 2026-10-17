"""
Независимая проверка кандидата xsec (Amihud illiquidity long-short), написана по спецификации правил,
без чтения кода автора.

Правила (из резюме автора):
  универсум на дне t: торгуется на t, >= 60 дневных баров истории, ADV30 (средний дневной quote-объём за
  30 дней вкл. t) >= $5M, только крипто (TradFi/стейблы исключены);
  ILLIQ = среднее за 30 дней |r_d| / QV_d (r — close-to-close), >= 21 валидных дней;
  n >= 20 -> k = floor(0.1 n): лонг k самых неликвидных +0.5/k, шорт k самых ликвидных -0.5/k;
  цель = среднее последних 7 дневных портфелей; сделка по open[t+1] тейкером, ежедневная ребалансировка.

Учёт:
  - веса — доли капитала на момент ребалансировки по open[d]; за бар d P&L = sum w*(C[d]/O[d]-1);
    гэп C[d-1] -> O[d] получают «дрейфовавшие» веса до ребалансировки;
  - издержки cost_per_side(ADV30[d-1]) * |w_target - w_drift| (дрейф между ребалансировками учитывается);
  - фандинг пересобран из сырых расчётов: расчёт в 00:00 дня d+1 относится к позиции бара d
    (позиция до ребалансировки в 00:00); знак: + ставка — лонг платит;
  - монета перестала торговаться -> выход по последнему close (с издержками), пессимистично -50% позиции.

Запуск:
  python3 -m scripts.edge_lab.verify_xsec main     # база, сравнение, робастность (панель 1d)
  python3 -m scripts.edge_lab.verify_xsec offset   # сдвиг времени ребалансировки (панель 1h, один раз)
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from scripts.edge_lab.common import (DAY, HOUR, IS, OOS, OUT, cost_per_side, load_funding, metrics, ms,
                                     panel, per_year, universe, window_idx)

OUTD = OUT / "xsec"

# --- не-крипто перпетуалы: металлы/нефть/газ/медь, FX, стейблы, акции/ETF/pre-IPO -----------------
NAMED_NONCRYPTO = {
    "XAU", "XAG", "XPD", "XPT", "PAXG", "XAUT", "COPPER", "NATGAS", "CL", "BZ", "USDBRL", "USDC",
    "OPENAI", "ANTHROPIC", "SPCX", "BRKB", "MRK", "EWZ", "XLE", "GTLB", "IWM", "V", "JPM", "DIS", "PYPL",
    "SAMSUNGEM", "HANMI", "POPMART", "SKUU", "SKDD", "BNC", "VST", "TXN", "CSOPSKHYNIX2L", "EWJ", "SPY",
    "NAVER", "MEITUAN", "LGELECTRONICS", "HK0625", "XBI", "KO", "EBAY", "PDD", "NVO", "HD", "EWT", "CXMT",
    "UVXY", "KODEX200", "GME", "TBT", "BYD", "CSOPSAMSUNG2L", "SAMSUNG", "CRM", "TTWO", "HYUNDAI",
}


def noncrypto_mask(p: dict) -> np.ndarray:
    """Имя из списка ИЛИ (листинг >= 2025-12-01 и «выходные» |r| < 0.55 будних — у акций/ETF рынок закрыт)."""
    u = universe()
    c, ts = p["close"], p["ts"]
    r = np.abs(c[1:] / c[:-1] - 1)
    we = (((ts[1:] // DAY) + 3) % 7) >= 5
    out = np.zeros(len(p["symbols"]), bool)
    for j, s in enumerate(p["symbols"]):
        base = s[:-4] if s.endswith("USDT") else s
        if base in NAMED_NONCRYPTO:
            out[j] = True
            continue
        if u[s]["first"][:10] >= "2025-12-01":
            a = r[:, j]
            ok = ~np.isnan(a)
            if (we & ok).sum() > 3 and (~we & ok).sum() > 3:
                ratio = np.mean(a[we & ok]) / max(np.mean(a[~we & ok]), 1e-12)
                out[j] = ratio < 0.55
    return out


def rolling_sum_cnt(x: np.ndarray, n: int):
    v = np.nan_to_num(x, nan=0.0, posinf=0.0)
    k = np.isfinite(x).astype(float)
    cs = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), v]), 0)
    cc = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), k]), 0)
    s = np.full(x.shape, np.nan)
    c = np.zeros(x.shape)
    s[n - 1:] = cs[n:] - cs[:-n]
    c[n - 1:] = cc[n:] - cc[:-n]
    # первые n-1 строк — по доступной истории
    s[: n - 1] = cs[1:n]
    c[: n - 1] = cc[1:n]
    return s, c


def load_data(freq_panel: dict | None = None) -> dict:
    p = freq_panel if freq_panel is not None else panel("1d")
    ts = p["ts"]
    T, N = p["close"].shape
    F = np.zeros((T, N))
    t0 = int(ts[0])
    step = int(ts[1] - ts[0])
    for j, s in enumerate(p["symbols"]):
        f = load_funding(str(s))
        if len(f["ts"]) == 0:
            continue
        tr = (f["ts"].astype(np.int64) + 30_000) // 60_000 * 60_000
        b = (tr - t0 - 1) // step  # расчёт ровно на границе бара -> предыдущий бар (позиция до ребалансировки)
        ok = (b >= 0) & (b < T)
        np.add.at(F[:, j], b[ok], f["rate"][ok])
    d = {"ts": ts, "symbols": p["symbols"], "O": p["open"].astype(float), "C": p["close"].astype(float),
         "QV": p["qvol"].astype(float), "F": F, "panel_funding": p["funding"].astype(float)}
    d["noncrypto"] = noncrypto_mask(p)
    return d


def signals(d: dict, win=30, minv=21, adv_min=5e6, age_min=60, q=0.1, H=7, nmin=20, excl=None,
            include_tradfi=False, fac="illiq", min_age_days=None, sign=+1):
    C, QV = d["C"], d["QV"]
    T, N = C.shape
    prev = np.vstack([np.full((1, N), np.nan), C[:-1]])
    r = C / prev - 1
    ok = np.isfinite(r) & (QV > 0)
    if fac == "illiq":
        x = np.where(ok, np.abs(r) / np.where(QV > 0, QV, np.nan), np.nan)
    elif fac == "vol":
        x = np.where(ok, np.abs(r), np.nan)
    elif fac == "size":  # 1/QV — малый объём = «неликвидный»
        x = np.where(QV > 0, 1.0 / QV, np.nan)
    else:
        raise ValueError(fac)
    s, c = rolling_sum_cnt(x, win)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(c >= minv, s / np.maximum(c, 1), np.nan)
    alive = np.isfinite(C)
    age = np.cumsum(alive, 0)
    sq, cq = rolling_sum_cnt(np.where(alive, QV, np.nan), 30)
    with np.errstate(invalid="ignore", divide="ignore"):
        adv = np.where(cq > 0, sq / np.maximum(cq, 1), np.nan)
    elig = alive & (age >= (min_age_days or age_min)) & (np.nan_to_num(adv) >= adv_min) & np.isfinite(score)
    if not include_tradfi:
        elig &= ~d["noncrypto"][None, :]
    if excl is not None:
        elig &= ~excl[None, :]
    P = np.zeros((T, N))
    nk = np.zeros(T, int)
    for t in range(T):
        idx = np.flatnonzero(elig[t])
        n = len(idx)
        if n < nmin:
            continue
        k = int(np.floor(q * n))
        if k < 1:
            continue
        o = idx[np.argsort(score[t, idx], kind="stable")]
        lo, hi = (o[-k:], o[:k]) if sign > 0 else (o[:k], o[-k:])
        P[t, lo] += 0.5 / k
        P[t, hi] -= 0.5 / k
        nk[t] = k
    if H > 1:
        cs = np.cumsum(np.vstack([np.zeros((1, N)), P]), 0)
        W = np.empty_like(P)
        W[H - 1:] = (cs[H:] - cs[:-H]) / H
        for t in range(H - 1):
            W[t] = cs[t + 1] / H
    else:
        W = P
    return W, adv, nk


def backtest(d: dict, W: np.ndarray, adv: np.ndarray, cost_mult=1.0, pess=False, lag=1, rebalance_every=1,
             reb_offset=0, cap_ret=None, funding="raw", min_trade=0.0, leg="both", collect=False):
    """W[t] — цель по закрытию t, исполнение на open[t+lag]. Возвращает дневной ряд и диагностику."""
    O, C = d["O"], d["C"]
    F = d["F"] if funding == "raw" else d["panel_funding"]
    T, N = C.shape
    if leg == "long":
        W = np.where(W > 0, W, 0.0)
    elif leg == "short":
        W = np.where(W < 0, W, 0.0)
    cps = cost_per_side(np.nan_to_num(adv, nan=0.0)) * cost_mult
    ret = np.zeros(T)
    price = np.zeros(T)
    fund = np.zeros(T)
    cost = np.zeros(T)
    turn = np.zeros(T)
    gross = np.zeros(T)
    npos = np.zeros(T, int)
    entries = 0
    delist_events = 0
    cap = np.full(T, np.nan)
    coin_pnl = np.zeros(N)
    coin_pnl_oos = np.zeros(N)
    i_oos = int(np.searchsorted(d["ts"], ms(OOS[0])))
    wd = np.zeros(N)
    held_days = []
    run = np.zeros(N, int)
    tgt_prev = np.zeros(N)
    for t in range(1, T):
        alive_now = np.isfinite(O[t]) & np.isfinite(C[t])
        # гэп C[t-1] -> O[t] на дрейфовавших весах
        g = np.where(np.isfinite(C[t - 1]) & alive_now, O[t] / C[t - 1] - 1, 0.0)
        if cap_ret is not None:
            g = np.clip(g, -cap_ret, cap_ret)
        G = float(np.sum(wd * g))
        if np.any(wd != 0):
            wd = wd * (1 + g) / (1 + G)
        # вылетевшие из торговли монеты: выход по последнему close
        dead = (wd != 0) & ~alive_now
        extra = 0.0
        c_exit = 0.0
        if dead.any():
            delist_events += int(dead.sum())
            c_exit = float(np.sum(np.abs(wd[dead]) * cps[t - 1, dead]))
            if pess:
                extra = -0.5 * float(np.sum(np.abs(wd[dead])))
            wd[dead] = 0.0
        # цель
        s = t - lag
        if s >= 0 and ((t - reb_offset) % rebalance_every == 0):
            tgt = np.where(alive_now, W[s], 0.0)
        else:
            tgt = np.where(alive_now, wd, 0.0) if rebalance_every > 1 else np.where(alive_now, W[max(s, 0)] * (s >= 0), 0.0)
        if min_trade > 0:
            dw = tgt - wd
            small = (np.abs(dw) < min_trade) & (tgt != 0) & (wd != 0)
            tgt = np.where(small, wd, tgt)
        dw = tgt - wd
        c = float(np.sum(np.abs(dw) * cps[t - 1])) + c_exit
        entries += int(np.sum((wd == 0) & (tgt != 0)))
        rb = np.where(alive_now, C[t] / O[t] - 1, 0.0)
        if cap_ret is not None:
            rb = np.clip(rb, -cap_ret, cap_ret)
        f = np.where(alive_now, F[t], 0.0)
        b = float(np.sum(tgt * rb))
        fp = -float(np.sum(tgt * f))
        R = (1 + G + extra) * (1 - c) * (1 + b + fp) - 1
        ret[t] = R
        price[t] = G + b + extra
        fund[t] = fp
        cost[t] = c
        turn[t] = float(np.sum(np.abs(dw)))
        gross[t] = float(np.sum(np.abs(tgt)))
        held = tgt != 0
        npos[t] = int(held.sum())
        if held.any():
            cap[t] = float(np.min(0.01 * adv[t - 1, held] / 24.0))
        pnl_i = wd * g + tgt * rb - tgt * f
        coin_pnl += np.nan_to_num(pnl_i)
        if t >= i_oos:
            coin_pnl_oos += np.nan_to_num(pnl_i)
        # длительность удержания
        ended = (run > 0) & ~held
        if ended.any():
            held_days.extend(run[ended].tolist())
        run = np.where(held, run + 1, 0)
        denom = 1 + b + fp
        wd = np.where(held, tgt * (1 + rb) / denom, 0.0) if denom > 0 else np.zeros(N)
        tgt_prev = tgt
    out = {"ret": ret, "price": price, "fund": fund, "cost": cost, "turn": turn, "gross": gross, "npos": npos,
           "entries": entries, "delist_events": delist_events, "cap": cap, "coin_pnl": coin_pnl,
           "coin_pnl_oos": coin_pnl_oos, "hold_mean": float(np.mean(held_days)) if held_days else np.nan}
    return out


def win_stats(ts, ret, window):
    a, b = window_idx(ts, window)
    return metrics(ts[a:b], ret[a:b])


def halfyears(ts, ret):
    dt = ts.astype("datetime64[ms]").astype("datetime64[M]").astype(int)
    y = dt // 12 + 1970
    h = (dt % 12) // 6 + 1
    out = {}
    for yy in np.unique(y):
        for hh in (1, 2):
            m = (y == yy) & (h == hh)
            if m.sum() == 0:
                continue
            out[f"{yy}H{hh}"] = round((float(np.prod(1 + ret[m])) - 1) * 100, 1)
    return out


def sharpe(r):
    r = np.asarray(r, float)
    return float(np.mean(r) / np.std(r) * np.sqrt(365)) if np.std(r) > 0 else 0.0


def block_boot_sharpe(r, nb=2000, block=20, seed=0):
    rng = np.random.default_rng(seed)
    n = len(r)
    nblk = int(np.ceil(n / block))
    res = np.empty(nb)
    for i in range(nb):
        st = rng.integers(0, n - block, nblk)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:n]
        res[i] = sharpe(r[idx])
    return res


def summarize(d, res, label=""):
    ts = d["ts"]
    r = res["ret"]
    i0, i1 = window_idx(ts, IS)
    o0, o1 = window_idx(ts, OOS)
    out = {"label": label, "IS": win_stats(ts, r, IS), "OOS": win_stats(ts, r, OOS)}
    out["IS_sharpe"] = out["IS"].get("sharpe")
    out["OOS_sharpe"] = out["OOS"].get("sharpe")
    return out


def run_main():
    t_start = time.time()
    d = load_data()
    ts = d["ts"]
    T, N = d["C"].shape
    syms = [str(s) for s in d["symbols"]]
    print("loaded", time.time() - t_start, "noncrypto excluded:", int(d["noncrypto"].sum()))
    o0, o1 = window_idx(ts, OOS)
    i0, i1 = window_idx(ts, IS)
    results = {}

    def go(label, sig_kw=None, bt_kw=None):
        sig_kw = sig_kw or {}
        bt_kw = bt_kw or {}
        W, adv, nk = signals(d, **sig_kw)
        res = backtest(d, W, adv, **bt_kw)
        s = summarize(d, res, label)
        s["sig"] = {k: (v if not isinstance(v, np.ndarray) else "mask") for k, v in sig_kw.items()}
        s["bt"] = bt_kw
        results[label] = s
        print(f"{label:38s} IS {s['IS_sharpe']:6.2f} cagr {s['IS']['cagr_pct']:7.1f} | OOS {s['OOS_sharpe']:6.2f} "
              f"cagr {s['OOS']['cagr_pct']:7.1f} dd {s['OOS']['max_dd_pct']:5.1f}", flush=True)
        return res, W, adv, nk

    # ---------------- база ----------------
    base, Wb, advb, nkb = go("base")
    r = base["ret"]
    # детали базы
    det = {}
    for nm, (a, b) in {"IS": (i0, i1), "OOS": (o0, o1)}.items():
        m = metrics(ts[a:b], r[a:b], trades=int(0))
        days = b - a
        det[nm] = {
            **m,
            "entries": None,
            "turnover_2way_per_day": round(float(np.mean(base["turn"][a:b])), 4),
            "gross_mean": round(float(np.mean(base["gross"][a:b])), 3),
            "npos_mean": round(float(np.mean(base["npos"][a:b])), 1),
            "price_pnl_sum_pct": round(float(np.sum(base["price"][a:b])) * 100, 1),
            "funding_pnl_sum_pct": round(float(np.sum(base["fund"][a:b])) * 100, 1),
            "cost_sum_pct": round(float(np.sum(base["cost"][a:b])) * 100, 1),
            "cost_per_year_pct": round(float(np.sum(base["cost"][a:b])) * 100 / (days / 365.25), 2),
            "funding_per_year_pct": round(float(np.sum(base["fund"][a:b])) * 100 / (days / 365.25), 2),
            "capacity_median_usd": round(float(np.nanmedian(base["cap"][a:b])), 0),
            "capacity_p10_usd": round(float(np.nanpercentile(base["cap"][a:b], 10)), 0),
        }
    det["per_year"] = per_year(ts, r)
    det["halfyears"] = halfyears(ts, r)
    det["hold_mean_days"] = round(base["hold_mean"], 1)
    det["entries_total"] = base["entries"]
    det["delist_events"] = base["delist_events"]
    # BTC beta
    jb = syms.index("BTCUSDT")
    btc = d["C"][1:, jb] / d["C"][:-1, jb] - 1
    btc = np.r_[0, np.nan_to_num(btc)]
    for nm, (a, b) in {"IS": (i0, i1), "OOS": (o0, o1)}.items():
        x, y = btc[a:b], r[a:b]
        det[nm]["beta_btc"] = round(float(np.cov(x, y)[0, 1] / np.var(x)), 3)
    # бутстрэп
    bs = block_boot_sharpe(r[o0:o1])
    det["OOS_boot"] = {"sharpe_p05": round(float(np.percentile(bs, 5)), 2), "sharpe_p50": round(float(np.median(bs)), 2),
                       "sharpe_p95": round(float(np.percentile(bs, 95)), 2), "P_sharpe_gt0": round(float(np.mean(bs > 0)), 3),
                       "P_sharpe_ge1": round(float(np.mean(bs >= 1)), 3)}
    det["OOS_t_stat"] = round(sharpe(r[o0:o1]) * np.sqrt((o1 - o0) / 365.25), 2)
    # 2024-25 vs 2026
    a26 = int(np.searchsorted(ts, ms("2026-01-01")))
    det["OOS_2024_25"] = metrics(ts[o0:a26], r[o0:a26])
    det["OOS_2026"] = metrics(ts[a26:o1], r[a26:o1])
    # вклад монет
    cp = base["coin_pnl_oos"]
    order = np.argsort(-cp)
    det["top_coins_oos"] = [(syms[j], round(float(cp[j]) * 100, 2)) for j in order[:10]]
    det["worst_coins_oos"] = [(syms[j], round(float(cp[j]) * 100, 2)) for j in order[::-1][:10]]
    det["oos_coin_pnl_sum_pct"] = round(float(cp.sum()) * 100, 1)
    cpf = base["coin_pnl"]
    det["top_coins_full"] = [(syms[j], round(float(cpf[j]) * 100, 2)) for j in np.argsort(-cpf)[:10]]
    # лучшие дни
    ro = r[o0:o1]
    top10 = np.argsort(-ro)[:10]
    det["OOS_top10_days"] = [(str(ts[o0 + i].astype("datetime64[ms]"))[:10], round(float(ro[i]) * 100, 2)) for i in top10]
    det["OOS_sharpe_ex_top10_days"] = round(sharpe(np.delete(ro, top10)), 2)
    det["OOS_cagr_ex_top10_days"] = round((float(np.prod(1 + np.delete(ro, top10))) ** (365.25 / (len(ro) - 10)) - 1) * 100, 1)
    ri = r[i0:i1]
    t10i = np.argsort(-ri)[:10]
    det["IS_sharpe_ex_top10_days"] = round(sharpe(np.delete(ri, t10i)), 2)
    results["base_detail"] = det
    print(json.dumps(det, indent=1, default=str)[:6000], flush=True)

    # сохранить дневной ряд (базовый вариант с учётом корректного фандинга)
    OUTD.mkdir(parents=True, exist_ok=True)
    np.savez(OUTD / "verified_daily.npz", ts=ts, ret=r,
             note=np.array("verify_xsec: illiq q0.1 H7 adv5M age60, gross 1.0 (0.5L/0.5S), 1x, taker+slippage "
                           "cost_per_side(ADV30[t-1]), raw funding timing, delisted exit at last close"))

    # сравнение с рядом автора
    try:
        z = np.load(OUTD / "best_daily.npz")
        ra = z["ret"]
        if len(ra) == len(r) and np.all(z["ts"] == ts):
            results["author_series"] = {
                "IS": win_stats(ts, ra, IS), "OOS": win_stats(ts, ra, OOS),
                "corr_full": round(float(np.corrcoef(ra, r)[0, 1]), 3),
                "corr_oos": round(float(np.corrcoef(ra[o0:o1], r[o0:o1])[0, 1]), 3),
                "per_year": per_year(ts, ra),
            }
            print("author series", results["author_series"], flush=True)
    except Exception as e:  # noqa: BLE001
        print("author series load failed", e)

    # ---------------- варианты учёта ----------------
    go("panel_funding_timing", bt_kw={"funding": "panel"})
    go("pessimistic_delist", bt_kw={"pess": True})
    go("cost_x2", bt_kw={"cost_mult": 2.0})
    go("cost_x3", bt_kw={"cost_mult": 3.0})
    go("lag2_exec_t+2", bt_kw={"lag": 2})
    go("include_tradfi", sig_kw={"include_tradfi": True})
    go("cap_ret_25pct", bt_kw={"cap_ret": 0.25})
    go("cap_ret_50pct", bt_kw={"cap_ret": 0.50})
    go("long_leg_only", bt_kw={"leg": "long"})
    go("short_leg_only", bt_kw={"leg": "short"})
    go("age_ge_365", sig_kw={"min_age_days": 365})
    go("age_ge_180", sig_kw={"min_age_days": 180})
    # ---------------- ±25% параметров ----------------
    for q in (0.075, 0.125, 0.05, 0.15, 0.2):
        go(f"q={q}", sig_kw={"q": q})
    for H in (5, 9, 1, 3, 14):
        go(f"H={H}", sig_kw={"H": H})
    for w, mv in ((22, 16), (38, 27), (14, 10), (60, 42)):
        go(f"win={w}", sig_kw={"win": w, "minv": mv})
    for a in (3.75e6, 6.25e6, 2e6, 1e7, 2e7):
        go(f"adv>={a/1e6:g}M", sig_kw={"adv_min": a})
    for ag in (45, 75):
        go(f"age>={ag}", sig_kw={"age_min": ag})
    # совместные ±25%
    for q in (0.075, 0.125):
        for H in (5, 9):
            for w, mv in ((22, 16), (38, 27)):
                go(f"joint q{q} H{H} w{w}", sig_kw={"q": q, "H": H, "win": w, "minv": mv})
    # недельная ребалансировка без перекрытия, разные дни недели
    for off in range(7):
        go(f"weekly_H1_offset{off}", sig_kw={"H": 1}, bt_kw={"rebalance_every": 7, "reb_offset": off})
    # порог минимальной сделки (имитация мин. ордера): 0.1% и 0.5% капитала
    go("min_trade_0.1pct", bt_kw={"min_trade": 0.001})
    go("min_trade_0.5pct", bt_kw={"min_trade": 0.005})
    # факторы-«контрольные»: чистая волатильность и чистый размер
    go("ctrl_vol_only", sig_kw={"fac": "vol"})
    go("ctrl_size_only(1/QV)", sig_kw={"fac": "size"})

    # ---------------- исключить топ-5 монет (по вкладу OOS и по всему периоду) ----------------
    top5 = np.argsort(-base["coin_pnl_oos"])[:5]
    ex = np.zeros(N, bool)
    ex[top5] = True
    go("drop_top5_oos_coins", sig_kw={"excl": ex})
    top5f = np.argsort(-base["coin_pnl"])[:5]
    exf = np.zeros(N, bool)
    exf[top5f] = True
    go("drop_top5_full_coins", sig_kw={"excl": exf})
    top10 = np.argsort(-base["coin_pnl_oos"])[:10]
    ex10 = np.zeros(N, bool)
    ex10[top10] = True
    go("drop_top10_oos_coins", sig_kw={"excl": ex10})
    results["dropped_top5_oos"] = [syms[j] for j in top5]
    results["dropped_top5_full"] = [syms[j] for j in top5f]

    # по годам для ключевых вариантов
    results["elapsed_s"] = round(time.time() - t_start, 1)
    # распределение OOS по всем вариантам робастности (не выбор!)
    keys = [k for k, v in results.items() if isinstance(v, dict) and "OOS_sharpe" in v]
    oo = np.array([results[k]["OOS_sharpe"] for k in keys])
    results["robust_grid_summary"] = {"n": len(oo), "median_oos": float(np.median(oo)), "share_pos": float(np.mean(oo > 0)),
                                      "share_ge1": float(np.mean(oo >= 1))}
    (OUTD / "verify_results.json").write_text(json.dumps(results, indent=1, default=str))
    print("done", results["elapsed_s"], results["robust_grid_summary"])


def run_offset():
    """Сдвиг времени дневного бара/ребалансировки: дневные бары, начинающиеся в час h UTC, из панели 1h."""
    t_start = time.time()
    p1 = panel("1h")
    ts1 = p1["ts"]
    syms = p1["symbols"]
    res_all = {}
    for h in (0, 4, 8, 12, 16, 20):
        # индекс дня: (ts - h*HOUR) // DAY
        day = (ts1 - h * HOUR) // DAY
        first_day = day[0]
        start = np.flatnonzero(day != np.r_[day[0] - 1, day[:-1]])
        dts = (day[start]) * DAY + h * HOUR
        o = p1["open"]
        c = p1["close"]
        qv = p1["qvol"]
        T1, N = o.shape
        # open = первый валидный open в окне, close = последний валидный close
        nd = len(start)
        ends = np.r_[start[1:], T1]
        O = np.full((nd, N), np.nan)
        C = np.full((nd, N), np.nan)
        QV = np.full((nd, N), np.nan)
        F = np.zeros((nd, N))
        valid = np.isfinite(c)
        for i in range(nd):
            a, b = start[i], ends[i]
            v = valid[a:b]
            any_ = v.any(0)
            if not any_.any():
                continue
            fi = np.argmax(v, 0)
            li = (b - a - 1) - np.argmax(v[::-1], 0)
            cols = np.flatnonzero(any_)
            O[i, cols] = o[a + fi[cols], cols]
            C[i, cols] = c[a + li[cols], cols]
            QV[i, cols] = np.nansum(qv[a:b, cols], 0)
        # фандинг по сырым расчётам (граница бара -> предыдущий бар)
        t0 = int(dts[0])
        for j, s in enumerate(syms):
            f = load_funding(str(s))
            if len(f["ts"]) == 0:
                continue
            tr = (f["ts"].astype(np.int64) + 30_000) // 60_000 * 60_000
            bb = (tr - t0 - 1) // DAY
            ok = (bb >= 0) & (bb < nd)
            np.add.at(F[:, j], bb[ok], f["rate"][ok])
        dp = {"ts": dts, "symbols": syms, "open": O, "close": C, "qvol": QV, "funding": F}
        d = {"ts": dts, "symbols": syms, "O": O, "C": C, "QV": QV, "F": F, "panel_funding": F}
        d["noncrypto"] = noncrypto_mask(dp)
        W, adv, nk = signals(d)
        res = backtest(d, W, adv)
        s = summarize(d, res, f"offset_{h}h")
        s["per_year"] = per_year(dts, res["ret"])
        s["halfyears"] = halfyears(dts, res["ret"])
        res_all[f"{h}h"] = s
        print(f"offset {h:2d}h  IS {s['IS_sharpe']:5.2f}  OOS {s['OOS_sharpe']:5.2f}  OOS cagr {s['OOS']['cagr_pct']:6.1f}  "
              f"{time.time() - t_start:.0f}s", flush=True)
    (OUTD / "verify_offset.json").write_text(json.dumps(res_all, indent=1, default=str))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    if mode == "main":
        run_main()
    elif mode == "offset":
        run_offset()
