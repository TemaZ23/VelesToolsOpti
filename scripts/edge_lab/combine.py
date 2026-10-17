"""
Комбинированный портфель из кандидатов, прошедших состязательную проверку (вердикт 'confirmed' или 'weakened'),
и прогноз роста счёта блочным бутстрэпом OOS.

Прошли проверку с вердиктом weakened (confirmed — ни одного):
  carry    — funding: статичный cash-and-carry BTC+ETH (лонг спот + шорт перп, маржа 0.5), verified_daily.npz
  xsec     — Amihud illiquidity L/S (q 0.1, H 7, ADV>=5M), исправленный ряд верификатора (без зомби-баров)
  listings — шорт нового крипто-перпа через 24 ч, SL 30% / TP 25% / 7 дней, хедж BTC, equity/20 на сделку
Не проверены (verify = null) и в портфель НЕ входят: trend, flow, seasonality (только корреляции для справки).

Логика портфеля:
  * веса inverse-vol считаются ТОЛЬКО по IS (2020-01-01..2023-12-31), дневная σ каждого рукава;
  * «строгий» вариант по трём рукавам почти целиком уходит в carry (σ carry ≈ 1% против 9–25%), т.е.
    превращается в долларовый депозит; carry сверх ставки T-bill в OOS = +0.08%/год (Sharpe 0.21), поэтому
    основной (MAIN) портфель — inverse-vol по двум рисковым рукавам xsec + listings, а carry — лишь
    «куда положить свободный USDT, если фандинг выше ставки»;
  * сумма весов = 1 (разделение капитала на два суб-счёта), плечо L масштабирует экспозицию обоих рукавов;
  * ребалансировка между рукавами: ежедневная (основной ряд) и ежемесячная (проверка);
  * стресс: издержки ×2 + −50% при делистинге (xsec) и издержки ×2 + стоп +1% + 25% глубже в бар (listings)
    — ряды из combine_stress.py (те же независимые реализации верификаторов).

Прогноз: кольцевой блочный бутстрэп OOS-дневных доходностей (блоки 20 дней, 10 000 путей, 10 лет), старт
$1000 и $5000, +$100 в конце каждого месяца, плечо 1x … пол-Келли (Келли = mean/var дневных OOS-доходностей).

Запуск: python3 -m scripts.edge_lab.combine_stress   (один раз, ≈10 с)
        python3 -m scripts.edge_lab.combine          (≈1–2 мин)
Выход:  output/edge_lab/combined.json, output/edge_lab/combine/combined_daily.npz, best_daily.npz
"""

from __future__ import annotations

import json
import time

import numpy as np

from scripts.edge_lab.common import DAY, IS, OOS, OUT, metrics, ms, panel, per_year, window_idx

OUTD = OUT / "combine"
REC = ("2025-07-01", "2026-10-01")  # последние 15 месяцев (в них listings перестал работать)
MID = ("2024-01-01", "2025-07-01")
N_PATHS = 10_000
BLOCK = 20
YEARS = 10
DPY = 365  # крипторынок торгуется каждый день
SEED = 20261017


# ------------------------------------------------------------------------------------------------
# данные
# ------------------------------------------------------------------------------------------------
def _load(path, key="ret"):
    z = np.load(OUT / path)
    return z["ts"].astype(np.int64), z[key].astype(float)


def load_all() -> dict:
    ts, carry = _load("funding/verified_daily.npz")
    z = np.load(OUT / "funding/verified_daily.npz")
    rf = z["rf_daily"].astype(float)
    carry_ex = z["ret_excess_tbill"].astype(float)
    tx, xsec = _load("xsec/verified_daily.npz")
    _, xsec_p50 = _load("xsec/verified_daily.npz", "ret_pess50")
    tl, lst = _load("listings/verified_daily.npz")
    assert np.array_equal(ts, tx) and np.array_equal(ts, tl), "сетки дней не совпадают"
    S = {"carry": carry, "xsec": xsec, "listings": lst}
    V = {"carry_excess_tbill": carry_ex, "xsec_pess50": xsec_p50}
    sp = OUTD / "stress_daily.npz"
    if sp.exists():
        s = np.load(sp)
        assert np.array_equal(s["ts"], ts)
        for k in ("xsec_cost2", "xsec_cost2_pess50", "lst_cost2", "lst_pess_combo", "lst_nohedge", "lst_nohedge_pess"):
            V[k] = s[k].astype(float)
        V["_check_xsec"] = float(np.max(np.abs(s["xsec_base_check"] - xsec)))
        V["_check_lst"] = float(np.max(np.abs(s["lst_base_check"] - lst)))
    else:
        print("WARNING: нет stress_daily.npz — запустите combine_stress.py")
    U = {}
    for fam in ("trend", "flow", "seasonality"):
        t, r = _load(f"{fam}/best_daily.npz")
        assert np.array_equal(t, ts)
        U[fam] = r
    t, r = _load("flow/tradecount_ts_daily.npz")
    a = np.zeros(len(ts))
    a[np.searchsorted(ts, t)] = r
    U["flow_tradecount"] = a
    # BTC buy&hold перп (close-to-close по дневной панели) — только для беты и сравнения
    p = panel("1d")
    j = list(p["symbols"]).index("BTCUSDT")
    pts = p["ts"].astype(np.int64)
    c = p["close"][:, j].astype(float)
    br = np.r_[0.0, c[1:] / c[:-1] - 1]
    btc = np.zeros(len(ts))
    pos = np.searchsorted(pts, ts)
    ok = (pos < len(pts)) & (pts[np.minimum(pos, len(pts) - 1)] == ts)
    btc[ok] = np.nan_to_num(br[pos[ok]])
    return {"ts": ts, "rf": rf, "S": S, "V": V, "U": U, "btc": btc}


# ------------------------------------------------------------------------------------------------
# метрики
# ------------------------------------------------------------------------------------------------
def halfyears(ts, r, start="2020"):
    y = ts.astype("datetime64[ms]").astype("datetime64[Y]").astype(int) + 1970
    m = ts.astype("datetime64[ms]").astype("datetime64[M]").astype(int) % 12
    out = {}
    for yy in np.unique(y):
        for h in (0, 1):
            k = (y == yy) & ((m >= 6) == bool(h))
            if k.sum() and str(yy) >= start:
                out[f"{yy}H{h + 1}"] = round((float(np.prod(1 + r[k])) - 1) * 100, 1)
    return out


def stat(ts, r, rf, window) -> dict:
    a, b = window_idx(ts, window)
    x, f = r[a:b], rf[a:b]
    m = metrics(ts[a:b], x)
    sd = float(np.std(x))
    m["vol_pct"] = round(sd * np.sqrt(365) * 100, 2)
    ex = x - f
    m["excess_sharpe_tbill"] = round(float(np.mean(ex) / np.std(ex) * np.sqrt(365)) if np.std(ex) > 0 else 0.0, 2)
    m["excess_cagr_pct"] = round((float(np.prod(1 + x) / np.prod(1 + f)) ** (365.25 / max(len(x), 1)) - 1) * 100, 2)
    m["t_stat"] = round(float(np.mean(x) / sd * np.sqrt(len(x))) if sd > 0 else 0.0, 2)
    return m


def full_stats(ts, r, rf, btc=None) -> dict:
    out = {"IS": stat(ts, r, rf, IS), "OOS": stat(ts, r, rf, OOS), "OOS_2024_2025H1": stat(ts, r, rf, MID),
           "OOS_last15m_2025H2_2026Q3": stat(ts, r, rf, REC), "per_year": per_year(ts, r),
           "half_years": halfyears(ts, r)}
    c, d = window_idx(ts, OOS)
    ro = r[c:d]
    top = np.argsort(-ro)[:10]
    rest = np.delete(ro, top)
    out["OOS_sharpe_ex_top10_days"] = round(float(np.mean(rest) / np.std(rest) * np.sqrt(365)), 2)
    out["OOS_cagr_ex_top10_days"] = round((float(np.prod(1 + rest)) ** (365.25 / len(rest)) - 1) * 100, 2)
    hy = {k: v for k, v in out["half_years"].items() if k >= "2024"}
    out["OOS_halfyears_positive"] = f"{sum(v > 0 for v in hy.values())}/{len(hy)}"
    if btc is not None:
        for nm, w in (("IS", IS), ("OOS", OOS)):
            a, b = window_idx(ts, w)
            out[nm]["beta_btc"] = round(float(np.cov(btc[a:b], r[a:b])[0, 1] / np.var(btc[a:b])), 3)
    return out


def corr(ts, series: dict, window) -> dict:
    a, b = window_idx(ts, window)
    ks = list(series)
    M = np.corrcoef(np.array([series[k][a:b] for k in ks]))
    return {"names": ks, "matrix": np.round(M, 3).tolist()}


# ------------------------------------------------------------------------------------------------
# портфели
# ------------------------------------------------------------------------------------------------
def iv_weights(ts, series: dict) -> dict:
    a, b = window_idx(ts, IS)
    inv = {k: 1.0 / float(np.std(v[a:b])) for k, v in series.items()}
    s = sum(inv.values())
    return {k: v / s for k, v in inv.items()}


def combine_daily(series: dict, w: dict) -> np.ndarray:
    return sum(w[k] * series[k] for k in w)


def combine_monthly(ts, series: dict, w: dict) -> np.ndarray:
    """Ребалансировка рукавов к весам в первый день месяца, внутри месяца рукава дрейфуют."""
    ks = list(w)
    R = np.array([series[k] for k in ks])
    mon = ts.astype("datetime64[ms]").astype("datetime64[M]").astype(int)
    v = np.array([w[k] for k in ks])
    out = np.zeros(len(ts))
    for t in range(len(ts)):
        if t > 0 and mon[t] != mon[t - 1]:
            v = np.array([w[k] for k in ks]) * v.sum()
        tot = v.sum()
        v = v * (1 + R[:, t])
        out[t] = v.sum() / tot - 1
    return out


# ------------------------------------------------------------------------------------------------
# прогноз: блочный бутстрэп
# ------------------------------------------------------------------------------------------------
def boot_idx(n: int, n_paths: int, horizon: int, block: int, rng) -> np.ndarray:
    """Кольцевой блочный бутстрэп: индексы дней [n_paths, horizon], блоки по `block` дней подряд."""
    nb = -(-horizon // block)
    starts = rng.integers(0, n, size=(n_paths, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(n_paths, nb * block)[:, :horizon].astype(np.int32)


X_MIN_USD = 3000.0  # минимальный номинал рукава xsec (позиции ~$45, дневные доборы >= $5); меньше — рукав в кэше
H_MIN_USD = 100.0   # минимальный ордер BTCUSDT: номинал сделки listings < $100 -> без хеджа BTC


def simulate(idx: np.ndarray, comp: dict, w: dict, levs, cap_1x: float, starts=(1000.0, 5000.0),
             contrib=100.0) -> dict:
    """Три режима счёта на одних и тех же бутстрэп-путях:
      plain     — MAIN-ряд при постоянном плече L без ограничений;
      capped    — то же, но экспозиция не больше ёмкости cap_1x (излишек счёта лежит под 0%);
      realistic — capped + минимальные ордера Binance: пока L*w_xsec*A < X_MIN_USD, рукав xsec не торгуется
                  (его доля в кэше под 0%); пока номинал сделки listings L*w_lst*A/20 < H_MIN_USD — без хеджа BTC.
    NAV-метрики (просадки, медианный CAGR) — для MAIN-ряда при плече L (plain)."""
    n_paths, H = idx.shape
    wx, wl = w["xsec"], w["listings"]
    yr_marks = {DPY * y: y for y in (1, 2, 3, 5, 10)}
    mon_end = set(int(round(m * DPY / 12)) - 1 for m in range(1, YEARS * 12 + 1))
    modes = ("plain", "capped", "realistic")
    out = {m: {} for m in modes}
    out["nav"] = {}
    for L in levs:
        key = f"{L:g}x"
        nav = np.ones(n_paths)
        peak = np.ones(n_paths)
        mdd = np.zeros(n_paths)
        dd50_5 = dd30_5 = None
        ruin = np.zeros(n_paths, bool)
        A = {(m, s): np.full(n_paths, s) for m in modes for s in starts}
        reach = {k: {th: np.full(n_paths, np.inf) for th in (1e5, 1e6)} for k in A}
        snaps = {k: {} for k in A}
        nav_snap = {}
        contributed = {s: s for s in starts}
        xs_on_days = {s: np.zeros(n_paths) for s in starts}
        for t in range(H):
            it = idx[:, t]
            rx, rlh, rln = comp["x"][it], comp["lh"][it], comp["ln"][it]
            rm = wx * rx + wl * rlh
            g = np.maximum(1 + L * rm, 0.0)
            nav *= g
            ruin |= nav <= 1e-6
            np.maximum(peak, nav, out=peak)
            np.maximum(mdd, 1 - nav / peak, out=mdd)
            if t == 5 * DPY - 1:
                dd50_5, dd30_5 = mdd >= 0.5, mdd >= 0.3
            is_m = t in mon_end
            if is_m:
                for s in starts:
                    contributed[s] += contrib
            for (m, s), a in A.items():
                if m == "plain":
                    a *= g
                else:
                    eff = np.minimum(L, cap_1x / np.maximum(a, 1e-9))
                    if m == "capped":
                        a *= np.maximum(1 + eff * rm, 0.0)
                    else:
                        on_x = (a * L * wx) >= X_MIN_USD
                        hedge = (a * L * wl / 20) >= H_MIN_USD
                        rr = wx * rx * on_x + wl * np.where(hedge, rlh, rln)
                        a *= np.maximum(1 + eff * rr, 0.0)
                        xs_on_days[s] += on_x
                if is_m:
                    a += contrib
                for th in (1e5, 1e6):
                    tt = reach[(m, s)][th]
                    hit = (a >= th) & np.isinf(tt)
                    tt[hit] = (t + 1) / DPY
            if (t + 1) in yr_marks:
                y = yr_marks[t + 1]
                nav_snap[y] = nav.copy()
                for k in A:
                    snaps[k][y] = (A[k].copy(), contributed[k[1]])
        out["nav"][key] = {
            "leverage": L,
            "nav_cagr_median_pct": {y: round((float(np.median(v)) ** (1 / y) - 1) * 100, 1) for y, v in nav_snap.items()},
            "P_maxDD_ge30pct_5y": round(float(np.mean(dd30_5)), 4),
            "P_maxDD_ge50pct_5y": round(float(np.mean(dd50_5)), 4),
            "P_maxDD_ge50pct_10y": round(float(np.mean(mdd >= 0.5)), 4),
            "P_ruin_10y": round(float(np.mean(ruin)), 4),
            "median_maxDD_10y_pct": round(float(np.median(mdd)) * 100, 1)}
        for m in modes:
            res = {}
            for s in starts:
                k = (m, s)
                acc = {}
                for y, (v, contr) in snaps[k].items():
                    acc[f"{y}y"] = {"p10": round(float(np.percentile(v, 10))), "median": round(float(np.median(v))),
                                    "p90": round(float(np.percentile(v, 90))), "contributed": round(contr),
                                    "P_below_contributed": round(float(np.mean(v < contr)), 4)}
                for th, nm in ((1e5, "100k"), (1e6, "1M")):
                    for yy in (5, 10):
                        acc[f"P_reach_{nm}_{yy}y"] = round(float(np.mean(reach[k][th] <= yy)), 4)
                if m == "realistic":
                    acc["xsec_sleeve_active_share_of_days"] = round(float(np.mean(xs_on_days[s])) / H, 3)
                res[f"start_{int(s)}"] = acc
            out[m][key] = res
    return out


def dd_within(R: np.ndarray, L: float, days: int, q=(50, 90, 95, 99)) -> dict:
    nav = np.cumprod(np.maximum(1 + L * R[:, :days], 0), axis=1)
    peak = np.maximum(np.maximum.accumulate(nav, axis=1), 1.0)  # пик с учётом стартового капитала 1.0
    mdd = np.max(1 - nav / peak, axis=1)
    ret = nav[:, -1] - 1
    return {"maxDD_pct": {f"p{k}": round(float(np.percentile(mdd, k)) * 100, 1) for k in q},
            "return_pct": {f"p{k}": round(float(np.percentile(ret, k)) * 100, 1) for k in (1, 5, 10, 50, 90)}}


def _dates(ts, mask):
    """Отрезки дат, где mask истинна: [(начало, конец), ...]."""
    idx = np.flatnonzero(mask)
    seg = []
    for i in idx:
        if seg and i == seg[-1][1] + 1:
            seg[-1][1] = i
        else:
            seg.append([i, i])
    f = lambda i: str(ts[i].astype("datetime64[ms]"))[:10]  # noqa: E731
    return [(f(a), f(b)) for a, b in seg]


def kill_history(ts, series: dict) -> dict:
    """Когда исторически сработали бы стоп-правила (1x): просадка от пика и скользящая 365-дн. доходность;
    для listings — среднее последних 100 сделок <= 0 (сделки из combine_stress.py)."""
    rules = {"MAIN": (0.20, 0.0), "xsec": (0.32, -0.17), "listings": (0.25, -0.09)}
    out = {}
    for nm, r in series.items():
        eq = np.cumprod(1 + r)
        dd = 1 - eq / np.maximum.accumulate(eq)
        r365 = np.full(len(r), np.nan)
        r365[365:] = eq[365:] / eq[:-365] - 1
        ddl, rl = rules[nm]
        out[nm] = {"rule_dd": ddl, "rule_r365": rl, "max_dd_full_pct": round(float(dd.max()) * 100, 1),
                   "dd_trigger_periods": _dates(ts, dd >= ddl),
                   "r365_min_pct": round(float(np.nanmin(r365)) * 100, 1),
                   "r365_trigger_periods": _dates(ts, np.nan_to_num(r365, nan=1.0) < rl)}
    tp = OUTD / "listings_trades.json"
    if tp.exists():
        tr = sorted(json.loads(tp.read_text()), key=lambda x: x["t_entry"])
        x = np.array([t["total"] for t in tr])
        tt = np.array([t["t_entry"] for t in tr], dtype=np.int64)
        st = np.array([t["reason"] == "stop" for t in tr], float)
        o = {}
        for n in (50, 100):
            m = np.convolve(x, np.ones(n) / n, "valid")
            o[f"rolling{n}_mean_le0_periods"] = _dates(tt[n - 1:], m <= 0)
            o[f"last{n}_mean_pct"] = round(float(x[-n:].mean()) * 100, 2)
            o[f"last{n}_stop_share"] = round(float(st[-n:].mean()), 3)
        o["n_trades"] = int(len(x))
        o["last_trade"] = str(tt[-1].astype("datetime64[ms]"))[:16]
        o["mean_trade_pct"] = {nm: round(float(x[(tt >= ms(w[0])) & (tt < ms(w[1]))].mean()) * 100, 2)
                               for nm, w in (("IS", IS), ("OOS", OOS), ("recent", REC))}
        out["listings_trades"] = o
    return out


def deterministic(start, contrib, rate, years):
    a = start
    for m in range(years * 12):
        a = a * (1 + rate) ** (1 / 12) + contrib
    return round(a)


# ------------------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    OUTD.mkdir(parents=True, exist_ok=True)
    D = load_all()
    ts, rf, S, V, U, btc = D["ts"], D["rf"], D["S"], D["V"], D["U"], D["btc"]
    res = {"meta": {
        "survivors": {"carry": "funding / weakened (return ≈ T-bill rate)", "xsec": "xsec / weakened",
                      "listings": "listings / weakened"},
        "excluded_unverified": ["trend (weak, verify=null)", "flow (no, verify=null)", "seasonality (no, verify=null)"],
        "series": {"carry": "output/edge_lab/funding/verified_daily.npz:ret",
                   "xsec": "output/edge_lab/xsec/verified_daily.npz:ret (zombie bars removed, settlement-price exit)",
                   "listings": "output/edge_lab/listings/verified_daily.npz:ret"},
        "IS": IS, "OOS": OOS, "recent": REC, "boot": {"paths": N_PATHS, "block_days": BLOCK, "years": YEARS,
                                                    "contrib_usd_month": 100, "starts_usd": [1000, 5000]},
        "stress_check_max_abs_diff": {k: V.get(k) for k in ("_check_xsec", "_check_lst")},
    }}
    # --- отдельные рукава ---
    res["sleeves"] = {k: full_stats(ts, v, rf, btc) for k, v in S.items()}
    res["variants"] = {k: full_stats(ts, v, rf, btc) for k, v in V.items() if not k.startswith("_")}
    res["unverified_reference"] = {k: full_stats(ts, v, rf, btc) for k, v in U.items()}
    res["btc_buy_hold_perp_price_only"] = full_stats(ts, btc, rf)
    allser = {**S, **U, "btc": btc}
    res["correlations"] = {"IS": corr(ts, allser, IS), "OOS": corr(ts, allser, OOS)}

    # --- портфели ---
    w3 = iv_weights(ts, S)
    w2 = iv_weights(ts, {k: S[k] for k in ("xsec", "listings")})
    res["weights_is_inverse_vol"] = {"strict_3way": {k: round(v, 4) for k, v in w3.items()},
                                     "main_risk_2way": {k: round(v, 4) for k, v in w2.items()}}
    a, b = window_idx(ts, IS)
    res["is_daily_vol_ann_pct"] = {k: round(float(np.std(v[a:b])) * np.sqrt(365) * 100, 2) for k, v in S.items()}
    P = {}
    P["strict_3way_iv"] = combine_daily(S, w3)
    P["MAIN_xsec_listings_iv"] = combine_daily(S, w2)
    P["MAIN_monthly_rebalance"] = combine_monthly(ts, S, w2)
    P["MAIN_xsec_pess50"] = combine_daily({"xsec": V["xsec_pess50"], "listings": S["listings"]}, w2)
    if "xsec_cost2" in V:
        P["MAIN_costs_x2"] = combine_daily({"xsec": V["xsec_cost2"], "listings": V["lst_cost2"]}, w2)
        P["MAIN_STRESS_cost2_pess50_stopfill"] = combine_daily(
            {"xsec": V["xsec_cost2_pess50"], "listings": V["lst_pess_combo"]}, w2)
    for f in (0.75, 1.25):
        wx = w2["xsec"] * f
        ww = {"xsec": wx, "listings": 1 - wx}
        P[f"MAIN_xsec_weight_x{f}"] = combine_daily(S, ww)
    P["equal_weight_xsec_listings"] = combine_daily(S, {"xsec": 0.5, "listings": 0.5})
    P["equal_weight_3way"] = combine_daily(S, {k: 1 / 3 for k in S})
    # контроль на отбор по OOS: все 5 рисковых IS-лучших кандидатов семейств (без учёта того, кто прошёл проверку)
    five = {"xsec": S["xsec"], "listings": S["listings"], "trend": U["trend"], "flow": U["flow"],
            "seasonality": U["seasonality"]}
    w5 = iv_weights(ts, five)
    res["weights_is_inverse_vol"]["control_all5_risky_family_candidates"] = {k: round(v, 4) for k, v in w5.items()}
    P["CONTROL_all5_family_IS_best_iv_no_OOS_selection"] = combine_daily(five, w5)
    res["portfolios"] = {k: full_stats(ts, v, rf, btc) for k, v in P.items()}
    main = P["MAIN_xsec_listings_iv"]
    stress = P.get("MAIN_STRESS_cost2_pess50_stopfill", P["MAIN_xsec_pess50"])

    # P&L вклад рукавов в MAIN по OOS (простая сумма дневных вкладов)
    c, d = window_idx(ts, OOS)
    res["MAIN_oos_contribution_pct_sum"] = {k: round(float(np.sum(w2[k] * S[k][c:d])) * 100, 2) for k in w2}
    e, f_ = window_idx(ts, REC)
    res["MAIN_recent_contribution_pct_sum"] = {k: round(float(np.sum(w2[k] * S[k][e:f_])) * 100, 2) for k in w2}

    # --- Келли и уровни плеча ---
    def kelly(x):
        return float(np.mean(x) / np.var(x))

    kel = {"MAIN_OOS": kelly(main[c:d]), "MAIN_IS": kelly(main[a:b]), "STRESS_OOS": kelly(stress[c:d]),
           "MAIN_recent15m": kelly(main[e:f_])}
    res["kelly_full"] = {k: round(v, 2) for k, v in kel.items()}
    half_k = kel["MAIN_OOS"] / 2
    levs = sorted({1.0, 1.5, 2.0, 3.0, round(half_k, 1)} if half_k > 3 else
                  {lv for lv in (1.0, 1.5, 2.0) if lv <= max(half_k, 1.0)} | {round(max(half_k, 1.0), 1)})
    res["leverage_levels"] = levs
    res["half_kelly_MAIN_OOS"] = round(half_k, 2)

    # --- бутстрэп ---
    rng = np.random.default_rng(SEED)
    cap_1x = min(100000 / w2["xsec"], 600000 / w2["listings"])
    res["capacity_1x_equivalent_usd"] = round(cap_1x)
    res["projection_rules"] = {
        "plain": "MAIN at constant leverage L, no limits",
        "capped": f"exposure <= ${cap_1x:,.0f} (xsec sleeve capacity $100k / w_xsec), excess capital idle at 0%",
        "realistic": f"capped + Binance minimum orders: xsec sleeve trades only if L*w_xsec*A >= ${X_MIN_USD:,.0f} "
                     f"(else that share idle at 0%); listings trades without BTC hedge while L*w_lst*A/20 < ${H_MIN_USD:,.0f}",
        "contributions": "+$100 at the end of every month (also in months with drawdowns)",
        "drawdown_metrics": "on the strategy NAV (time-weighted, no contributions), plain mode"}
    res["projection"] = {}
    win = {"base_OOS_2024_2026": OOS, "stress_OOS_cost2_pess50_stopfill": OOS, "recent_15m_2025H2_2026Q3": REC}
    comps = {"base_OOS_2024_2026": ("xsec", "listings", "lst_nohedge"),
             "stress_OOS_cost2_pess50_stopfill": ("xsec_cost2_pess50", "lst_pess_combo", "lst_nohedge_pess"),
             "recent_15m_2025H2_2026Q3": ("xsec", "listings", "lst_nohedge")}
    allv = {**S, **V}
    for nm in win:
        tt = time.time()
        i0, i1 = window_idx(ts, win[nm])
        cx, ch, cn = comps[nm]
        comp = {"x": allv[cx][i0:i1], "lh": allv[ch][i0:i1], "ln": allv[cn][i0:i1]}
        r = w2["xsec"] * comp["x"] + w2["listings"] * comp["lh"]
        n = len(r)
        idx = boot_idx(n, N_PATHS, YEARS * DPY, BLOCK, rng)
        sim = simulate(idx, comp, w2, levs, cap_1x)
        R1 = r[idx[:, :DPY]]
        one_year = {f"{L:g}x": dd_within(R1, L, DPY) for L in levs}
        Rs = r[idx[:2000, :n]]
        shs = Rs.mean(1) / Rs.std(1) * np.sqrt(365)
        res["projection"][nm] = {"n_days": int(n), "components": comps[nm],
                                 "mean_daily_pct": round(float(np.mean(r)) * 100, 4),
                                 "ann_vol_pct": round(float(np.std(r)) * np.sqrt(365) * 100, 2),
                                 "sharpe": round(float(np.mean(r) / np.std(r) * np.sqrt(365)), 2),
                                 "sharpe_boot_same_length": {"p05": round(float(np.percentile(shs, 5)), 2),
                                                             "p50": round(float(np.median(shs)), 2),
                                                             "p95": round(float(np.percentile(shs, 95)), 2),
                                                             "P_le0": round(float(np.mean(shs <= 0)), 3),
                                                             "P_ge1": round(float(np.mean(shs >= 1)), 3)},
                                 "kelly_full": round(kelly(r), 2), "nav": sim["nav"],
                                 "accounts_plain": sim["plain"], "accounts_capped": sim["capped"],
                                 "accounts_realistic": sim["realistic"], "one_year_dist": one_year}
        print(f"boot {nm}: {time.time() - tt:.0f}s", flush=True)
        del idx, R1, Rs
    # пороги для стоп-правил: распределения 6- и 12-месячных результатов рукавов и MAIN при 1x (бутстрэп OOS)
    kill = {}
    for nm, r in (("MAIN", main), ("xsec", S["xsec"]), ("listings", S["listings"])):
        ro = r[c:d]
        idx = boot_idx(len(ro), N_PATHS, DPY, BLOCK, rng)
        R1 = ro[idx]
        kill[nm] = {"6m": dd_within(R1, 1.0, 182), "12m": dd_within(R1, 1.0, DPY)}
    res["kill_switch_reference_1x_base_OOS_boot"] = kill
    res["kill_switch_history"] = kill_history(ts, {"MAIN": main, "xsec": S["xsec"], "listings": S["listings"]})
    res["deterministic_reference"] = {
        f"start_{s}": {f"{y}y": {"save_0pct": deterministic(s, 100, 0.0, y), "cash_4pct": deterministic(s, 100, 0.04, y)}
                       for y in (1, 2, 3, 5, 10)} for s in (1000, 5000)}

    # --- ёмкость/минимальный счёт (из отчётов семейств) ---
    res["capacity_notes"] = {
        "xsec_binding_trade_usd_median": 1615, "xsec_binding_trade_usd_p10": 811,
        "xsec_sleeve_equity_cap_usd": 100000, "listings_trade_usd_median": 31000, "listings_trade_usd_p10": 4400,
        "MAIN_account_cap_usd_at_1x": round(100000 / w2["xsec"]),
        "min_account_note": "xsec sleeve needs >= ~$3k of its own notional (min order $5, BTC $100, ETH $20); "
                            "listings BTC hedge needs >= $100 per trade (sleeve >= ~$2k at K=20)",
        "MAIN_min_account_usd_at_1x_approx": round(X_MIN_USD / w2["xsec"]),
    }
    res["elapsed_s"] = round(time.time() - t0, 1)
    (OUT / "combined.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    np.savez(OUTD / "combined_daily.npz", ts=ts, **{k: v for k, v in P.items()}, rf_daily=rf,
             note=np.array("daily net returns, fraction of total account at 1x; MAIN = inverse-vol(IS) "
                           f"xsec {w2['xsec']:.3f} + listings {w2['listings']:.3f}, daily rebalanced"))
    np.savez(OUTD / "best_daily.npz", ts=ts, ret=main,
             note=np.array(f"MAIN combined: {w2['xsec']:.3f} of capital in xsec Amihud L/S (gross 1.0 of sleeve) + "
                           f"{w2['listings']:.3f} in listings short (equity/20 per trade + equal BTC hedge); 1x"))
    # краткая печать
    for k, v in res["portfolios"].items():
        print(f"{k:40s} IS SR {v['IS']['sharpe']:5.2f} CAGR {v['IS']['cagr_pct']:6.1f} | OOS SR {v['OOS']['sharpe']:5.2f} "
              f"CAGR {v['OOS']['cagr_pct']:6.1f} DD {v['OOS']['max_dd_pct']:5.1f} exSR {v['OOS']['excess_sharpe_tbill']:5.2f}"
              f" | rec SR {v['OOS_last15m_2025H2_2026Q3']['sharpe']:5.2f}")
    print("weights", res["weights_is_inverse_vol"], "kelly", res["kelly_full"], "levs", levs)
    print(f"done {res['elapsed_s']}s")


if __name__ == "__main__":
    main()
