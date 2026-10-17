"""
xsec: дополнительные проверки устойчивости для кандидата, выбранного по IS (фактор неликвидности Амихуда),
и для лучшего по IS моментума.

    PYTHONPATH=. python3 -m scripts.edge_lab.xsec_robust

Пишет output/edge_lab/xsec/robust.json и добавляет секцию "robustness_extra" в results.json.

Проверки (всё — без изменения правил выбора; OOS только отчитывается):
  1. Разбиение OOS на 2024–2025 и 2026 (янв–сен); скользящие 365-дневные доходности.
  2. Регрессия дневных доходностей стратегии и её ног на BTC (альфа/бета, t-статистика), IS и OOS.
  3. Плацебо с той же структурой и сопоставимым оборотом: случайный, но «устойчивый» скор монеты
     (перевыбор раз в 30 дней), 50 сидов.
  4. Нейтрализация по размеру и волатильности: двойная сортировка (внутри половин по ADV / по σ),
     остаток ранга неликвидности после кросс-секционной регрессии на ранги ln ADV и σ30.
  5. Отсечение хвостов: дневная доходность каждой монеты обрезана до ±25% / ±50%.
  6. Возраст монеты ≥ 180 / 365 дней (исключение свежих листингов).
  7. Поправка на множественное тестирование (deflated Sharpe, Bailey & López de Prado).
  8. Концентрация ценового P&L по монето-дням (доля лучших 20 / 50 / 1% монето-дней).
"""

from __future__ import annotations

import json
import math

import numpy as np
from numba import njit

from scripts.edge_lab.common import IS, OOS, ms
from scripts.edge_lab.xsec import KEEP, XOUT, build_W, score, uni_of
from scripts.edge_lab.xsec_ctx import Ctx
from scripts.edge_lab.xsec_engine import backtest, ls_weights, overlap, summarize, to_daily, window_stats

BEST = {"hyp": "c_illiq", "fac": "illiq", "sign": 1, "q": 0.1, "H": 7, "bn": 0, "uni": "adv5"}
MOM = {"hyp": "a_mom", "fac": "mom", "L": 7, "skip": 0, "H": 3, "q": 0.1, "wt": "eq", "bn": 1, "uni": "top50", "sign": 1}
SUB = {"OOS_2024_2025": ("2024-01-01", "2026-01-01"), "OOS_2026_jan_sep": ("2026-01-01", "2026-10-01")}
SHORT = ("sharpe", "cagr_pct", "max_dd_pct", "gross_sharpe", "ann_ret_pct", "ann_cost_pct", "turnover_per_day", "tstat")


def st(ts, bt, extra_windows=True) -> dict:
    s = summarize(ts, bt)
    out = {"IS": {k: s["IS"].get(k) for k in SHORT}, "OOS": {k: s["OOS"].get(k) for k in SHORT},
           "oos_half_years": s["oos_half_years"]}  # fmt: skip
    if extra_windows:
        for nm, w in SUB.items():
            ws = window_stats(ts, bt, w)
            out[nm] = {k: ws.get(k) for k in SHORT}
    return out


# ----------------------------------------------------------------------------- statistics helpers
def ols(y: np.ndarray, x: np.ndarray) -> dict:
    """y = a + b x + e; t-статистики по обычным (не HAC) ошибкам — дневные доходности почти не автокоррелированы."""
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ coef
    s2 = e @ e / (len(y) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return {"alpha_ann_pct": round(float(coef[0]) * 365 * 100, 2), "alpha_t": round(float(coef[0] / se[0]), 2),
            "beta": round(float(coef[1]), 3), "beta_t": round(float(coef[1] / se[1]), 1),
            "r2": round(float(1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))), 3)}  # fmt: skip


def norm_ppf(p: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def deflated_sharpe(dr: np.ndarray, n_trials: int) -> dict:
    """Deflated Sharpe ratio (Bailey & López de Prado): вероятность, что истинный SR выбранной стратегии больше
    ожидаемого максимума оценок SR из n_trials независимых шумовых стратегий (нулевая дисперсия оценки SR ≈ 1/T).
    Конфигурации сильно коррелированы, поэтому эффективное число попыток меньше общего — даём несколько значений."""
    g = 0.5772156649
    sr = dr.mean() / dr.std()
    T = len(dr)
    sk = float(((dr - dr.mean()) ** 3).mean() / dr.std() ** 3)
    ku = float(((dr - dr.mean()) ** 4).mean() / dr.std() ** 4)
    sd0 = 1.0 / math.sqrt(T)
    e_max = 0.0 if n_trials <= 1 else sd0 * ((1 - g) * norm_ppf(1 - 1 / n_trials) + g * norm_ppf(1 - 1 / (n_trials * math.e)))
    den = math.sqrt(max(1 - sk * sr + (ku - 1) / 4 * sr * sr, 1e-12))
    dsr = norm_cdf((sr - e_max) * math.sqrt(T - 1) / den)
    return {"n_trials": n_trials, "days": T, "sr_ann": round(sr * math.sqrt(365), 2),
            "expected_max_sr_ann_under_null": round(e_max * math.sqrt(365), 2),
            "prob_true_sr_gt_expected_max": round(dsr, 3), "skew": round(sk, 2), "kurt": round(ku, 1)}  # fmt: skip


def concentration(P: np.ndarray) -> dict:
    """P — ценовой P&L позиций по монето-дням (доля капитала). Доля итога от лучших монето-дней."""
    x = P[P != 0]
    tot = float(x.sum())
    srt = np.sort(x)[::-1]
    return {"total_price_pnl_pct": round(tot * 100, 1), "coin_days": int(len(x)),
            "top20_coin_days_pct": round(float(srt[:20].sum()) * 100, 1), "top50_coin_days_pct": round(float(srt[:50].sum()) * 100, 1),
            "top1pct_coin_days_pct": round(float(srt[: max(1, len(x) // 100)].sum()) * 100, 1),
            "rest_after_top50_pct": round((tot - float(srt[:50].sum())) * 100, 1)}  # fmt: skip


# ----------------------------------------------------------------------------- return clipping
@njit(cache=True)
def clipped_prices(O, C, lim):
    """Синтетические open/close, у которых доходность open→open (и open→close) каждой монеты обрезана до ±lim."""
    T, N = O.shape
    O2 = np.full((T, N), np.nan)
    C2 = np.full((T, N), np.nan)
    for i in range(N):
        prev = -1
        for t in range(T):
            if not np.isfinite(O[t, i]):
                prev = -1
                continue
            if prev < 0:
                O2[t, i] = O[t, i]
            else:
                r = O[t, i] / O[prev, i] - 1.0
                r = min(max(r, -lim), lim)
                O2[t, i] = O2[prev, i] * (1.0 + r)
            if np.isfinite(C[t, i]):
                rc = C[t, i] / O[t, i] - 1.0
                rc = min(max(rc, -lim), lim)
                C2[t, i] = O2[t, i] * (1.0 + rc)
            prev = t
    return O2, C2


# ----------------------------------------------------------------------------- cross-sectional helpers
def xs_rank(S: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Ранг в [0, 1] внутри допустимой вселенной на каждую дату (NaN вне вселенной)."""
    X = np.where(u & np.isfinite(S), S, np.nan)
    order = np.argsort(np.where(np.isnan(X), np.inf, X), axis=1)
    rk = np.empty_like(X)
    np.put_along_axis(rk, order, np.arange(X.shape[1], dtype=float)[None, :].repeat(X.shape[0], 0), axis=1)
    n = np.isfinite(X).sum(axis=1, keepdims=True)
    with np.errstate(all="ignore"):
        out = np.where(np.isfinite(X), rk / np.maximum(n - 1, 1), np.nan)
    return out


def resid_score(S, controls, u):
    """Остаток кросс-секционной регрессии ранга S на ранги контролей (по каждой дате отдельно)."""
    R = xs_rank(S, u)
    Cs = [xs_rank(c, u) for c in controls]
    out = np.full_like(R, np.nan)
    for t in range(R.shape[0]):
        m = np.isfinite(R[t])
        for c in Cs:
            m &= np.isfinite(c[t])
        if m.sum() < 20:
            continue
        X = np.column_stack([np.ones(m.sum())] + [c[t, m] for c in Cs])
        b, *_ = np.linalg.lstsq(X, R[t, m], rcond=None)
        out[t, m] = R[t, m] - X @ b
    return out


def double_sort_W(S, split, u, q, H, min_n=10):
    """Две половины вселенной по контролю split (медиана на дату); внутри каждой L/S по S; половины по 50%."""
    X = np.where(u & np.isfinite(split), split, np.nan)
    with np.errstate(all="ignore"):
        med = np.nanmedian(X, axis=1, keepdims=True)
    lo = u & np.isfinite(X) & (X < med)
    hi = u & np.isfinite(X) & (X >= med)
    one = np.ones(S.shape)
    W = 0.5 * ls_weights(S, lo, one, q, min_n, 0, 0) + 0.5 * ls_weights(S, hi, one, q, min_n, 0, 0)
    return overlap(W, H)


def main() -> None:
    c = Ctx(0)
    ts = c.ts
    out: dict = {"cfg": BEST}
    W = build_W(c, BEST)
    bt = backtest(W, c.ex)
    out["base"] = st(ts, bt)
    dts, dr = to_daily(ts, bt["ret"])

    # 1. rolling 365-day returns
    lr = np.log1p(dr)
    cs = np.r_[0, np.cumsum(lr)]
    roll = np.expm1(cs[365:] - cs[:-365])
    i_oos = int(np.searchsorted(dts[364:], ms(OOS[0])))
    out["rolling_365d"] = {"share_positive_all": round(float((roll > 0).mean()), 3),
                           "min_pct": round(float(roll.min()) * 100, 1), "median_pct": round(float(np.median(roll)) * 100, 1),
                           "share_positive_windows_ending_in_OOS": round(float((roll[i_oos:] > 0).mean()), 3),
                           "min_pct_windows_ending_in_OOS": round(float(roll[i_oos:].min()) * 100, 1)}  # fmt: skip

    # 2. BTC regression (bar s: open[s] -> open[s+1], same interval as strategy returns)
    O = c.ex["open"]
    btc = np.r_[O[1:, c.j_btc] / O[:-1, c.j_btc] - 1, 0.0]
    btc = np.nan_to_num(btc)
    WL, WS = np.where(W > 0, W, 0.0), np.where(W < 0, W, 0.0)
    legs = {"long_short": bt["ret"], "long_leg": backtest(WL, c.ex)["ret"], "short_leg": backtest(WS, c.ex)["ret"]}
    reg = {}
    for nm, r in legs.items():
        for w, (a, b) in (("IS", IS), ("OOS", OOS)):
            i0, i1 = int(np.searchsorted(ts, ms(a))), int(np.searchsorted(ts, ms(b)))
            reg[f"{nm}_{w}"] = ols(r[i0:i1], btc[i0:i1])
    out["btc_regression"] = reg

    # 3. persistent random placebo (similar turnover)
    u = uni_of(c, BEST)
    one = np.ones((c.T, c.N))
    pl = []
    for seed in range(50):
        rng = np.random.default_rng(1000 + seed)
        S = rng.standard_normal((c.T // 30 + 2, c.N))[(np.arange(c.T) + seed % 30) // 30]
        Wr = overlap(ls_weights(S, u, one, BEST["q"], 20, 0, 0), BEST["H"])
        b = backtest(Wr, c.ex)
        I, Oo = window_stats(ts, b, IS), window_stats(ts, b, OOS)
        pl.append((I["sharpe"], Oo["sharpe"], I["turnover_per_day"], Oo["turnover_per_day"]))
    pl = np.array(pl)
    out["placebo_persistent_random"] = {
        "n": len(pl), "IS_sharpe_median": round(float(np.median(pl[:, 0])), 2), "IS_sharpe_p95": round(float(np.percentile(pl[:, 0], 95)), 2),
        "IS_sharpe_max": round(float(pl[:, 0].max()), 2), "OOS_sharpe_median": round(float(np.median(pl[:, 1])), 2),
        "OOS_sharpe_p95": round(float(np.percentile(pl[:, 1], 95)), 2), "OOS_sharpe_max": round(float(pl[:, 1].max()), 2),
        "turnover_per_day_IS_median": round(float(np.median(pl[:, 2])), 4), "turnover_per_day_OOS_median": round(float(np.median(pl[:, 3])), 4),
        "strategy_turnover_IS": out["base"]["IS"]["turnover_per_day"], "strategy_turnover_OOS": out["base"]["OOS"]["turnover_per_day"],
        "share_placebo_IS_sharpe_ge_strategy": round(float(np.mean(pl[:, 0] >= out["base"]["IS"]["sharpe"])), 3),
        "share_placebo_OOS_sharpe_ge_strategy": round(float(np.mean(pl[:, 1] >= out["base"]["OOS"]["sharpe"])), 3),
    }  # fmt: skip
    print("placebo", out["placebo_persistent_random"], flush=True)

    # 4. size / vol neutralisation
    S = score(c, BEST)
    with np.errstate(all="ignore"):
        lsize = np.log(c.adv)
    vol = c.vol(30)
    neu = {}
    neu["double_sort_by_ADV"] = st(ts, backtest(double_sort_W(S, lsize, u, BEST["q"], BEST["H"]), c.ex))
    neu["double_sort_by_vol30"] = st(ts, backtest(double_sort_W(S, vol, u, BEST["q"], BEST["H"]), c.ex))
    Rs = resid_score(S, [lsize, vol], u)
    neu["resid_vs_rank_lnADV_and_vol30"] = st(ts, backtest(overlap(ls_weights(Rs, u, one, BEST["q"], 20, 0, 0), BEST["H"]), c.ex))
    # same structure on plain vol (high-vol long) and plain small size for comparison
    neu["plain_high_vol30_long"] = st(ts, backtest(overlap(ls_weights(vol, u, one, BEST["q"], 20, 0, 0), BEST["H"]), c.ex))
    neu["plain_small_ADV_long"] = st(ts, backtest(overlap(ls_weights(-lsize, u, one, BEST["q"], 20, 0, 0), BEST["H"]), c.ex))
    out["neutralisation"] = neu
    print("neutralisation done", flush=True)

    # 5. return clipping
    clip = {}
    for lim in (0.25, 0.5):
        O2, C2 = clipped_prices(c.ex["open"], c.ex["close"], lim)
        ex = dict(c.ex)
        ex["open"], ex["close"] = O2, C2
        clip[f"clip_{int(lim * 100)}pct"] = st(ts, backtest(W, ex))
    out["return_clipping"] = clip

    # 6. coin age
    ages = {}
    for a in (180, 365):
        cfg = dict(BEST, min_adv=5e6, min_age=a)
        ages[f"min_age_{a}d"] = st(ts, backtest(build_W(c, cfg), c.ex))
    out["coin_age_filter"] = ages

    # 7. multiple-testing deflation (IS: выбор из многих конфигураций; OOS: одна проверка)
    res = json.loads((XOUT / "results.json").read_text())
    i1 = int(np.searchsorted(dts, ms(IS[1])))
    j0, j1 = int(np.searchsorted(dts, ms(OOS[0]))), int(np.searchsorted(dts, ms(OOS[1])))
    i0 = int(np.argmax(np.abs(dr) > 0))
    dd = {}
    for n in (len(res["configs"]), 300, 50, 10):
        dd[f"IS_trials_{n}"] = deflated_sharpe(dr[i0:i1], n)
    dd["OOS_single_test"] = deflated_sharpe(dr[j0:j1], 1)
    out["deflated_sharpe"] = dd

    # 8. concentration of price P&L by coin-days
    Oe = c.ex["open"]
    rr = np.full_like(Oe, np.nan)
    rr[:-1] = Oe[1:] / Oe[:-1] - 1
    P = np.zeros_like(Oe)
    P[1:] = W[:-1] * np.nan_to_num(rr[1:])
    k0, k1, k2 = (int(np.searchsorted(ts, ms(x))) for x in (IS[0], IS[1], OOS[1]))
    out["pnl_concentration"] = {"IS": concentration(P[k0:k1]), "OOS": concentration(P[k1:k2])}

    # momentum IS-best: same core checks
    Wm = build_W(c, MOM)
    btm = backtest(Wm, c.ex)
    mo = {"cfg": MOM, "base": st(ts, btm)}
    regm = {}
    for w, (a, b) in (("IS", IS), ("OOS", OOS)):
        i0_, i1_ = int(np.searchsorted(ts, ms(a))), int(np.searchsorted(ts, ms(b)))
        regm[w] = ols(btm["ret"][i0_:i1_], btc[i0_:i1_])
    mo["btc_regression"] = regm
    ex2 = dict(c.ex)
    ex2["cps"] = c.ex["cps"] * 2
    mo["cost_x2"] = st(ts, backtest(Wm, ex2), False)
    mo["pessimistic_delist"] = st(ts, backtest(Wm, c.ex, pess=0.5), False)
    out["momentum_is_best"] = mo

    # correlation of illiq candidate with momentum IS-best (daily, OOS)
    _, dm = to_daily(ts, btm["ret"])
    out["corr_illiq_vs_mom_OOS"] = round(float(np.corrcoef(dr[j0:j1], dm[j0:j1])[0, 1]), 3)

    (XOUT / "robust.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    res["robustness_extra"] = out
    (XOUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, default=float))
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
