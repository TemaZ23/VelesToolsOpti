"""
xsec: сводка по всем гипотезам, выбор лучшего кандидата ТОЛЬКО по IS, проверки устойчивости и итоговые файлы.

    PYTHONPATH=. python3 -m scripts.edge_lab.xsec report

Пишет output/edge_lab/xsec/results.json и best_daily.npz (дневные чистые доходности кандидата 2020–2026,
доля капитала при валовой экспозиции 1.0: 0.5 лонг + 0.5 шорт, плечо 1×).
"""

from __future__ import annotations

import collections
import copy
import json

import numpy as np

from scripts.edge_lab.common import IS, OOS, metrics, ms, per_year
from scripts.edge_lab.xsec import KEEP, UNIS, XOUT, build_W, run_cfg
from scripts.edge_lab.xsec_ctx import Ctx
from scripts.edge_lab.xsec_data import load_twap
from scripts.edge_lab.xsec_diag import contributions
from scripts.edge_lab.xsec_engine import backtest, capacity, half_years, ls_weights, overlap, summarize, to_daily, window_stats

PARTS = ("part_a.json", "part_c.json", "part_b_1h.json", "part_b_4h.json", "part_b_1d.json", "part_d.json")
_CTX: dict = {}


def ctx(off: int = 0) -> Ctx:
    if off not in _CTX:
        _CTX[off] = Ctx(off)
    return _CTX[off]


def load_parts() -> list:
    recs = []
    for f in PARTS:
        p = XOUT / f
        if not p.exists():
            print("missing", f)
            continue
        d = json.loads(p.read_text())
        recs += d["configs"] if isinstance(d, dict) else d
    return recs


def hyp_summary(recs: list) -> dict:
    by = collections.defaultdict(list)
    for r in recs:
        by[r["cfg"]["hyp"]].append(r)
    out = {}
    for h, rs in sorted(by.items()):
        isS = np.array([r["IS"]["sharpe"] for r in rs])
        oS = np.array([r["OOS"]["sharpe"] for r in rs])
        best = max(rs, key=lambda r: r["IS"]["sharpe"])
        out[h] = {
            "configs": len(rs),
            "is_sharpe_median": round(float(np.median(isS)), 2),
            "is_sharpe_max": round(float(isS.max()), 2),
            "oos_sharpe_median": round(float(np.median(oS)), 2),
            "oos_share_positive": round(float(np.mean(oS > 0)), 2),
            "corr_is_oos": round(float(np.corrcoef(isS, oS)[0, 1]), 2) if len(rs) > 2 else None,
            "gross_is_median": round(float(np.median([r["IS"]["gross_sharpe"] for r in rs])), 2),
            "gross_oos_median": round(float(np.median([r["OOS"]["gross_sharpe"] for r in rs])), 2),
            "ann_cost_oos_median_pct": round(float(np.median([r["OOS"]["ann_cost_pct"] for r in rs])), 2),
            "ann_gross_oos_median_pct": round(float(np.median([r["OOS"]["ann_gross_pct"] for r in rs])), 2),
            "is_best": best,
        }
    return out


# ----------------------------------------------------------------------------- robustness helpers
def perturb(cfg: dict) -> list:
    """±25% по каждому числовому параметру (целые — с округлением, минимум на 1)."""
    out = []
    base = copy.deepcopy(cfg)
    if "min_adv" not in base and base.get("uni") in UNIS:
        base.update(UNIS[base["uni"]])
    if base.get("fac") == "illiq":
        base.setdefault("n", 30)
    for k in ("L", "H", "q", "n", "min_adv", "min_age", "rebal"):
        if k not in base or base[k] in (None, 0):
            continue
        v = base[k]
        for f in (0.75, 1.25):
            c = copy.deepcopy(base)
            if isinstance(v, int) and not isinstance(v, bool):
                nv = int(round(v * f))
                if nv == v:
                    nv = v + (1 if f > 1 else -1)
                if nv < 1:
                    continue
            else:
                nv = round(v * f, 4)
            c[k] = nv
            c["perturb"] = f"{k}x{f}"
            out.append(c)
    return out


def stats2(ts, bt) -> dict:
    s = summarize(ts, bt)
    return {"IS": {k: s["IS"].get(k) for k in KEEP}, "OOS": {k: s["OOS"].get(k) for k in KEEP},
            "oos_half_years": s["oos_half_years"]}  # fmt: skip


def block_bootstrap_sharpe(x: np.ndarray, block: int = 20, reps: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(x)
    nb = int(np.ceil(n / block))
    out = np.empty(reps)
    for r in range(reps):
        st = rng.integers(0, n - block, nb)
        y = np.concatenate([x[s : s + block] for s in st])[:n]
        out[r] = y.mean() / y.std() * np.sqrt(365) if y.std() > 0 else 0
    return {"p05": round(float(np.percentile(out, 5)), 2), "p50": round(float(np.percentile(out, 50)), 2),
            "p95": round(float(np.percentile(out, 95)), 2), "share_le_0": round(float(np.mean(out <= 0)), 3)}  # fmt: skip


def monthly(ts, ret) -> dict:
    mon = ts.astype("datetime64[ms]").astype("datetime64[M]")
    out = {}
    for m in np.unique(mon):
        out[str(m)] = round((float(np.prod(1 + ret[mon == m])) - 1) * 100, 2)
    return out


def analyze_daily(cfg: dict) -> dict:
    c0 = ctx(0)
    rec, W, bt = run_cfg(c0, cfg, full=True)
    res = {"cfg": cfg, "base": rec}
    ts = c0.ts
    dts, dr = to_daily(ts, bt["ret"])
    res["daily"] = (dts, dr)
    full = metrics(dts, dr, int(bt["entries"].sum()))
    res["full_period"] = full
    res["monthly"] = monthly(dts, dr)
    # P&L split
    for w, (a, b) in (("IS", IS), ("OOS", OOS)):
        i0, i1 = int(np.searchsorted(ts, ms(a))), int(np.searchsorted(ts, ms(b)))
        pr, fu, co = (float(bt[k][i0:i1].sum()) for k in ("price", "fund", "cost"))
        res[f"pnl_split_{w}"] = {"price_pct": round(pr * 100, 1), "funding_pct": round(fu * 100, 1), "costs_pct": round(-co * 100, 1),
                                 "funding_share_of_gross": round(fu / (pr + fu), 3) if (pr + fu) else None}  # fmt: skip
    res["capacity_OOS"] = capacity(W, c0.adv, ts, OOS)
    res["capacity_IS"] = capacity(W, c0.adv, ts, IS)
    # correlation with BTC
    btc = np.nan_to_num(c0.ex["open"][1:, c0.j_btc] / c0.ex["open"][:-1, c0.j_btc] - 1)
    btc = np.r_[btc, 0.0]
    i0, i1 = int(np.searchsorted(ts, ms(OOS[0]))), int(np.searchsorted(ts, ms(OOS[1])))
    res["corr_btc_OOS"] = round(float(np.corrcoef(bt["ret"][i0:i1], btc[i0:i1])[0, 1]), 3)
    res["bootstrap_sharpe_OOS"] = block_bootstrap_sharpe(dr[np.searchsorted(dts, ms(OOS[0])) : np.searchsorted(dts, ms(OOS[1]))])
    res["bootstrap_sharpe_IS"] = block_bootstrap_sharpe(dr[: np.searchsorted(dts, ms(IS[1]))])
    # variants
    v = {}
    v["pessimistic_delist_-50%"] = stats2(ts, backtest(W, c0.ex, pess=0.5))
    for f in (1.5, 2.0, 3.0):
        ex = dict(c0.ex)
        ex["cps"] = c0.ex["cps"] * f
        v[f"cost_x{f}"] = stats2(ts, backtest(W, ex))
    c1 = ctx(1)
    v["exec_delay_1h"] = stats2(c1.ts, backtest(W[: c1.T], c1.ex))
    tw = load_twap()
    n = min(len(tw["ts"]), c0.T)
    ex = {k: (vv[:n] if isinstance(vv, np.ndarray) and len(vv) == c0.T else vv) for k, vv in c0.ex.items()}
    ex["open"] = tw["twap"][:n]
    ex["close"] = np.where(np.isfinite(tw["twap"][:n]), c0.ex["close"][:n], np.nan)
    from scripts.edge_lab.xsec_engine import fill_gaps

    ex["open"], ex["close"] = fill_gaps(ex["open"], ex["close"], 7)
    v["exec_twap_0-4h"] = stats2(ts[:n], backtest(W[:n], ex))
    for off in (8, 16):
        cc = ctx(off)
        v[f"day_boundary_{off}h"] = stats2(cc.ts, backtest(build_W(cc, cfg), cc.ex))
    # practical: rebalance only every R days (weights drift in between), minimum trade band
    for R in (7,):
        cR = dict(cfg, H=1, rebal=R)
        WR = build_W(c0, cR)
        tm = (np.arange(c0.T) - 1) % R == 0  # bar s trades target W[s-1] formed on a rebalance day
        v[f"rebal_every_{R}d_drift"] = stats2(ts, backtest(WR, c0.ex, tmask=tm))
        v[f"rebal_every_{R}d_drift_band0.5%"] = stats2(ts, backtest(WR, c0.ex, tmask=tm, band=0.005))
    v["band_0.5%"] = stats2(ts, backtest(W, c0.ex, band=0.005))
    WL, WS = np.where(W > 0, W, 0), np.where(W < 0, W, 0)
    v["long_leg_only"] = stats2(ts, backtest(WL, c0.ex))
    v["short_leg_only"] = stats2(ts, backtest(WS, c0.ex))
    # remove the 10 best OOS contributors (ex-post stress)
    con = contributions(c0, W, OOS)
    worst_case = W.copy()
    top = [s for s, _ in con["top"]][:8]
    cols = np.isin(c0.symbols, top)
    worst_case[:, cols] = 0
    v["without_top8_OOS_contributors"] = stats2(ts, backtest(worst_case, c0.ex))
    res["variants"] = v
    res["contrib_IS"] = contributions(c0, W, IS)
    res["contrib_OOS"] = con
    # ±25% perturbations
    pr = []
    for pc in perturb(cfg):
        r = run_cfg(c0, pc)
        pr.append({"perturb": pc["perturb"], "IS_sharpe": r["IS"]["sharpe"], "OOS_sharpe": r["OOS"]["sharpe"],
                   "OOS_cagr": r["OOS"]["cagr_pct"], "OOS_dd": r["OOS"]["max_dd_pct"],
                   "oos_half_years_pos": sum(x > 0 for x in r["oos_half_years"].values())})  # fmt: skip
    res["perturbations"] = pr
    # random placebo with identical structure
    u_cfg = dict(cfg)
    rs = []
    from scripts.edge_lab.xsec import uni_of

    u = uni_of(c0, u_cfg)
    for seed in range(20):
        rng = np.random.default_rng(seed)
        S = rng.standard_normal((c0.T, c0.N))
        Wr = overlap(ls_weights(S, u, np.ones_like(S), cfg["q"], 20, cfg.get("mode", 0), 0), cfg["H"])
        b = backtest(Wr, c0.ex)
        I, O = window_stats(ts, b, IS), window_stats(ts, b, OOS)
        rs.append((I["sharpe"], O["sharpe"], I["gross_sharpe"], O["gross_sharpe"]))
    rs = np.array(rs)
    res["random_placebo"] = {"n": 20, "IS_net_sharpe_median": round(float(np.median(rs[:, 0])), 2),
                             "OOS_net_sharpe_median": round(float(np.median(rs[:, 1])), 2),
                             "OOS_net_sharpe_max": round(float(rs[:, 1].max()), 2),
                             "IS_gross_sharpe_median": round(float(np.median(rs[:, 2])), 2),
                             "OOS_gross_sharpe_median": round(float(np.median(rs[:, 3])), 2)}  # fmt: skip
    return res


def benchmarks() -> dict:
    c0 = ctx(0)
    out = {}
    W = np.zeros((c0.T, c0.N))
    W[:, c0.j_btc] = 1.0
    out["BTC_long_perp"] = stats2(c0.ts, backtest(W, c0.ex))
    u = c0.universe(min_adv=5e6, min_age=60)
    cnt = u.sum(axis=1, keepdims=True)
    We = np.where(u & (cnt >= 20), 1.0 / np.maximum(cnt, 1), 0.0)
    out["EW_alts_long_only_adv5"] = stats2(c0.ts, backtest(We, c0.ex))
    return out


def main() -> None:
    recs = load_parts()
    hs = hyp_summary(recs)
    allS = np.array([r["OOS"]["sharpe"] for r in recs])
    daily_recs = [r for r in recs if r["cfg"]["hyp"][:2] in ("a_", "c_")]
    best = max(daily_recs, key=lambda r: r["IS"]["sharpe"])
    best_any = max(recs, key=lambda r: r["IS"]["sharpe"])
    print("IS-best (daily families):", best["cfg"], best["IS"]["sharpe"], best["OOS"]["sharpe"])
    print("IS-best (any):", best_any["cfg"], best_any["IS"]["sharpe"], best_any["OOS"]["sharpe"])
    an = analyze_daily(best["cfg"])
    dts, dr = an.pop("daily")
    np.savez(XOUT / "best_daily.npz", ts=dts.astype(np.int64), ret=dr,
             note=np.array("net daily return, gross exposure 1.0 (0.5 long + 0.5 short perps), leverage 1x"))  # fmt: skip
    # also the momentum IS-best for reference
    mom_best = hs.get("a_mom", {}).get("is_best")
    res = {
        "family": "xsec",
        "selection_rule": "max IS (2020-2023) net Sharpe among all daily-family configs; OOS only reported",
        "all_configs": {"count": len(recs), "oos_sharpe_median": round(float(np.median(allS)), 2),
                        "oos_share_positive": round(float(np.mean(allS > 0)), 3)},  # fmt: skip
        "hypotheses": {h: {k: v for k, v in s.items()} for h, s in hs.items()},
        "best_candidate": an,
        "best_any_family_by_IS": best_any,
        "benchmarks": benchmarks(),
        "momentum_is_best": mom_best,
        "configs": recs,
    }
    (XOUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, default=float))
    print(json.dumps({k: v for k, v in an.items() if k not in ("monthly",)}, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
