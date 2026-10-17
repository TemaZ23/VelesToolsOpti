"""
flow: сводка семейства -> output/edge_lab/flow/results.json и best_daily.npz.

    PYTHONPATH=. python3 -m scripts.edge_lab.flow_report

Собирает part_a/b/c/d.json и robust_*.json. Для каждой гипотезы — число конфигураций, медианы IS/OOS Sharpe,
доля OOS > 0 и конфигурация, лучшая по IS Sharpe (минимум сделок в IS: 30 для (c), 20 для (d)).
Лучший кандидат семейства = гипотеза, чья IS-лучшая конфигурация имеет максимальный IS Sharpe
(OOS при выборе не используется). Печатает markdown-таблицы для notes.md.
"""

from __future__ import annotations

import json
import shutil

import numpy as np

from scripts.edge_lab.common import OUT

ODIR = OUT / "flow"


def _s(d, *ks, default=float("nan")):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def hyp_rows_a(part_a: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    imb = {"tbr_dm", "tbr_z", "netflow", "div"}
    for r in part_a["xs"]:
        name = "a_xs_imbalance" if r["signal"] in imb else "a_xs_tradecount"
        cfg = {k: r[k] for k in ("freq", "uni", "signal", "q", "dir")}
        out.setdefault(name, []).append({"cfg": cfg, "IS": r["net"].get("IS", {}), "OOS": r["net"].get("OOS", {}),
                                         "gross_IS": r["gross"].get("IS", {}), "gross_OOS": r["gross"].get("OOS", {}),
                                         "turnover_per_rebal": r["turnover_per_rebal"]})  # fmt: skip
    for r in part_a["ts"]:
        name = "a_ts_imbalance" if r["signal"] in imb else "a_ts_tradecount"
        cfg = {k: r[k] for k in ("freq", "signal", "thr", "dir")}
        out.setdefault(name, []).append({"cfg": cfg, "IS": r["net"].get("IS", {}), "OOS": r["net"].get("OOS", {}),
                                         "gross_IS": r["gross"].get("IS", {}), "gross_OOS": r["gross"].get("OOS", {}),
                                         "turnover_per_rebal": r["turnover_per_rebal"], "active_share": r["active_share"]})  # fmt: skip
    return out


def summarize_hyp(rows: list[dict], min_tr: int | None) -> dict:
    isr = np.array([_s(r, "IS", "sharpe") for r in rows], float)
    oos = np.array([_s(r, "OOS", "sharpe") for r in rows], float)
    ok = np.isfinite(isr) & np.isfinite(oos)
    cand = [r for r in rows if np.isfinite(_s(r, "IS", "sharpe")) and (min_tr is None or _s(r, "IS", "trades", default=0) >= min_tr)]
    best = max(cand, key=lambda r: r["IS"]["sharpe"]) if cand else None
    return {
        "configs": len(rows),
        "is_sharpe_median": round(float(np.median(isr[ok])), 2),
        "is_sharpe_max": round(float(np.max(isr[ok])), 2),
        "oos_sharpe_median": round(float(np.median(oos[ok])), 2),
        "oos_share_positive": round(float(np.mean(oos[ok] > 0)), 3),
        "corr_is_oos": round(float(np.corrcoef(isr[ok], oos[ok])[0, 1]), 2),
        "is_best": best,
    }


def main() -> None:
    pa = json.loads((ODIR / "part_a.json").read_text())
    pb = json.loads((ODIR / "part_b.json").read_text())
    pc = json.loads((ODIR / "part_c.json").read_text())
    pd = json.loads((ODIR / "part_d.json").read_text())
    hyps: dict[str, dict] = {}
    for name, rows in hyp_rows_a(pa).items():
        hyps[name] = {"part": "a", "rows": rows, "summary": summarize_hyp(rows, None)}
    for name, rows in pc.items():
        hyps[name] = {"part": "c", "rows": rows, "summary": summarize_hyp(rows, 30)}
    for name in ("cascade_bounce", "squeeze_fade"):
        hyps[name] = {"part": "d", "rows": pd[name], "summary": summarize_hyp(pd[name], 20)}
    # распределение по ВСЕМ стратегическим конфигурациям семейства
    all_is, all_oos = [], []
    for h in hyps.values():
        for r in h["rows"]:
            a, b = _s(r, "IS", "sharpe"), _s(r, "OOS", "sharpe")
            if np.isfinite(a) and np.isfinite(b):
                all_is.append(a)
                all_oos.append(b)
    all_is, all_oos = np.array(all_is), np.array(all_oos)
    fam = {"configs": int(len(all_oos)), "oos_sharpe_median": round(float(np.median(all_oos)), 2),
           "oos_share_positive": round(float(np.mean(all_oos > 0)), 3),
           "corr_is_oos": round(float(np.corrcoef(all_is, all_oos)[0, 1]), 2)}  # fmt: skip
    # выбор лучшего по IS
    best_name = max(hyps, key=lambda k: _s(hyps[k]["summary"]["is_best"], "IS", "sharpe", default=-99))
    robust = {}
    for name in hyps:
        pth = ODIR / f"robust_{name}.json"
        if pth.exists():
            robust[name] = json.loads(pth.read_text())
    tsc = json.loads((ODIR / "ts_check.json").read_text()) if (ODIR / "ts_check.json").exists() else None
    src = ODIR / f"robust_{best_name}_daily.npz"
    if src.exists():
        shutil.copy(src, ODIR / "best_daily.npz")
    res = {
        "family": "flow",
        "selection_rule": "IS-лучшая конфигурация каждой гипотезы по IS Sharpe (мин. 30/20 сделок в IS для c/d); "
                          "лучший кандидат семейства — максимум IS Sharpe среди них; OOS при выборе не используется",
        "family_oos_distribution": fam,
        "best_hypothesis": best_name,
        "hypotheses": {k: {"part": v["part"], **v["summary"]} for k, v in hyps.items()},
        "robust": robust,
        "secondary_candidate_tradecount_ts": {
            "note": "IS-лучшая конфигурация гипотезы a_ts_tradecount (не лучшая по IS в семействе); проверки устойчивости — "
                    "scripts/edge_lab/flow_ts_check.py; суточные доходности — tradecount_ts_daily.npz",
            "check": tsc,
        },
        "event_study_b": {"tables": pb["tables"], "by_year": pb["by_year"], "by_pos": pb["by_pos"]},
        "event_study_d": pd["event_study"],
        "ic_a": pa["ic"],
        "buckets_a": pa["buckets"],
        "buy_hold": pd["buy_hold"],
        "all_configs": {k: [{"cfg": r["cfg"], "IS": {kk: r["IS"].get(kk) for kk in ("sharpe", "cagr_pct", "max_dd_pct", "trades")},
                             "OOS": {kk: r["OOS"].get(kk) for kk in ("sharpe", "cagr_pct", "max_dd_pct", "trades")}}
                            for r in v["rows"]] for k, v in hyps.items()},  # fmt: skip
    }
    (ODIR / "results.json").write_text(json.dumps(res, default=float))
    print("family", fam, "best", best_name)
    print("\n| Гипотеза | Конфигов | IS SR мед / макс | OOS SR мед | доля OOS>0 | IS-лучший: IS SR → OOS SR | OOS CAGR % | OOS сделок |")
    print("|---|---|---|---|---|---|---|---|")
    for k, v in hyps.items():
        s = v["summary"]
        b = s["is_best"] or {}
        print(f"| {k} | {s['configs']} | {s['is_sharpe_median']} / {s['is_sharpe_max']} | {s['oos_sharpe_median']} | "
              f"{s['oos_share_positive'] * 100:.0f}% | {_s(b, 'IS', 'sharpe')} → {_s(b, 'OOS', 'sharpe')} | {_s(b, 'OOS', 'cagr_pct')} | "
              f"{_s(b, 'OOS', 'trades', default='-')} |")  # fmt: skip
    for k, v in hyps.items():
        print(k, "IS-best cfg:", (v["summary"]["is_best"] or {}).get("cfg"))


if __name__ == "__main__":
    main()
