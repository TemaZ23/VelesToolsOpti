"""
Перебор «циклических» стратегий и честная проверка: параметры выбираются на 2017-08 … 2022-12 (IS),
проверяются на 2023-01 … 2026-09 (OOS), который при выборе не виден.

    PYTHONPATH=. python3 -m scripts.cycle_lab.research
"""

from __future__ import annotations

import itertools
import json

import numpy as np

from .backtest import COINS, IS, OOS, OUT, REGIMES, market, run

LEVS = [1, 2, 3, 5, 10, 20]
STOPS = [0.03, 0.05, 0.08, 0.12, 0.2, 0.3]
FULL = ("2017-08-01", "2026-10-01")
YEARS = [(f"{y}-01-01", f"{y + 1}-01-01") for y in range(2018, 2026)] + [("2026-01-01", "2026-10-01")]


def configs():
    for regime, lev, stop, trailing, reentry, short in itertools.product(
        REGIMES, LEVS, STOPS, (False, True), (0, 1), (True, False)
    ):
        if stop >= 1 / lev:  # стоп дальше ликвидации — это уже не стоп
            continue
        yield {"regime": regime, "lev": lev, "stop": stop, "trailing": trailing, "reentry": reentry, "short": short}


def key(cfg: dict) -> str:
    return (f"{cfg['regime']} x{cfg['lev']} стоп {cfg['stop'] * 100:.0f}%{' трейл' if cfg['trailing'] else ''}"
            f"{' повторный вход' if cfg['reentry'] else ''}{'' if cfg['short'] else ' без шорта'}")  # fmt: skip


def main() -> None:
    markets = {c: market(c) for c in COINS}
    # «oracle» — идеальные вершины и дно цикла BTC задним числом, на альтах те же даты.
    signals = {(c, name): fn(m) for c, m in markets.items() for name, fn in REGIMES.items()}

    rows = []
    for cfg in configs():
        row = {"cfg": cfg, "name": key(cfg), "coins": {}}
        for c, m in markets.items():
            sig = signals[c, cfg["regime"]]
            res = {}
            for wname, w in (("is", IS), ("oos", OOS), ("full", FULL)):
                r = run(m, sig, w, cfg["lev"], cfg["stop"], cfg["trailing"], cfg["reentry"], cfg["short"])
                if r is not None:
                    res[wname] = r
            row["coins"][c] = res
        for w in ("is", "oos", "full"):
            vals = [row["coins"][c][w] for c in COINS if w in row["coins"][c]]
            row[w] = {
                "median_cagr": float(np.median([v["cagr_pct"] for v in vals])),
                "median_dd": float(np.median([v["max_dd_pct"] for v in vals])),
                "ruined": int(sum(v["ruined_at"] is not None for v in vals)),
                "coins": len(vals),
                "btc": row["coins"]["BTC"].get(w),
            }
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "grid.json").write_text(json.dumps(rows, ensure_ascii=False, default=float))
    print("конфигураций:", len(rows))

    summary = {"by_leverage": {}, "years": {}, "lev_curve": {}}
    # Для каждого плеча — лучшая по IS (медиана CAGR по монетам) конфигурация без «оракула».
    for lev in LEVS:
        cands = [r for r in rows if r["cfg"]["lev"] == lev and r["cfg"]["regime"] != "oracle"]
        best = max(cands, key=lambda r: r["is"]["median_cagr"])
        best_btc = max(cands, key=lambda r: r["is"]["btc"]["cagr_pct"])
        oracle = [r for r in rows if r["cfg"]["lev"] == lev and r["cfg"]["regime"] == "oracle"]
        best_oracle = max(oracle, key=lambda r: r["full"]["btc"]["cagr_pct"])
        summary["by_leverage"][lev] = {
            "best_is_all_coins": {"name": best["name"], "is": best["is"], "oos": best["oos"],
                                  "oos_coins": {c: best["coins"][c].get("oos") for c in COINS}},  # fmt: skip
            "best_is_btc": {"name": best_btc["name"], "is": best_btc["is"]["btc"], "oos": best_btc["oos"]["btc"]},
            "oracle_btc_full": {"name": best_oracle["name"], "full": best_oracle["full"]["btc"]},
        }
        print(f"\nплечо x{lev}")
        print(f"  лучшая по IS (медиана 9 монет): {best['name']}")
        print(f"    IS  медиана CAGR {best['is']['median_cagr']:.0f}% DD {best['is']['median_dd']:.0f}% разорено {best['is']['ruined']}/{best['is']['coins']}")  # fmt: skip
        print(f"    OOS медиана CAGR {best['oos']['median_cagr']:.0f}% DD {best['oos']['median_dd']:.0f}% разорено {best['oos']['ruined']}/{best['oos']['coins']}")  # fmt: skip
        b = best_btc
        print(f"  лучшая по IS на BTC: {b['name']}")
        print(f"    IS  BTC CAGR {b['is']['btc']['cagr_pct']:.0f}% DD {b['is']['btc']['max_dd_pct']:.0f}% худший мес {b['is']['btc']['worst_month_pct']:.0f}%")  # fmt: skip
        print(f"    OOS BTC CAGR {b['oos']['btc']['cagr_pct']:.0f}% DD {b['oos']['btc']['max_dd_pct']:.0f}% худший мес {b['oos']['btc']['worst_month_pct']:.0f}% разорён {b['oos']['btc']['ruined_at']}")  # fmt: skip
        o = best_oracle["full"]["btc"]
        print(f"  оракул (идеальные циклы, вся история BTC): {best_oracle['name']} CAGR {o['cagr_pct']:.0f}% DD {o['max_dd_pct']:.0f}% разорён {o['ruined_at']}")  # fmt: skip

    # Кривая по плечу для «классики»: SMA200 и кросс 50/200 на BTC, стоп 8%/12% трейл.
    m = markets["BTC"]
    for regime in ("sma200", "cross50_200", "donchian55", "oracle"):
        for stop, trailing in ((0.05, False), (0.12, True), (0.2, True)):
            curve = {}
            for lev in LEVS:
                if stop >= 1 / lev:
                    continue
                r = run(m, signals["BTC", regime], FULL, lev, stop, trailing, 1, True)
                curve[lev] = {k: r[k] for k in ("cagr_pct", "geo_month_pct", "max_dd_pct", "worst_month_pct", "ruined_at", "trades", "win_pct")}  # fmt: skip
            summary["lev_curve"][f"{regime} стоп {stop * 100:.0f}%{' трейл' if trailing else ''}"] = curve

    # По годам — оракул и SMA200 на BTC при x1/x3/x5/x10.
    for regime in ("oracle", "sma200", "cross50_200"):
        for lev in (1, 3, 5, 10):
            yrs = {}
            for a, b in YEARS:
                r = run(m, signals["BTC", regime], (a, b), lev, 0.08 if lev < 10 else 0.05, True, 1, True)
                yrs[a[:4]] = {"year_pct": round((r["final_x"] - 1) * 100, 0), "max_dd_pct": r["max_dd_pct"],
                              "ruined_at": r["ruined_at"]} if r else None  # fmt: skip
            summary["years"][f"{regime} x{lev}"] = yrs
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    print("\nкривые по плечу (BTC, вся история 2017-08 … 2026-09):")
    for name, curve in summary["lev_curve"].items():
        cells = "  ".join(
            f"x{lev}: {v['cagr_pct']:.0f}%/г DD {v['max_dd_pct']:.0f}%{' 💀' + v['ruined_at'] if v['ruined_at'] else ''}"
            for lev, v in curve.items()
        )
        print(f"  {name}: {cells}")
    print("\nпо годам (доходность за год, %):")
    for name, yrs in summary["years"].items():
        print(f"  {name}: " + "  ".join(f"{y} {v['year_pct']:.0f}" if v else f"{y} —" for y, v in yrs.items()))


if __name__ == "__main__":
    main()
