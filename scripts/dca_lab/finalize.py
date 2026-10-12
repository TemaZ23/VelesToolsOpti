"""
Финализация: из логов эволюции берём лучшие по IN-SAMPLE кандидаты (OOS в отборе не участвует),
проверяем их на out-of-sample, чувствительность к параметрам и собираем configs.json для отчёта.

    PYTHONPATH=. python3 -m scripts.dca_lab.finalize --long output/dca_lab/evolve_long_is21.json \
        [--crash output/dca_lab/evolve_crash_is21.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import WORK_COINS
from .optimize import CRASH_SPACE, LONG_SPACE, crash_sleeve, long_sleeve, series_for
from .portfolio import load_market, minute_of, run_batch
from .regime import schedule_from_spec
from .run import IS_FROM, IS_TO, OOS_FROM, OOS_TO
from .strategy import PortfolioConfig, Sleeve, long_strategy_22x

OUT = Path("output/dca_lab")
DD_REGIME = {"kind": "dd", "window": 30, "dd": 20, "low": 0.3}


def top_params(path: str, n: int = 5) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        log = json.load(f)
    return [t["params"] for t in log["top"][:n]]


def sensitivity(market, build, params: dict, space, t_from: str, t_to: str, rel: float = 0.1) -> list[dict]:
    """Сдвиг каждого непрерывного параметра на ±rel диапазона → изменение эффективности."""
    cfgs = [build(params)]
    labels = ["base"]
    for p in space:
        if p.kind != "float":
            continue
        for sgn in (-1, 1):
            q = dict(params)
            q[p.name] = float(np.clip(q[p.name] + sgn * rel * (p.hi - p.lo), p.lo, p.hi))
            cfgs.append(build(q))
            labels.append(f"{p.name}{'+' if sgn > 0 else '-'}")
    res = run_batch(market, cfgs, minute_of(t_from), minute_of(t_to))
    base_eff = res[0].efficiency
    return [
        {"param": lb, "eff": round(r.efficiency, 4), "delta_pct": round((r.efficiency / base_eff - 1) * 100, 1)}
        for lb, r in zip(labels, res)
    ]


def _dump(sleeves: list[Sleeve], pf: PortfolioConfig, regime: dict | None = None) -> dict:
    return {"sleeves": [s.to_dict() for s in sleeves], "portfolio": pf.to_dict(), "regime": regime}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", required=True)
    ap.add_argument("--crash", default=None)
    args = ap.parse_args()
    market = load_market(WORK_COINS, series=series_for("long", "crash"), verbose=False)
    pf = PortfolioConfig(deposit=1000.0)
    configs: dict[str, dict] = {"AUTHOR_22X": _dump([Sleeve(long_strategy_22x(), 0.75, 20, 60, 5)], pf)}
    report: dict = {"candidates": {}}

    longs = top_params(args.long)
    cands = {f"LONG_top{i + 1}": ([long_sleeve(p)], pf) for i, p in enumerate(longs)}
    crashes: list[dict] = []
    if args.crash:
        crashes = top_params(args.crash)
        cands.update({f"CRASH_top{i + 1}": ([crash_sleeve(p)], pf) for i, p in enumerate(crashes)})
    names = list(cands)
    for wname, (a, b) in {"IS": (IS_FROM, IS_TO), "OOS": (OOS_FROM, OOS_TO)}.items():
        res = run_batch(market, [cands[n] for n in names], minute_of(a), minute_of(b))
        for n, r in zip(names, res):
            report["candidates"].setdefault(n, {})[wname] = r.summary()
    for n, d in report["candidates"].items():
        print(n, {w: (v["eff"], v["avg_month_pct"], v["dd_abs_pct"]) for w, v in d.items()}, flush=True)

    best_long = longs[0]
    report["long_params"] = best_long
    report["long_sensitivity"] = sensitivity(
        market, lambda p: ([long_sleeve(p)], pf), best_long, LONG_SPACE, IS_FROM, IS_TO
    )
    configs["LONG_OPT"] = _dump([long_sleeve(best_long)], pf)
    configs["LONG_OPT+BTC_DD_FILTER"] = _dump([long_sleeve(best_long)], pf, DD_REGIME)

    if crashes:
        best_crash = crashes[0]
        report["crash_params"] = best_crash
        report["crash_sensitivity"] = sensitivity(
            market, lambda p: ([crash_sleeve(p)], pf), best_crash, CRASH_SPACE, IS_FROM, IS_TO
        )
        configs["CRASH_OPT"] = _dump([crash_sleeve(best_crash)], pf)
        # Комбинация: доля «ловца» относительно основного бота подбирается ТОЛЬКО на IS
        weights = (0.25, 0.5, 0.75, 1.0)
        mixes = [([long_sleeve(best_long), crash_sleeve(best_crash, share=round(0.75 * w, 4))], pf) for w in weights]
        res = run_batch(market, mixes, minute_of(IS_FROM), minute_of(IS_TO))
        report["mix_is"] = [{"w": w, **r.summary()} for w, r in zip(weights, res)]
        best_i = int(np.argmax([r.efficiency for r in res]))
        report["mix_weight"] = weights[best_i]
        configs["LONG_OPT+CRASH_OPT"] = _dump(mixes[best_i][0], pf)
        print("mix IS:", [(w, round(r.efficiency, 4)) for w, r in zip(weights, res)], flush=True)

    # Итоговые конфигурации (с режим-фильтром, где он задан) на IS/OOS
    ctrl = {}
    for n, d in configs.items():
        sl = [Sleeve.from_dict(s) for s in d["sleeves"]]
        ctrl[n] = (sl, PortfolioConfig(**d["portfolio"]), schedule_from_spec(d.get("regime")))
    for wname, (a, b) in {"IS": (IS_FROM, IS_TO), "OOS": (OOS_FROM, OOS_TO)}.items():
        res = run_batch(
            market, [(c[0], c[1]) for c in ctrl.values()], minute_of(a), minute_of(b), [c[2] for c in ctrl.values()]
        )
        report.setdefault("final", {})[wname] = {n: r.summary() for n, r in zip(ctrl, res)}

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "finalize.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=float)
    with open(OUT / "configs.json", "w", encoding="utf-8") as f:
        json.dump(configs, f, ensure_ascii=False, indent=1)
    print("сохранено:", OUT / "finalize.json", OUT / "configs.json")


if __name__ == "__main__":
    main()
