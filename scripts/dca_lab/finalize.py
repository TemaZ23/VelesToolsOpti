"""
Финализация: из логов эволюции берём лучшие по IN-SAMPLE кандидаты (OOS в отборе не участвует),
проверяем их на out-of-sample, чувствительность к параметрам и собираем configs.json для отчёта.

    PYTHONPATH=. python3 -m scripts.dca_lab.finalize --long output/dca_lab/evolve_long_is1.json \
        [--crash output/dca_lab/evolve_crash_is1.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import WORK_COINS
from .optimize import (
    CRASH_SPACE,
    LONG_SPACE,
    crash_sleeve,
    long_sleeve,
    series_for,
)
from .portfolio import load_market, minute_of, run_batch
from .run import IS_FROM, IS_TO, OOS_FROM, OOS_TO
from .strategy import PortfolioConfig, Sleeve, long_strategy_22x

OUT = Path("output/dca_lab")


def top_params(path: str, n: int = 5) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        log = json.load(f)
    return [t["params"] for t in log["top"][:n]]


def sensitivity(market, build, params: dict, space, t_from: str, t_to: str, rel: float = 0.1) -> list[dict]:
    """Сдвиг каждого непрерывного параметра на ±rel диапазона → изменение эффективности."""
    base_cfg = build(params)
    cfgs = [base_cfg]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", required=True)
    ap.add_argument("--crash", default=None)
    args = ap.parse_args()
    market = load_market(WORK_COINS, series=series_for("long", "crash"), verbose=False)
    pf = PortfolioConfig(deposit=1000.0)
    author = [Sleeve(long_strategy_22x(), 0.75, 20, 60, 5)]

    report: dict = {"candidates": {}}
    configs: dict[str, tuple[list[Sleeve], PortfolioConfig]] = {"AUTHOR_22X": (author, pf)}

    longs = top_params(args.long)
    cands = {f"LONG_top{i + 1}": ([long_sleeve(p)], pf) for i, p in enumerate(longs)}
    if args.crash:
        crashes = top_params(args.crash)
        cands.update({f"CRASH_top{i + 1}": ([crash_sleeve(p)], pf) for i, p in enumerate(crashes)})
    names = list(cands)
    for wname, (a, b) in {"IS": (IS_FROM, IS_TO), "OOS": (OOS_FROM, OOS_TO)}.items():
        res = run_batch(market, [cands[n] for n in names], minute_of(a), minute_of(b))
        for n, r in zip(names, res):
            report["candidates"].setdefault(n, {})[wname] = r.summary()
    print(json.dumps({n: {w: (v["eff"], v["avg_month_pct"], v["dd_abs_pct"]) for w, v in d.items()} for n, d in report["candidates"].items()}, ensure_ascii=False, indent=1))

    best_long = longs[0]
    configs["LONG_OPT"] = ([long_sleeve(best_long)], pf)
    report["long_params"] = best_long
    report["long_sensitivity"] = sensitivity(
        market, lambda p: ([long_sleeve(p)], pf), best_long, LONG_SPACE, IS_FROM, IS_TO
    )

    if args.crash:
        best_crash = top_params(args.crash, 1)[0]
        report["crash_params"] = best_crash
        configs["CRASH_OPT"] = ([crash_sleeve(best_crash)], pf)
        report["crash_sensitivity"] = sensitivity(
            market, lambda p: ([crash_sleeve(p)], pf), best_crash, CRASH_SPACE, IS_FROM, IS_TO
        )
        # Комбинации: доля «ловца» относительно основного бота подбирается ТОЛЬКО на IS
        mixes = []
        for w in (0.25, 0.5, 0.75, 1.0):
            sl = [long_sleeve(best_long), crash_sleeve(best_crash, share=round(0.75 * w, 4))]
            mixes.append((w, (sl, pf)))
        res = run_batch(market, [m[1] for m in mixes], minute_of(IS_FROM), minute_of(IS_TO))
        report["mix_is"] = [{"w": w, **r.summary()} for (w, _), r in zip(mixes, res)]
        best_w, best_cfg = max(zip(mixes, res), key=lambda x: x[1].efficiency)[0]
        configs["LONG_OPT+CRASH_OPT"] = best_cfg
        report["mix_weight"] = best_w

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "finalize.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=float)
    raw = {n: {"sleeves": [s.to_dict() for s in sl], "portfolio": p.to_dict()} for n, (sl, p) in configs.items()}
    with open(OUT / "configs.json", "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=1)
    print("сохранено:", OUT / "finalize.json", OUT / "configs.json")


if __name__ == "__main__":
    main()
