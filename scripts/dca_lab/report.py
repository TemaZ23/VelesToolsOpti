"""
Оценка конфигураций и отчёт.

    PYTHONPATH=. python3 -m scripts.dca_lab.report --configs output/dca_lab/configs.json

configs.json — словарь {имя: {"sleeves": [...], "portfolio": {...}}} (см. strategy.save_config).
Для каждой конфигурации считаются:
  * метрики на окнах (in-sample, out-of-sample, весь период, окно автора);
  * профиль «размер бота → доходность / просадка / ликвидация» с реинвестом и без;
  * риск разорения по дате старта: запуск в начале каждого месяца до конца данных.
"""

from __future__ import annotations

import argparse
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .data import WORK_COINS
from .optimize import series_for
from .portfolio import Market, Result, date_of, load_market, minute_of, run_batch, run_sleeves
from .regime import schedule_from_spec
from .strategy import PortfolioConfig, Sleeve

WINDOWS = {
    "IS 2021-03-01…2025-04-20": ("2021-03-01", "2025-04-20"),
    "Окно автора 2023-04-20…2026-04-20": ("2023-04-20", "2026-04-20"),
    "OOS 2025-04-20…2026-10-01": ("2025-04-20", "2026-10-01"),
    "Весь период 2021-03-01…2026-10-01": ("2021-03-01", "2026-10-01"),
}


def scaled(sleeves: list[Sleeve], k: float) -> list[Sleeve]:
    out = copy.deepcopy(sleeves)
    for s in out:
        s.bot_share = round(s.bot_share * k, 4)
    return out


Config = tuple[list[Sleeve], PortfolioConfig, "np.ndarray | None"]


def load_configs(path: str) -> dict[str, Config]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {
        name: (
            [Sleeve.from_dict(s) for s in d["sleeves"]],
            PortfolioConfig(**d["portfolio"]),
            schedule_from_spec(d.get("regime")),
        )
        for name, d in raw.items()
    }


def _batch(market: Market, cfgs: list[Config], a: str, b: str) -> list[Result]:
    return run_batch(market, [(c[0], c[1]) for c in cfgs], minute_of(a), minute_of(b), [c[2] for c in cfgs])


def window_table(market: Market, configs: dict) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    names = list(configs)
    for wname, (a, b) in WINDOWS.items():
        res = _batch(market, [configs[n] for n in names], a, b)
        out[wname] = {n: r.summary() for n, r in zip(names, res)}
    return out


def size_profile(
    market: Market, cfg: Config, ks: list[float], t_from: str, t_to: str
) -> list[dict]:
    """Масштаб размера ботов → метрики (без реинвеста и с реинвестом)."""
    sleeves, pf, sched = cfg
    cfgs = []
    for compound in (False, True):
        for k in ks:
            p = copy.deepcopy(pf)
            p.compound = compound
            cfgs.append((scaled(sleeves, k), p, sched))
    res = _batch(market, cfgs, t_from, t_to)
    rows = []
    for (sl, p, _), r in zip(cfgs, res):
        s = r.summary()
        rows.append(
            {
                "compound": p.compound,
                "bot_share": [x.bot_share for x in sl],
                "avg_month_pct": s["avg_month_pct"],
                "cmgr_pct": s["cmgr_pct"],
                "max_dd_pct": s["max_dd_pct"],
                "dd_abs_pct": s["dd_abs_pct"],
                "worst_month_pct": s["worst_month_pct"],
                "liquidated": s["liquidated"],
                "liq_date": date_of(int(r.stats[5])) if r.liquidated else None,
            }
        )
    return rows


def start_date_risk(
    market: Market,
    cfg_in: Config,
    ks: list[float],
    first: str = "2021-03-01",
    last: str = "2026-04-01",
    horizon_days: int | None = None,
    compound: bool = True,
) -> list[dict]:
    """Запуск в начале каждого месяца: доля ликвидаций, медианный и худший результат."""
    sleeves, pf, sched = cfg_in
    starts = []
    y, m = map(int, first[:7].split("-"))
    ly, lm = map(int, last[:7].split("-"))
    while (y, m) <= (ly, lm):
        starts.append(f"{y:04d}-{m:02d}-01")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    rows = []
    for k in ks:
        p = copy.deepcopy(pf)
        p.compound = compound
        cfg = (scaled(sleeves, k), p)

        def one(st: str, cfg=cfg) -> dict:
            t0 = minute_of(st)
            t1 = t0 + horizon_days * 1440 if horizon_days else market.t_end
            r = run_sleeves(market, cfg[0], cfg[1], t0, min(t1, market.t_end), block_sched=sched)
            return {
                "start": st,
                "liquidated": int(r.liquidated),
                "cmgr_pct": r.cmgr_pct,
                "avg_month_pct": r.avg_month_pct,
                "max_dd_pct": r.max_dd_pct,
            }

        # движок отпускает GIL (nogil) — прогоны по датам старта идут параллельно
        with ThreadPoolExecutor(max_workers=4) as pool:
            outcomes = list(pool.map(one, starts))
        liq = [o for o in outcomes if o["liquidated"]]
        alive = [o for o in outcomes if not o["liquidated"]]
        rows.append(
            {
                "k": k,
                "bot_share": [s.bot_share for s in cfg[0]],
                "starts": len(outcomes),
                "liquidations": len(liq),
                "liq_starts": [o["start"] for o in liq],
                "median_cmgr_pct": float(np.median([o["cmgr_pct"] for o in outcomes])),
                "median_cmgr_alive_pct": float(np.median([o["cmgr_pct"] for o in alive])) if alive else None,
                "worst_dd_alive_pct": max((o["max_dd_pct"] for o in alive), default=None),
                "outcomes": outcomes,
            }
        )
    return rows


def monthly_table(market: Market, configs: dict, t_from: str, t_to: str) -> dict[str, list[tuple[str, float]]]:
    names = list(configs)
    res = _batch(market, [configs[n] for n in names], t_from, t_to)
    return {n: r.monthly_returns() for n, r in zip(names, res)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--out", default="output/dca_lab/report_data.json")
    ap.add_argument("--ks", default="0.5,1,1.5,2,3,4")
    args = ap.parse_args()
    configs = load_configs(args.configs)
    market = load_market(WORK_COINS, series=series_for("long", "crash"), verbose=False)
    ks = [float(x) for x in args.ks.split(",")]
    data: dict = {"windows": window_table(market, configs), "sizes": {}, "start_risk": {}, "start_risk_12m": {}, "monthly": {}}
    data["monthly"] = monthly_table(market, configs, "2021-03-01", "2026-10-01")
    for name, cfg in configs.items():
        data["sizes"][name] = size_profile(market, cfg, ks, "2021-03-01", "2026-10-01")
        data["start_risk"][name] = start_date_risk(market, cfg, ks)
        data["start_risk_12m"][name] = start_date_risk(market, cfg, ks, last="2025-09-01", horizon_days=365)
        print("готово:", name, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, default=float)
    print("сохранено:", args.out)


if __name__ == "__main__":
    main()
