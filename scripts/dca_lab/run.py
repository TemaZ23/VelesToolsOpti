"""
CLI DCA Lab.

    python -m scripts.dca_lab.run baseline            # стратегия автора на WORK
    python -m scripts.dca_lab.run optimize --gens 40  # эволюционный поиск на in-sample
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .data import STRESS_COINS, WORK_COINS
from .optimize import crash_problem, evolve, long_problem, series_for, short_problem
from .portfolio import load_market, minute_of, run
from .strategy import long_strategy_22x, long_strategy_22x_portfolio

OUT_DIR = Path(__file__).resolve().parents[2] / "output" / "dca_lab"

IS_FROM, IS_TO = "2021-03-01", "2025-04-20"
OOS_FROM, OOS_TO = "2025-04-20", "2026-10-01"


def cmd_baseline(args: argparse.Namespace) -> None:
    m = load_market(WORK_COINS, series=series_for("long"), verbose=False)
    bot, pf = long_strategy_22x(), long_strategy_22x_portfolio()
    for name, (a, b) in {"author": ("2023-04-20", "2026-04-20"), "IS": (IS_FROM, IS_TO), "OOS": (OOS_FROM, OOS_TO)}.items():
        r = run(m, bot, pf, minute_of(a), minute_of(b), record_deals=True)
        s = r.summary()
        s["concurrent_mae_pct"] = round(r.concurrent_mae_pct(), 1)
        print(name, json.dumps(s, ensure_ascii=False))


def cmd_optimize(args: argparse.Namespace) -> None:
    coins = STRESS_COINS if args.stress else WORK_COINS
    m = load_market(coins, series=series_for(args.problem), verbose=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
    log = OUT_DIR / f"evolve_{args.problem}_{tag}.json"
    problem = {"long": long_problem, "short": short_problem, "crash": crash_problem}[args.problem]()
    pop = evolve(
        m,
        problem,
        minute_of(args.t_from),
        minute_of(args.t_to),
        generations=args.gens,
        pop=args.pop,
        children=args.children,
        seed=args.seed,
        log_path=str(log),
        time_budget_s=args.budget,
        min_deals=args.min_deals,
    )
    print("лучший:", json.dumps(pop[0].summary, ensure_ascii=False))
    print("лог:", log)


def main() -> None:
    ap = argparse.ArgumentParser(description="DCA Lab — симулятор и оптимизатор ботов Veles")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("baseline").set_defaults(fn=cmd_baseline)
    op = sub.add_parser("optimize")
    op.add_argument("--problem", choices=["long", "short", "crash"], default="long")
    op.add_argument("--min-deals", type=int, default=1000)
    op.add_argument("--gens", type=int, default=40)
    op.add_argument("--pop", type=int, default=24)
    op.add_argument("--children", type=int, default=24)
    op.add_argument("--seed", type=int, default=7)
    op.add_argument("--from", dest="t_from", default=IS_FROM)
    op.add_argument("--to", dest="t_to", default=IS_TO)
    op.add_argument("--stress", action="store_true")
    op.add_argument("--budget", type=float, default=None, help="лимит времени, сек")
    op.add_argument("--tag", default=None)
    op.set_defaults(fn=cmd_optimize)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
