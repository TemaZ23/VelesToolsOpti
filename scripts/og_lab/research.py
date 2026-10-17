"""
Поиск и проверка стратегий для OG/USDT.

    PYTHONPATH=. python3 -m scripts.og_lab.research bench
    PYTHONPATH=. python3 -m scripts.og_lab.research evolve --problem fade --gens 40
    PYTHONPATH=. python3 -m scripts.og_lab.research final

Данные: OGSPOT — спот Binance с 2021 года (сигналы и цены, шорт на споте нельзя — это
гипотетическая история, нужная для длинной выборки), OG — фьючерс OGUSDT с мая 2025
(на нём бот и будет торговать).

Окна: подбор (IS) 2021-01 … 2024-06 на споте; проверка (OOS) 2024-07 … 2026-09 на споте;
отдельно фьючерс 2025-05 … 2026-09. Отбор финалистов — только по IS, OOS не подсматривается.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.dca_lab.optimize import evolve, score
from scripts.dca_lab.portfolio import load_market, minute_of, run_sleeves
from scripts.dca_lab.strategy import PortfolioConfig, Sleeve

from .strategies import BOT_SHARE, LEVERAGE, SERIES, build_bot, constant_short_bot, fade_problem, long_problem

OUT = Path(__file__).resolve().parents[2] / "output" / "og_lab"
IS = ("2021-01-01", "2024-07-01")
OOS = ("2024-07-01", "2026-10-01")
FUT = ("2025-05-14", "2026-10-01")
YEARS = [(f"{y}-01-01", f"{y + 1}-01-01") for y in range(2021, 2026)] + [("2026-01-01", "2026-10-01")]
PROBLEMS = {"fade": fade_problem, "long": long_problem}


def market():
    return load_market(["OGSPOT", "OG"], series=SERIES, verbose=False)


def run_bot(m, bot, coin: str, window: tuple[str, str], share: float = BOT_SHARE, leverage: float = LEVERAGE):
    sleeve = Sleeve(bot, share, leverage, 1, 0, [coin])
    return run_sleeves(m, [sleeve], PortfolioConfig(bot_share=share, leverage=leverage), minute_of(window[0]), minute_of(window[1]))


def brief(res) -> dict:
    s = res.summary()
    keys = ("months", "avg_month_pct", "dd_abs_pct", "eff", "worst_month_pct", "neg_months", "deals", "tp", "sl",
            "pf", "avg_deal_h", "fees_pct", "funding_pct", "open_upnl_pct", "liquidated")  # fmt: skip
    return {k: s[k] for k in keys}


def evaluate_everywhere(m, bot) -> dict:
    out = {
        "is_spot": brief(run_bot(m, bot, "OGSPOT", IS)),
        "oos_spot": brief(run_bot(m, bot, "OGSPOT", OOS)),
        "futures": brief(run_bot(m, bot, "OG", FUT)),
        "years_spot": {},
    }
    for a, b in YEARS:
        r = run_bot(m, bot, "OGSPOT", (a, b))
        out["years_spot"][a[:4]] = {k: brief(r)[k] for k in ("avg_month_pct", "dd_abs_pct", "deals")}
    return out


def cmd_bench(_args) -> None:
    m = market()
    res = {"const_short": evaluate_everywhere(m, constant_short_bot())}
    for name, factory in PROBLEMS.items():
        prob = factory()
        res[f"baseline_{name}"] = evaluate_everywhere(m, build_bot(prob.baseline, name == "fade", name))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bench.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    for k, v in res.items():
        print(k, json.dumps({w: (v[w] if w != "years_spot" else v[w]) for w in v}, ensure_ascii=False, default=float)[:900])


def cmd_evolve(args) -> None:
    m = market()
    prob = PROBLEMS[args.problem](coin="OGSPOT")
    log = OUT / f"evolve_{args.problem}_s{args.seed}.json"
    OUT.mkdir(parents=True, exist_ok=True)
    evolve(
        m,
        prob,
        minute_of(IS[0]),
        minute_of(IS[1]),
        generations=args.gens,
        pop=args.pop,
        children=args.pop,
        seed=args.seed,
        log_path=str(log),
        min_deals=args.min_deals,
    )
    print("saved", log)


def cmd_final(args) -> None:
    """Финалисты: лучшие по IS из каждого прогона эволюции → IS/OOS/фьючерс/годы."""
    m = market()
    rows = []
    for path in sorted(OUT.glob("evolve_*_s*.json")):
        data = json.loads(path.read_text())
        short = data["problem"] == "og_fade"
        for rank, cand in enumerate(data["top"][: args.top]):
            bot = build_bot(cand["params"], short, data["problem"])
            ev = evaluate_everywhere(m, bot)
            rows.append({"run": path.stem, "rank": rank, "is_score": round(cand["score"], 4), "params": cand["params"], **ev})
            print(
                f"{path.stem} #{rank} IS {cand['score']:.3f} | IS {ev['is_spot']['avg_month_pct']:.2f}%/мес DD {ev['is_spot']['dd_abs_pct']:.1f}% "
                f"| OOS {ev['oos_spot']['avg_month_pct']:.2f}%/мес DD {ev['oos_spot']['dd_abs_pct']:.1f}% "
                f"| FUT {ev['futures']['avg_month_pct']:.2f}%/мес DD {ev['futures']['dd_abs_pct']:.1f}% "
                f"| сделок IS/OOS {ev['is_spot']['deals']}/{ev['oos_spot']['deals']}",
                flush=True,
            )
    (OUT / "final.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=float))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bench")
    ev = sub.add_parser("evolve")
    ev.add_argument("--problem", choices=tuple(PROBLEMS), required=True)
    ev.add_argument("--gens", type=int, default=40)
    ev.add_argument("--pop", type=int, default=32)
    ev.add_argument("--seed", type=int, default=7)
    ev.add_argument("--min-deals", type=int, default=60)
    fi = sub.add_parser("final")
    fi.add_argument("--top", type=int, default=6)
    args = parser.parse_args()
    {"bench": cmd_bench, "evolve": cmd_evolve, "final": cmd_final}[args.cmd](args)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()
