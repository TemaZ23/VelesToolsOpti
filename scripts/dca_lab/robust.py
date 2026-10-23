"""
Робастная доводка LONG_OPT («супер-движок»): ищем не максимум на истории, а настройки,
которые держатся при смене периода и набора монет.

Отличия от optimize.evolve:
* каждый кандидат считается в том виде, в каком его выставят в Veles (округление до шагов
  интерфейса — veles_round), поэтому проверяется ровно то, что попадёт в бота;
* целевая функция — устойчивая: эффективность на всём окне, среднее двух худших из пяти
  временных фолдов и худшая из двух непересекающихся половин монет; ликвидация в любом
  прогоне — отказ; сделок заметно меньше, чем у базы, — штраф;
* старт — от LONG_OPT (Veles-округлённого), мутации мелкие: это доводка, а не новый поиск;
* финальный выбор — по плато: кандидат оценивается вместе с соседями (±7% диапазона по
  каждому параметру и случайные сдвиги всех сразу). Острый пик проигрывает ровному плато;
* база участвует в отборе на равных: если ничего не обошло её по плато, остаётся она;
* окно отбора — IS 2021-03..2025-04 на монетах WORK. OOS 2025-04..2026-10 и 83 монеты
  вне WORK (holdout) в отборе не участвуют — только в финальной проверке.

    PYTHONPATH=. python3 -m scripts.dca_lab.robust search  [--gens 45] [--budget-min 80]
    PYTHONPATH=. python3 -m scripts.dca_lab.robust plateau
    PYTHONPATH=. python3 -m scripts.dca_lab.robust validate --coins work
    PYTHONPATH=. python3 -m scripts.dca_lab.robust validate --coins holdout
    PYTHONPATH=. python3 -m scripts.dca_lab.robust settings   # Markdown-настройки победителя

Тяжёлые шаги запускайте строго по одному (WORK ≈ 7 ГБ, holdout ≈ 8 ГБ памяти).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .data import CACHE_DIR, WORK_COINS, coin_to_symbol
from .optimize import LONG_SPACE, P, build_long_bot, decode, encode, fold_efficiency, mutate, crossover, series_for
from .portfolio import Market, Result, load_market, minute_of, run_batch
from .run import IS_FROM, IS_TO, OOS_FROM, OOS_TO
from .strategy import BotConfig, Condition, Order, PortfolioConfig, Sleeve, long_strategy_22x

OUT = Path("output/dca_lab")
STATE = OUT / "robust_state.json"
FULL_FROM, FULL_TO = IS_FROM, OOS_TO
N_FOLDS = 5
SPLIT_SEED = 2026

# Пространство доводки: LONG_SPACE с расширенными границами там, где LONG_OPT упёрся в край
_WIDER = {"g_first": (1.0, 14.0), "g_pow": (0.6, 3.0), "o_mfi5": (4.0, 40.0), "g_depth": (25.0, 70.0)}
ROBUST_SPACE: list[P] = [replace(p, lo=_WIDER[p.name][0], hi=_WIDER[p.name][1]) if p.name in _WIDER else p for p in LONG_SPACE]
NAMES = [p.name for p in ROBUST_SPACE]


# ═══════════════════════════════════════════════════════════════════════
# ОКРУГЛЕНИЕ ДО ШАГОВ ИНТЕРФЕЙСА VELES
# ═══════════════════════════════════════════════════════════════════════


def _step(v: float, step: float) -> float:
    """Округление к ближайшему шагу, половина — вверх (без банковского округления round)."""
    return round(float(np.floor(v / step + 0.5 + 1e-9)) * step, 4)


def _round_cond(c: Condition) -> Condition:
    kind = c.series.split("@")[0]
    if kind == "VOL" or c.value in (0, 0.5):
        return c
    if kind == "ATRP":
        return Condition(c.series, c.op, _step(c.value, 0.01))
    return Condition(c.series, c.op, _step(c.value, 1.0))


def veles_round(bot: BotConfig) -> BotConfig:
    """Отступы — шаг 0.5% (до 1% — 0.1%), объёмы — 0.1% (сумма 100), пороги — целые."""
    orders = []
    for i, o in enumerate(bot.orders):
        ind = _step(o.indent, 0.1) if o.indent < 1 else _step(o.indent, 0.5)
        if orders and ind <= orders[-1].indent:
            ind = orders[-1].indent + 0.5
        orders.append(Order(ind, _step(o.volume, 0.1), [_round_cond(c) for c in o.conditions]))
    rest = round(100 - sum(o.volume for o in orders[:-1]), 2)
    orders[-1] = Order(orders[-1].indent, rest, orders[-1].conditions)
    return bot.clone(
        entry=[_round_cond(c) for c in bot.entry],
        orders=orders,
        pull_up=max(_step(bot.pull_up, 0.1), 0.1),
        tp_min_pnl=max(_step(bot.tp_min_pnl, 0.1), 0.1),
        tp_pct=_step(bot.tp_pct, 0.1),
        sl_indent=_step(bot.sl_indent, 0.5),
        sl_pct=_step(bot.sl_pct, 1.0),
        tp_conditions=[_round_cond(c) for c in bot.tp_conditions],
        sl_conditions=[_round_cond(c) for c in bot.sl_conditions],
    )


def robust_bot(p: dict, name: str = "LONG_OPT_V2") -> BotConfig:
    return veles_round(build_long_bot(p, name))


def sleeve_of(p: dict, coins: list[str] | None = None, share: float = 0.75) -> Sleeve:
    return Sleeve(robust_bot(p), share, 20.0, int(round(p["block"])), int(p["dyn_order"]), coins)


def load_base() -> dict:
    """Параметры LONG_OPT (лучший по IS из прошлой оптимизации) — точка старта и эталон."""
    with open(OUT / "finalize.json", encoding="utf-8") as f:
        return json.load(f)["long_params"]


def coin_halves(market: Market) -> tuple[list[str], list[str]]:
    coins = sorted(market.coins)
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(coins))
    half = len(coins) // 2
    return [coins[i] for i in order[:half]], [coins[i] for i in order[half:]]


# ═══════════════════════════════════════════════════════════════════════
# УСТОЙЧИВАЯ ЦЕЛЕВАЯ ФУНКЦИЯ
# ═══════════════════════════════════════════════════════════════════════


def robust_parts(full: Result, a: Result, b: Result) -> dict:
    folds = fold_efficiency(full, N_FOLDS)
    worst2 = float(np.mean(sorted(folds)[:2])) if folds else 0.0
    return {
        "eff": full.efficiency,
        "worst2_folds": worst2,
        "halves_min": min(a.efficiency, b.efficiency),
        "folds": [round(x, 4) for x in folds],
        "deals": int(full.stats[6]),
        "liquidated": bool(full.liquidated or a.liquidated or b.liquidated),
    }


def robust_value(parts: dict, min_deals: int) -> float:
    if parts["liquidated"]:
        return -10.0
    s = 0.3 * parts["eff"] + 0.3 * parts["worst2_folds"] + 0.4 * parts["halves_min"]
    if parts["deals"] < min_deals:
        s *= parts["deals"] / min_deals
    return float(s)


class Evaluator:
    """Кеширует оценки по округлённой конфигурации: одинаковые боты не пересчитываются."""

    def __init__(self, market: Market, t_from: str, t_to: str, min_deals: int):
        self.market = market
        self.a, self.b = coin_halves(market)
        self.t_from, self.t_to = minute_of(t_from), minute_of(t_to)
        self.min_deals = min_deals
        self.cache: dict[str, dict] = {}
        self.runs = 0

    @staticmethod
    def key(p: dict) -> str:
        sl = sleeve_of(p)
        return json.dumps(sl.to_dict(), sort_keys=True)

    def evaluate(self, params: list[dict], chunk: int = 4) -> list[dict]:
        keys = [self.key(p) for p in params]
        todo = []
        for k, p in zip(keys, params):
            if k not in self.cache and k not in [t[0] for t in todo]:
                todo.append((k, p))
        for s in range(0, len(todo), chunk):
            part = todo[s : s + chunk]
            cfgs = []
            for _, p in part:
                cfgs += [([sleeve_of(p)], PortfolioConfig()), ([sleeve_of(p, self.a)], PortfolioConfig()),
                         ([sleeve_of(p, self.b)], PortfolioConfig())]  # fmt: skip
            res = run_batch(self.market, cfgs, self.t_from, self.t_to)
            self.runs += len(cfgs)
            for i, (k, _) in enumerate(part):
                parts = robust_parts(res[3 * i], res[3 * i + 1], res[3 * i + 2])
                parts["score"] = robust_value(parts, self.min_deals)
                parts["summary"] = res[3 * i].summary()
                self.cache[k] = parts
        return [self.cache[k] for k in keys]


# ═══════════════════════════════════════════════════════════════════════
# ШАГ 1. ЭВОЛЮЦИЯ ОТ LONG_OPT
# ═══════════════════════════════════════════════════════════════════════


def _save_state(state: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1, default=float)
    tmp.replace(STATE)


def _load_state() -> dict:
    if STATE.exists():
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def cmd_search(args: argparse.Namespace) -> None:
    market = load_market(WORK_COINS, series=series_for("long"), verbose=False)
    base = load_base()
    ev0 = Evaluator(market, IS_FROM, IS_TO, 1)
    base_eval = ev0.evaluate([base])[0]
    min_deals = int(base_eval["deals"] * 0.6)
    ev = Evaluator(market, IS_FROM, IS_TO, min_deals)
    ev.cache = {k: {**v, "score": robust_value(v, min_deals)} for k, v in ev0.cache.items()}
    print(f"база LONG_OPT (Veles-округл.): score {robust_value(base_eval, min_deals):.4f} "
          f"eff {base_eval['eff']:.4f} folds {base_eval['folds']} halves {base_eval['halves_min']:.4f} "
          f"сделок {base_eval['deals']} → минимум сделок {min_deals}", flush=True)  # fmt: skip

    rng = np.random.default_rng(args.seed)
    x0 = encode(base, ROBUST_SPACE)
    init = [x0]
    while len(init) < args.pop:
        wide = len(init) < args.pop // 4
        init.append(mutate(x0, rng, 0.12 if wide else 0.05, 0.5 if wide else 0.3, set(), ROBUST_SPACE))
    pop = [(x, decode(x, ROBUST_SPACE)) for x in init]
    scores = ev.evaluate([p for _, p in pop])
    popn = sorted(zip(pop, scores), key=lambda t: -t[1]["score"])
    t0 = time.time()
    history = []
    for gen in range(args.gens):
        sigma = 0.08 * (1 - gen / args.gens) + 0.02
        kids = []
        for _ in range(args.children):
            a = popn[min(int(rng.exponential(args.pop / 4)), len(popn) - 1)][0][0]
            if rng.random() < 0.35:
                b = popn[rng.integers(0, len(popn))][0][0]
                x = mutate(crossover(a, b, rng), rng, sigma, 0.2, set(), ROBUST_SPACE)
            else:
                x = mutate(a, rng, sigma, 0.25, set(), ROBUST_SPACE)
            kids.append((x, decode(x, ROBUST_SPACE)))
        ks = ev.evaluate([p for _, p in kids])
        merged = popn + list(zip(kids, ks))
        uniq, seen = [], set()
        for item in sorted(merged, key=lambda t: -t[1]["score"]):
            k = Evaluator.key(item[0][1])
            if k not in seen:
                seen.add(k)
                uniq.append(item)
        popn = uniq[: args.pop]
        best = popn[0][1]
        history.append({"gen": gen, "score": best["score"], "eff": best["eff"], "runs": ev.runs})
        print(f"gen {gen:3d} | {time.time() - t0:5.0f}s | score {best['score']:.4f} | eff {best['eff']:.4f} | "
              f"folds {best['folds']} | halves {best['halves_min']:.4f} | {best['summary']['avg_month_pct']:.2f}%/мес "
              f"DD {best['summary']['dd_abs_pct']:.1f}% | сделок {best['deals']}", flush=True)  # fmt: skip
        _save_state({
            "base": base, "base_eval": base_eval, "min_deals": min_deals, "history": history,
            "top": [{"params": c[0][1], **{k: v for k, v in c[1].items()}} for c in popn],
            "evaluated": len(ev.cache), "runs": ev.runs,
        })  # fmt: skip
        if args.budget_min and time.time() - t0 > args.budget_min * 60:
            print("лимит времени", flush=True)
            break


# ═══════════════════════════════════════════════════════════════════════
# ШАГ 2. ВЫБОР ПО ПЛАТО
# ═══════════════════════════════════════════════════════════════════════


def neighbours(p: dict, rng: np.random.Generator, rel: float = 0.07, n_random: int = 10) -> list[dict]:
    out = []
    for s in ROBUST_SPACE:
        if s.kind == "cat":
            continue
        for sgn in (-1, 1):
            q = dict(p)
            v = q[s.name] + sgn * rel * (s.hi - s.lo)
            q[s.name] = float(np.clip(v, s.lo, s.hi)) if s.kind == "float" else int(np.clip(round(v), s.lo, s.hi))
            if q[s.name] != p[s.name]:
                out.append(q)
    x = encode(p, ROBUST_SPACE)
    cats = {i for i, s in enumerate(ROBUST_SPACE) if s.kind == "cat"}
    for _ in range(n_random):
        y = x + rng.normal(0, rel / 2, len(x))
        y[list(cats)] = x[list(cats)]
        out.append(decode(np.clip(y, 0, 1), ROBUST_SPACE))
    return out


def _distinct_top(top: list[dict], n: int) -> list[dict]:
    """Лучшие кандидаты, отличающиеся друг от друга хотя бы на 3% нормированного расстояния."""
    out: list[dict] = []
    for c in top:
        x = encode(c["params"], ROBUST_SPACE)
        if all(np.abs(x - encode(o["params"], ROBUST_SPACE)).mean() > 0.03 for o in out):
            out.append(c)
        if len(out) >= n:
            break
    return out


def cmd_plateau(args: argparse.Namespace) -> None:
    state = _load_state()
    market = load_market(WORK_COINS, series=series_for("long"), verbose=False)
    ev = Evaluator(market, IS_FROM, IS_TO, state["min_deals"])
    cands = [{"name": "LONG_OPT", "params": state["base"]}]
    for i, c in enumerate(_distinct_top(state["top"], args.top)):
        if Evaluator.key(c["params"]) != Evaluator.key(state["base"]):
            cands.append({"name": f"cand{i + 1}", "params": c["params"]})
    rng = np.random.default_rng(11)
    rows = []
    for c in cands:
        own = ev.evaluate([c["params"]])[0]
        nb = ev.evaluate(neighbours(c["params"], rng))
        nscores = np.array([n["score"] for n in nb])
        row = {
            "name": c["name"],
            "params": c["params"],
            "own": own["score"],
            "nb_mean": float(nscores.mean()),
            "nb_q20": float(np.quantile(nscores, 0.2)),
            "nb_liq": int(sum(n["liquidated"] for n in nb)),
            "n_nb": len(nb),
            "own_parts": {k: v for k, v in own.items() if k != "summary"},
            "summary": own["summary"],
        }
        row["plateau"] = 0.4 * row["own"] + 0.3 * row["nb_mean"] + 0.3 * row["nb_q20"]
        rows.append(row)
        print(f"{c['name']:9s} own {row['own']:.4f} | соседи: среднее {row['nb_mean']:.4f} q20 {row['nb_q20']:.4f} "
              f"ликв. {row['nb_liq']}/{row['n_nb']} | плато {row['plateau']:.4f}", flush=True)  # fmt: skip
    rows.sort(key=lambda r: -r["plateau"])
    state["plateau"] = rows
    state["winner"] = rows[0]["name"]
    state["winner_params"] = rows[0]["params"]
    _save_state(state)
    print("победитель по плато:", rows[0]["name"], flush=True)


# ═══════════════════════════════════════════════════════════════════════
# ШАГ 3. ПРОВЕРКА НА ДАННЫХ, НЕ УЧАСТВОВАВШИХ В ОТБОРЕ
# ═══════════════════════════════════════════════════════════════════════


def holdout_coins() -> list[str]:
    work = {coin_to_symbol(c) for c in WORK_COINS}
    skip = {"BTCUSDT", "OGSPOTUSDT"}
    return sorted(p.stem[:-4] for p in CACHE_DIR.glob("*USDT.npz") if p.stem not in work and p.stem not in skip)


def _configs(state: dict) -> dict[str, tuple[list[Sleeve], PortfolioConfig]]:
    pf = PortfolioConfig()
    out = {"AUTHOR_22X": ([Sleeve(long_strategy_22x(), 0.75, 20, 60, 5)], pf), "LONG_OPT": ([sleeve_of(state["base"])], pf)}
    for r in state.get("plateau", []):
        if r["name"] != "LONG_OPT":
            out[r["name"]] = ([sleeve_of(r["params"])], pf)
    return out


def cmd_validate(args: argparse.Namespace) -> None:
    state = _load_state()
    coins = WORK_COINS if args.coins == "work" else holdout_coins()
    market = load_market(list(coins), series=series_for("long"), verbose=False)
    cfgs = _configs(state)
    names = list(cfgs)
    windows = {"IS": (IS_FROM, IS_TO), "OOS": (OOS_FROM, OOS_TO), "FULL": (FULL_FROM, FULL_TO)}
    if args.coins == "work":
        windows = {"OOS": (OOS_FROM, OOS_TO), "FULL": (FULL_FROM, FULL_TO)}
    out: dict = {"coins": market.coins, "windows": {}}
    stress = {"taker_fee": 0.08, "funding_8h": 0.015}
    for wname, (a, b) in windows.items():
        res = run_batch(market, [cfgs[n] for n in names], minute_of(a), minute_of(b))
        out["windows"][wname] = {n: {**r.summary(), "folds": [round(x, 4) for x in fold_efficiency(r, 3)]} for n, r in zip(names, res)}
        print(wname, {n: (r.efficiency.__round__(4), round(r.avg_month_pct, 2), round(r.dd_abs_pct, 1), int(r.liquidated)) for n, r in zip(names, res)}, flush=True)
    # стресс издержек на OOS: дороже taker и фандинг
    a, b = windows["OOS"]
    sres = run_batch(market, [(cfgs[n][0], replace(cfgs[n][1], **stress)) for n in names], minute_of(a), minute_of(b))
    out["oos_cost_stress"] = {n: r.summary() for n, r in zip(names, sres)}
    print("OOS + издержки", {n: (round(r.efficiency, 4), round(r.avg_month_pct, 2)) for n, r in zip(names, sres)}, flush=True)
    state.setdefault("validate", {})[args.coins] = out
    _save_state(state)


def cmd_settings(args: argparse.Namespace) -> None:
    from .veles_format import sleeve_ru

    state = _load_state()
    p = state["winner_params"]
    sl = sleeve_of(p)
    sl.bot.name = "LONG_OPT v2" if state["winner"] != "LONG_OPT" else "LONG_OPT"
    print(sleeve_ru(sl))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--gens", type=int, default=45)
    s.add_argument("--pop", type=int, default=20)
    s.add_argument("--children", type=int, default=20)
    s.add_argument("--seed", type=int, default=17)
    s.add_argument("--budget-min", type=float, default=90)
    p = sub.add_parser("plateau")
    p.add_argument("--top", type=int, default=6)
    v = sub.add_parser("validate")
    v.add_argument("--coins", choices=("work", "holdout"), default="work")
    sub.add_parser("settings")
    args = ap.parse_args()
    {"search": cmd_search, "plateau": cmd_plateau, "validate": cmd_validate, "settings": cmd_settings}[args.cmd](args)


if __name__ == "__main__":
    main()
