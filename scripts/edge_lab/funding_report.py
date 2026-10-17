"""
Сборка итогов семейства funding: output/edge_lab/funding/results.json и best_daily.npz.

    PYTHONPATH=. python3 -m scripts.edge_lab.funding_report

Лучший кандидат: статичный cash-and-carry BTC+ETH (лонг спот + шорт перп, 50/50 капитала, маржа перпа 1/2
номинала, кросс-маржа USDT-M). Выбор по IS: carry с наибольшим IS Sharpe — статичный BTC+ETH; маржа m=0.5 —
наименьшая, при которой вся окрестность ±25% (0.375..0.625) прошла IS без ликвидаций (у m=1/3 → 0.25 две
ликвидации в IS). Нужны part_*.json всех этапов, включая funding_robust (carry, alt) и funding_tick.
"""

from __future__ import annotations

import json

import numpy as np

from scripts.edge_lab.common import metrics, per_year
from scripts.edge_lab.funding import OUTD, half_years, summarize_configs


def load(name: str) -> dict:
    return json.loads((OUTD / name).read_text())


def brief(r: dict) -> dict:
    return {k: r[k] for k in ("cfg", "is", "oos", "trades") if k in r}


def main() -> None:
    xs, tm, ca, mi, m1 = (load(f"part_{n}.json") for n in ("xs", "timing", "carry", "micro", "m1"))
    rc, ra, tk = (load(f"part_{n}.json") for n in ("robust_carry", "robust_alt", "tick"))
    # ---------------- лучший кандидат: статичный carry BTC+ETH, m=0.5
    st = rc["best"]
    with np.load(OUTD / "robust_carry_m05_daily.npz") as z:
        bts, bret = z["ts"], z["ret"]
        bfund, bbasis, bcost = z["fund"], z["basis"], z["cost"]
    np.savez(OUTD / "best_daily.npz", ts=bts, ret=bret, fund=bfund, basis=bbasis, cost=bcost)
    m_full = metrics(bts, bret)
    carry_rows = (ca["configs"] + [dict(v, cfg={"static": k}) for k, v in ca["static"].items()]
                  + [dict(v, cfg={"robust": v["variant"]}) for v in rc["variants"]] + rc["onoff"]["configs"])  # fmt: skip
    is_best = max(carry_rows, key=lambda r: r["is"]["sharpe"])
    cap = rc["capacity_per_leg"]
    best = {
        "name": "Статичный cash-and-carry BTC+ETH (лонг спот / шорт перп USDT-M, маржа 50% номинала)",
        "params": "BTC и ETH по 50% капитала; на монету: спот на капитал_монеты/1.5, шорт перпа того же количества "
                  "монет, маржа USDT-M = 0.5 номинала (кросс-маржа на обе монеты); раз в день (по 00:00 UTC) — если "
                  "кошелёк USDT-M по монете < 37.5% или > 100% её номинала, обе ноги приводятся к спот=E/(1.5), "
                  "маржа=0.5*номинал; без ротации и порогов",
        "is": st["is"], "oos": st["oos"], "full": st["full"], "full_check": m_full, "per_year": st["per_year"],
        "oos_half_years": st["oos_half_years"], "liquidation_days": st["liq"], "entries": st["trades_entries"],
        "band_rebalances": st["band_rebalances"],
        "pnl_split_full_ann_pct": {"funding": st["full"]["ann_fund_pct"], "basis_price": st["full"]["ann_price_pct"],
                                   "costs": -st["full"]["ann_cost_pct"]},
        "capacity_usd_per_leg_oos": cap,
        "exposure": "капитал = спот + маржа перпа = 1.5 * номинал; на 1 капитала ≈ 0.667 лонг спот + 0.667 шорт перп "
                    "(брутто 1.33), net delta ≈ 0; ret в best_daily.npz — доля всего капитала (свободного кэша нет)",
        "robustness": rc["variants"], "onoff_threshold": {k: v for k, v in rc["onoff"].items() if k != "configs"},
        "engine_check": rc["engine_check"], "rolling_180d_apr_pct": rc["rolling_180d_apr_pct"],
        "reverse_carry_upper_bound_pct_per_year": rc["reverse_carry_upper_bound_pct_per_year"],
        "mark_missing_days": rc["mark_missing_days"], "intraday_adverse_moves": rc["intraday_adverse_moves"],
        "is_selection": {"rule": "max IS Sharpe среди всех carry-конфигов -> статичный BTC+ETH; маржа — минимальная, "
                                 "у которой окрестность ±25% без ликвидаций в IS",
                         "argmax_cfg": is_best["cfg"], "argmax_is_sharpe": is_best["is"]["sharpe"],
                         "argmax_oos": {k: is_best["oos"][k] for k in ("cagr_pct", "sharpe", "max_dd_pct")},
                         "note": "IS Sharpe статичных вариантов 8.6–8.8 (разница — шум); multi-assets (залог BTC/ETH в "
                                 "USDT-M, капитал 1.05*номинал, без модели ликвидации) — оптимистичная граница: "
                                 "OOS CAGR 6.85%"},
        "oos_all_carry_configs": summarize_configs(carry_rows),
    }
    alt = tm["best_is_by_asset"]["ALT"]
    hyps = [
        {"name": "(a) cash-and-carry: ротация по трейлинг-фандингу (спот+перп, вкл. альты)",
         "configs": len(ca["configs"]), "summary": ca["summary_all"], "best_is": brief(ca["best_is"]),
         "best_is_per_year": ca["best_is"]["per_year"], "best_is_oos_half_years": ca["best_is"]["oos_half_years"],
         "summary_by_margin": ca["summary_by_m"], "basis_selected_pct": ca.get("basis_selected_pct"),
         "intraday_up_jump_selected": ca.get("intraday_up_jump_selected"),
         "note": "ликвидации по mark high; при пропуске mark — high последней цены (консервативно)"},
        {"name": "(a) cash-and-carry: статичный BTC/ETH (+ проверки устойчивости, вкл/выкл по фандингу)",
         "configs": len(ca["static"]) + len(rc["variants"]) + len(rc["onoff"]["configs"]),
         "static": {k: {"is": v["is"], "oos": v["oos"], "per_year": v["per_year"]} for k, v in ca["static"].items()},
         "robust_variants": rc["variants"], "onoff": {k: v for k, v in rc["onoff"].items() if k != "configs"}},
        {"name": "(b) кросс-секционный фактор фандинга (шорт высокого / лонг низкого)", "configs": len(xs["configs"]),
         "summary": xs["summary_all"], "summary_by_dir": xs["summary_by_dir"], "best_is": xs["best_is"]},
        {"name": "(c) экстремумы фандинга как сигнал (BTC/ETH/альты)", "configs": len(tm["configs"]),
         "summary": tm["summary_all"], **{k: v for k, v in tm.items() if k.startswith("summary_")},
         "best_is_by_asset": tm["best_is_by_asset"], "baselines": {k: v for k, v in tm.items() if k.startswith("baseline")},
         "control_price_reversal": tm.get("control_price_reversal"), "alt_robustness": ra},
        {"name": "(d) микроструктура выплат, часовые бары (НЕРЕАЛИСТИЧНО: вход/выход по первой сделке после выплаты)",
         "configs": len(mi["configs"]), "summary_taker": mi["summary_taker"], "summary_maker": mi["summary_maker"],
         "best_is_taker": brief(mi["best_is_taker"]), "bucket_table": mi["bucket_table"]},
        {"name": "(d) микроструктура выплат, минутки (вход не раньше чем через 1 мин после выплаты)",
         "configs": len(m1["configs"]), "summary": m1["summary"], "summary_by_iv": m1["summary_by_iv"],
         "best_is": brief(m1["best_is"]), "profile_bp": m1["profile_bp"], "through_realistic": m1["through_realistic"]},
        {"name": "(d) микроструктура выплат, сделки aggTrades (±120 с, выборка 235 событий)",
         "configs": len(tk["through_table"]) + len(tk["post_momentum_table"]), **tk},
    ]
    real_rows = xs["configs"] + tm["configs"] + carry_rows + m1["configs"]
    res = {
        "family": "funding",
        "data_notes": [
            "В архиве klines после делистинга идут бары с нулевым объёмом и замороженной ценой (~150 символов, "
            "флаг delisted в universe.json только у 31). Здесь бар считается торговым только при qvol>0; "
            "«настоящий делистинг» — последний торговый бар раньше чем за 7 дней до конца выборки.",
            "Дневной high последней цены перпа содержит фитили (BTC 2021-07-26 +36%, ETH 2020-03-13 +200%), "
            "которых нет в mark price; ликвидации в carry считаются по mark price (markPriceKlines 1d). В архиве mark "
            "есть пропуски (BTC 9 дней, ETH 3, в т.ч. 2021-07-26): в проверках устойчивости при пропуске берётся "
            "high спота, в исходной сетке ротации — high последней цены (консервативно).",
            "Фандинг для P&L: выплаты в (open_k, open_k+1] — позиция, открытая по open, не получает выплату в "
            "этот же момент; сигнал — panel funding[<=t].",
            "Спот Binance: дневные свечи 481 перпа, у которых есть спот (с учётом префиксов 1000*), фильтр "
            "|базис| < 5% против неверного маппинга.",
            "Издержки входа в первый день выборки завышены (ADV ещё не накоплен → худший тариф проскальзывания): "
            "−0.7% в 2020 у carry; консервативно, оставлено как есть.",
        ],
        "best": best,
        "second_candidate_alt_contrarian": {
            "verdict": "weak: OOS Sharpe 0.91 (<1), max DD 57%, 81% OOS-прибыли — 5 монет (ALPACA 2025 — 43%), "
                       "без топ-10 монет OOS < 0; при издержках x1.5 / без выплаты 00:00 / пессимистичном делистинге "
                       "Sharpe 0.38–0.80",
            "cfg": alt["cfg"], "is": alt["is"], "oos": alt["oos"], "trades": alt["trades"],
            "per_year": alt.get("per_year"), "oos_half_years": alt.get("oos_half_years"),
            "pessimistic_delist_oos": alt.get("pessimistic_delist_oos"),
            "random_entry_baseline": alt.get("random_entry_baseline"),
            "control_price_reversal": tm.get("control_price_reversal"), "robustness": ra},
        "hypotheses": hyps,
        "oos_all_configs_family_realistic": summarize_configs(real_rows),
        "oos_all_configs_family_incl_unrealistic_hourly_micro": summarize_configs(real_rows + mi["configs"]),
        "configs": {"xs": xs["configs"], "timing": tm["configs"], "carry": ca["configs"], "carry_static": ca["static"],
                    "carry_robust": rc["variants"], "carry_onoff": rc["onoff"]["configs"],
                    "micro_hourly": mi["configs"], "micro_1m": m1["configs"]},
    }
    (OUTD / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"best_oos": best["oos"], "cap": cap, "is_argmax": best["is_selection"],
                      "carry_all": best["oos_all_carry_configs"], "family_real": res["oos_all_configs_family_realistic"],
                      "family_all": res["oos_all_configs_family_incl_unrealistic_hourly_micro"]},
                     indent=1, default=float, ensure_ascii=False))  # fmt: skip
    print("check full metrics", m_full, per_year(bts, bret), half_years(bts, bret))


if __name__ == "__main__":
    main()
