"""
Сборка итогов семейства funding: output/edge_lab/funding/results.json и best_daily.npz.

    PYTHONPATH=. python3 -m scripts.edge_lab.funding_report

Лучший кандидат (выбран по IS среди carry-конфигов с консервативной моделью счёта): статичный
cash-and-carry BTC+ETH (лонг спот + шорт перп, 50/50 капитала, маржа перпа 1/3 номинала, кросс-маржа).
"""

from __future__ import annotations

import json

import numpy as np

from scripts.edge_lab.common import OOS, metrics, per_year, rolling_mean_past, window_idx
from scripts.edge_lab.funding import OUTD, half_years, prep, summarize_configs


def load(name: str) -> dict:
    return json.loads((OUTD / name).read_text())


def brief(r: dict) -> dict:
    return {k: r[k] for k in ("cfg", "is", "oos", "trades") if k in r}


def main() -> None:
    xs, tm, ca, mi, m1 = (load(f"part_{n}.json") for n in ("xs", "timing", "carry", "micro", "m1"))
    d = prep()
    ts, syms = d["ts"], d["syms"]
    # ---------------- лучший кандидат: статичный carry BTC+ETH, m=1/3
    st = ca["static"]["BTC+ETH|m1/3"]
    with np.load(OUTD / "carry_static_BTC_ETH_m13.npz") as z:
        bts, bret = z["ts"], z["ret"]
    assert np.array_equal(bts, ts)
    a, b = window_idx(ts, ("2020-01-01", "2026-10-01"))
    np.savez(OUTD / "best_daily.npz", ts=bts[a:b], ret=bret[a:b])
    sadv = np.nan_to_num(rolling_mean_past(d["aux"]["spot_qvol"], 30))
    oa, ob = window_idx(ts, OOS)
    cap = {}
    for s in ("BTCUSDT", "ETHUSDT"):
        j = syms.index(s)
        cap[s] = float(np.median(0.01 * np.minimum(d["adv"][oa:ob, j], sadv[oa:ob, j]) / 24))
    candidates_carry = ca["configs"] + [dict(v, cfg={"static": k}) for k, v in ca["static"].items()]
    is_best = max(candidates_carry, key=lambda r: r["is"]["sharpe"])
    best = {
        "name": "Статичный cash-and-carry BTC+ETH (лонг спот / шорт перп USDT-M, маржа 1/3)",
        "params": "BTC и ETH по 50% капитала; на монету: спот = капитал/(1+1/3), шорт перпа того же количества, "
                  "маржа USDT-M = 1/3 номинала (кросс); перебалансировка ног, когда маржа монеты выходит из "
                  "[25%, 66.7%] номинала; без ротации и порогов",
        "is": st["is"], "oos": st["oos"], "full": st["full"], "per_year": st["per_year"],
        "oos_half_years": st["oos_half_years"], "trades": st["trades"], "liquidation_days": st["liquidation_days"],
        "pnl_split_full_ann_pct": {"funding": st["full"]["ann_fund_pct"], "basis_price": st["full"]["ann_price_pct"],
                                   "costs": -st["full"]["ann_cost_pct"]},
        "capacity_usd_per_leg_oos_median": cap,
        "exposure": "капитал = спот + маржа перпа = 1.333 * номинал; брутто ≈ 0.75 лонг спот + 0.75 шорт перп "
                    "на 1 капитала; net delta ≈ 0",
        "variants": {k: {"is_cagr": v["is"]["cagr_pct"], "is_sharpe": v["is"]["sharpe"],
                         "oos_cagr": v["oos"]["cagr_pct"], "oos_sharpe": v["oos"]["sharpe"],
                         "oos_dd": v["oos"]["max_dd_pct"], "liq_days": v["liquidation_days"],
                         "per_year": v["per_year"]} for k, v in ca["static"].items()},
        "is_selection": {"rule": "max IS Sharpe среди всех carry-конфигов (ротация 1152 + статичные 9)",
                         "argmax": is_best["cfg"], "argmax_is_sharpe": is_best["is"]["sharpe"],
                         "note": "argmax — multiasset (залог BTC/ETH в USDT-M, оптимистичная модель); "
                                 "как основной взят ближайший консервативный вариант m=1/3 (IS Sharpe 8.66)"},
        "oos_all_carry_configs": summarize_configs(candidates_carry),
    }
    # ---------------- второй кандидат: (c) лонг альтов при экстремально низком фандинге
    alt = tm["best_is_by_asset"]["ALT"]
    hyps = [
        {"name": "(a) cash-and-carry: ротация альтов по трейлинг-фандингу (спот+перп)",
         "configs": len(ca["configs"]), "summary": ca["summary_all"], "best_is": brief(ca["best_is"]),
         "best_is_per_year": ca["best_is"]["per_year"], "best_is_oos_half_years": ca["best_is"]["oos_half_years"],
         "summary_by_margin": ca["summary_by_m"]},
        {"name": "(a) cash-and-carry: статичный BTC/ETH", "configs": len(ca["static"]),
         "static": {k: {"is": v["is"], "oos": v["oos"], "per_year": v["per_year"]} for k, v in ca["static"].items()}},
        {"name": "(b) кросс-секционный фактор фандинга (шорт высокого / лонг низкого)", "configs": len(xs["configs"]),
         "summary": xs["summary_all"], "summary_by_dir": xs["summary_by_dir"], "best_is": xs["best_is"]},
        {"name": "(c) экстремумы фандинга как сигнал (BTC/ETH/альты)", "configs": len(tm["configs"]),
         "summary": tm["summary_all"], **{k: v for k, v in tm.items() if k.startswith("summary_")},
         "best_is_by_asset": tm["best_is_by_asset"], "baselines": {k: v for k, v in tm.items() if k.startswith("baseline")},
         "control_price_reversal": tm.get("control_price_reversal")},
        {"name": "(d) микроструктура выплат, часовые бары (НЕРЕАЛИСТИЧНО: вход/выход по первой сделке после выплаты)",
         "configs": len(mi["configs"]), "summary_taker": mi["summary_taker"], "summary_maker": mi["summary_maker"],
         "best_is_taker": brief(mi["best_is_taker"]), "bucket_table": mi["bucket_table"]},
        {"name": "(d) микроструктура выплат, минутки (реалистичный вход через >=1 мин после выплаты)",
         "configs": len(m1["configs"]), "summary": m1["summary"], "summary_by_iv": m1["summary_by_iv"],
         "best_is": brief(m1["best_is"]), "profile_bp": m1["profile_bp"], "through_realistic": m1["through_realistic"]},
    ]
    all_rows = xs["configs"] + tm["configs"] + candidates_carry + m1["configs"]
    res = {
        "family": "funding",
        "data_notes": [
            "В архиве klines после делистинга идут бары с нулевым объёмом и замороженной ценой (~150 символов, "
            "флаг delisted в universe.json только у 31). Здесь бар считается торговым только при qvol>0; "
            "«настоящий делистинг» — последний торговый бар раньше чем за 7 дней до конца выборки.",
            "Дневной high последней цены перпа содержит фитили (BTC 2021-07-26 +36%, ETH 2020-03-13 +200%), "
            "которых нет в mark price; ликвидации в carry считаются по mark price (markPriceKlines 1d).",
            "Фандинг для P&L: выплаты в (open_k, open_k+1] — позиция, открытая по open, не получает выплату в "
            "этот же момент; сигнал — panel funding[<=t].",
            "Спот Binance: дневные свечи 481 перпа, у которых есть спот (с учётом префиксов 1000*), фильтр "
            "|базис| < 5% против неверного маппинга.",
        ],
        "best": best,
        "second_candidate_alt_contrarian": {
            "cfg": alt["cfg"], "is": alt["is"], "oos": alt["oos"], "trades": alt["trades"],
            "per_year": alt.get("per_year"), "oos_half_years": alt.get("oos_half_years"),
            "pessimistic_delist_oos": alt.get("pessimistic_delist_oos"),
            "random_entry_baseline": alt.get("random_entry_baseline"),
            "control_price_reversal": tm.get("control_price_reversal")},
        "hypotheses": hyps,
        "oos_all_configs_family": summarize_configs(all_rows),
        "configs": {"xs": xs["configs"], "timing": tm["configs"], "carry": ca["configs"], "carry_static": ca["static"],
                    "micro_hourly": mi["configs"], "micro_1m": m1["configs"]},
    }
    (OUTD / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"best_oos": best["oos"], "cap": cap, "is_argmax": best["is_selection"],
                      "carry_all": best["oos_all_carry_configs"], "family_all": res["oos_all_configs_family"]},
                     indent=1, default=float))  # fmt: skip
    m = metrics(bts[a:b], bret[a:b])
    print("check full metrics", m, per_year(bts[a:b], bret[a:b]), half_years(bts, bret))


if __name__ == "__main__":
    main()
