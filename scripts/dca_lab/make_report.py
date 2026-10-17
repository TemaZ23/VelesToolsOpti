"""
Таблицы для итогового отчёта из report_data.json (report.py) и configs.json.

    PYTHONPATH=. python3 -m scripts.dca_lab.make_report > output/dca_lab/tables.md
"""

from __future__ import annotations

import argparse
import json

from .strategy import Sleeve
from .veles_format import sleeve_ru


def _f(v, nd: int = 1, suffix: str = "") -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}{suffix}"
    return str(v)


def windows_md(data: dict) -> str:
    out = []
    for wname, rows in data["windows"].items():
        out.append(f"#### {wname}\n")
        out.append("| Конфигурация | Доходность, %/мес | Просадка, % депо | Эффективность | Худший месяц | Сделок | Стопов | Ликвидация |")
        out.append("|---|---|---|---|---|---|---|---|")
        for name, s in rows.items():
            out.append(
                f"| {name} | {_f(s['avg_month_pct'], 2)} | {_f(s['dd_abs_pct'])} | {_f(s['eff'], 3)} | "
                f"{_f(s['worst_month_pct'], 1)}% | {s['deals']} | {s['sl']} | {'да' if s['liquidated'] else 'нет'} |"
            )
        out.append("")
    return "\n".join(out)


def sizes_md(data: dict, name: str) -> str:
    rows = data["sizes"][name]
    out = [
        "| Депозит на бота | Реинвест | Доходность, %/мес (простая) | Сложный рост, %/мес | Макс. просадка от пика | Ликвидация |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        share = ", ".join(f"{x:g}%" for x in r["bot_share"])
        liq = f"да ({r['liq_date']})" if r["liquidated"] else "нет"
        simple = "—" if r["compound"] or r["liquidated"] else _f(r["avg_month_pct"], 1)
        cmgr = "—" if r["liquidated"] else _f(r["cmgr_pct"], 1)
        dd = "100% (ликвидация)" if r["liquidated"] else f"{_f(r['max_dd_pct'])}%"
        out.append(f"| {share} | {'да' if r['compound'] else 'нет'} | {simple} | {cmgr} | {dd} | {liq} |")
    return "\n".join(out)


def start_risk_md(data: dict, name: str, key: str = "start_risk") -> str:
    rows = data[key][name]
    out = [
        "| Депозит на бота | Стартов | Ликвидаций | Медианный рост, %/мес | Худшая просадка у выживших | Даты стартов с ликвидацией |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        share = ", ".join(f"{x:g}%" for x in r["bot_share"])
        liq_dates = ", ".join(s[:7] for s in r["liq_starts"][:8]) + (" …" if len(r["liq_starts"]) > 8 else "")
        out.append(
            f"| {share} | {r['starts']} | {r['liquidations']} ({r['liquidations'] / r['starts'] * 100:.0f}%) | "
            f"{_f(r['median_cmgr_pct'], 1)} | {_f(r['worst_dd_alive_pct'])}% | {liq_dates or '—'} |"
        )
    return "\n".join(out)


def monthly_md(data: dict, names: list[str]) -> str:
    """Помесячный результат в % НАЧАЛЬНОГО депозита (режим без реинвеста)."""
    months = [m for m, _ in data["monthly"][names[0]]]
    per_dep: dict[str, list[float]] = {}
    for n in names:
        eq = 1.0
        vals = []
        for _, r in data["monthly"][n]:
            new = eq * (1 + r / 100)
            vals.append((new - eq) * 100)
            eq = new
        per_dep[n] = vals
    out = ["| Месяц | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for i, m in enumerate(months):
        out.append(f"| {m} | " + " | ".join(f"{per_dep[n][i]:+.1f}%" for n in names) + " |")
    return "\n".join(out)


def settings_md(configs_path: str, names: list[str]) -> str:
    with open(configs_path, encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for n in names:
        out.append(f"## {n}\n")
        for s in raw[n]["sleeves"]:
            out.append(sleeve_ru(Sleeve.from_dict(s)))
            out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="output/dca_lab/report_data.json")
    ap.add_argument("--configs", default="output/dca_lab/configs.json")
    args = ap.parse_args()
    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    names = list(data["sizes"])
    print("## Сравнение по окнам\n")
    print(windows_md(data))
    for n in names:
        print(f"## {n}: размер бота → доходность и риск (2021-03…2026-10)\n")
        print(sizes_md(data, n))
        print(f"\n### {n}: риск по дате старта (реинвест, до 2026-10)\n")
        print(start_risk_md(data, n))
        if "start_risk_12m" in data:
            print(f"\n### {n}: риск по дате старта, горизонт 12 месяцев (реинвест)\n")
            print(start_risk_md(data, n, "start_risk_12m"))
        print()
    print("## Помесячный результат (0.75% на бота, без реинвеста, % от начального депозита)\n")
    print(monthly_md(data, names))
    print()
    print(settings_md(args.configs, names))


if __name__ == "__main__":
    main()
