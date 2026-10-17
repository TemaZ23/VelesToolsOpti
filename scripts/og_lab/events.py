"""
Событийное исследование «памп сдувается» без ограничений бота: шорт на закрытии часа, который вырос
больше чем на X%, выход через N часов или по стопу +S% (по максимуму часа), комиссия и
проскальзывание 0.25% на круг. Показывает, есть ли у эффекта преимущество после издержек, и
держится ли оно в разные периоды.

    PYTHONPATH=. python3 -m scripts.og_lab.events
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.dca_lab.data import load_symbol

from .analyze import resample

OUT = Path(__file__).resolve().parents[2] / "output" / "og_lab"
COST = 0.0025  # комиссии (2 × 0.055%) + проскальзывание на неликвидной монете
PERIODS = {"2021-2022": ("2021-01-01", "2023-01-01"), "2023-24H1": ("2023-01-01", "2024-07-01"),
           "2024H2-2025": ("2024-07-01", "2026-01-01"), "2026": ("2026-01-01", "2026-10-01")}  # fmt: skip


def trades(h: dict, thr: float, hold: int, stop: float | None) -> list[tuple[int, float]]:
    c, hi = h["close"], h["high"]
    out = []
    i = 1
    n = len(c)
    while i < n - 1:
        if c[i] / c[i - 1] - 1 > thr:
            entry = c[i]
            exit_ret = None
            for k in range(1, hold + 1):
                j = i + k
                if j >= n:
                    break
                if stop is not None and hi[j] >= entry * (1 + stop):
                    exit_ret = -stop
                    break
                if k == hold:
                    exit_ret = 1 - c[j] / entry
            if exit_ret is None:
                break
            out.append((int(h["ts"][i]), exit_ret - COST))
            i += hold  # одна сделка за раз
        else:
            i += 1
    return out


def main() -> None:
    d = load_symbol("OGUSDT", "spot")
    h = resample(d["ts"], d["ohlc"], d["qvol"], 60)
    rows = []
    for thr in (0.05, 0.08, 0.12, 0.2):
        for hold in (4, 12, 24, 72):
            for stop in (None, 0.1, 0.2, 0.4):
                t = trades(h, thr, hold, stop)
                if not t:
                    continue
                row = {"thr": thr, "hold_h": hold, "stop": stop, "n": len(t)}
                for name, (a, b) in PERIODS.items():
                    lo = np.datetime64(a).astype("datetime64[ms]").astype(np.int64)
                    hi_ = np.datetime64(b).astype("datetime64[ms]").astype(np.int64)
                    r = np.array([x for ts, x in t if lo <= ts < hi_])
                    row[name] = {
                        "n": int(len(r)),
                        "mean_pct": round(float(r.mean() * 100), 2) if len(r) else None,
                        "win": round(float((r > 0).mean()), 2) if len(r) else None,
                        "sum_pct": round(float(r.sum() * 100), 1) if len(r) else None,
                    }
                rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "events.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    # Лучшие по первому периоду и что с ними дальше
    def mean_of(row, p):
        return row[p]["mean_pct"] if row[p]["mean_pct"] is not None else -99
    ranked = sorted(rows, key=lambda r: -mean_of(r, "2021-2022"))
    print("Топ по 2021-2022 (сделка: средний %, доля плюсовых, число) → следующие периоды:")
    for r in ranked[:12]:
        cells = " | ".join(
            f"{p}: {r[p]['mean_pct']}% w{r[p]['win']} n{r[p]['n']}" for p in PERIODS
        )
        print(f"рост>{int(r['thr']*100)}% держать {r['hold_h']}ч стоп {r['stop']}: {cells}")
    stable = [r for r in rows if all((r[p]["mean_pct"] or -1) > 0 and r[p]["n"] >= 5 for p in PERIODS)]
    print(f"\nВариантов, прибыльных в КАЖДОМ периоде (n≥5): {len(stable)} из {len(rows)}")
    for r in sorted(stable, key=lambda r: -min(r[p]['mean_pct'] for p in PERIODS))[:8]:
        print(f"  рост>{int(r['thr']*100)}% {r['hold_h']}ч стоп {r['stop']}: " + " | ".join(f"{p}: {r[p]['mean_pct']}% n{r[p]['n']}" for p in PERIODS))


if __name__ == "__main__":
    main()
