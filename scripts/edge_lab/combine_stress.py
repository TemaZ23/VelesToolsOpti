"""
Стресс-ряды для комбинированного портфеля (scripts/edge_lab/combine.py).

Пересчитывает дневные ряды двух «ослабленных, но не опровергнутых» кандидатов (xsec, listings) теми же
независимыми реализациями верификаторов (verify_xsec.py, verify_listings.py — только импорт, без правок)
в пессимистичных режимах исполнения, чтобы комбинированный портфель и прогноз можно было оценить не только
на базовых допущениях:

  xsec_cost2        — Amihud L/S, исправленные данные (без зомби-баров, выход по цене расчёта), издержки ×2
  xsec_cost2_pess50 — то же + −50% каждой позиции в монете, которая перестала торговаться (>7 дней)
  lst_cost2         — шорт новых листингов, издержки ×2
  lst_pess_combo    — издержки ×2 + стоп с проскальзыванием 1% + исполнение стопа на 25% глубже в бар
  lst_nohedge(_pess)— то же без хеджа BTC (для малого счёта, где номинал сделки < минимума BTCUSDT $100)

Запуск: python3 -m scripts.edge_lab.combine_stress   (≈ 2–5 мин, один процесс, панель 1d + часовые окна событий)
Выход:  output/edge_lab/combine/stress_daily.npz (ts + ряды выше), stress_summary.json
"""

from __future__ import annotations

import json
import time

import numpy as np

from scripts.edge_lab.common import IS, OOS, OUT, metrics, ms, window_idx

OUTD = OUT / "combine"


def xsec_series() -> dict[str, np.ndarray]:
    from scripts.edge_lab import verify_xsec as vx

    d = vx.load_data()
    ts = d["ts"]
    syms = [str(s) for s in d["symbols"]]
    sf = vx.OUTD / "verify_settle.json"
    d_set = dict(d)
    d_set["C"] = d["C"].copy()
    n_adj = 0
    if sf.exists():
        for rec in json.loads(sf.read_text()):
            if "index_avg30" not in rec:
                continue
            j = syms.index(rec["sym"])
            t = int(np.searchsorted(ts, ms(rec["day"])))
            d_set["C"][t, j] = rec["index_avg30"]
            n_adj += 1
    W, adv, _ = vx.signals(d_set)
    out = {"ts": ts}
    out["xsec_base_check"] = vx.backtest(d_set, W, adv)["ret"]
    out["xsec_cost2"] = vx.backtest(d_set, W, adv, cost_mult=2.0)["ret"]
    out["xsec_cost2_pess50"] = vx.backtest(d_set, W, adv, cost_mult=2.0, pess=True)["ret"]
    print("xsec done, settle-adjusted events:", n_adj, flush=True)
    return out


def listings_series() -> dict[str, np.ndarray]:
    from scripts.edge_lab import verify_listings as vl

    events = vl.find_events()
    crypto = [e for e in events if e["cls"] == "crypto"]
    wins = {(e["sym"], e["L"]): vl.event_window(e["sym"], e["L"]) for e in crypto}
    btc = vl.BTC()
    out = {}
    base = vl.run(crypto, wins, btc)
    out["lst_ts"] = base["dts"]
    trades = [{"sym": r["sym"], "t_entry": int(r["t_entry"]), "total": float(r["total"]), "reason": r["reason"]}
              for r in base["used"]]
    (OUTD / "listings_trades.json").write_text(json.dumps(trades))
    out["lst_base_check"] = base["dret"]
    out["lst_cost2"] = vl.run(crypto, wins, btc, cmul=2.0)["dret"]
    out["lst_pess_combo"] = vl.run(crypto, wins, btc, cmul=2.0, stop_slip=0.01, stop_frac=0.25)["dret"]
    # без хеджа BTC (малый счёт: номинал сделки < $100 — минимального ордера BTCUSDT)
    out["lst_nohedge"] = vl.run(crypto, wins, btc, hedge=False)["dret"]
    out["lst_nohedge_pess"] = vl.run(crypto, wins, btc, hedge=False, cmul=2.0, stop_slip=0.01, stop_frac=0.25)["dret"]
    print("listings done, crypto events:", len(crypto), flush=True)
    return out


def main():
    t0 = time.time()
    OUTD.mkdir(parents=True, exist_ok=True)
    xs = xsec_series()
    ls = listings_series()
    ts = xs["ts"]
    lts = ls.pop("lst_ts")
    # выравнивание по дням: оба ряда начинаются 2020-01-01, шаг 1 день
    res = {"ts": ts}
    for k, v in xs.items():
        if k != "ts":
            res[k] = v
    for k, v in ls.items():
        a = np.zeros(len(ts))
        pos = np.searchsorted(ts, lts)
        ok = (pos < len(ts)) & (ts[np.minimum(pos, len(ts) - 1)] == lts)
        a[pos[ok]] = v[ok]
        res[k] = a
    np.savez(OUTD / "stress_daily.npz", **res)
    summ = {}
    for k, v in res.items():
        if k == "ts":
            continue
        a, b = window_idx(ts, IS)
        c, d = window_idx(ts, OOS)
        summ[k] = {"IS": metrics(ts[a:b], v[a:b]), "OOS": metrics(ts[c:d], v[c:d])}
        print(f"{k:20s} IS SR {summ[k]['IS']['sharpe']:5.2f}  OOS SR {summ[k]['OOS']['sharpe']:5.2f} "
              f"CAGR {summ[k]['OOS']['cagr_pct']:6.1f}", flush=True)
    (OUTD / "stress_summary.json").write_text(json.dumps(summ, indent=1))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
