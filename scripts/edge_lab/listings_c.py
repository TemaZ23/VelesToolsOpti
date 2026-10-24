"""
listings (c): поведение перед делистингом и фактор «возраста» контракта (молодые vs старые монеты).

    PYTHONPATH=. python3 -m scripts.edge_lab.listings_c

Фактор возраста (дневные бары, 1d-панель):
- возраст монеты на дне t = число дней с начала текущего торгового сегмента (листинг/релистинг),
  «живой» день = qvol > 0 (мёртвые контракты в архиве продолжаются плоскими барами);
- сигнал и допуск (возраст, ADV за 30 прошлых дней) — по данным до закрытия дня t−1,
  позиция держится с open[t] до open[t+1] (доходность open→open);
- фандинг: точное соответствие удержанию (open[t]+ε, open[t+1]+ε] — расчёт в 00:00 дня t+1
  относится к дню t (сдвиг на 2 с), из сырых ставок;
- если монета перестала торговаться — выход по последнему close; пессимистично −50% позиции;
- издержки cost_per_side(ADV) на оборот при каждом ребалансе.
"""

from __future__ import annotations

import itertools
import json
import time

import numpy as np

from scripts.edge_lab.common import DAY, IS, OOS, OUT, cost_per_side, load_funding, metrics, ms, panel, per_year, rolling_mean_past
from scripts.edge_lab.listings_data import category

ODIR = OUT / "listings"
EVF = ODIR.parents[2] / "cache" / "edge_lab" / "listings" / "events.npz"


# ============================================================================ пред-делистинг
def pre_delisting() -> dict:
    z = np.load(EVF, allow_pickle=False)
    c, o, qv, f = (z[f"dl_{k}"].astype(float) for k in ("c", "o", "qv", "f"))
    cat, sym, end, start = z["dl_cat"], z["dl_sym"], z["dl_end"], z["dl_start"]
    btc_c = z["btc_c"]
    t0 = int(z["t0"])
    DW = c.shape[1] - 1
    m = (cat == "crypto") & ((end - start) > 24 * 90)  # прожил > 90 дней (чтобы окно было до листинга)
    out = {"n_delistings_all": int(len(cat)), "n_crypto_aged90": int(m.sum())}
    # ffill close
    idx = np.where(~np.isnan(c), np.arange(c.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    cf = np.take_along_axis(c, idx, axis=1)
    rows = {}
    for dd in (60, 30, 14, 7, 3, 1):
        a = DW - 24 * dd
        r = cf[:, DW] / cf[:, a] - 1
        br = btc_c[end] / btc_c[end - 24 * dd] - 1
        fs = np.nansum(f[:, a + 1 :], axis=1)
        sel = m & np.isfinite(r)
        rows[f"last_{dd}d"] = {
            "n": int(sel.sum()),
            "ret_mean": round(float(np.mean(r[sel])) * 100, 2),
            "ret_median": round(float(np.median(r[sel])) * 100, 2),
            "ex_btc_mean": round(float(np.mean((r - br)[sel])) * 100, 2),
            "ex_btc_median": round(float(np.median((r - br)[sel])) * 100, 2),
            "neg_pct": round(float(np.mean(r[sel] < 0)) * 100, 1),
            "funding_sum_mean_pct": round(float(np.mean(fs[sel])) * 100, 3),
        }
    out["windows"] = rows
    # объём: последние 7 дней vs дни 30..60 до конца
    v7 = np.nanmean(qv[:, DW - 24 * 7 :], axis=1)
    v60 = np.nanmean(qv[:, : DW - 24 * 30], axis=1)
    out["volume_last7_vs_d30_60_median"] = round(float(np.nanmedian((v7 / v60)[m])), 2)
    # крупнейшее падение за сутки в последние 14 дней (прокси даты анонса)
    dmin = []
    for i in np.where(m)[0]:
        x = cf[i, DW - 24 * 14 :: 24]
        dr = x[1:] / x[:-1] - 1
        dmin.append(np.nanmin(dr) if len(dr) else np.nan)
    out["worst_day_last14_median_pct"] = round(float(np.nanmedian(dmin)) * 100, 1)
    out["by_year"] = {}
    yr = (t0 + end * 3_600_000).astype("datetime64[ms]").astype("datetime64[Y]").astype(int) + 1970
    r30 = cf[:, DW] / cf[:, DW - 24 * 30] - 1
    for y in np.unique(yr[m]):
        s = m & (yr == y)
        out["by_year"][str(y)] = {"n": int(s.sum()), "ret30_median": round(float(np.median(r30[s])) * 100, 1)}
    out["list"] = [{"sym": str(sym[i]), "last": str((t0 + end[i] * 3_600_000).astype("datetime64[ms]"))[:10], "ret30_pct": round(float(r30[i]) * 100, 1)} for i in np.where(m)[0]]
    return out


# ============================================================================ фактор возраста
def _daily_funding_hold(ts: np.ndarray, syms: list[str]) -> np.ndarray:
    """F[t, i] — сумма ставок за удержание (open[t]+ε, open[t+1]+ε]."""
    T, N = len(ts), len(syms)
    F = np.zeros((T, N))
    t0 = int(ts[0])
    for j, s in enumerate(syms):
        fd = load_funding(s)
        if not len(fd["ts"]):
            continue
        b = (fd["ts"] - 2000 - t0) // DAY
        ok = (b >= 0) & (b < T)
        np.add.at(F[:, j], b[ok].astype(np.int64), fd["rate"][ok])
    return F


def age_factor() -> dict:
    t_start = time.time()
    p = panel("1d")
    ts, syms = p["ts"], list(p["symbols"])
    o, c, qv = p["open"], p["close"], p["qvol"]
    T, N = o.shape
    alive = np.nan_to_num(qv) > 0
    meta = json.loads((ODIR.parents[2] / "cache" / "edge_lab" / "listings" / "meta.json").read_text())
    t0h = ms("2020-01-01")
    # возраст = дни с начала текущего сегмента
    age = np.full((T, N), np.nan)
    for j, s in enumerate(syms):
        for a, b in meta[s]["segments"]:
            da = (t0h + a * 3_600_000 - int(ts[0])) // DAY
            db = (t0h + b * 3_600_000 - int(ts[0])) // DAY
            da, db = max(int(da), 0), min(int(db), T - 1)
            pre = a < 72  # торговался до начала архива — возраст неизвестен (старая монета)
            age[da : db + 1, j] = np.arange(db - da + 1) + (10_000 if pre else 0)
    cat = np.array([category(s) for s in syms])
    crypto = cat == "crypto"
    adv = rolling_mean_past(np.where(alive, qv, np.nan), 30)
    F = _daily_funding_hold(ts, syms)
    # доходность удержания в день t: open[t+1]/open[t]-1; если t+1 не живой — выход по close[t]
    nxt_alive = np.r_[alive[1:], np.zeros((1, N), bool)]
    o_next = np.r_[o[1:], np.full((1, N), np.nan)]
    R = np.where(nxt_alive, o_next / o - 1, c / o - 1)
    R = np.where(alive, R, 0.0)
    R = np.nan_to_num(R)
    died = alive & ~nxt_alive
    btc = syms.index("BTCUSDT")
    out = {}

    # --- описательно: средняя избыточная (к равновзв. рынку) 7-дневная доходность по корзинам возраста
    buckets = [(1, 7), (7, 30), (30, 90), (90, 180), (180, 365), (365, 730), (730, 100_000)]
    elig_base = alive & crypto
    lo7 = np.log1p(np.clip(R, -0.99, None))
    cum = np.cumsum(np.r_[np.zeros((1, N)), lo7], axis=0)
    desc = {}
    for adv_min in (2e6,):
        a_lag = np.r_[np.full((1, N), np.nan), adv[:-1]]  # ADV на закрытие t−1
        age_lag = np.r_[np.full((1, N), np.nan), age[:-1]]
        elig = elig_base & (np.nan_to_num(a_lag) >= adv_min) & np.isfinite(age_lag)
        res = {}
        for name, win in (("IS", IS), ("OOS", OOS)):
            i0, i1 = np.searchsorted(ts, ms(win[0])), np.searchsorted(ts, ms(win[1]))
            series = {b: [] for b in buckets}
            for t in range(max(i0, 31), min(i1, T - 8), 7):
                fwd = np.exp(cum[t + 7] - cum[t]) - 1  # удержание дни t..t+6
                e = elig[t]
                if e.sum() < 10:
                    continue
                mkt = np.mean(fwd[e])
                for b in buckets:
                    mb = e & (age_lag[t] >= b[0]) & (age_lag[t] < b[1])
                    if mb.sum() >= 3:
                        series[b].append(np.mean(fwd[mb]) - mkt)
            res[name] = {
                f"{b[0]}-{b[1] if b[1] < 100_000 else 'inf'}d": {
                    "weeks": len(v), "ex_mkt_7d_mean_pct": round(float(np.mean(v)) * 100, 2) if v else None,
                    "t": round(float(np.mean(v) / (np.std(v, ddof=1) / np.sqrt(len(v)))), 2) if len(v) > 3 else None,
                }
                for b, v in series.items()
            }  # fmt: skip
        desc[f"adv_min_{int(adv_min / 1e6)}M"] = res
    out["age_buckets_excess_7d"] = desc
    print("корзины возраста:", json.dumps(desc, ensure_ascii=False), flush=True)

    # --- торгуемая версия: шорт равновзвешенной корзины молодых монет, хедж BTC или старые монеты
    def run(a_lo, a_hi, hedge, adv_min, reb, pess=0):
        a_lag = np.r_[np.full((1, N), np.nan), adv[:-1]]
        age_lag = np.r_[np.full((1, N), np.nan), age[:-1]]
        elig = elig_base & (np.nan_to_num(a_lag) >= adv_min) & np.isfinite(age_lag)
        young = elig & (age_lag >= a_lo) & (age_lag < a_hi)
        old = elig & (age_lag >= 730)
        cst = cost_per_side(np.nan_to_num(a_lag))
        w = np.zeros(N)
        rets = np.zeros(T)
        n_tr = 0
        for t in range(1, T):
            if (t % reb == 0) or t == 1:
                nw = np.zeros(N)
                ny = young[t].sum()
                if ny >= 3:
                    nw[young[t]] = -0.5 / ny
                    if hedge == "btc":
                        nw[btc] += 0.5
                    else:
                        no = old[t].sum()
                        if no >= 3:
                            nw[old[t]] += 0.5 / no
                # мёртвые позиции уже закрыты
                nw = np.where(alive[t], nw, 0.0)
                dw = np.abs(nw - w)
                rets[t] -= float(np.sum(dw * np.where(np.isfinite(cst[t]), cst[t], 0.004)))
                n_tr += int(np.sum(dw > 1e-9))
                w = nw
            w = np.where(alive[t], w, 0.0)
            pnl = w * R[t] - w * F[t]
            if pess:
                pnl = pnl - np.where(died[t] & (w != 0), 0.5 * np.abs(w), 0.0)
            rets[t] += float(np.sum(pnl))
            # дрейф весов
            gw = w * (1 + R[t])
            tot = 1 + float(np.sum(w * R[t]))
            w = np.where(died[t], 0.0, gw / tot if tot > 0 else gw)
        i0, i1 = np.searchsorted(ts, ms(IS[0])), np.searchsorted(ts, ms(IS[1]))
        j0, j1 = np.searchsorted(ts, ms(OOS[0])), np.searchsorted(ts, ms(OOS[1]))
        return {"is": metrics(ts[i0:i1], rets[i0:i1], None), "oos": metrics(ts[j0:j1], rets[j0:j1], None)}, rets, n_tr

    rows = []
    for (a_lo, a_hi), hedge, adv_min, reb in itertools.product(
        [(7, 30), (7, 90), (30, 90), (30, 180), (7, 180), (90, 365)], ["btc", "old"], [2e6, 1e7], [7]
    ):
        r, _, ntr = run(a_lo, a_hi, hedge, adv_min, reb)
        rows.append({"age": [a_lo, a_hi], "hedge": hedge, "adv_min": adv_min, "reb_days": reb, "n_trades": ntr, **r})
        print(f"  age {a_lo}-{a_hi} {hedge} adv {adv_min / 1e6:.0f}M: IS sh {r['is']['sharpe']} cagr {r['is']['cagr_pct']} | OOS sh {r['oos']['sharpe']} cagr {r['oos']['cagr_pct']}", flush=True)
    is_sh = np.array([r["is"]["sharpe"] for r in rows])
    is_dd = np.array([r["is"]["max_dd_pct"] for r in rows])
    oos_sh = np.array([r["oos"]["sharpe"] for r in rows])
    b = int(np.lexsort((is_dd, -is_sh))[0])
    best = rows[b]
    _, rb, _ = run(best["age"][0], best["age"][1], best["hedge"], best["adv_min"], best["reb_days"])
    _, rbp, _ = run(best["age"][0], best["age"][1], best["hedge"], best["adv_min"], best["reb_days"], pess=1)
    out["age_ls"] = {
        "rows": rows,
        "n_configs": len(rows),
        "best_is": best,
        "best_per_year": per_year(ts, rb),
        "best_pessimistic_oos": metrics(ts[np.searchsorted(ts, ms(OOS[0])):], rbp[np.searchsorted(ts, ms(OOS[0])):], None),
        "oos_sharpe_median": round(float(np.median(oos_sh)), 2),
        "oos_sharpe_share_positive": round(float(np.mean(oos_sh > 0)) * 100, 1),
        "note": "0.5 капитала шорт молодых (равные веса) + 0.5 лонг BTC или старых монет (возраст ≥ 730 д), ребаланс раз в 7 дней",
    }  # fmt: skip
    np.savez(ODIR / "age_ls_best_daily.npz", ts=ts, ret=rb)
    print(f"возраст: лучший IS {best['age']} {best['hedge']} → OOS sh {best['oos']['sharpe']}; время {time.time() - t_start:.0f} с", flush=True)
    return out



def delist_short() -> dict:
    """Шорт за N дней до последнего живого часа (время делистинга Binance объявляет заранее) до последнего close.
    Только описательно: в IS (2020-2023) всего ~16 событий, выбор параметров на IS невозможен."""
    z = np.load(EVF, allow_pickle=False)
    o, h, c, qv, f = (z[f"dl_{k}"].astype(float) for k in ("o", "h", "c", "qv", "f"))
    cat, end, start = z["dl_cat"], z["dl_end"], z["dl_start"]
    t0 = int(z["t0"])
    btc_o, btc_c = z["btc_o"], z["btc_c"]
    DW = c.shape[1] - 1
    m = (cat == "crypto") & ((end - start) > 24 * 90)
    yr = (t0 + end * 3_600_000).astype("datetime64[ms]").astype("datetime64[Y]").astype(int) + 1970
    # где был худший день последних 14 суток (прокси даты анонса), в днях до конца
    idx = np.where(~np.isnan(c), np.arange(c.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    cf = np.take_along_axis(c, idx, axis=1)
    worst_pos = []
    for i in np.where(m)[0]:
        x = cf[i, DW - 24 * 14 :: 24]
        dr = x[1:] / x[:-1] - 1
        worst_pos.append(14 - int(np.nanargmin(dr)) - 1 if np.isfinite(dr).any() else -1)
    worst_pos = np.array(worst_pos)
    out = {"worst_day_days_before_end_hist": {str(k): int(np.sum(worst_pos == k)) for k in range(14)}}
    rows = {}
    for N in (1, 2, 3, 5, 7):
        e = DW - 24 * N + 1
        pe = o[:, e]
        ok = m & np.isfinite(pe) & np.isfinite(c[:, DW])
        adv = np.nanmean(qv[:, max(0, e - 720) : e], axis=1) * 24
        cst = cost_per_side(np.nan_to_num(adv))
        fs = np.nansum(f[:, e + 1 : DW + 1] * np.where(np.isnan(o[:, e + 1 : DW + 1]), cf[:, e : DW], o[:, e + 1 : DW + 1]), axis=1) / pe
        r = (pe - c[:, DW]) / pe + fs - cst - cst * c[:, DW] / pe
        mae = np.nanmax(h[:, e : DW + 1], axis=1) / pe - 1
        br = btc_c[end] / btc_o[end - 24 * N + 1] - 1
        rh = r - br  # с хеджем BTC тем же номиналом (без учёта издержек хеджа)
        row = {"n": int(ok.sum()), "net_mean_pct": round(float(np.mean(r[ok])) * 100, 2), "net_median_pct": round(float(np.median(r[ok])) * 100, 2),
               "win_pct": round(float(np.mean(r[ok] > 0)) * 100, 1), "hedged_mean_pct": round(float(np.mean(rh[ok])) * 100, 2),
               "funding_mean_pct": round(float(np.mean(fs[ok])) * 100, 2), "mae_p90_pct": round(float(np.percentile(mae[ok], 90)) * 100, 1),
               "mae_max_pct": round(float(np.max(mae[ok])) * 100, 1), "worst_trade_pct": round(float(np.min(r[ok])) * 100, 1)}  # fmt: skip
        row["by_year"] = {str(y): {"n": int((ok & (yr == y)).sum()), "mean_pct": round(float(np.mean(r[ok & (yr == y)])) * 100, 2)} for y in np.unique(yr[ok])}
        # только события, где «анонс» (худший день) был раньше окна входа
        mm = ok.copy()
        mm[np.where(m)[0]] &= worst_pos >= N
        row["announce_before_entry"] = {"n": int(mm.sum()), "net_mean_pct": round(float(np.mean(r[mm])) * 100, 2) if mm.any() else None}
        rows[f"N{N}"] = row
    out["rows"] = rows
    return out


def main() -> None:
    ODIR.mkdir(parents=True, exist_ok=True)
    res = {"pre_delisting": pre_delisting()}
    print(json.dumps({k: v for k, v in res["pre_delisting"].items() if k != "list"}, ensure_ascii=False, indent=1), flush=True)
    res["delist_short"] = delist_short()
    print(json.dumps(res["delist_short"], ensure_ascii=False), flush=True)
    res["age_factor"] = age_factor()
    (ODIR / "part_c.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
