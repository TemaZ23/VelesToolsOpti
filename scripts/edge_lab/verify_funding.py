"""
Независимая проверка (adversarial review) лучшего кандидата семейства funding:
статичный cash-and-carry BTC+ETH (лонг спот + шорт USDT-M перп, маржа 50% номинала).

Реализация написана по текстовой спецификации правил, без чтения кода автора.

    PYTHONPATH=. python3 -m scripts.edge_lab.verify_funding download   # спот 1h, mark 1h, DTB3
    PYTHONPATH=. python3 -m scripts.edge_lab.verify_funding run        # симуляция + робастность

Данные (свои загрузки, cache/edge_lab/funding/verify/):
    spot_<SYM>.npz  — спот 1h свечи Binance (ts, ohlc, qvol)
    mark_<SYM>.npz  — mark-price 1h свечи USDT-M (ts, ohlc)
    dtb3.csv        — 3-месячные T-bill (FRED DTB3), безрисковая ставка USD
Перп 1h и фандинг — common.load_hourly / load_funding.

Модель (часовая сетка, капитал C = 1):
    на монету i: Q_i монет спот = Q_i монет шорт перп; M_i — эквити USDT-M (маржа + нереализ. PnL + фандинг).
    старт 2020-01-01 00:00: вес монеты w_i·C, Q_i = w_i·C / (S + m·F), M_i = m·Q_i·F (минус издержки).
    фандинг: в момент выплаты τ шорт получает rate·Q·mark(τ) (позиция, держащаяся до τ).
    проверка (по умолчанию раз в сутки в 00:00 UTC по цене открытия часа): ratio = M_i/(Q_i·F);
    вне [lo, hi] → Q_new = E_i/(S + m·F), E_i = Q_i·S + M_i; торгуем |ΔQ| на споте и перпе с издержками.
    ликвидация: кросс-маржа по обеим монетам, Σ(M_i − Q_i·(markHigh_i − markOpen_i)) < mmr·Σ Q_i·markHigh_i
    (одновременные часовые максимумы mark-цены — пессимистично). При ликвидации эквити USDT-M → 0,
    спот остаётся, хедж восстанавливается тем же часом по закрытию.
"""

from __future__ import annotations

import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from numba import njit

from scripts.edge_lab.common import CACHE, DAY, HOUR, IS, OOS, OUT, load_funding, load_hourly, metrics, ms, per_year
from scripts.edge_lab.universe import _get, _read_csv

VDIR = CACHE / "funding" / "verify"
ODIR = OUT / "funding"
COINS = ["BTCUSDT", "ETHUSDT"]
T0, T1 = ms("2020-01-01"), ms("2026-10-01")
MONTHS = [f"{y}-{m:02d}" for y in range(2020, 2027) for m in range(1, 13) if f"{y}-{m:02d}" <= "2026-09"]
BASE = "https://data.binance.vision/"


# ----------------------------------------------------------------------------- download
def _fetch_month(args):
    url, cols = args
    blob = _get(url)
    return _read_csv(blob, cols) if blob else np.empty((0, len(cols)))


def download() -> None:
    VDIR.mkdir(parents=True, exist_ok=True)
    jobs = {
        "spot": ("data/spot/monthly/klines/{s}/1h/{s}-1h-{m}.zip", (0, 1, 2, 3, 4, 7)),
        "mark": ("data/futures/um/monthly/markPriceKlines/{s}/1h/{s}-1h-{m}.zip", (0, 1, 2, 3, 4)),
    }
    for kind, (tmpl, cols) in jobs.items():
        for s in COINS:
            path = VDIR / f"{kind}_{s}.npz"
            if path.exists():
                continue
            urls = [(BASE + tmpl.format(s=s, m=m), cols) for m in MONTHS]
            with ThreadPoolExecutor(6) as ex:
                parts = list(ex.map(_fetch_month, urls))
            a = np.concatenate([p for p in parts if len(p)])
            a = a[np.argsort(a[:, 0], kind="stable")]
            a = a[np.r_[True, a[1:, 0] != a[:-1, 0]]]
            out = {"ts": a[:, 0].astype(np.int64), "ohlc": a[:, 1:5]}
            if kind == "spot":
                out["qvol"] = a[:, 5]
            np.savez(path, **out)
            print(kind, s, len(a), np.datetime64(int(a[0, 0]), "ms"), np.datetime64(int(a[-1, 0]), "ms"), flush=True)
    fp = VDIR / "dtb3.csv"
    if not fp.exists():
        blob = _get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3")
        fp.write_bytes(blob)
        print("dtb3", len(blob))


# ----------------------------------------------------------------------------- data
def _grid(ts_src, ohlc, ts):
    """Раскладка на часовую сетку ts; пропуски — forward fill по прошлым данным (open=high=low=close=prev close)."""
    T = len(ts)
    out = np.full((T, 4), np.nan)
    idx = (ts_src - ts[0]) // HOUR
    ok = (idx >= 0) & (idx < T) & (ts_src % HOUR == 0)
    out[idx[ok].astype(np.int64)] = ohlc[ok]
    missing = np.isnan(out[:, 3])
    last = np.nan
    for t in range(T):
        if np.isnan(out[t, 3]):
            out[t] = last
        else:
            last = out[t, 3]
    return out, int(missing.sum())


def load_data() -> dict:
    ts = np.arange(T0, T1, HOUR, dtype=np.int64)
    T = len(ts)
    d = {"ts": ts, "missing": {}}
    S, F, MK, FR, SV, FV = (np.zeros((T, 2, 4)), np.zeros((T, 2, 4)), np.zeros((T, 2, 4)),
                            np.zeros((T, 2)), np.zeros((T, 2)), np.zeros((T, 2)))  # fmt: skip
    for i, s in enumerate(COINS):
        h = load_hourly(s)
        F[:, i], mf = _grid(h["ts"], h["ohlc"], ts)
        fv = np.zeros(T)
        idx = (h["ts"] - T0) // HOUR
        ok = (idx >= 0) & (idx < T)
        fv[idx[ok]] = h["qvol"][ok]
        FV[:, i] = fv
        with np.load(VDIR / f"spot_{s}.npz") as z:
            S[:, i], msp = _grid(z["ts"], z["ohlc"], ts)
            sv = np.zeros(T)
            idx = (z["ts"] - T0) // HOUR
            ok = (idx >= 0) & (idx < T)
            sv[idx[ok]] = z["qvol"][ok]
            SV[:, i] = sv
        with np.load(VDIR / f"mark_{s}.npz") as z:
            mk, mmk = _grid(z["ts"], z["ohlc"], ts)
        # mark до начала архива — перп как замена
        nomark = np.isnan(mk[:, 3])
        mk[nomark] = F[nomark, i]
        MK[:, i] = mk
        f = load_funding(s)
        fb = (f["ts"] - T0) // HOUR  # выплата в τ (с миллисекундами) → час, открывающийся в τ
        ok = (fb >= 0) & (fb < T)
        np.add.at(FR[:, i], fb[ok].astype(np.int64), f["rate"][ok])
        d["missing"][s] = {"perp_h": mf, "spot_h": msp, "mark_h": mmk, "mark_filled_by_perp_h": int(nomark.sum()),
                           "funding_n": int(ok.sum())}
    d.update(S=S, F=F, MK=MK, FR=FR, SV=SV, FV=FV)
    # безрисковая ставка (T-bill 3m, % годовых) → доля за календарный день
    rows = (VDIR / "dtb3.csv").read_text().strip().splitlines()[1:]
    rd, rv = [], []
    for r in rows:
        a, b = r.split(",")
        if b.strip() not in ("", "."):
            rd.append(ms(a))
            rv.append(float(b))
    rd, rv = np.array(rd), np.array(rv)
    days = np.arange(T0, T1, DAY, dtype=np.int64)
    k = np.searchsorted(rd, days, side="right") - 1  # последнее известное значение (без заглядывания)
    d["rf_daily"] = rv[np.clip(k, 0, None)] / 100 / 365
    d["days"] = days
    return d


# ----------------------------------------------------------------------------- simulator
@njit(cache=True, error_model="numpy")
def _sim(S, F, MK, FR, check, w0, m, lo, hi, c_spot, c_perp, mmr, rebal_to_w, fund_mult):
    T = S.shape[0]
    Q = np.zeros(2)
    M = np.zeros(2)
    eq = np.zeros(T)
    fund_acc = np.zeros(T)
    cost_acc = np.zeros(T)
    turn_acc = np.zeros(T)
    nreb = np.zeros(T)
    liqf = np.zeros(T)
    lastF = np.zeros(2)
    fund_by = np.zeros(2)
    for t in range(T):
        so = S[t, :, 0]
        fo = F[t, :, 0]
        if t == 0:
            for i in range(2):
                Q[i] = w0[i] / (so[i] + m * fo[i])
                notional_s = Q[i] * so[i]
                notional_f = Q[i] * fo[i]
                c = notional_s * c_spot + notional_f * c_perp
                M[i] = m * notional_f - c
                cost_acc[t] += c
                turn_acc[t] += notional_s + notional_f
                lastF[i] = fo[i]
            nreb[t] = 1
        else:
            # gap prev close → this open
            for i in range(2):
                M[i] -= Q[i] * (fo[i] - lastF[i])
                lastF[i] = fo[i]
            # фандинг, выплаченный в начале часа t, — на позицию, удерживаемую до τ
            for i in range(2):
                fpay = fund_mult * FR[t, i] * Q[i] * MK[t, i, 0]
                M[i] += fpay
                fund_acc[t] += fpay
                fund_by[i] += fpay
            if check[t]:
                any_out = False
                for i in range(2):
                    r = M[i] / (Q[i] * fo[i])
                    if r < lo or r > hi:
                        any_out = True
                if any_out:
                    Etot = 0.0
                    for i in range(2):
                        Etot += Q[i] * so[i] + M[i]
                    for i in range(2):
                        r = M[i] / (Q[i] * fo[i])
                        if (r < lo or r > hi) or rebal_to_w:
                            Ei = Q[i] * so[i] + M[i]
                            if rebal_to_w:
                                Ei = w0[i] * Etot
                            Qn = Ei / (so[i] + m * fo[i])
                            dq = abs(Qn - Q[i])
                            c = dq * so[i] * c_spot + dq * fo[i] * c_perp
                            Q[i] = Qn
                            M[i] = Ei - Qn * so[i] - c
                            cost_acc[t] += c
                            turn_acc[t] += dq * (so[i] + fo[i])
                    nreb[t] = 1
        # ликвидация внутри часа (кросс-маржа, одновременные максимумы mark)
        worst = 0.0
        mm = 0.0
        for i in range(2):
            worst += M[i] - Q[i] * (MK[t, i, 1] - MK[t, i, 0])
            mm += mmr * Q[i] * MK[t, i, 1]
        liq = worst < mm
        if liq:
            liqf[t] = 1.0
        # закрытие часа
        for i in range(2):
            fc = F[t, i, 3]
            M[i] -= Q[i] * (fc - lastF[i])
            lastF[i] = fc
        if liq:
            # перп закрыт, эквити USDT-M потеряно, спот остался → восстановить хедж по закрытию
            for i in range(2):
                sc = S[t, i, 3]
                fc = F[t, i, 3]
                Ei = Q[i] * sc  # только спот
                Qn = Ei / (sc + m * fc)
                dq_s = abs(Qn - Q[i])
                c = dq_s * sc * c_spot + Qn * fc * c_perp
                Q[i] = Qn
                M[i] = Ei - Qn * sc - c
                cost_acc[t] += c
                turn_acc[t] += dq_s * sc + Qn * fc
            nreb[t] = 1
        e = 0.0
        for i in range(2):
            e += Q[i] * S[t, i, 3] + M[i]
        eq[t] = e
    return eq, fund_acc, cost_acc, turn_acc, nreb, liqf, fund_by


def run_config(d, m=0.5, lo=0.375, hi=1.0, c_spot=0.0012, c_perp=0.0007, mmr=0.02, check_hour=0,
               every_h=24, w=(0.5, 0.5), rebal_to_w=False, fund_mult=1.0, start=None):
    ts = d["ts"]
    hod = (ts // HOUR) % 24
    if every_h == 24:
        check = hod == check_hour
    else:
        check = (hod % every_h) == (check_hour % every_h)
    S, F, MK, FR = d["S"], d["F"], d["MK"], d["FR"]
    w0 = np.array(w, float)
    if start is not None:
        k = int(np.searchsorted(ts, ms(start)))
        S, F, MK, FR, check, ts = S[k:], F[k:], MK[k:], FR[k:], check[k:], ts[k:]
    eq, fa, ca, ta, nr, liqf, fby = _sim(np.ascontiguousarray(S), np.ascontiguousarray(F), np.ascontiguousarray(MK),
                                         np.ascontiguousarray(FR), check, w0, m, lo, hi, c_spot, c_perp, mmr,
                                         rebal_to_w, fund_mult)  # fmt: skip
    # дневная агрегация: эквити на закрытии последнего часа суток
    day = ts // DAY
    last = np.r_[day[1:] != day[:-1], True]
    deq = eq[last]
    dts = (day[last] * DAY).astype(np.int64)
    dret = deq / np.r_[1.0, deq[:-1]] - 1
    first = np.r_[True, day[1:] != day[:-1]]
    ids = np.cumsum(first) - 1
    agg = lambda x: np.bincount(ids, weights=x, minlength=len(dts))  # noqa: E731
    return {"ts": dts, "ret": dret, "eq": deq, "fund": agg(fa), "cost": agg(ca), "turn": agg(ta),
            "reb": agg(nr), "nliq": int(liqf.sum()), "fund_by": fby,
            "liq_times": [str(np.datetime64(int(x), "ms")) for x in ts[liqf > 0]]}  # fmt: skip


# ----------------------------------------------------------------------------- reporting
def _win(ts, a, b):
    return (ts >= ms(a)) & (ts < ms(b))


def stats(r, d=None, excess=False):
    ts, ret = r["ts"], r["ret"].copy()
    if excess:
        k = ((ts - T0) // DAY).astype(int)
        ret = ret - d["rf_daily"][k]
    out = {}
    for name, (a, b) in {"is": IS, "oos": OOS, "full": ("2020-01-01", "2026-10-01")}.items():
        sel = _win(ts, a, b)
        mm = metrics(ts[sel], ret[sel], trades=int(r["reb"][sel].sum()))
        out[name] = mm
    return out


def half_years(r, excess_rf=None):
    ts, ret = r["ts"], r["ret"]
    if excess_rf is not None:
        ret = ret - excess_rf
    out = {}
    for y in range(2020, 2027):
        for h, (a, b) in enumerate([(f"{y}-01-01", f"{y}-07-01"), (f"{y}-07-01", f"{y + 1}-01-01")]):
            sel = _win(ts, a, b)
            if sel.sum():
                out[f"{y}H{h + 1}"] = round((float(np.prod(1 + ret[sel])) - 1) * 100, 2)
    return out


def attribution(r, a, b):
    sel = _win(r["ts"], a, b)
    years = sel.sum() / 365.25
    eqp = np.r_[1.0, r["eq"][:-1]][sel]
    fund = float(np.sum(r["fund"][sel] / eqp)) / years * 100
    cost = float(np.sum(r["cost"][sel] / eqp)) / years * 100
    tot = float(np.sum(r["ret"][sel])) / years * 100
    return {"funding_pct_yr": round(fund, 2), "costs_pct_yr": round(-cost, 2),
            "price_basis_pct_yr": round(tot - fund + cost, 2), "total_simple_pct_yr": round(tot, 2),
            "turnover_pct_of_equity_per_day": round(float(np.mean(r["turn"][sel] / eqp)) * 100, 3)}  # fmt: skip


def main_run() -> None:
    t0 = time.time()
    d = load_data()
    print("data", d["missing"], f"{time.time() - t0:.1f}s", flush=True)
    res = {"data_missing": d["missing"]}

    base = run_config(d)
    res["base"] = {"stats": stats(base), "per_year": per_year(base["ts"], base["ret"]), "half_years": half_years(base),
                   "nliq": base["nliq"], "attr_full": attribution(base, "2020-01-01", "2026-10-01"),
                   "attr_is": attribution(base, *IS), "attr_oos": attribution(base, *OOS),
                   "rebalance_days": {"is": int(base["reb"][_win(base["ts"], *IS)].sum()),
                                      "oos": int(base["reb"][_win(base["ts"], *OOS)].sum())}}  # fmt: skip
    k = ((base["ts"] - T0) // DAY).astype(int)
    rf = d["rf_daily"][k]
    res["base"]["excess_vs_tbill"] = {"stats": stats(base, d, excess=True), "half_years": half_years(base, rf),
                                      "per_year_rf_pct": per_year(base["ts"], rf)}  # fmt: skip
    print(json.dumps(res["base"], indent=1, ensure_ascii=False), flush=True)
    np.savez(ODIR / "verified_daily.npz", ts=base["ts"], ret=base["ret"])

    # их файл best_daily — сверка
    with np.load(ODIR / "best_daily.npz") as z:
        theirs = {k2: z[k2] for k2 in z.files}
    res["theirs_files"] = list(theirs)
    tts, tret = theirs["ts"], theirs["ret"]
    common_ts, ia, ib = np.intersect1d(base["ts"], tts, return_indices=True)
    diff = base["ret"][ia] - tret[ib]
    res["compare_theirs"] = {"n_common": int(len(common_ts)), "corr": float(np.corrcoef(base["ret"][ia], tret[ib])[0, 1]),
                             "mean_diff_bp_day": float(np.mean(diff) * 1e4), "mad_bp_day": float(np.mean(np.abs(diff)) * 1e4),
                             "their_stats": {nm: metrics(tts[_win(tts, a, b)], tret[_win(tts, a, b)]) for nm, (a, b) in
                                             {"is": IS, "oos": OOS}.items()}}  # fmt: skip
    print("compare", json.dumps(res["compare_theirs"], indent=1), flush=True)

    # ------------------------------------------------------------------ robustness grid
    rows = []

    def rec(name, **kw):
        r = run_config(d, **kw)
        st = stats(r)
        ex = stats(r, d, excess=True)
        row = {"name": name, "params": {k2: (list(v) if isinstance(v, tuple) else v) for k2, v in kw.items()},
               "is": st["is"], "oos": st["oos"], "nliq": r["nliq"], "liq_times": r["liq_times"], "oos_excess_tbill": ex["oos"],
               "oos_half_years": {k2: v for k2, v in half_years(r).items() if k2 >= "2024"}}  # fmt: skip
        rows.append(row)
        isr = st["is"] or {"cagr_pct": float("nan"), "sharpe": float("nan")}
        print(f"{name:42s} IS cagr {isr['cagr_pct']:6.2f} sh {isr['sharpe']:6.2f} | OOS cagr {st['oos']['cagr_pct']:5.2f}"
              f" sh {st['oos']['sharpe']:6.2f} dd {st['oos']['max_dd_pct']:5.2f} | ex-rf OOS cagr {ex['oos']['cagr_pct']:5.2f}"
              f" sh {ex['oos']['sharpe']:5.2f} | liq {r['nliq']} reb {int(r['reb'].sum())}", flush=True)
        return r

    rec("base m0.5 [0.375,1.0] 00UTC")
    for mm_ in (0.375, 0.625):
        rec(f"m={mm_} band scaled", m=mm_, lo=0.75 * mm_, hi=2 * mm_)
    for lo_ in (0.28, 0.47):
        rec(f"lo={lo_}", lo=lo_)
    for hi_ in (0.75, 1.25):
        rec(f"hi={hi_}", hi=hi_)
    rec("m=0.25 band scaled (their liq case)", m=0.25, lo=0.1875, hi=0.5)
    rec("m=0.25 mmr0.5%", m=0.25, lo=0.1875, hi=0.5, mmr=0.005)
    for hh in (4, 8, 12, 16, 20):
        rec(f"check hour {hh:02d} UTC", check_hour=hh)
    rec("check every hour", every_h=1)
    rec("check every 8h (01,09,17)", every_h=8, check_hour=1)
    rec("costs x2", c_spot=0.0024, c_perp=0.0014)
    rec("costs x3", c_spot=0.0036, c_perp=0.0021)
    rec("spot fee BNB 0.075%+slip", c_spot=0.00095)
    rec("rebalance to 50/50 at each rebalance", rebal_to_w=True)
    rec("BTC only", w=(1.0, 0.0))
    rec("ETH only", w=(0.0, 1.0))
    rec("funding x0.75 (haircut)", fund_mult=0.75)
    rec("start 2020-07-01", start="2020-07-01")
    rec("start 2024-01-01 (fresh OOS entry)", start="2024-01-01")
    res["robustness"] = rows

    # BTC-only/ETH-only: w=(1,0) делит на ноль в другой монете — защищено ниже
    # top-10 days removal (OOS)
    sel = _win(base["ts"], *OOS)
    r_oos = base["ret"][sel].copy()
    order = np.argsort(r_oos)[::-1]
    r_drop = r_oos.copy()
    r_drop[order[:10]] = 0.0
    res["oos_drop_top10_days"] = metrics(base["ts"][sel], r_drop)
    res["oos_top10_days_pct"] = [round(float(x) * 100, 3) for x in r_oos[order[:10]]]
    # bootstrap-like: block sharpe stability, share of positive months excess
    ex = base["ret"] - rf
    res["oos_excess_monthly_pos_share"] = metrics(base["ts"][sel], ex[sel])["pos_months_pct"]

    # capacity: 1% средне-часового объёма (OOS), спот и перп
    hsel = (d["ts"] >= ms(OOS[0])) & (d["ts"] < ms(OOS[1]))
    res["capacity_usd_1pct_avg_hour_oos"] = {s: {"perp": round(float(d["FV"][hsel, i].mean()) * 0.01),
                                                 "spot": round(float(d["SV"][hsel, i].mean()) * 0.01),
                                                 "spot_p05_hour": round(float(np.percentile(d["SV"][hsel, i], 5)) * 0.01)}
                                             for i, s in enumerate(COINS)}  # fmt: skip
    # rolling funding APR (per coin, 90d) at end
    res["funding_apr_by_year"] = {}
    yrs = (d["ts"].astype("datetime64[ms]").astype("datetime64[Y]").astype(int) + 1970)
    for i, s in enumerate(COINS):
        res["funding_apr_by_year"][s] = {str(y): round(float(d["FR"][yrs == y, i].sum()) / (np.sum(yrs == y) / 24 / 365.25) * 100, 2)
                                         for y in np.unique(yrs)}  # fmt: skip
    res["elapsed_s"] = round(time.time() - t0, 1)
    (ODIR / "verify_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    print("saved", ODIR / "verify_results.json", f"{time.time() - t0:.1f}s")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "download":
        download()
    else:
        main_run()
