"""
(d) Проверка микроструктуры на минутках: окна 1m-свечей перпа [s-60м, s+120м) вокруг выплат с экстремальным
фандингом (|f_prev| или |f_cur| >= 0.2%, ADV >= $10M). Нужна, чтобы понять, успевает ли бот войти ПОСЛЕ
выплаты до основного движения (вход не раньше открытия минуты s+1).

    PYTHONPATH=. python3 -m scripts.edge_lab.funding_m1 fetch   # data.binance.vision daily 1m klines -> окна
    PYTHONPATH=. python3 -m scripts.edge_lab.funding_m1 study   # профиль и стратегия «после выплаты»
"""

from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from scripts.edge_lab.common import CACHE, IS, OOS, cost_per_side, metrics, ms, per_year, universe
from scripts.edge_lab.funding import OUTD, half_years, save_json, summarize_configs
from scripts.edge_lab.funding_micro import load_records
from scripts.edge_lab.universe import BASE, _get

WIN = CACHE / "funding" / "derived" / "m1_windows.npz"
PRE, POST = 60, 120
MIN = 60_000
DAYMS = 86_400_000


def events():
    D = load_records()
    m = (D["adv"] >= 1e7) & ((np.abs(D["f_prev"]) >= 0.002) | (np.abs(D["f_cur"]) >= 0.002))
    idx = np.where(m)[0]
    return D, idx


def _day_rows(sym: str, day: int) -> np.ndarray | None:
    ds = str(np.datetime64(int(day) * DAYMS, "ms").astype("datetime64[D]"))
    blob = _get(f"{BASE}data/futures/um/daily/klines/{sym}/1m/{sym}-1m-{ds}.zip")
    if not blob:
        return None
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        text = zf.read(zf.namelist()[0]).decode()
    if text and not text[:1].isdigit():
        text = text[text.find("\n") + 1:]
    if not text.strip():
        return None
    a = np.loadtxt(io.StringIO(text), delimiter=",", usecols=(0, 1, 2, 3, 4, 7), ndmin=2)
    if len(a) and a[0, 0] > 1e14:
        a[:, 0] = np.floor(a[:, 0] / 1000)
    return a


def fetch(threads: int = 10) -> None:
    D, idx = events()
    syms = sorted(universe())
    ev_ts = D["ts"][idx].astype(np.int64)
    ev_sym = D["sym"][idx].astype(np.int64)
    need: dict[tuple[int, int], list[int]] = {}
    for n, (s, t) in enumerate(zip(ev_sym, ev_ts)):
        for d in {int((t - PRE * MIN) // DAYMS), int((t + POST * MIN - MIN) // DAYMS)}:
            need.setdefault((int(s), d), []).append(n)
    print("событий", len(idx), "сим-дней", len(need), flush=True)
    W = np.full((len(idx), PRE + POST, 5), np.nan, dtype=np.float32)
    t0 = time.time()
    with ThreadPoolExecutor(threads) as ex:
        futs = {ex.submit(_day_rows, syms[s], d): (s, d) for s, d in need}
        for k, f in enumerate(as_completed(futs), 1):
            key = futs[f]
            try:
                a = f.result()
            except Exception:  # noqa: BLE001
                a = None
            if a is not None:  # сразу раскладываем по окнам событий, сам день не храним
                for n in need[key]:
                    rel = ((a[:, 0] - (ev_ts[n] - PRE * MIN)) // MIN).astype(np.int64)
                    ok = (rel >= 0) & (rel < PRE + POST)
                    W[n, rel[ok], :] = a[ok][:, [1, 2, 3, 4, 5]]
            if k % 500 == 0:
                print(k, len(need), f"{time.time() - t0:.0f}s", flush=True)
    np.savez(WIN, idx=idx, W=W)
    print("сохранено", WIN, "заполнено окон", int(np.isfinite(W[:, PRE, 0]).sum()), "из", len(idx))


def study() -> None:
    D, idx0 = events()
    with np.load(WIN) as z:
        idx, W = z["idx"], z["W"].astype(np.float64)
    assert np.array_equal(idx, idx0)
    O, H, Lo, C, V = (W[:, :, k] for k in range(5))
    ts = D["ts"][idx]
    fprev, fcur, adv, iv = D["f_prev"][idx], D["f_cur"][idx], D["adv"][idx], D["interval_h"][idx]
    is_m = ts < ms(IS[1])
    p0 = O[:, PRE]  # первая цена после выплаты (open минуты s)
    res = {"n_events": int(len(idx)), "filled": int(np.isfinite(p0).sum())}
    # профиль: накопленная доходность от open минуты s до close минуты s+k, а также до выплаты
    prof = {}
    for name, m in (("neg_prev", fprev <= -0.002), ("pos_prev", fprev >= 0.002), ("neg_cur", fcur <= -0.002),
                    ("pos_cur", fcur >= 0.002)):  # fmt: skip
        m = m & np.isfinite(p0) & (adv >= 1e7)
        row = {}
        for per, pm in (("is", is_m), ("oos", ~is_m)):
            mm = m & pm
            pts = {}
            for k in (-60, -30, -10, -5, -1):
                pts[f"pre{k}"] = round(float(np.nanmean(p0[mm] / O[mm, PRE + k] - 1)) * 1e4, 1)
            for k in (0, 1, 2, 5, 10, 30, 55, 59, 119):
                pts[f"m{k}"] = round(float(np.nanmean(C[mm, PRE + k] / p0[mm] - 1)) * 1e4, 1)
            pts["n"] = int(mm.sum())
            row[per] = pts
        prof[name] = row
        print(name, row, flush=True)
    res["profile_bp"] = prof
    # стратегия: после выплаты в сторону ставки (f<0 -> шорт), вход по open минуты s+e, выход по close s+x
    days0 = int(ms("2020-01-01") // DAYMS)
    dts = np.arange(days0, int(ms("2026-10-01") // DAYMS), dtype=np.int64) * DAYMS
    rows = []
    for A in (1e7, 5e7, 2e8):
        for thr in (0.002, 0.003, 0.005):
            for selby in ("prev", "cur"):
                for sign in ("neg", "both"):
                    for e in (1, 2, 5):
                        for x in (14, 29, 54):
                            for ivs in ("all", "4-8h"):
                                f = fprev if selby == "prev" else fcur
                                sel = (adv >= A) & ((f <= -thr) if sign == "neg" else (np.abs(f) >= thr))
                                if ivs == "4-8h":
                                    sel &= iv >= 4
                                pin = O[:, PRE + e]
                                pout = C[:, PRE + x]
                                direction = np.sign(f)  # f<0 -> шорт: в сторону ставки (перезаход толпы после выплаты)
                                gross = direction * (pout / pin - 1)
                                net = gross - 2 * cost_per_side(adv)
                                ok = sel & np.isfinite(net)
                                if ok.sum() < 30:
                                    continue
                                e_ts, e_net, e_f = ts[ok], net[ok], np.abs(f[ok])
                                order = np.lexsort((-e_f, e_ts))
                                e_ts, e_net = e_ts[order], e_net[order]
                                first = np.r_[True, e_ts[1:] != e_ts[:-1]]
                                grp = np.cumsum(first) - 1
                                rank = np.arange(len(e_ts)) - np.where(first)[0][grp]
                                keep = rank < 5  # не больше 5 событий на одну выплату, 20% капитала на событие
                                e_ts, e_net = e_ts[keep], e_net[keep]
                                di = (e_ts // DAYMS).astype(np.int64) - days0
                                inr = (di >= 0) & (di < len(dts))
                                dr = np.bincount(di[inr], 0.2 * e_net[inr], minlength=len(dts))[: len(dts)]
                                st = {"cfg": dict(A=A, thr=thr, selby=selby, sign=sign, e=e, x=x, iv=ivs)}
                                for per, win in (("is", IS), ("oos", OOS)):
                                    a, b = np.searchsorted(dts, ms(win[0])), np.searchsorted(dts, ms(win[1]))
                                    ev = (e_ts >= ms(win[0])) & (e_ts < ms(win[1]))
                                    st[per] = metrics(dts[a:b], dr[a:b], int(ev.sum()))
                                    st[per]["mean_event_bp"] = round(float(np.mean(e_net[ev])) * 1e4, 2) if ev.any() else None
                                    st[per]["median_event_bp"] = round(float(np.median(e_net[ev])) * 1e4, 2) if ev.any() else None
                                    st[per]["win_rate"] = round(float(np.mean(e_net[ev] > 0)), 3) if ev.any() else None
                                st["per_year"] = per_year(dts, dr)
                                st["oos_half_years"] = half_years(dts, dr)
                                rows.append(st)
    # «через выплату» реалистично: вход за k минут до выплаты против знака ставки (получаем фандинг), выход по
    # close минуты s+j (j=0 — первая минута после выплаты). Отбор только по f_prev.
    rows_thr = []
    for A in (1e7, 5e7, 2e8):
        for thr in (0.001, 0.002, 0.005):
            for k in (1, 5, 15):
                for j in (0, 1, 4):
                    sel = (adv >= A) & (np.abs(fprev) >= thr)
                    dirn = -np.sign(fprev)
                    gross = dirn * (C[:, PRE + j] / O[:, PRE - k] - 1) - dirn * fcur
                    net = gross - 2 * cost_per_side(adv)
                    st = {"cfg": dict(A=A, thr=thr, k=k, j=j)}
                    for per, pm in (("is", is_m), ("oos", ~is_m)):
                        ok = sel & pm & np.isfinite(net)
                        st[per] = {"n": int(ok.sum()), "mean_gross_bp": round(float(np.mean(gross[ok])) * 1e4, 2),
                                   "mean_net_bp": round(float(np.mean(net[ok])) * 1e4, 2),
                                   "mean_funding_bp": round(float(np.mean(np.abs(fcur[ok]))) * 1e4, 2),
                                   "win_rate": round(float(np.mean(net[ok] > 0)), 3)} if ok.any() else {"n": 0}  # fmt: skip
                    rows_thr.append(st)
    res["through_realistic"] = rows_thr
    for r in rows_thr:
        if r["cfg"]["A"] == 5e7:
            print("through", r["cfg"], r["is"], r["oos"])
    res["summary"] = summarize_configs(rows)
    res["summary_by_iv"] = {v: summarize_configs([r for r in rows if r["cfg"]["iv"] == v]) for v in ("all", "4-8h")}
    best = max(rows, key=lambda r: r["is"]["sharpe"])
    res["best_is"] = best
    keys = ("A", "thr", "selby", "sign", "e", "x", "iv")
    res["best_neighbors"] = [{"cfg": r["cfg"], "is_sh": r["is"]["sharpe"], "oos_sh": r["oos"]["sharpe"],
                              "oos_cagr": r["oos"]["cagr_pct"]} for r in rows
                             if sum(r["cfg"][k] != best["cfg"][k] for k in keys) == 1]  # fmt: skip
    res["top15_is"] = sorted(rows, key=lambda r: -r["is"]["sharpe"])[:15]
    res["configs"] = rows
    save_json("part_m1.json", res)
    print(json.dumps({"summary": res["summary"], "by_iv": res["summary_by_iv"]}, indent=1))
    for r in res["top15_is"]:
        print(r["cfg"], "IS", r["is"]["sharpe"], r["is"]["cagr_pct"], r["is"]["mean_event_bp"], r["is"]["trades"], "OOS",
              r["oos"]["sharpe"], r["oos"]["cagr_pct"], r["oos"]["mean_event_bp"], r["oos"]["median_event_bp"],
              r["oos"]["win_rate"], r["oos"]["trades"], r["oos"]["max_dd_pct"])  # fmt: skip


if __name__ == "__main__":
    {"fetch": fetch, "study": study}[sys.argv[1] if len(sys.argv) > 1 else "study"]()
