"""
(d) Секундная проверка: успевает ли розничный бот забрать фандинг «через выплату» и выйти до того, как цена
сдвинется на размер ставки. Минутки (funding_m1.py) показали, что основное движение (~|ставка|) проходит в
первую минуту после выплаты, но не показали, за сколько секунд. Здесь — сделки (aggTrades) в окне ±120 с.

    PYTHONPATH=. python3 -m scripts.edge_lab.funding_tick fetch   # выборка событий -> окна сделок (кеш)
    PYTHONPATH=. python3 -m scripts.edge_lab.funding_tick study   # профиль по секундам и P&L «через выплату»

Выборка: выплаты с |f_prev| >= 0.2% (ставка ПРЕДЫДУЩЕЙ выплаты — известна заранее), ADV перпа >= $50M,
не в 00:00 UTC (чтобы окно лежало в одном дневном файле), не более одного события на символ-день,
случайная стратифицированная выборка IS/OOS. Сделка «через выплату»: вход против знака ставки (получаем
фандинг) по последней сделке за k секунд до выплаты, выход по первой сделке не раньше s + d секунд.
"""

from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from scripts.edge_lab.common import CACHE, IS, cost_per_side, ms, universe
from scripts.edge_lab.funding import OUTD, save_json
from scripts.edge_lab.funding_micro import load_records
from scripts.edge_lab.universe import BASE, _get

WIN = CACHE / "funding" / "derived" / "tick_windows.npz"
HALF = 120_000  # мс
MAX_ZIP = 60e6  # большие файлы (самые ликвидные дни) пропускаем ради трафика
POST = (0, 250, 500, 1000, 2000, 5000, 10000, 30000, 60000)  # мс после выплаты (окно — до +120 с)
PRE = (1000, 5000, 30000, 60000)  # мс до выплаты


def sample(n_is: int = 90, n_oos: int = 160, seed: int = 7):
    D = load_records()
    hour = (D["ts"] % 86_400_000) // 3_600_000
    m = (D["adv"] >= 5e7) & (np.abs(D["f_prev"]) >= 0.002) & (hour != 0) & np.isfinite(D["f_cur"])
    idx = np.where(m)[0]
    day = (D["ts"][idx] // 86_400_000).astype(np.int64)
    key = D["sym"][idx].astype(np.int64) * 100_000 + day
    _, first = np.unique(key, return_index=True)  # одно событие на символ-день
    idx = idx[first]
    rng = np.random.default_rng(seed)
    is_m = D["ts"][idx] < ms(IS[1])
    a = rng.choice(idx[is_m], size=min(n_is, int(is_m.sum())), replace=False)
    b = rng.choice(idx[~is_m], size=min(n_oos, int((~is_m).sum())), replace=False)
    return D, np.sort(np.r_[a, b])


def _line_time(buf: bytes, pos: int) -> tuple[int, int]:
    """Время сделки (transact_time) первой полной строки, начинающейся после pos; (-1, len) если нет."""
    nl = buf.find(b"\n", pos)
    if nl < 0:
        return -1, len(buf)
    end = buf.find(b"\n", nl + 1)
    line = buf[nl + 1: end if end > 0 else len(buf)]
    parts = line.split(b",")
    if len(parts) < 6 or not parts[5].strip().isdigit():
        return -1, nl + 1
    t = int(parts[5])
    return (t // 1000 if t > 1e14 else t), nl + 1


def _find(buf: bytes, t: int) -> int:
    lo, hi = 0, len(buf)
    while hi - lo > 4096:
        mid = (lo + hi) // 2
        tm, _ = _line_time(buf, mid)
        if tm < 0 or tm >= t:
            hi = mid
        else:
            lo = mid
    return lo


def window(sym: str, s: int) -> np.ndarray | None:
    ds = str(np.datetime64(int(s), "ms").astype("datetime64[D]"))
    blob = _get(f"{BASE}data/futures/um/daily/aggTrades/{sym}/{sym}-aggTrades-{ds}.zip")
    if not blob or len(blob) > MAX_ZIP:
        return None
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        buf = zf.read(zf.namelist()[0])
    del blob
    a, b = _find(buf, s - HALF), _find(buf, s + HALF)
    b = buf.find(b"\n", min(len(buf), b + 4096 * 64))  # с запасом
    chunk = buf[a: b if b > 0 else len(buf)]
    rows = []
    for line in chunk.split(b"\n"):
        p = line.split(b",")
        if len(p) < 7 or not p[5].strip().isdigit():
            continue
        t = int(p[5])
        t = t // 1000 if t > 1e14 else t
        if s - HALF <= t < s + HALF:
            rows.append((t - s, float(p[1]), float(p[2])))
    return np.array(rows) if rows else None


def fetch(threads: int = 4) -> None:
    D, idx = sample()
    syms = sorted(universe())
    print("событий в выборке", len(idx), flush=True)
    out: dict[int, np.ndarray] = {}
    t0 = time.time()
    with ThreadPoolExecutor(threads) as ex:
        futs = {ex.submit(window, syms[int(D["sym"][i])], int(D["ts"][i])): int(i) for i in idx}
        for k, f in enumerate(as_completed(futs), 1):
            i = futs[f]
            try:
                w = f.result()
            except Exception as exc:  # noqa: BLE001
                print("err", i, repr(exc)[:120], flush=True)
                w = None
            if w is not None and len(w) >= 20:
                out[i] = w
            if k % 25 == 0:
                print(k, len(idx), "ok", len(out), f"{time.time() - t0:.0f}s", flush=True)
    keys = np.array(sorted(out), dtype=np.int64)
    lens = np.array([len(out[k]) for k in keys], dtype=np.int64)
    flat = np.concatenate([out[k] for k in keys]) if len(keys) else np.empty((0, 3))
    np.savez(WIN, idx=keys, lens=lens, trades=flat, sample=idx)
    print("сохранено", WIN, "окон", len(keys), "сделок", len(flat), f"{time.time() - t0:.0f}s")


def _px_after(w: np.ndarray, d: int) -> float:
    j = np.searchsorted(w[:, 0], d, side="left")
    return float(w[j, 1]) if j < len(w) else np.nan


def _px_before(w: np.ndarray, d: int) -> float:
    j = np.searchsorted(w[:, 0], -d, side="left") - 1  # последняя сделка строго раньше s - d
    return float(w[j, 1]) if j >= 0 else np.nan


def study() -> None:
    D = load_records()
    with np.load(WIN) as z:
        keys, lens, flat = z["idx"], z["lens"], z["trades"]
    starts = np.r_[0, np.cumsum(lens)[:-1]]
    n = len(keys)
    fprev, fcur, adv, ts = D["f_prev"][keys], D["f_cur"][keys], D["adv"][keys], D["ts"][keys]
    dirn = -np.sign(fprev)  # против знака ставки: получаем фандинг
    pre = np.full((n, len(PRE)), np.nan)
    post = np.full((n, len(POST)), np.nan)
    last = np.full(n, np.nan)
    for e in range(n):
        w = flat[starts[e]: starts[e] + lens[e]]
        w = w[np.argsort(w[:, 0], kind="stable")]
        last[e] = _px_before(w, 0)
        for k, dd in enumerate(PRE):
            pre[e, k] = _px_before(w, dd)
        for k, dd in enumerate(POST):
            post[e, k] = _px_after(w, dd)
    is_m = ts < ms(IS[1])
    res = {"n_windows": int(n), "n_is": int(is_m.sum()), "n_oos": int((~is_m).sum()),
           "sample_rule": "|f_prev|>=0.2%, ADV>=$50M, not 00:00 UTC, 1 event per symbol-day, random"}  # fmt: skip
    # профиль: направленная доходность от последней сделки до выплаты к цене через d мс (б.п.)
    prof = {}
    for name, pm in (("is", is_m), ("oos", ~is_m), ("all", np.ones(n, bool))):
        r = dirn[pm, None] * (post[pm] / last[pm, None] - 1) * 1e4
        prof[name] = {f"+{d}ms": round(float(np.nanmean(r[:, k])), 1) for k, d in enumerate(POST)}
        prof[name]["median_+1000ms"] = round(float(np.nanmedian(r[:, POST.index(1000)])), 1)
        prof[name]["abs_fcur_bp"] = round(float(np.mean(np.abs(fcur[pm]))) * 1e4, 1)
        prof[name]["n"] = int(pm.sum())
        print(name, prof[name], flush=True)
    res["profile_dir_bp_vs_last_pre"] = prof
    # P&L «через выплату»: вход по последней сделке за k до s, выход по первой сделке после s + d
    cps = cost_per_side(adv)
    table = []
    for kk, k in enumerate(PRE):
        for dk, d in enumerate(POST):
            gross = dirn * (post[:, dk] / pre[:, kk] - 1) - dirn * fcur
            row = {"enter_before_ms": k, "exit_after_ms": d}
            for name, pm in (("is", is_m), ("oos", ~is_m)):
                g = gross[pm]
                ok = np.isfinite(g)
                net_t = g[ok] - 2 * cps[pm][ok]  # тейкер обе ноги
                net_m = g[ok] - 0.0002 - cps[pm][ok]  # вход мейкером (оптимистично: всегда исполнен), выход тейкером
                se = float(np.std(net_t) / np.sqrt(ok.sum())) if ok.sum() > 1 else np.nan
                row[name] = {"n": int(ok.sum()), "gross_bp": round(float(np.mean(g[ok])) * 1e4, 1),
                             "net_taker_bp": round(float(np.mean(net_t)) * 1e4, 1),
                             "net_taker_t": round(float(np.mean(net_t) / se), 2) if se and se > 0 else None,
                             "net_maker_entry_bp": round(float(np.mean(net_m)) * 1e4, 1),
                             "win_rate_taker": round(float(np.mean(net_t > 0)), 3)}  # fmt: skip
            table.append(row)
    res["through_table"] = table
    for r in table:
        if r["enter_before_ms"] in (5000, 60000):
            print("enter -%5d exit +%6d | IS gross %6.1f net %6.1f (t %5s) mk %6.1f | OOS gross %6.1f net %6.1f (t %5s) mk %6.1f"
                  % (r["enter_before_ms"], r["exit_after_ms"], r["is"]["gross_bp"], r["is"]["net_taker_bp"],
                     r["is"]["net_taker_t"], r["is"]["net_maker_entry_bp"], r["oos"]["gross_bp"], r["oos"]["net_taker_bp"],
                     r["oos"]["net_taker_t"], r["oos"]["net_maker_entry_bp"]), flush=True)  # fmt: skip
    # «после выплаты по направлению перезахода»: вход по первой сделке после s+e, выход после s+x, сторона = знак
    # ставки (f<0 -> шорт, т.к. уклонявшиеся шорты перезаходят) — для бота с задержкой ~0.5–2 с
    mom = []
    for e_ms in (500, 1000, 2000):
        for x_ms in (5000, 10000, 30000, 60000):
            ge = -dirn * (post[:, POST.index(x_ms)] / post[:, POST.index(e_ms)] - 1)
            row = {"enter_after_ms": e_ms, "exit_after_ms": x_ms}
            for name, pm in (("is", is_m), ("oos", ~is_m)):
                g = ge[pm]
                ok = np.isfinite(g)
                net = g[ok] - 2 * cps[pm][ok]
                se = float(np.std(net) / np.sqrt(ok.sum())) if ok.sum() > 1 else np.nan
                row[name] = {"n": int(ok.sum()), "gross_bp": round(float(np.mean(g[ok])) * 1e4, 1),
                             "net_taker_bp": round(float(np.mean(net)) * 1e4, 1),
                             "net_taker_t": round(float(np.mean(net) / se), 2) if se and se > 0 else None}  # fmt: skip
            mom.append(row)
            print("post-mom enter +%5d exit +%6d | IS gross %6.1f net %6.1f (t %5s) | OOS gross %6.1f net %6.1f (t %5s)"
                  % (e_ms, x_ms, row["is"]["gross_bp"], row["is"]["net_taker_bp"], row["is"]["net_taker_t"],
                     row["oos"]["gross_bp"], row["oos"]["net_taker_bp"], row["oos"]["net_taker_t"]), flush=True)  # fmt: skip
    res["post_momentum_table"] = mom
    res["mean_cost_per_side_bp"] = round(float(np.mean(cps)) * 1e4, 1)
    # сколько сделок проходит в первую секунду (конкуренция)
    first_sec = np.array([np.sum((flat[starts[e]: starts[e] + lens[e], 0] >= 0) &
                                 (flat[starts[e]: starts[e] + lens[e], 0] < 1000)) for e in range(n)])  # fmt: skip
    res["trades_in_first_second_median"] = float(np.median(first_sec))
    save_json("part_tick.json", res)


if __name__ == "__main__":
    {"fetch": fetch, "study": study}[sys.argv[1] if len(sys.argv) > 1 else "study"]()
