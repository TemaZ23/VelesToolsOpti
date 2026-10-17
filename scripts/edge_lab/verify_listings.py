"""
Независимая проверка (adversarial review) кандидата семейства listings:
шорт нового крипто-перпа через X часов после листинга, SL +S, TP -TP, выход по времени через H часов,
хедж лонгом BTCUSDT на тот же номинал, позиция = equity/K.

Реализация написана с нуля по текстовой спецификации, без чтения кода автора.
Данные — только common.load_hourly / load_funding (часовые свечи по символу, точные времена фандинга).

Запуск:  python3 -m scripts.edge_lab.verify_listings            (база + робастность)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.edge_lab.common import (  # noqa: E402
    DAY, HOUR, IS, OOS, OUT, TAKER_FEE, MAKER_FEE, cost_per_side, load_funding, load_hourly, metrics, ms,
    per_year, universe,
)

OUTD = OUT / "listings"
WIN = 24 * 30  # часов окна после листинга, которые храним по событию

# ----------------------------------------------------------------------------------------------
# Классификация (собственная, по названиям + проверка цен; не из кода автора)
# ----------------------------------------------------------------------------------------------
INDEX = {"BTCDOMUSDT", "DEFIUSDT", "FOOTBALLUSDT", "BLUEBIRDUSDT", "ALLUSDT"}
# переименования/миграции с разрывом < 60 дней и непрерывной ценой (старый -> новый)
RENAMES = {"RENDERUSDT": "RNDR", "POLUSDT": "MATIC", "SUSDT": "FTM", "AUSDT": "EOS", "FORMUSDT": "BNX",
           "COMBOUSDT": "COCOS", "GRAMUSDT": "TON", "1000BTTCUSDT": "BTT", "FRAXUSDT": "FXS", "SKYUSDT": "MKR",
           "DUSDT": "DAR", "KAIAUSDT": "KLAY", "MANTRAUSDT": "OM", "GUSDT": "GAL"}
TRADFI_BASES = """XAU XAG XPD XPT XAUT COPPER BZ CL NATGAS USDBRL
TSLA HOOD INTC AMZN CRCL MSTR COIN PLTR EWY EWJ PAYP GOOGL META NVDA AAPL QQQ SPY TSM MU SNDK AVGO BABA MSFT
AMD QCOM USAR CSCO DIS HD LITE ORCL UBER MRVL SOXL BRKB CRWV FLNC JPM V WMT DRAM RKLB CBRS SPCX OPENAI ARM BE
COHR NBIS WDC QNTX BBX EWT LLY NOK NVO HYUNDAI SAMSUNG SKHYNIX ANTHROPIC CRM DELL IBM IREN NOW ONDS AAOI AMAT
AXTI BX CRDO CRWD HPE IWM COST EBAY HIMS NFLX URNM DKNG EWZ GME RIVN XLE ZM ADBE ASML BMNR GLW STXX UVXY ALAB
CIEN KLAC KORU LRCX SMCI SONY MVLL SQQQ TQQQ CAT FLEX STRC TER TTWO TXN BNC FWDI INTW SNXX WEN XBI APP GEV SNOW
VRT SKHY MUU SOXS TZA MINIMAX ZHIPU PANW SOFI HK0700 HK1810 TENCENT POPMART BITO TBT TMF GS PYPL SMH GIGADEV KO
RDDT CSOPSAMSUNG2L CSOPSKHYNIX2L KUAISHOU MEITUAN HANMI KODEX200 LGELECTRONICS NAVER SAMSUNGEM ZHONGJI GDX LYTE
NET SHOP VST CXMT UNITREE DJT MRNA SKDD SKUU IONQ MARA MRK PDD TEM HK0625 DDOG GTLB MDB NVDL TEAM TSLL ZS GPRO BYD
HK0992 AGPU AMC ANET APLD HUT PATH OURA CVNA OKLO RUM TWST XOM ACN BWET CRML MP NKE UNH SECZ SHAZ BOT BSP KSTR
MOONSHOT PENG""".split()
TRADFI = {b + "USDT" for b in TRADFI_BASES}


def find_events(min_gap_days: int = 60) -> list[dict]:
    """Листинги: первый час с объёмом > 0; релистинги — возобновление торговли после ≥ min_gap_days без объёма."""
    u = universe()
    data_start = ms("2020-01-01")
    ev = []
    for s in sorted(u):
        d = load_hourly(s)
        t, q = d["ts"], d["qvol"]
        act = np.where(q > 0)[0]
        if len(act) == 0:
            continue
        starts = [act[0]] + list(act[1:][np.diff(t[act]) >= min_gap_days * DAY])
        for k, i in enumerate(starts):
            if t[i] <= data_start:  # торговался до начала данных — не листинг
                continue
            kind = "relist" if k > 0 else "new"
            cls = "crypto"
            if s in INDEX:
                cls = "index"
            elif s in TRADFI:
                cls = "tradfi"
            elif s in RENAMES and kind == "new":
                cls = "rename"
            ev.append({"sym": s, "L": int(t[i]), "kind": kind, "cls": cls})
    return sorted(ev, key=lambda e: (e["L"], e["sym"]))


# ----------------------------------------------------------------------------------------------
# Данные по событию: часовое окно, выровненное на сетку часов от L
# ----------------------------------------------------------------------------------------------
def event_window(sym: str, L: int, nh: int = WIN) -> dict:
    d = load_hourly(sym)
    t = d["ts"]
    grid = L + HOUR * np.arange(nh, dtype=np.int64)
    pos = np.searchsorted(t, grid)
    ok = (pos < len(t)) & (t[np.minimum(pos, len(t) - 1)] == grid)
    o = np.full(nh, np.nan); h = o.copy(); lo = o.copy(); c = o.copy(); qv = np.zeros(nh)
    p = pos[ok]
    o[ok], h[ok], lo[ok], c[ok] = d["ohlc"][p, 0], d["ohlc"][p, 1], d["ohlc"][p, 2], d["ohlc"][p, 3]
    qv[ok] = d["qvol"][p]
    act = qv > 0
    # нет торговли (нет бара или нулевой объём) -> бар «пустой»; цены не используем
    o[~act] = np.nan; h[~act] = np.nan; lo[~act] = np.nan; c[~act] = np.nan
    # фандинг: ставка, отнесённая к часу settle (floor), только settle строго после L
    f = load_funding(sym)
    fr = np.zeros(nh)
    if len(f["ts"]):
        fb = (f["ts"] - L) // HOUR
        m = (fb >= 0) & (fb < nh)
        np.add.at(fr, fb[m].astype(np.int64), f["rate"][m])
    # последний час с объёмом в истории символа (для делистинга)
    all_act = np.where(d["qvol"] > 0)[0]
    last_active = int(t[all_act[-1]])
    return {"o": o, "h": h, "l": lo, "c": c, "qv": qv, "fr": fr, "last_active": last_active}


class BTC:
    def __init__(self):
        d = load_hourly("BTCUSDT")
        self.t0 = int(d["ts"][0])
        n = int((d["ts"][-1] - self.t0) // HOUR) + 1
        idx = ((d["ts"] - self.t0) // HOUR).astype(np.int64)
        self.o = np.full(n, np.nan); self.c = np.full(n, np.nan); self.qv = np.zeros(n)
        self.o[idx] = d["ohlc"][:, 0]; self.c[idx] = d["ohlc"][:, 3]; self.qv[idx] = d["qvol"]
        # заполнение дыр предыдущей ценой (только прошлые значения)
        for a in (self.o, self.c):
            m = np.isnan(a)
            ii = np.where(~m, np.arange(n), 0)
            np.maximum.accumulate(ii, out=ii)
            a[:] = a[ii]
        self.fr = np.zeros(n)
        f = load_funding("BTCUSDT")
        fb = ((f["ts"] - self.t0) // HOUR).astype(np.int64)
        m = (fb >= 0) & (fb < n)
        np.add.at(self.fr, fb[m], f["rate"][m])
        cs = np.r_[0, np.cumsum(self.qv)]
        self.cs = cs

    def i(self, t_ms: int) -> int:
        return int((t_ms - self.t0) // HOUR)

    def adv(self, t_ms: int) -> float:
        """ADV (USD/сутки) по 720 прошлым часам до t (не включая бар t)."""
        k = self.i(t_ms)
        a = max(0, k - 720)
        return float((self.cs[k] - self.cs[a]) / max(1, k - a) * 24)


# ----------------------------------------------------------------------------------------------
# Сделка по одному событию (на единицу номинала)
# ----------------------------------------------------------------------------------------------
def coin_cost(w: dict, j: int, age_h: int, cmul: float) -> float:
    """Издержки на сторону при сделке в начале бара j (ADV по прошлым барам < j, максимум 720)."""
    a = max(0, j - 720)
    past = w["qv"][a:j]
    adv = float(np.mean(past) * 24) if len(past) else 0.0
    slip = cost_per_side(adv) - TAKER_FEE
    if 24 <= age_h <= 72:
        slip = max(slip, 0.0015)
    elif age_h < 24:
        slip = max(slip, 0.004)
    return (TAKER_FEE + slip) * cmul


def sim_trade(w: dict, L: int, btc: BTC, X: int, H: int, S: float, TP: float, cmul: float = 1.0,
              hedge: bool = True, stop_slip: float = 0.003, intrabar: str = "stop_first",
              dead_loss: float | None = None, stop_frac: float = 0.0, fund_entry_bar: bool = False):
    """Возвращает словарь: бары (часы от L) входа/выхода, путь P&L на единицу номинала по часам
    (на конец каждого часа, включая издержки и фандинг) и разложение."""
    e = X
    if e >= len(w["o"]) or np.isnan(w["o"][e]):
        # бар входа пустой — берём первый активный бар после X (в пределах 6 ч)
        nxt = np.where(~np.isnan(w["o"][e:e + 6]))[0]
        if len(nxt) == 0:
            return None
        e = e + int(nxt[0])
    pe = w["o"][e]
    stop, tpx = pe * (1 + S), pe * (1 - TP)
    last_bar = int((w["last_active"] - L) // HOUR)  # последний активный час (может быть > окна)
    x_planned = e + H  # выход по open бара e+H
    reason, xb, px = None, None, None
    # поиск выхода
    end = min(x_planned, len(w["o"]))
    for j in range(e, end):
        if j > last_bar:
            break
        hj, lj, oj = w["h"][j], w["l"][j], w["o"][j]
        if np.isnan(hj):
            continue
        hit_s = hj >= stop
        hit_t = lj < tpx
        if hit_s and hit_t:
            if intrabar == "stop_first":
                hit_t = False
            else:
                hit_s = False
        if hit_s:
            reason, xb = "stop", j
            px = max(oj, stop + stop_frac * (hj - stop)) * (1 + stop_slip)
            break
        if hit_t:
            reason, xb, px = "tp", j, tpx
            break
    if reason is None:
        if last_bar < x_planned:  # монета перестала торговаться до планового выхода
            xb = last_bar
            if xb >= len(w["c"]):
                return None
            px = w["c"][xb]
            reason = "dead"
        else:
            xb = x_planned
            if xb >= len(w["o"]):
                return None
            px = w["o"][xb]
            if np.isnan(px):  # пустой бар выхода — первый активный после
                nx = np.where(~np.isnan(w["o"][xb:]))[0]
                if len(nx) == 0:
                    return None
                xb = xb + int(nx[0]); px = w["o"][xb]
            reason = "time"
    # путь MTM: для бара j (e..xb) — значение на конец бара (для выхода по open — на начало бара xb)
    nb = xb - e + 1
    c = w["c"][e:xb + 1].copy()
    # пустые бары — переносим предыдущую цену
    for k in range(nb):
        if np.isnan(c[k]):
            c[k] = c[k - 1] if k > 0 else pe
    price_path = (pe - c) / pe
    price_path[-1] = (pe - px) / pe
    if reason == "dead" and dead_loss is not None:
        price_path[-1] = -dead_loss  # пессимистично: -50% на позицию
    # издержки
    age_e = e
    c_in = coin_cost(w, e, age_e, cmul)
    if reason == "tp":
        c_out = MAKER_FEE * cmul
    else:
        c_out = coin_cost(w, xb, xb, cmul)
    cost = np.zeros(nb)
    cost[0] -= c_in
    cost[-1] -= c_out * (px / pe)
    # фандинг: settle в барах e+1..xb (вход по open бара e — после settle этого бара);
    # для выхода по open бара xb settle в начале бара xb тоже считаем (консервативно для шорта при fr<0)
    fund = np.zeros(nb)
    for k in range(0 if fund_entry_bar else 1, nb):
        j = e + k
        r = w["fr"][j]
        if r != 0.0:
            pj = w["o"][j] if not np.isnan(w["o"][j]) else c[k - 1]
            fund[k] += r * pj / pe  # шорт получает +r
    pnl_coin = price_path.copy()
    pnl_coin[1:] = np.diff(price_path)
    inc = pnl_coin + cost + fund
    pre = w["qv"][max(0, e - 24):e]
    cap = 0.01 * float(np.mean(pre)) if len(pre) else 0.0  # 1% среднего часового объёма за 24 ч до входа
    both = int(np.sum((w["h"][e:xb + 1] >= stop) & (w["l"][e:xb + 1] < tpx)))
    out = {"e": e, "xb": xb, "reason": reason, "cap": cap, "both": both, "pe": pe, "px": px, "price": float(price_path[-1]),
           "cost": float(cost.sum()), "fund": float(fund.sum()), "inc": inc}
    # BTC хедж
    if hedge:
        te = L + e * HOUR
        bi = btc.i(te)
        be = btc.o[bi]
        if reason in ("time",):
            bx_i = bi + nb - 1
            bxp = btc.o[bx_i]
        else:  # стоп/тейк/делист внутри бара xb — закрываем BTC по close этого бара
            bx_i = bi + nb - 1
            bxp = btc.c[bx_i]
        bc = btc.c[bi:bi + nb].copy()
        bpath = (bc - be) / be
        bpath[-1] = (bxp - be) / be
        binc = bpath.copy(); binc[1:] = np.diff(bpath)
        cb = cost_per_side(btc.adv(te)) * cmul
        bcost = np.zeros(nb); bcost[0] -= cb; bcost[-1] -= cb * bxp / be
        bf = np.zeros(nb)
        for k in range(1, nb):
            r = btc.fr[bi + k]
            if r != 0.0:
                bf[k] -= r * btc.o[bi + k] / be  # лонг платит +r
        out["inc"] = inc + binc + bcost + bf
        out["btc"] = float(bpath[-1]); out["btc_cost"] = float(bcost.sum()); out["btc_fund"] = float(bf.sum())
    else:
        out["btc"] = out["btc_cost"] = out["btc_fund"] = 0.0
    out["total"] = float(out["inc"].sum())
    return out


# ----------------------------------------------------------------------------------------------
# Портфель: позиция = equity(в момент входа)/K, почасовая переоценка, дневные доходности
# ----------------------------------------------------------------------------------------------
T0, T1 = ms("2020-01-01"), ms("2026-10-01")
NH = (T1 - T0) // HOUR


def portfolio(trades: list[dict], K: float, drop_syms: set | None = None):
    pnl = np.zeros(NH + 400)
    order = sorted(trades, key=lambda r: r["t_entry"])
    eq0 = 1.0
    used = []
    for r in order:
        if drop_syms and r["sym"] in drop_syms:
            continue
        h0 = (r["t_entry"] - T0) // HOUR
        if h0 < 0 or h0 >= NH:
            continue
        eq = eq0 + pnl[:h0].sum()
        if eq <= 0:
            break
        n = eq / K
        inc = r["inc"]
        pnl[h0:h0 + len(inc)] += n * inc
        r["usd"] = float(n * inc.sum())
        used.append(r)
    pnl = pnl[:NH]
    eq = eq0 + np.cumsum(pnl)
    eq = np.maximum(eq, 1e-9)
    deq = np.r_[eq0, eq[23::24]]
    dret = deq[1:] / deq[:-1] - 1
    dts = T0 + DAY * np.arange(len(dret), dtype=np.int64)
    return dts, dret, used


def stats(dts, dret, used, window):
    a, b = ms(window[0]), ms(window[1])
    m = (dts >= a) & (dts < b)
    tr = [r for r in used if a <= r["t_entry"] < b]
    out = metrics(dts[m], dret[m], len(tr))
    if tr:
        tot = np.array([r["total"] for r in tr])
        out["avg_trade_pct"] = round(float(tot.mean()) * 100, 2)
        out["t_trade"] = round(float(tot.mean() / (tot.std(ddof=1) / np.sqrt(len(tot)))), 2) if len(tot) > 2 else 0.0
        out["stop_share"] = round(float(np.mean([r["reason"] == "stop" for r in tr])), 3)
        out["tp_share"] = round(float(np.mean([r["reason"] == "tp" for r in tr])), 3)
    return out


def half_years(dts, dret, start="2024-01-01"):
    out = {}
    y0 = int(start[:4])
    for y in range(y0, 2027):
        for hh, (m0, m1) in enumerate((("01", "07"), ("07", "13"))):
            a = ms(f"{y}-{m0}-01")
            b = ms(f"{y}-{m1}-01") if m1 != "13" else ms(f"{y + 1}-01-01")
            m = (dts >= a) & (dts < b) & (dts < T1)
            if m.sum() == 0:
                continue
            out[f"{y}H{hh + 1}"] = round((float(np.prod(1 + dret[m])) - 1) * 100, 1)
    return out


# ----------------------------------------------------------------------------------------------
def build_trades(events, wins, btc, X, H, S, TP, **kw):
    trades = []
    for ev in events:
        w = wins[(ev["sym"], ev["L"])]
        r = sim_trade(w, ev["L"], btc, X, H, S, TP, **kw)
        if r is None:
            continue
        r["sym"] = ev["sym"]; r["L"] = ev["L"]
        r["t_entry"] = ev["L"] + r["e"] * HOUR
        trades.append(r)
    return trades


def run(events, wins, btc, X=24, H=168, S=0.30, TP=0.25, K=20, drop_syms=None, **kw):
    tr = build_trades(events, wins, btc, X, H, S, TP, **kw)
    dts, dret, used = portfolio(tr, K, drop_syms)
    return {"is": stats(dts, dret, used, IS), "oos": stats(dts, dret, used, OOS), "dts": dts, "dret": dret,
            "used": used}


def main():
    t0 = time.time()
    events = find_events()
    cnt = {}
    for e in events:
        cnt[e["cls"]] = cnt.get(e["cls"], 0) + 1
    print("events", len(events), cnt, f"{time.time() - t0:.0f}s")
    crypto = [e for e in events if e["cls"] == "crypto"]
    wins = {(e["sym"], e["L"]): event_window(e["sym"], e["L"]) for e in events}
    btc = BTC()
    print("windows loaded", f"{time.time() - t0:.0f}s")
    res = {"events": cnt, "n_crypto": len(crypto),
           "n_crypto_is": sum(ms(IS[0]) <= e["L"] < ms(IS[1]) for e in crypto),
           "n_crypto_oos": sum(ms(OOS[0]) <= e["L"] < ms(OOS[1]) for e in crypto)}

    base = run(crypto, wins, btc)
    res["base"] = {"is": base["is"], "oos": base["oos"], "per_year": per_year(base["dts"], base["dret"]),
                   "half_years": half_years(base["dts"], base["dret"], "2020-01-01")}
    print("BASE IS ", base["is"]); print("BASE OOS", base["oos"])
    print("per_year", res["base"]["per_year"]); print("half", res["base"]["half_years"])
    used = base["used"]
    res["base"]["decomp_usd"] = {k: round(float(sum(r["usd"] * r[k] / r["total"] if r["total"] else 0 for r in used)), 3)
                                 for k in ("price", "cost", "fund", "btc", "btc_cost", "btc_fund")}
    res["base"]["exit_reasons"] = {k: int(sum(r["reason"] == k for r in used)) for k in ("time", "stop", "tp", "dead")}
    res["base"]["avg_hold_h"] = round(float(np.mean([r["xb"] - r["e"] for r in used])), 1)
    np.savez(OUTD / "verified_daily.npz", ts=base["dts"], ret=base["dret"])

    # --- экспозиция, ёмкость ---
    caps = np.array([r["cap"] for r in used])
    res["base"]["capacity_usd"] = {"median": round(float(np.median(caps))), "p10": round(float(np.percentile(caps, 10))),
                                   "oos_median": round(float(np.median([r["cap"] for r in used if r["t_entry"] >= ms(OOS[0])])))}
    cnt_open = np.zeros(NH + 400)
    for r in used:
        h0 = (r["t_entry"] - T0) // HOUR
        cnt_open[h0:h0 + r["xb"] - r["e"]] += 1
    cnt_open = cnt_open[:NH]
    res["base"]["open_positions"] = {"mean": round(float(cnt_open.mean()), 2), "max": int(cnt_open.max()),
                                     "mean_oos": round(float(cnt_open[(ms(OOS[0]) - T0) // HOUR:].mean()), 2)}
    res["base"]["ambiguous_bars_trades"] = int(sum(r["both"] > 0 for r in used))
    print("capacity", res["base"]["capacity_usd"], "open", res["base"]["open_positions"],
          "ambig", res["base"]["ambiguous_bars_trades"], "decomp", res["base"]["decomp_usd"], res["base"]["exit_reasons"])

    REC = ("2025-07-01", "2026-10-01")
    MID = ("2024-01-01", "2025-07-01")

    def summ(rr):
        return {"is_sh": rr["is"].get("sharpe"), "oos_sh": rr["oos"].get("sharpe"), "oos_cagr": rr["oos"].get("cagr_pct"),
                "oos_dd": rr["oos"].get("max_dd_pct"),
                "mid_sh": stats(rr["dts"], rr["dret"], rr["used"], MID).get("sharpe"),
                "rec_sh": stats(rr["dts"], rr["dret"], rr["used"], REC).get("sharpe"),
                "rec_cagr": stats(rr["dts"], rr["dret"], rr["used"], REC).get("cagr_pct"),
                "rec_trade": stats(rr["dts"], rr["dret"], rr["used"], REC).get("avg_trade_pct"),
                "oos_trade": rr["oos"].get("avg_trade_pct"), "oos_t": rr["oos"].get("t_trade")}

    res["base"]["mid"] = stats(base["dts"], base["dret"], base["used"], MID)
    res["base"]["recent"] = stats(base["dts"], base["dret"], base["used"], REC)
    print("MID", res["base"]["mid"]); print("RECENT", res["base"]["recent"])

    # --- ±25% по всем 4 параметрам (81 комбинация) ---
    grid = []
    for X in (18, 24, 30):
        for H in (126, 168, 210):
            for S in (0.225, 0.30, 0.375):
                for TP in (0.1875, 0.25, 0.3125):
                    rr = run(crypto, wins, btc, X, H, S, TP)
                    grid.append({"X": X, "H": H, "S": S, "TP": TP, **summ(rr)})
    oos = np.array([g["oos_sh"] for g in grid]); rec = np.array([g["rec_sh"] for g in grid])
    res["pm25"] = {"n": len(grid), "oos_median": float(np.median(oos)), "oos_min": float(oos.min()),
                   "oos_share_ge1": float(np.mean(oos >= 1)), "oos_share_pos": float(np.mean(oos > 0)),
                   "rec_median": float(np.median(rec)), "rec_share_pos": float(np.mean(rec > 0)), "grid": grid}
    print("pm25", {k: v for k, v in res["pm25"].items() if k != "grid"})

    # --- сдвиг времени входа ---
    res["entry_shift"] = {}
    for X in (12, 20, 22, 23, 25, 26, 28, 36, 48):
        res["entry_shift"][X] = summ(run(crypto, wins, btc, X=X))
    print("entry_shift", {k: (v["oos_sh"], v["rec_sh"]) for k, v in res["entry_shift"].items()})

    # --- издержки, исполнение, хедж, делистинг ---
    variants = {
        "cost_x2": dict(cmul=2.0), "cost_x3": dict(cmul=3.0),
        "stop_slip_1pct": dict(stop_slip=0.01), "stop_slip_2pct": dict(stop_slip=0.02),
        "stop_fill_25pct_into_bar": dict(stop_frac=0.25), "stop_fill_50pct_into_bar": dict(stop_frac=0.5),
        "tp_first_if_ambiguous": dict(intrabar="tp_first"), "fund_incl_entry_bar": dict(fund_entry_bar=True),
        "no_hedge": dict(hedge=False), "dead_minus50": dict(dead_loss=0.5),
        "pessimistic_combo": dict(cmul=2.0, stop_slip=0.01, stop_frac=0.25),
    }
    res["variants"] = {}
    for k, kw in variants.items():
        res["variants"][k] = summ(run(crypto, wins, btc, **kw))
        print(k, res["variants"][k])

    # --- без топ-5 монет по вкладу в P&L, без топ-10 дней ---
    contrib = {}
    for r in used:
        contrib[r["sym"]] = contrib.get(r["sym"], 0.0) + r["usd"]
    top5 = sorted(contrib, key=contrib.get, reverse=True)[:5]
    top5_oos = {}
    for r in used:
        if r["t_entry"] >= ms(OOS[0]):
            top5_oos[r["sym"]] = top5_oos.get(r["sym"], 0.0) + r["usd"]
    top5o = sorted(top5_oos, key=top5_oos.get, reverse=True)[:5]
    res["drop_top5"] = {"syms": top5, **summ(run(crypto, wins, btc, drop_syms=set(top5)))}
    res["drop_top5_oos"] = {"syms": top5o, **summ(run(crypto, wins, btc, drop_syms=set(top5o)))}
    print("drop_top5", res["drop_top5"]); print("drop_top5_oos", res["drop_top5_oos"])
    d2 = base["dret"].copy()
    for w_ in (IS, OOS):
        m = (base["dts"] >= ms(w_[0])) & (base["dts"] < ms(w_[1]))
        idx = np.where(m)[0]
        top = idx[np.argsort(d2[idx])[-10:]]
        d2[top] = 0.0
    res["drop_top10_days"] = {"is": metrics(base["dts"][(base["dts"] < ms(IS[1]))], d2[base["dts"] < ms(IS[1])]),
                              "oos": metrics(base["dts"][base["dts"] >= ms(OOS[0])], d2[base["dts"] >= ms(OOS[0])])}
    print("drop_top10_days", res["drop_top10_days"])

    # --- бутстрэп сделок OOS и последних 15 мес. (среднее на сделку) ---
    rng = np.random.default_rng(0)
    for nm, w_ in (("oos", OOS), ("recent", REC), ("is", IS)):
        tot = np.array([r["total"] for r in used if ms(w_[0]) <= r["t_entry"] < ms(w_[1])])
        bs = rng.choice(tot, (5000, len(tot))).mean(1)
        res[f"boot_{nm}"] = {"n": len(tot), "mean_pct": round(float(tot.mean()) * 100, 2),
                             "ci90": [round(float(np.percentile(bs, 5)) * 100, 2), round(float(np.percentile(bs, 95)) * 100, 2)],
                             "p_le0": round(float(np.mean(bs <= 0)), 4)}
        print("boot", nm, res[f"boot_{nm}"])

    # --- по кварталам: среднее на сделку и доля стопов ---
    q = {}
    for r in used:
        k = str(np.datetime64(r["t_entry"], "ms").astype("datetime64[M]"))
        k = k[:4] + "Q" + str((int(k[5:7]) - 1) // 3 + 1)
        q.setdefault(k, []).append(r)
    res["by_quarter"] = {k: {"n": len(v), "avg_trade_pct": round(float(np.mean([r["total"] for r in v])) * 100, 2),
                             "stop_share": round(float(np.mean([r["reason"] == "stop" for r in v])), 2)} for k, v in sorted(q.items())}
    print("by_quarter", res["by_quarter"])
    # --- неопределённость OOS Sharpe: блочный бутстрэп дневных доходностей (блок 20 дней) ---
    dts, dr = base["dts"], base["dret"]
    m = (dts >= ms(OOS[0])) & (dts < ms(OOS[1]))
    x = dr[m]
    nb = len(x) // 20
    shs = []
    for _ in range(3000):
        st = rng.integers(0, len(x) - 20, nb)
        y = np.concatenate([x[a:a + 20] for a in st])
        shs.append(y.mean() / y.std() * np.sqrt(365))
    shs = np.array(shs)
    res["oos_sharpe_block_boot"] = {"ci90": [round(float(np.percentile(shs, 5)), 2), round(float(np.percentile(shs, 95)), 2)],
                                    "p_ge1": round(float(np.mean(shs >= 1)), 3), "p_le0": round(float(np.mean(shs <= 0)), 3)}
    # OOS без лучшего полугодия (2025H1)
    m2 = m & ~((dts >= ms("2025-01-01")) & (dts < ms("2025-07-01")))
    res["oos_ex_2025H1"] = metrics(dts[m2], dr[m2])
    # тест на «распад»: средняя сделка 2020-01..2025-06 против 2025-07..2026-09
    a_ = np.array([r["total"] for r in used if r["t_entry"] < ms(REC[0])])
    b_ = np.array([r["total"] for r in used if r["t_entry"] >= ms(REC[0])])
    se = np.sqrt(a_.var(ddof=1) / len(a_) + b_.var(ddof=1) / len(b_))
    res["decay_test"] = {"early_n": len(a_), "early_mean_pct": round(float(a_.mean()) * 100, 2), "recent_n": len(b_),
                         "recent_mean_pct": round(float(b_.mean()) * 100, 2), "t_diff": round(float((a_.mean() - b_.mean()) / se), 2)}
    print("block boot", res["oos_sharpe_block_boot"], "oos_ex_2025H1", res["oos_ex_2025H1"], "decay", res["decay_test"])
    # без хеджа: по годам и полугодиям (для счёта < $2k хедж BTC невозможен: мин. ордер BTCUSDT 100 USDT)
    nh = run(crypto, wins, btc, hedge=False)
    res["no_hedge_detail"] = {"is": nh["is"], "oos": nh["oos"], "per_year": per_year(nh["dts"], nh["dret"]),
                              "half_years": half_years(nh["dts"], nh["dret"], "2024-01-01")}
    c2 = run(crypto, wins, btc, cmul=2.0)
    res["cost_x2_detail"] = {"per_year": per_year(c2["dts"], c2["dret"]), "half_years": half_years(c2["dts"], c2["dret"], "2024-01-01")}
    print("no_hedge_detail", res["no_hedge_detail"]); print("cost_x2_detail", res["cost_x2_detail"])
    json.dump(res, open(OUTD / "verify_results.json", "w"), indent=1, default=float)
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
