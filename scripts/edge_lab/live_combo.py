"""
Бумажная торговля (paper trading) комбинированного плана из output/edge_lab/REPORT.md.

ТОЛЬКО ПЕЧАТАЕТ намеченные ордера. Ключей API нет и не нужно: используются только публичные эндпоинты
Binance USDT-M (exchangeInfo, klines, ticker/price, fundingRate). Кода отправки ордеров здесь нет.

Рукава (доли капитала — веса inverse-vol по IS из output/edge_lab/combined.json):
  1) xsec  — Amihud illiquidity long/short, ежедневно после 00:05 UTC:
       универсум: USDT-M крипто-перпы в статусе TRADING (без TradFi/индексов/стейблов), >= 60 дневных баров
       с объёмом, ADV30 (средний дневной quote-объём за 30 дней, включая вчерашний закрытый день) >= $5M;
       ILLIQ = среднее за 30 дней |r_d| / QV_d (r — close-to-close), >= 21 валидных дней;
       n >= 20 -> k = floor(0.1 n): лонг k самых неликвидных по +0.5/k, шорт k самых ликвидных по -0.5/k;
       цель = среднее 7 последних дневных портфелей; сделка по рынку сразу после 00:00 UTC.
  2) listings — шорт нового крипто-перпа через 24 ч после листинга (onboardDate), SL +30% (STOP_MARKET),
       TP -25% (лимит reduce-only), выход по рынку через 168 ч; одновременно лонг BTCUSDT на тот же номинал;
       номинал = equity_listings * leverage / 20. Запускать раз в час (в начале часа).
  3) carry-check — справка: годовой фандинг BTC/ETH за 30 дней на капитал cash-and-carry (маржа 0.5,
       капитал = 1.5 x номинал) против ставки кэша (--cash-rate). Это НЕ край, а «куда положить свободный USDT».

Примеры:
  python3 -m scripts.edge_lab.live_combo --equity 5000 --leverage 1
  python3 -m scripts.edge_lab.live_combo --equity 5000 --positions my_positions.json   # {"SYMBOL": signed_usd}
  python3 -m scripts.edge_lab.live_combo --offline 2026-09-29        # самопроверка на локальном кеше (без сети)

Самопроверка --offline сравнивает целевые веса xsec, посчитанные этим скриптом из «сырых» дневных рядов, с
весами независимой реализации верификатора (verify_xsec.signals) на ту же дату; расхождение должно быть 0.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAPI = "https://fapi.binance.com"
DAY = 86_400_000
HOUR = 3_600_000

# параметры кандидатов (как в бэктесте; не менять без нового IS-теста)
XS = dict(win=30, minv=21, adv_min=5e6, age_min=60, q=0.1, H=7, nmin=20)
LS = dict(X_h=24, H_h=168, stop=0.30, tp=0.25, K=20)
DEFAULT_W_XSEC = 0.2627  # из combined.json -> weights_is_inverse_vol.main_risk_2way
MIN_ORDER_USD = 10.0     # не отправлять сделки меньше этого (минимальный номинал Binance $5, у ETH $20, у BTC $100)

STABLES = {"USDC", "FDUSD", "TUSD", "BUSD", "USDP", "DAI", "USDE", "USD1", "RLUSD", "EUR", "EURI", "USDS", "PYUSD"}


def _noncrypto_names() -> tuple[set, set, set]:
    from scripts.edge_lab.verify_listings import INDEX, RENAMES, TRADFI_BASES
    from scripts.edge_lab.verify_xsec import NAMED_NONCRYPTO
    return set(NAMED_NONCRYPTO) | set(TRADFI_BASES) | STABLES, set(INDEX), set(RENAMES)


# ------------------------------------------------------------------------------------------------
# расчёт целевых весов xsec из матриц дневных данных (одинаково для online и offline)
# ------------------------------------------------------------------------------------------------
def xsec_targets(C: np.ndarray, QV: np.ndarray, excluded: np.ndarray, alive_now: np.ndarray, p=XS):
    """C, QV — [D, N] дневные close и quote-объём (NaN/0 — монета не торговалась), последняя строка —
    последний ЗАКРЫТЫЙ день t. excluded — [N] не-крипто. alive_now — [N] торгуется ли сейчас.
    Возвращает целевые веса W [N] (доли капитала рукава при плече 1) и диагностику."""
    D, N = C.shape
    C = np.where(np.isfinite(QV) & (QV > 0), C, np.nan)
    QV = np.where(np.isfinite(C), QV, np.nan)
    prev = np.vstack([np.full((1, N), np.nan), C[:-1]])
    r = C / prev - 1
    ok = np.isfinite(r) & (QV > 0)
    x = np.where(ok, np.abs(r) / np.where(QV > 0, QV, np.nan), np.nan)
    alive = np.isfinite(C)
    age = np.cumsum(alive, 0)
    P = []
    info = []
    for s in range(D - p["H"], D):
        lo = max(0, s - p["win"] + 1)
        xs = x[lo:s + 1]
        cnt = np.isfinite(xs).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            score = np.where(cnt >= p["minv"], np.nansum(xs, 0) / np.maximum(cnt, 1), np.nan)
            qv = np.where(alive[lo:s + 1], QV[lo:s + 1], np.nan)
            cq = np.isfinite(qv).sum(0)
            adv = np.where(cq > 0, np.nansum(qv, 0) / np.maximum(cq, 1), np.nan)
        elig = alive[s] & (age[s] >= p["age_min"]) & (np.nan_to_num(adv) >= p["adv_min"]) & np.isfinite(score)
        elig &= ~excluded
        Ps = np.zeros(N)
        idx = np.flatnonzero(elig)
        n = len(idx)
        k = int(math.floor(p["q"] * n)) if n >= p["nmin"] else 0
        if k >= 1:
            o = idx[np.argsort(score[idx], kind="stable")]
            Ps[o[-k:]] += 0.5 / k
            Ps[o[:k]] -= 0.5 / k
        P.append(Ps)
        info.append({"n": n, "k": k})
    W = np.sum(P, 0) / p["H"]
    W = np.where(alive_now, W, 0.0)
    return W, info


# ------------------------------------------------------------------------------------------------
# сеть (публичные эндпоинты)
# ------------------------------------------------------------------------------------------------
def get(path: str, params: dict | None = None, retries: int = 3):
    q = "&".join(f"{k}={v}" for k, v in (params or {}).items())
    url = f"{FAPI}{path}" + (f"?{q}" if q else "")
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            if i == retries - 1:
                raise RuntimeError(f"GET {url}: {e}") from e
            time.sleep(1.5 * (i + 1))


def filters(sym: dict) -> dict:
    f = {x["filterType"]: x for x in sym.get("filters", [])}
    return {"step": float(f.get("MARKET_LOT_SIZE", f.get("LOT_SIZE", {})).get("stepSize", 0) or
                          f.get("LOT_SIZE", {}).get("stepSize", 0)),
            "tick": float(f.get("PRICE_FILTER", {}).get("tickSize", 0)),
            "min_notional": float(f.get("MIN_NOTIONAL", {}).get("notional", 5))}


def is_noncrypto_online(sym: dict, names: set) -> bool:
    base = sym["baseAsset"].upper()
    if base in names or sym["symbol"][:-4] in names:
        return True
    ut = str(sym.get("underlyingType", "COIN")).upper()
    if ut not in ("COIN", ""):
        return True
    sub = " ".join(map(str, sym.get("underlyingSubType", []))).upper()
    return any(t in sub for t in ("TRADFI", "STOCK", "EQUITY", "COMMODITY", "METAL", "FOREX", "INDEX"))


def weekend_ratio_noncrypto(ts: np.ndarray, C: np.ndarray) -> bool:
    """Та же эвристика, что в бэктесте: у акций/ETF |r| по выходным < 0.55 от будних."""
    r = np.abs(C[1:] / C[:-1] - 1)
    we = (((ts[1:] // DAY) + 3) % 7) >= 5
    ok = np.isfinite(r)
    if (we & ok).sum() > 3 and (~we & ok).sum() > 3:
        return float(np.mean(r[we & ok]) / max(np.mean(r[~we & ok]), 1e-12)) < 0.55
    return False


def online_data(now_ms: int):
    names, index, renames = _noncrypto_names()
    ex = get("/fapi/v1/exchangeInfo")
    syms = [s for s in ex["symbols"] if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"]
    last_closed = (now_ms // DAY) * DAY - DAY  # открытие последнего закрытого дневного бара
    D = 100
    days = last_closed - DAY * np.arange(D - 1, -1, -1, dtype=np.int64)
    N = len(syms)
    C = np.full((D, N), np.nan)
    QV = np.zeros((D, N))
    excl = np.zeros(N, bool)
    meta = []
    for j, s in enumerate(syms):
        kl = get("/fapi/v1/klines", {"symbol": s["symbol"], "interval": "1d", "limit": 99,
                                     "endTime": last_closed + DAY - 1})
        for k in kl:
            t = int(k[0])
            if int(k[6]) >= now_ms:  # незакрытый бар
                continue
            i = int((t - days[0]) // DAY)
            if 0 <= i < D:
                C[i, j] = float(k[4])
                QV[i, j] = float(k[7])
        onboard = int(s.get("onboardDate", 0))
        excl[j] = is_noncrypto_online(s, names) or s["symbol"] in index
        if not excl[j] and onboard >= int(np.datetime64("2025-12-01", "ms").astype(np.int64)):
            excl[j] = weekend_ratio_noncrypto(days, C[:, j])
        meta.append({"symbol": s["symbol"], "onboard": onboard, "delivery": int(s.get("deliveryDate", 0)),
                     **filters(s), "is_rename": s["symbol"] in renames,
                     "noncrypto": bool(is_noncrypto_online(s, names) or s["symbol"] in index)})
        time.sleep(0.03)
    return {"days": days, "C": C, "QV": QV, "excl": excl, "alive_now": np.ones(N, bool), "meta": meta}


def prices_now() -> dict:
    return {x["symbol"]: float(x["price"]) for x in get("/fapi/v1/ticker/price")}


# ------------------------------------------------------------------------------------------------
# печать ордеров
# ------------------------------------------------------------------------------------------------
def round_step(q: float, step: float) -> float:
    return math.floor(abs(q) / step) * step * (1 if q >= 0 else -1) if step > 0 else q


def print_xsec_orders(W, meta, px, equity_sleeve, lev, positions):
    tgt = {m["symbol"]: W[j] * equity_sleeve * lev for j, m in enumerate(meta) if W[j] != 0}
    names = set(tgt) | set(positions)
    rows = []
    for s in sorted(names):
        cur = float(positions.get(s, 0.0))
        t = tgt.get(s, 0.0)
        d = t - cur
        if abs(d) < MIN_ORDER_USD and t != 0:
            continue
        m = next((x for x in meta if x["symbol"] == s), None)
        price = px.get(s)
        qty = round_step(d / price, m["step"]) if (price and m) else float("nan")
        mn = m["min_notional"] if m else 5.0
        flag = "" if abs(d) >= mn else f"  < minNotional {mn:g}: пропустить или копить"
        if m and m["delivery"] and m["delivery"] < 4_000_000_000_000:
            flag += "  [!] назначен делистинг " + str(np.datetime64(m["delivery"], "ms"))[:16]
        rows.append((s, cur, t, d, qty, flag))
    print(f"\n=== xsec: Amihud L/S, рукав ${equity_sleeve:,.0f} x{lev:g}; целевых позиций {len(tgt)}; "
          f"ордеров {len(rows)} (порог ${MIN_ORDER_USD:g}) ===")
    if equity_sleeve * lev < 3000:
        print("[!] номинал рукава < $3000: доборы H=7 меньше минимальных ордеров; в прогнозе combined.json "
              "(режим realistic) рукав xsec при таком размере НЕ торгуется — держите эту долю в кэше")
    print(f"{'symbol':16s} {'текущая $':>11s} {'цель $':>11s} {'ордер $':>11s} {'qty':>14s}")
    for s, cur, t, d, q, flag in rows:
        side = "BUY " if d > 0 else "SELL"
        print(f"{s:16s} {cur:11.2f} {t:11.2f} {side}{abs(d):7.2f} {q:14.6g}{flag}")
    g_long = sum(v for v in tgt.values() if v > 0)
    g_short = -sum(v for v in tgt.values() if v < 0)
    print(f"итого лонг ${g_long:,.2f}, шорт ${g_short:,.2f}; исполнять по рынку сразу после 00:00 UTC")


def listing_plan(events, now_ms, equity_sleeve, lev, px=None, btc_min=100.0):
    """events: [{'symbol','L'(ms)}] крипто-листинги. Печатает действия на текущий час."""
    notional = equity_sleeve * lev / LS["K"]
    print(f"\n=== listings: шорт новых перпов, рукав ${equity_sleeve:,.0f} x{lev:g}, номинал сделки ${notional:,.2f} ===")
    if notional < btc_min:
        print(f"[!] номинал < ${btc_min:g} (мин. ордер BTCUSDT): хедж BTC невозможен. Без хеджа OOS Sharpe 1.06, "
              "последние 15 мес. ~0 — лучше увеличить рукав или не торговать этот рукав.")
    any_ev = False
    for e in sorted(events, key=lambda x: x["L"]):
        entry = e["L"] + LS["X_h"] * HOUR
        exit_ = entry + LS["H_h"] * HOUR
        if now_ms < e["L"] or now_ms >= exit_ + HOUR:
            continue
        any_ev = True
        st = "ожидание входа"
        if entry <= now_ms < entry + HOUR:
            st = "ВХОД СЕЙЧАС"
        elif entry + HOUR <= now_ms < exit_:
            st = "позиция открыта (если вход был)"
        elif exit_ <= now_ms < exit_ + HOUR:
            st = "ВЫХОД ПО ВРЕМЕНИ СЕЙЧАС"
        line = (f"{e['symbol']:16s} листинг {str(np.datetime64(e['L'], 'ms'))[:16]}  вход {str(np.datetime64(entry, 'ms'))[:16]}"
                f"  выход {str(np.datetime64(exit_, 'ms'))[:16]}  -> {st}")
        print(line)
        if st == "ВХОД СЕЙЧАС" and px:
            p = px.get(e["symbol"])
            if p:
                print(f"    SELL {e['symbol']} market ${notional:,.2f} (~{notional / p:.6g} шт.); "
                      f"STOP_MARKET closePosition @ {p * (1 + LS['stop']):.6g}; reduce-only BUY LIMIT @ {p * (1 - LS['tp']):.6g}")
                print(f"    BUY  BTCUSDT market ${notional:,.2f} (хедж; закрыть вместе с шортом)")
    if not any_ev:
        print("нет листингов в окне [листинг, листинг+24ч+168ч]")


def carry_check(cash_rate: float, fund: dict):
    print("\n=== carry-check (справка, не край) ===")
    for s, rates in fund.items():
        apr_notional = float(np.sum(rates)) * 365 / 30
        apr_capital = apr_notional / 1.5
        verdict = "выше кэша" if apr_capital > cash_rate else "НИЖЕ кэша -> держать USDT в Simple Earn/кэше"
        print(f"{s}: фандинг за 30 дн. {apr_notional * 100:5.2f}% годовых на номинал, {apr_capital * 100:5.2f}% на капитал "
              f"(маржа 0.5) против кэша {cash_rate * 100:.2f}% -> {verdict}")


# ------------------------------------------------------------------------------------------------
# offline: самопроверка на локальном кеше
# ------------------------------------------------------------------------------------------------
def offline(date: str, equity: float, lev: float, w_x: float, positions: dict):
    from scripts.edge_lab import verify_xsec as vx
    from scripts.edge_lab.common import load_funding, ms

    d = vx.load_data()  # дневная панель без «зомби»-баров + маска не-крипто, как в бэктесте
    ts = d["ts"]
    t = int(np.searchsorted(ts, ms(date)))
    assert ts[t] == ms(date), "дата вне панели"
    W_ref, _, _ = vx.signals(d)
    lo = max(0, t - 99)
    C = d["C"][lo:t + 1]
    QV = np.nan_to_num(d["QV"][lo:t + 1])
    # «торгуется сейчас» = есть бар t+1 (в бэктесте цель исполняется по open[t+1] только для живых монет)
    alive_now = np.isfinite(d["O"][t + 1]) if t + 1 < len(ts) else np.isfinite(d["C"][t])
    W, info = xsec_targets(C, QV, d["noncrypto"], alive_now)
    # возраст в бэктесте считается от 2020-01-01; окно 100 дней даёт то же условие age>=60, если монета
    # торгуется >= 60 дней внутри окна — проверяем совпадение
    ref = np.where(alive_now, W_ref[t], 0.0)
    diff = float(np.max(np.abs(W - ref)))
    print(f"[offline {date}] xsec: n={info[-1]['n']} k={info[-1]['k']}, позиций {int((W != 0).sum())}, "
          f"max|W - W_backtest| = {diff:.2e} {'OK' if diff < 1e-12 else 'РАСХОЖДЕНИЕ'}")
    syms = [str(s) for s in d["symbols"]]
    meta = [{"symbol": s, "step": 0.0, "tick": 0.0, "min_notional": 5.0, "delivery": 0} for s in syms]
    px = {s: float(d["C"][t, j]) for j, s in enumerate(syms) if np.isfinite(d["C"][t, j])}
    print_xsec_orders(W, meta, px, equity * w_x, lev, positions)
    # listings: события из той же разметки, что в бэктесте
    from scripts.edge_lab.verify_listings import find_events
    now_ms = ms(date) + DAY  # момент запуска — 00:00 UTC следующего дня
    ev = [{"symbol": e["sym"], "L": e["L"]} for e in find_events() if e["cls"] == "crypto"
          and now_ms - 9 * DAY <= e["L"] <= now_ms]
    listing_plan(ev, now_ms, equity * (1 - w_x), lev)
    fund = {}
    for s in ("BTCUSDT", "ETHUSDT"):
        f = load_funding(s)
        m = (f["ts"] > now_ms - 30 * DAY) & (f["ts"] <= now_ms)
        fund[s] = f["rate"][m]
    carry_check(0.04, fund)
    return diff


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--equity", type=float, default=5000.0, help="весь капитал плана, USDT")
    ap.add_argument("--leverage", type=float, default=1.0, help="множитель экспозиции обоих рукавов (рекомендовано 1)")
    ap.add_argument("--w-xsec", type=float, default=DEFAULT_W_XSEC, help="доля капитала в рукаве xsec")
    ap.add_argument("--positions", type=str, default=None, help="JSON {symbol: текущий номинал USD со знаком}")
    ap.add_argument("--cash-rate", type=float, default=0.04, help="доступная ставка по USDT (доля в год)")
    ap.add_argument("--offline", type=str, default=None, help="YYYY-MM-DD: самопроверка на локальном кеше")
    ap.add_argument("--skip-xsec", action="store_true")
    a = ap.parse_args()
    positions = json.loads(Path(a.positions).read_text()) if a.positions else {}
    print("РЕЖИМ БУМАЖНОЙ ТОРГОВЛИ: ордера только печатаются, ничего не отправляется.")
    if a.leverage > 2:
        print(f"[!] плечо {a.leverage:g} > 2: в стресс-сценарии вероятность просадки -50% за 5 лет >= 23% (см. REPORT.md)")
    if a.offline:
        offline(a.offline, a.equity, a.leverage, a.w_xsec, positions)
        return
    now_ms = int(time.time() * 1000)
    try:
        px = prices_now()
        if not a.skip_xsec:
            od = online_data(now_ms)
            W, info = xsec_targets(od["C"], od["QV"], od["excl"], od["alive_now"])
            print(f"xsec: универсум n={info[-1]['n']}, k={info[-1]['k']}")
            print_xsec_orders(W, od["meta"], px, a.equity * a.w_xsec, a.leverage, positions)
            meta = od["meta"]
        else:
            ex = get("/fapi/v1/exchangeInfo")
            nm_, ix_, rn_ = _noncrypto_names()
            meta = [{"symbol": s["symbol"], "onboard": int(s.get("onboardDate", 0)), "is_rename": s["symbol"] in rn_,
                     "noncrypto": bool(is_noncrypto_online(s, nm_) or s["symbol"] in ix_)}
                    for s in ex["symbols"] if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT"
                    and s.get("status") == "TRADING"]
        # новые листинги: только крипто; TradFi-перп в первые дни эвристикой выходных не отличить —
        # проверяйте категорию монеты вручную перед входом
        ev = [{"symbol": m["symbol"], "L": (m["onboard"] // HOUR) * HOUR} for m in meta
              if m["onboard"] >= now_ms - 9 * DAY and not m.get("noncrypto") and not m.get("is_rename")]
        listing_plan(ev, now_ms, a.equity * (1 - a.w_xsec), a.leverage, px)
        print("  (релистинги после >= 60 дней простоя скрипт не видит — проверяйте анонсы Binance вручную)")
        fund = {}
        for s in ("BTCUSDT", "ETHUSDT"):
            fr = get("/fapi/v1/fundingRate", {"symbol": s, "startTime": now_ms - 30 * DAY, "limit": 1000})
            fund[s] = np.array([float(x["fundingRate"]) for x in fr])
        carry_check(a.cash_rate, fund)
    except RuntimeError as e:
        print(f"Ошибка сети: {e}\n(Binance API недоступен с этого IP? Проверьте доступ или используйте --offline.)")
        sys.exit(2)


if __name__ == "__main__":
    main()
