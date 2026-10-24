"""
Семейство «listings»: подготовка событий листинга / делистинга (часовые окна вокруг события).

    PYTHONPATH=. python3 -m scripts.edge_lab.listings_data

Особенности данных (проверено вручную, см. notes.md):
- в архиве Binance мёртвые контракты продолжаются «плоскими» барами с нулевым объёмом
  (SRM, FTT, HNT ... до 2024-05-28 или до конца архива), поэтому «живой» бар = qvol > 0;
  настоящих делистингов не 31 (флаг universe.json), а ~160;
- часть контрактов была приостановлена и позже залистирована заново (CVC, CTK, CVX, LIT, SLP, TLM, ICP):
  простой >= 60 суток => новый сегмент = событие «relist»;
- 2025-12 .. 2026-09: сотни TradFi-перпетуалов (акции, ETF, металлы, нефть, FX) — отдельная категория;
- переименования / миграции токенов (MATIC->POL, FTM->S, EOS->A ...) — не новые листинги.

Выход: cache/edge_lab/listings/events.npz — окна [E, W] от бара листинга L (относительный час 0..W-1),
cache/edge_lab/listings/meta.json — сегменты, категории, делистинги.
"""

from __future__ import annotations

import json
import time

import numpy as np

from scripts.edge_lab.common import CACHE, HOUR, load_funding, load_hourly, ms, panel, rolling_mean_past, universe

OUTDIR = CACHE / "listings"
T0 = ms("2020-01-01")
T1 = ms("2026-10-01")
NH = (T1 - T0) // HOUR
W = 2200  # часов в окне события (90 дней + запас на задержку входа)
DEAD_CUT = ms("2026-09-25")
RELIST_GAP_H = 24 * 60

INDEX = {"BTCDOMUSDT", "DEFIUSDT", "BLUEBIRDUSDT", "FOOTBALLUSDT", "USDCUSDT"}
# новый символ -> старый (миграция/ребрендинг; цена непрерывна, это не «новая монета»)
RENAMES = {
    "POLUSDT": "MATICUSDT", "RENDERUSDT": "RNDRUSDT", "SUSDT": "FTMUSDT", "AUSDT": "EOSUSDT",
    "FORMUSDT": "BNXUSDT", "GRAMUSDT": "TONUSDT", "COMBOUSDT": "COCOSUSDT", "SKYUSDT": "MKRUSDT",
    "GUSDT": "GALUSDT", "DUSDT": "DARUSDT", "KAIAUSDT": "KLAYUSDT", "VICUSDT": "TOMOUSDT",
    "1000BTTCUSDT": "BTTUSDT",
}  # fmt: skip
TRADFI = set("""
XAU XAG XPD XPT XAUT PAXG COPPER BZ CL NATGAS USDBRL TSLA HOOD INTC AMZN CRCL MSTR COIN PLTR EWY EWJ PAYP
GOOGL META NVDA AAPL QQQ SPY TSM MU SNDK AVGO BABA MSFT AMD QCOM USAR CSCO DIS HD LITE ORCL UBER MRVL SOXL
BRKB CRWV FLNC JPM V WMT DRAM RKLB CBRS SPCX OPENAI ARM BE COHR NBIS WDC QNTX ASTS BBX EWT LLY NOK NVO
HYUNDAI SAMSUNG SKHYNIX ANTHROPIC CRM DELL IBM IREN NOW ONDS AAOI AMAT AXTI BX CRDO CRWD HPE IWM COST
EBAY HIMS NFLX URNM DKNG EWZ GME RIVN XLE ZM ADBE ASML BMNR GLW STXX UVXY ALAB CIEN KLAC KORU LRCX SMCI
SONY MVLL SQQQ TQQQ CAT FLEX KSTR STRC TER TTWO TXN BNC FWDI INTW SNXX XBI APP GEV SNOW VRT SKHY MUU SOXS
TZA MINIMAX ZHIPU PANW PENG SHAZ SOFI HK0700 HK1810 TENCENT POPMART BITO TBT TMF GS PYPL SMH GIGADEV KO
RDDT CSOPSAMSUNG2L CSOPSKHYNIX2L KUAISHOU MEITUAN HANMI KODEX200 LGELECTRONICS NAVER SAMSUNGEM ZHONGJI GDX
LYTE NET SHOP VST CXMT UNITREE DJT MRNA RAM SKDD SKUU IONQ MARA MRK PDD TEM HK0625 DDOG GTLB MDB NVDL TEAM
TSLL ZS GPRO BYD HK0992 AGPU AMC ANET APLD HUT PATH MOONSHOT OURA CVNA OKLO RUM TWST XOM ACN BWET CRML MP
NKE SECZ UNH DATAIP BSP
""".split())  # fmt: skip


def category(sym: str) -> str:
    base = sym[:-4]
    if sym in INDEX:
        return "index"
    if sym in RENAMES:
        return "rename"
    if base in TRADFI:
        return "tradfi"
    return "crypto"


def _grid(sym: str) -> dict:
    d = load_hourly(sym)
    gi = (d["ts"] - T0) // HOUR
    ok = (gi >= 0) & (gi < NH)
    gi = gi[ok].astype(np.int64)
    o, h, l, c = (d["ohlc"][ok, k] for k in range(4))
    qv = d["qvol"][ok]
    f = load_funding(sym)
    fg = (f["ts"] - T0) // HOUR
    fok = (fg >= 0) & (fg < NH)
    return {"gi": gi, "o": o, "h": h, "l": l, "c": c, "qv": qv, "fgi": fg[fok].astype(np.int64), "frate": f["rate"][fok]}


def _segments(gi: np.ndarray, qv: np.ndarray) -> list[tuple[int, int]]:
    act = gi[qv > 0]
    if len(act) == 0:
        return []
    br = np.where(np.diff(act) > RELIST_GAP_H)[0]
    starts = np.r_[act[0], act[br + 1]]
    ends = np.r_[act[br], act[-1]]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _window(g: dict, L: int, end: int, n: int = W, start_off: int = 0) -> dict[str, np.ndarray]:
    """Окно [L+start_off, L+start_off+n) по глобальной часовой сетке; бары вне сегмента и с нулевым объёмом — NaN."""
    a = L + start_off
    out = {k: np.full(n, np.nan, np.float32) for k in ("o", "h", "l", "c", "qv")}
    fund = np.zeros(n, np.float32)
    m = (g["gi"] >= a) & (g["gi"] < a + n) & (g["gi"] <= end) & (g["qv"] > 0)
    r = g["gi"][m] - a
    for k in ("o", "h", "l", "c", "qv"):
        out[k][r] = g[k][m]
    fm = (g["fgi"] >= a) & (g["fgi"] < a + n) & (g["fgi"] <= end)
    np.add.at(fund, g["fgi"][fm] - a, g["frate"][fm])
    out["f"] = fund
    return out


def main() -> None:
    t_start = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    uni = universe()
    syms = sorted(uni)
    meta, events, delists = {}, [], []
    grids = {}
    for s in syms:
        g = _grid(s)
        segs = _segments(g["gi"], g["qv"])
        cat = category(s)
        meta[s] = {"cat": cat, "segments": segs}
        for k, (a, b) in enumerate(segs):
            pre_existing = k == 0 and a < 72  # торговался до начала архива (2020-01-01)
            if not pre_existing:
                events.append({"sym": s, "L": a, "end": b, "relist": k > 0, "cat": cat})
            if b * HOUR + T0 < DEAD_CUT and s not in RENAMES.values():
                delists.append({"sym": s, "end": b, "start": a, "cat": cat})
        if cat in ("crypto", "rename") or s == "BTCUSDT":
            grids[s] = g
    print(f"символов {len(syms)}, событий листинга {len(events)}, делистингов {len(delists)}", flush=True)

    # BTC на глобальной сетке (пропуски — ffill)
    gb = grids["BTCUSDT"]
    btc = {k: np.full(NH, np.nan) for k in ("o", "h", "l", "c", "qv")}
    for k in btc:
        btc[k][gb["gi"]] = gb[k]
    for k in ("o", "h", "l", "c"):
        v = btc[k]
        idx = np.where(~np.isnan(v), np.arange(NH), 0)
        np.maximum.accumulate(idx, out=idx)
        btc[k] = v[idx]
    btc_f = np.zeros(NH)
    np.add.at(btc_f, gb["fgi"], gb["frate"])

    E = len(events)
    arr = {k: np.full((E, W), np.nan, np.float32) for k in ("o", "h", "l", "c", "qv")}
    arr["f"] = np.zeros((E, W), np.float32)
    for i, ev in enumerate(events):
        g = grids.get(ev["sym"]) or _grid(ev["sym"])
        w = _window(g, ev["L"], ev["end"])
        for k in arr:
            arr[k][i] = w[k]

    # Контроль: та же дата входа, случайная «зрелая» криптомонета (возраст >= 365 д, ADV30 >= $5 млн
    # по данным до дня листинга, жива на дату). Три зерна.
    p = panel("1d")
    dts = p["ts"]
    psyms = list(p["symbols"])
    alive_d = np.nan_to_num(p["qvol"]) > 0
    age_d = np.cumsum(alive_d, axis=0)
    adv_d = rolling_mean_past(np.where(alive_d, p["qvol"], np.nan), 30)
    is_crypto = np.array([category(s) == "crypto" for s in psyms])
    # Контроль 2 (по волатильности): возраст >= 180 д, ADV30 >= $5 млн, верхние 20% по 30-дневной
    # волатильности дневных доходностей (по данным до дня листинга).
    lr = np.diff(np.log(np.where(alive_d, p["close"], np.nan)), axis=0, prepend=np.nan)
    lr = np.where(alive_d, lr, np.nan)
    vol_d = np.sqrt(rolling_mean_past(lr**2, 30))
    rng = np.random.default_rng(12345)
    n_seeds = 3
    ctrls = {}
    for kind in ("ct", "cv"):
        ctrl = {k: np.full((n_seeds, E, W), np.nan, np.float32) for k in ("o", "h", "l", "c", "qv")}
        ctrl["f"] = np.zeros((n_seeds, E, W), np.float32)
        ctrl_sym = np.full((n_seeds, E), "", dtype=object)
        for i, ev in enumerate(events):
            tL = T0 + ev["L"] * HOUR
            di = int(np.searchsorted(dts, tL, side="right")) - 2  # прошлый полный день
            if di < 0:
                continue
            if kind == "ct":
                cand = np.where(alive_d[di] & (age_d[di] >= 365) & (np.nan_to_num(adv_d[di]) >= 5e6) & is_crypto)[0]
            else:
                base = alive_d[di] & (age_d[di] >= 180) & (np.nan_to_num(adv_d[di]) >= 5e6) & is_crypto & np.isfinite(vol_d[di])
                cand = np.where(base)[0]
                if len(cand) >= 10:
                    thr = np.percentile(vol_d[di, cand], 80)
                    cand = cand[vol_d[di, cand] >= thr]
            cand = [j for j in cand if psyms[j] != ev["sym"]]
            if not cand:
                continue
            pick = rng.choice(cand, size=n_seeds, replace=len(cand) < n_seeds)
            for sd in range(n_seeds):
                s2 = psyms[int(pick[sd])]
                g = grids.get(s2) or _grid(s2)
                segs = meta[s2]["segments"]
                end = next((b for a, b in segs if a <= ev["L"] <= b), segs[-1][1])
                w = _window(g, ev["L"], end)
                for k in ctrl:
                    ctrl[k][sd, i] = w[k]
                ctrl_sym[sd, i] = s2
        ctrls[kind] = (ctrl, ctrl_sym)
    ctrl, ctrl_sym = ctrls["ct"]
    ctrl2, ctrl2_sym = ctrls["cv"]

    # Делистинги: окно 60 дней до последнего живого бара (+ BTC берётся по глобальной сетке)
    DW = 24 * 60
    D = len(delists)
    darr = {k: np.full((D, DW + 1), np.nan, np.float32) for k in ("o", "h", "l", "c", "qv")}
    darr["f"] = np.zeros((D, DW + 1), np.float32)
    for i, ev in enumerate(delists):
        g = grids.get(ev["sym"]) or _grid(ev["sym"])
        w = _window(g, ev["end"], ev["end"], n=DW + 1, start_off=-DW)
        for k in darr:
            darr[k][i] = w[k]

    np.savez_compressed(
        OUTDIR / "events.npz",
        **{f"ev_{k}": v for k, v in arr.items()},
        **{f"ct_{k}": v for k, v in ctrl.items()},
        **{f"cv_{k}": v for k, v in ctrl2.items()}, cv_sym=ctrl2_sym.astype(str),
        **{f"dl_{k}": v for k, v in darr.items()},
        ev_L=np.array([e["L"] for e in events]), ev_end=np.array([e["end"] for e in events]),
        ev_sym=np.array([e["sym"] for e in events]), ev_cat=np.array([e["cat"] for e in events]),
        ev_relist=np.array([e["relist"] for e in events]), ct_sym=ctrl_sym.astype(str),
        dl_end=np.array([e["end"] for e in delists]), dl_start=np.array([e["start"] for e in delists]),
        dl_sym=np.array([e["sym"] for e in delists]), dl_cat=np.array([e["cat"] for e in delists]),
        btc_o=btc["o"], btc_h=btc["h"], btc_l=btc["l"], btc_c=btc["c"], btc_qv=btc["qv"], btc_f=btc_f,
        t0=np.int64(T0),
    )  # fmt: skip
    (OUTDIR / "meta.json").write_text(json.dumps(meta))
    cats = {}
    for e in events:
        cats[e["cat"]] = cats.get(e["cat"], 0) + 1
    print("категории событий:", cats, "relist:", sum(e["relist"] for e in events))
    print(f"готово за {time.time() - t_start:.0f} с")


if __name__ == "__main__":
    main()
