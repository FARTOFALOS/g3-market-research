"""Пересчёт 2026-09-23: достигается ли дальняя граница RIZ чаще, чем обычная линия
на том же расстоянии от цены.

Объект: уникальный уровень (минута T0, цена дальней границы, сторона), T0 2021–2025.
Расстояние = |t0_close − дальняя граница|. Плацебо A: случайная минута тех же лет,
линия на том же расстоянии и с той же стороны от её закрытия. Плацебо B: то же,
но минута подобрана по децили среднего размаха 30 минут и по квинтилю размаха
самой свечи, как у T0. Горизонты в записанных минутах ленты.

Volume не читается. Запуск из корня репозитория:
    python -B base/022/far_reach_placebo.py
Результат: base/022/far_reach_placebo.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.ndimage import maximum_filter1d, minimum_filter1d

REPO = Path(__file__).resolve().parents[2]
COLS = ["zone_top", "zone_bottom", "t0_spine_pos", "t0_ts_ns", "t0_exit_side", "t0_close"]
HORIZONS = {"60": 60, "1380": 1380, "6900": 6900, "27600": 27600}


def fwd(arr, w, fn):
    """fn по окну (t+1 .. t+w)."""
    out = fn(arr, size=w, origin=-(w // 2), mode="nearest")
    res = np.empty_like(arr)
    res[:-1] = out[1:]
    res[-1] = arr[-1]
    return res


def suffix(arr, ufunc):
    r = ufunc.accumulate(arr[::-1])[::-1]
    out = np.empty_like(arr)
    out[:-1] = r[1:]
    out[-1] = np.nan
    return out


def reach(mn, mx, pos, north, level):
    return np.where(north, mn[pos] <= level, mx[pos] >= level)


def run(inst: str) -> dict:
    mk = REPO / "data" / "market" / inst
    H, L, C = (np.load(mk / f"{n}.npy") for n in ("high", "low", "close"))
    ts = np.load(mk / "close_ts_utc_ns.npy")
    N = len(C)
    P = pd.concat([pq.read_table(REPO / "data" / "field" / inst / "cells" / f"tf_{tf:04d}" / "passports.parquet",
                                 columns=COLS).to_pandas() for tf in range(1, 1441)], ignore_index=True)
    P["year"] = pd.to_datetime(P.t0_ts_ns, utc=True).dt.year
    P = P[(P.year >= 2021) & (P.year <= 2025)]
    north = (P.t0_exit_side == "north").to_numpy()
    far = np.where(north, P.zone_bottom, P.zone_top)
    P = P.assign(north=north, far=far).drop_duplicates(["t0_spine_pos", "far", "north"])
    t0 = P.t0_spine_pos.to_numpy().astype(np.int64)
    north, far = P.north.to_numpy(), P.far.to_numpy()
    dist = np.abs(P.t0_close.to_numpy() - far)

    rng = H - L
    r30 = pd.Series(rng).rolling(30, min_periods=30).mean().to_numpy()
    yr = pd.to_datetime(ts, utc=True).year
    pool = np.flatnonzero((yr >= 2021) & (yr <= 2025) & ~np.isnan(r30))
    rs = np.random.default_rng(11)
    pick_a = rs.choice(pool, size=len(t0), replace=True)

    ok_t0 = ~np.isnan(r30[t0])
    q = np.quantile(r30[t0][ok_t0], np.linspace(0, 1, 11))
    q2 = np.quantile(rng[t0], np.linspace(0, 1, 6))
    key = lambda p: (np.clip(np.searchsorted(q, r30[p], side="right") - 1, 0, 9) * 10
                     + np.clip(np.searchsorted(q2, rng[p], side="right") - 1, 0, 4))
    k_t0, k_pool = key(t0), key(pool)
    pick_b = np.full(len(t0), -1)
    for b in np.unique(k_t0[ok_t0]):
        idx = np.flatnonzero((k_t0 == b) & ok_t0)
        cand = pool[k_pool == b]
        if len(cand):
            pick_b[idx] = rs.choice(cand, size=len(idx))
    ok_b = pick_b >= 0
    pb = np.maximum(pick_b, 0)

    lvl_a = np.where(north, C[pick_a] - dist, C[pick_a] + dist)
    lvl_b = np.where(north, C[pb] - dist, C[pb] + dist)
    out = {"unique_levels": int(len(t0)), "median_distance_points": float(np.median(dist)), "horizons_minutes": {}}
    for name, w in HORIZONS.items():
        mn, mx = fwd(L, w, minimum_filter1d), fwd(H, w, maximum_filter1d)
        oka = (t0 + w < N) & (pick_a + w < N)
        okb = ok_b & (t0 + w < N) & (pb + w < N)
        out["horizons_minutes"][name] = {
            "riz_far_pct": round(100 * float(reach(mn, mx, t0, north, far)[oka].mean()), 1),
            "line_random_minute_pct": round(100 * float(reach(mn, mx, pick_a, north, lvl_a)[oka].mean()), 1),
            "riz_far_pct_matched_support": round(100 * float(reach(mn, mx, t0, north, far)[okb].mean()), 1),
            "line_same_volatility_and_candle_pct": round(100 * float(reach(mn, mx, pb, north, lvl_b)[okb].mean()), 1)}
    smn, smx = suffix(L, np.minimum), suffix(H, np.maximum)
    out["horizons_minutes"]["archive_end"] = {
        "riz_far_pct": round(100 * float(reach(smn, smx, t0, north, far).mean()), 1),
        "line_random_minute_pct": round(100 * float(reach(smn, smx, pick_a, north, lvl_a).mean()), 1)}
    return out


def main():
    res = {inst: run(inst) for inst in ("NQ", "ES", "YM")}
    (REPO / "base" / "022" / "far_reach_placebo.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for inst, r in res.items():
        print(inst, json.dumps(r["horizons_minutes"], ensure_ascii=False))


if __name__ == "__main__":
    main()
