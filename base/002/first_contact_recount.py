"""Пересчёт 2026-09-23: когда приходит первое касание выходной границы после T0
и чем оно отличается от обычной свечи.

Касание = диапазон минуты содержит выходную границу (как `first_exit_contact_v1`),
минута T0 не считается, счёт k — в записанных минутах ленты после T0. Уникальный
уровень = (минута T0, цена выходной границы, сторона): копии соседних ТФ убраны.

Плацебо: для каждой сцены NQ 2021–2025 берётся обычная минута той же децили
размаха, что и свеча T0; линия проводится на том же расстоянии за её закрытием
(против направления её тела, как у T0). Сравнивается доля касания следующей
минутой. Отдельно: свеча T0 против среднего размаха 30 минут до неё.

Volume не читается. Запуск из корня репозитория:
    python -B base/002/first_contact_recount.py
Результат: base/002/first_contact_recount.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
KMAX = 1440
BINS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 15), (16, 60), (61, 240), (241, 1440)]
COLS = ["tf_minutes", "zone_top", "zone_bottom", "t0_spine_pos", "t0_ts_ns",
        "t0_exit_side", "t0_close"]


def load(inst: str):
    mk = REPO / "data" / "market" / inst
    arr = {n: np.load(mk / f"{n}.npy") for n in ("open", "high", "low", "close", "close_ts_utc_ns", "session_id")}
    parts = [pq.read_table(REPO / "data" / "field" / inst / "cells" / f"tf_{tf:04d}" / "passports.parquet",
                           columns=COLS).to_pandas() for tf in range(1, 1441)]
    return arr, pd.concat(parts, ignore_index=True)


def first_contact(arr, P):
    H, L = arr["high"], arr["low"]
    N = len(H)
    north = (P["t0_exit_side"] == "north").to_numpy()
    e = np.where(north, P["zone_top"].to_numpy(), P["zone_bottom"].to_numpy())
    t0 = P["t0_spine_pos"].to_numpy().astype(np.int64)
    k_hit = np.full(len(P), -1, dtype=np.int64)
    away = np.zeros(len(P))
    active = np.arange(len(P))
    for k in range(1, KMAX + 1):
        if active.size == 0:
            break
        pos = t0[active] + k
        ok = pos < N
        act, pos = active[ok], pos[ok]
        hi, lo, ee = H[pos], L[pos], e[act]
        touch = (lo <= ee) & (hi >= ee)
        aw = np.where(north[act], hi - ee, ee - lo)
        nt = ~touch
        away[act[nt]] = np.maximum(away[act[nt]], aw[nt])
        k_hit[act[touch]] = k
        active = act[~touch]
    ts = arr["close_ts_utc_ns"]
    hit = k_hit > 0
    el = np.full(len(P), np.nan)
    el[hit] = (ts[t0[hit] + k_hit[hit]] - ts[t0[hit]]) / 60e9
    other_session = np.zeros(len(P), dtype=bool)
    other_session[hit] = arr["session_id"][t0[hit] + k_hit[hit]] != arr["session_id"][t0[hit]]
    return pd.DataFrame({
        "tf": P["tf_minutes"].to_numpy(), "year": pd.to_datetime(P["t0_ts_ns"], utc=True).dt.year.to_numpy(),
        "north": north, "e": e, "t0": t0, "d0": np.abs(P["t0_close"].to_numpy() - e),
        "rng0": H[t0] - L[t0], "k": k_hit, "away": away, "elapsed_min": el, "other_session": other_session})


def dist(df):
    out = {"n": int(len(df))}
    for a, b in BINS:
        out[f"{a}-{b}"] = round(100 * float(((df.k >= a) & (df.k <= b)).mean()), 1)
    out["none_within_1440"] = round(100 * float((df.k < 0).mean()), 1)
    return out


def main():
    result = {"definition": "first minute after T0 whose range contains the exit boundary; k in recorded minutes",
              "instruments": {}}
    for inst in ("NQ", "ES", "YM"):
        arr, P = load(inst)
        r = first_contact(arr, P)
        u = r.sort_values("tf").drop_duplicates(["t0", "e", "north"])
        r5, u5 = r[(r.year >= 2021) & (r.year <= 2025)], u[(u.year >= 2021) & (u.year <= 2025)]
        block = {"all_rows": dist(r), "unique_levels": dist(u), "rows_2021_2025": dist(r5),
                 "unique_2021_2025": dist(u5), "by_tf_2021_2025": {}}
        for lo, hi in ((1, 5), (6, 30), (31, 120), (121, 480), (481, 1440)):
            block["by_tf_2021_2025"][f"{lo}-{hi}"] = dist(r5[(r5.tf >= lo) & (r5.tf <= hi)])
        h = u5[u5.k > 0]
        block["unique_2021_2025_medians_points"] = {
            "t0_close_to_boundary": float(u5.d0.median()), "t0_candle_range": float(u5.rng0.median()),
            "contact_in_other_session_pct": round(100 * float(h.other_session.mean()), 1)}
        if inst == "NQ":
            block["placebo_next_minute_touch_pct"] = placebo(arr, u5)
            block["groups_2021_2025"] = groups(arr, u5)
            block["t0_candle_vs_prev30"] = candle_vs_prev30(arr, u5)
        result["instruments"][inst] = block
        print(inst, json.dumps(block["unique_2021_2025"], ensure_ascii=False))
    out = REPO / "base" / "002" / "first_contact_recount.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


def session_pool(arr, lo_year=2021, hi_year=2025):
    ts, sid = arr["close_ts_utc_ns"], arr["session_id"]
    yr = pd.to_datetime(ts, utc=True).year
    c = np.flatnonzero((yr >= lo_year) & (yr <= hi_year))[:-2]
    return c[(sid[c + 1] == sid[c]) & ((ts[c + 1] - ts[c]) == 60_000_000_000)]


def placebo(arr, u):
    O, H, L, C = arr["open"], arr["high"], arr["low"], arr["close"]
    rng = H - L
    cand = session_pool(arr)
    edges = np.quantile(u.rng0, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(edges, u.rng0.to_numpy(), side="right") - 1, 0, 9)
    rs = np.random.default_rng(7)
    hits = 0
    for d in range(10):
        sub = u[dec == d]
        pool = cand[(rng[cand] >= edges[d]) & (rng[cand] <= edges[d + 1])]
        pick = rs.choice(pool, size=len(sub), replace=True)
        up = C[pick] >= O[pick]
        line = np.where(up, C[pick] - sub.d0.to_numpy(), C[pick] + sub.d0.to_numpy())
        hits += int(((L[pick + 1] <= line) & (H[pick + 1] >= line)).sum())
    return {"riz_unique_2021_2025": round(100 * float((u.k == 1).mean()), 1),
            "ordinary_minute_same_candle_same_distance": round(100 * hits / len(u), 1)}


def groups(arr, u):
    H, L = arr["high"], arr["low"]
    rows = {}
    for name, a, b in (("+1", 1, 1), ("+2..+3", 2, 3), ("+4..+15", 4, 15), ("+16..+60", 16, 60), ("+61..+1440", 61, 1440)):
        g = u[(u.k >= a) & (u.k <= b)]
        dep = np.maximum(g.d0, g.away)
        rows[name] = {"share_pct": round(100 * len(g) / len(u), 1), "n": int(len(g)),
                      "departure_points_median": float(dep.median()), "elapsed_min_median": float(g.elapsed_min.median())}
    g = u[u.k < 0]
    rows["none_within_1440"] = {"share_pct": round(100 * len(g) / len(u), 1), "n": int(len(g))}
    return rows


def candle_vs_prev30(arr, u):
    H, L = arr["high"], arr["low"]
    rng = H - L
    prev30 = pd.Series(rng).shift(1).rolling(30, min_periods=30).mean().to_numpy()
    a = rng[u.t0.to_numpy()] / prev30[u.t0.to_numpy()]
    pool = session_pool(arr)
    pool = pool[prev30[pool] > 0]
    p = np.random.default_rng(3).choice(pool, 200_000)
    b = rng[p] / prev30[p]
    return {"t0_median_ratio": round(float(np.nanmedian(a)), 2), "ordinary_median_ratio": round(float(np.nanmedian(b)), 2),
            "t0_share_above_2x_pct": round(100 * float(np.nanmean(a > 2)), 1),
            "ordinary_share_above_2x_pct": round(100 * float(np.nanmean(b > 2)), 1)}


if __name__ == "__main__":
    main()
