"""Шаг 2 входной проверки 091-TEMPORAL (FREEZE_091_TEMPORAL §7.4, «почти точно, измерено»).

A. Регрессия обвязки: поле + каноническая лента, отложенный кусок 086 (T0 2025-11-01 ... 2026-05-04).
   Обвязка обязана повторить сохранённые выходы: close_break 084 (work/084/xray_NQ_evaluation.parquet)
   и классы исполнения 091 (work/091/exec091.json, NQ_holdout).
B. RIZ линзы вместо поля на той же ленте — всё то же до сделки.
C. Источник: бары lynx (NQM26) на общем отрезке 2026-03-15 21:00 UTC ... конец канонической ленты
   вместо канонических; RIZ строит линза (прогрев с 2024 года, до отрезка — каноническая лента).
   Сравнение с полем на канонической ленте по каждому звену: RIZ, close_break, a/b, клетка, исход.
Исходов после 2026-05-04 не читает.

    python -B base/091/temporal/gate_091_level.py
Результат: base/091/temporal/gate_091_level.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import harness091 as H  # noqa: E402
import lens_riz as LR  # noqa: E402

NS_DAY = 86_400_000_000_000
MKT = REPO / "data/market/NQ"


def market(root: Path):
    return {n: np.load(root / f"{n}.npy") for n in ("open", "high", "low", "close", "close_ts_utc_ns")}


def xnys_close(extra_forward=False) -> np.ndarray:
    cal = pd.read_parquet(REPO / "setups/S-04/calendar.parquet").close_ns.to_numpy().astype(np.int64)
    if extra_forward:
        sys.path.insert(0, str(REPO / "setups/S-17"))
        import forward as S17F  # noqa: E402
        fc = S17F.forward_calendar().close_ns.to_numpy().astype(np.int64)
        cal = np.unique(np.concatenate([cal, fc]))
    return cal


def lens_films(lens: pd.DataFrame) -> pd.DataFrame:
    north = lens.t0_exit_side == "north"
    return pd.DataFrame({"riz_id": lens.riz_id, "tf_minutes": lens.tf_minutes, "side": lens.t0_exit_side,
                         "exit_boundary": np.where(north, lens.zone_top, lens.zone_bottom),
                         "t0_spine_pos": lens.t0_spine_pos.astype(np.int64), "t0_ts_ns": lens.t0_ts_ns,
                         "t0_day": (lens.t0_ts_ns.to_numpy() // NS_DAY).astype(np.int32)})


def field_films(lo_ns, hi_ns) -> pd.DataFrame:
    f = pd.read_parquet(REPO / "work/081a/paths/films_NQ_evaluation.parquet",
                        columns=["riz_id", "tf_minutes", "side", "exit_boundary", "t0_spine_pos", "t0_ts_ns", "t0_day"])
    return f[(f.t0_ts_ns >= lo_ns) & (f.t0_ts_ns < hi_ns)].reset_index(drop=True)


def compare(ref, new, key="riz_id"):
    """Сравнение двух прогонов обвязки по звеньям."""
    xr, gr, er = ref
    xn, gn, en = new
    out = {}
    m = xr[[key, "r_close_break"]].merge(xn[[key, "r_close_break"]], on=key, how="outer", indicator=True)
    out["films"] = {"ref": int(len(xr)), "new": int(len(xn)), "both": int((m._merge == "both").sum())}
    b = m[m._merge == "both"]
    out["close_break_same"] = int((b.r_close_break_x == b.r_close_break_y).sum())
    out["close_break_ref"] = int((xr.r_close_break >= 0).sum())
    out["close_break_new"] = int((xn.r_close_break >= 0).sum())
    mg = gr[[key, "q", "a", "b"]].merge(gn[[key, "q", "a", "b"]], on=key, how="inner")
    out["geometry_both"] = int(len(mg))
    out["geometry_same_q_a_b"] = int(((mg.q_x == mg.q_y) & (mg.a_x == mg.a_y) & (mg.b_x == mg.b_y)).sum())
    me = er[[key, "cls", "code"]].merge(en[[key, "cls", "code"]], on=key, how="outer", indicator=True)
    out["cell_ref"] = int(len(er)); out["cell_new"] = int(len(en))
    out["cell_both"] = int((me._merge == "both").sum())
    bb = me[me._merge == "both"]
    out["cell_same_outcome_class"] = int((bb.cls_x == bb.cls_y).sum())
    out["cell_same_fork_code"] = int((bb.code_x == bb.code_y).sum())
    out["cell_counts_ref"] = er.cls.value_counts().to_dict()
    out["cell_counts_new"] = en.cls.value_counts().to_dict()
    return out


def main():
    mk = market(MKT)
    hi_, lo_, cl_, op_, ts_ = mk["high"], mk["low"], mk["close"], mk["open"], mk["close_ts_utc_ns"]
    kind_ts = H.gap_kind_from_ts(ts_)
    sys.path.insert(0, str(REPO / "base/084"))
    import race084  # noqa: E402
    _, kind_frozen = race084.load_market("NQ")
    res = {"kind_equivalence": {"frozen_gap_bars": int((kind_frozen != 0).sum()),
                                "timestamp_gap_bars": int((kind_ts != 0).sum()),
                                "identical": bool(np.array_equal(kind_frozen != 0, kind_ts != 0))}}
    cal = xnys_close()

    # A. регрессия на поле, отложенный кусок 086
    lo_ns = pd.Timestamp("2025-11-01", tz="UTC").value; hi_ns = pd.Timestamp("2026-05-05", tz="UTC").value
    ff = field_films(lo_ns, hi_ns)
    A = H.run(ff, hi_, lo_, cl_, op_, ts_, kind_frozen, cal)
    xr = pd.read_parquet(REPO / "work/084/xray_NQ_evaluation.parquet", columns=["riz_id", "r_close_break"])
    chk = A[0][["riz_id", "r_close_break"]].merge(xr, on="riz_id", suffixes=("", "_frozen"))
    e91 = json.loads((REPO / "work/091/exec091.json").read_text(encoding="utf-8"))["periods"]["NQ_holdout"]
    res["A_regression"] = {
        "films": int(len(ff)),
        "close_break_equal_to_frozen_084": int((chk.r_close_break == chk.r_close_break_frozen).sum()),
        "close_break_compared": int(len(chk)),
        "cell_N": int(len(A[2])), "cell_N_frozen_091": e91["N_cell"],
        "counts": A[2].cls.value_counts().to_dict(), "counts_frozen_091": e91["counts"],
        "mean_p0": round(float(A[2].p0.mean()), 4), "mean_p0_frozen_091": e91["mean_p0"]}

    # B. линза вместо поля, та же лента
    lens = pd.read_parquet(REPO / "work/091t/lens_canonical_NQ.parquet")
    lf = lens_films(lens)
    lf = lf[(lf.t0_ts_ns >= lo_ns) & (lf.t0_ts_ns < hi_ns)].reset_index(drop=True)
    B = H.run(lf, hi_, lo_, cl_, op_, ts_, kind_ts, cal)
    res["B_lens_vs_field_same_tape"] = compare(A, B)

    # C. бары lynx на общем отрезке
    cut = pd.Timestamp("2026-03-15 21:00", tz="UTC").value
    lx = pq.read_table(REPO / "data/forward/_incoming/lynx1231_equity_index_minute.parquet").to_pandas()
    g = lx[lx.contract_code == "NQM26"].copy()
    g["close_ns"] = (pd.to_datetime(g.timestamp, unit="ms") + pd.Timedelta(minutes=1)).dt.tz_localize(
        "America/Chicago", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC").dt.tz_localize(None) \
        .to_numpy(dtype="datetime64[ns]").astype("int64")
    g = g.drop_duplicates("close_ns").set_index("close_ns")
    lo_pos = int(np.searchsorted(ts_, cut))
    win = ts_[lo_pos:]
    have = np.isin(win, g.index.to_numpy())
    hyb = {k: mk[k].copy() for k in ("open", "high", "low", "close")}
    gg = g.loc[win[have]]
    for k in ("open", "high", "low", "close"):
        v = hyb[k][lo_pos:]; v[have] = gg[k].to_numpy(); hyb[k][lo_pos:] = v
    res["C_source"] = {"window_minutes": int(win.size), "lynx_minutes_on_grid": int(have.sum()),
                       "bars_changed_by_source": {k: int((hyb[k][lo_pos:] != mk[k][lo_pos:]).sum())
                                                  for k in ("open", "high", "low", "close")}}
    store = LR.load_store(MKT)
    for k in ("open", "high", "low", "close"):
        store[k] = hyb[k]
    warm = int(np.searchsorted(ts_, pd.Timestamp("2024-01-01", tz="UTC").value))
    from_pos = int(np.searchsorted(ts_, pd.Timestamp("2026-03-16", tz="UTC").value))
    lensC = pd.concat([LR.lens_riz(store, "NQ", tf, warm, from_pos) for tf in range(1, 1441)], ignore_index=True)
    lensC.to_parquet(REPO / "work/091t/lens_hybrid_lynx_NQ.parquet", index=False)
    lc = lens_films(lensC)
    ffC = field_films(pd.Timestamp("2026-03-16", tz="UTC").value, hi_ns)
    refC = H.run(ffC, hi_, lo_, cl_, op_, ts_, kind_ts, cal)
    newC = H.run(lc, hyb["high"], hyb["low"], hyb["close"], hyb["open"], ts_, kind_ts, cal)
    res["C_lynx_bars_lens_riz_vs_field_canonical"] = compare(refC, newC)
    rset = set(ffC.riz_id); nset = set(lc.riz_id)
    res["C_lynx_bars_lens_riz_vs_field_canonical"]["riz"] = {
        "field": len(rset), "lens_on_lynx": len(nset), "same_riz_id": len(rset & nset)}
    res = json.loads(json.dumps(res, default=lambda o: int(o) if isinstance(o, (np.integer,)) else str(o)))
    (HERE / "gate_091_level.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
