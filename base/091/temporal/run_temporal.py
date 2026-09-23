"""091-TEMPORAL: единственный прогон замороженного оператора по ленте после 2026-05-04.

Объявление и входная проверка — FREEZE_091_TEMPORAL §7.5–§7.6 (закоммичены до запуска).

    python -B base/091/temporal/run_temporal.py
Результат: base/091/temporal/temporal_result.json, temporal_signals.csv, temporal_daily.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import harness091 as H  # noqa: E402
import lens_riz as LR  # noqa: E402
from gate_091_level import lens_films, xnys_close  # noqa: E402

FWD = REPO / "data/forward/market/NQ"
USD = 20.0


def main():
    manifest = json.loads((FWD / "manifest.json").read_text(encoding="utf-8"))
    cutoff = int(manifest["canonical_prefix"]["rows"])          # первая минута продолжения
    store = LR.load_store(FWD)
    ts = store["close_ts_utc_ns"]
    op, hi, lo, cl = (store[k] for k in ("open", "high", "low", "close"))
    warm = int(np.searchsorted(ts, pd.Timestamp("2024-01-01", tz="UTC").value))
    lens = pd.concat([LR.lens_riz(store, "NQ", tf, warm, cutoff) for tf in range(1, 1441)], ignore_index=True)
    (REPO / "work/091t").mkdir(parents=True, exist_ok=True)
    lens.to_parquet(REPO / "work/091t/lens_forward_NQ.parquet", index=False)
    films = lens_films(lens)
    kind = H.gap_kind_from_ts(ts)
    cal = xnys_close(extra_forward=True)
    x, g, e = H.run(films, hi, lo, cl, op, ts, kind, cal)
    s = H.single_unit(e)

    taken = s.status.str.startswith("taken_")
    unk = s.status == "taken_unknown"
    s["entry_day"] = (ts[np.minimum(s.q.to_numpy() + 1, ts.size - 1)] // 86_400_000_000_000).astype(np.int64)
    days = np.unique(ts[cutoff:] // 86_400_000_000_000)
    res = {
        "territory": {"first_minute_utc": str(pd.Timestamp(int(ts[cutoff]), tz="UTC")),
                      "last_minute_utc": str(pd.Timestamp(int(ts[-1]), tz="UTC")), "utc_days_on_tape": int(days.size)},
        "forward_corpus_id": manifest["corpus_id"],
        "riz_after_cutoff": int(len(lens)), "close_break_films": int((x.r_close_break >= 0).sum()),
        "cell_signals": int(len(e)), "signal_classes": e.cls.value_counts().to_dict(),
        "fork_codes": {H.F.NAMES[k]: int((e.code == k).sum()) for k in H.F.NAMES},
        "mean_d": round(float(e.d.mean()), 2) if len(e) else None,
        "mean_s": round(float(e.s.mean()), 2) if len(e) else None,
        "mean_p0": round(float(e.p0.mean()), 4) if len(e) else None,
        "single_unit_status": s.status.value_counts().to_dict(),
        "taken": int(taken.sum()),
    }
    if taken.any():
        t = s[taken]
        res["target_first_rate_resolved"] = round(float((t.status == "taken_target_first").sum()
                                                        / max(1, (t.status != "taken_unknown").sum())), 4)
        res["participation_rate"] = round(float(taken.sum() / len(s)), 4)
        res["skipped_overlap_rate"] = round(float((s.status == "skipped_overlap").mean()), 4)
        res["unknown_rate_taken"] = round(float(unk.sum() / taken.sum()), 4)
        res["gap_through_b_exits"] = int(((t.code == H.GAP_B)).sum())
        per = {}
        for name, col in (("identified", "net_identified"), ("favorable", "net_favorable"), ("adverse", "net_adverse")):
            v = t[col].to_numpy(); ok = np.isfinite(v)
            dd = pd.Series(v[ok]).groupby(t.entry_day.to_numpy()[ok]).agg(["sum", "count"])
            vb = dd["sum"].reindex(days, fill_value=0.0).to_numpy(); nb = dd["count"].reindex(days, fill_value=0).to_numpy()
            per[name] = {"n": int(ok.sum()), "net_per_position_pts": round(float(v[ok].mean()), 4) if ok.any() else None,
                         "ci95_widest_day_blocks": H.day_block_ci(vb, nb.astype(float)) if ok.sum() > 1 else [None, None],
                         "total_pts": round(float(v[ok].sum()), 2),
                         "per_utc_day_pts": round(float(v[ok].sum() / days.size), 4),
                         "total_usd_one_contract": round(float(v[ok].sum() * USD), 0)}
        res["net"] = per
        lo_adv = per["adverse"]["ci95_widest_day_blocks"][0]; hi_fav = per["favorable"]["ci95_widest_day_blocks"][1]
        if lo_adv is not None and lo_adv > 0:
            res["decision_7_3"] = "POSITIVE temporal result"
        elif hi_fav is not None and hi_fav < 0:
            res["decision_7_3"] = "FAILS: frozen 091 action economically fails on this territory"
        else:
            res["decision_7_3"] = "UNRESOLVED"
        daily = pd.DataFrame({"utc_day": days})
        for name in ("identified", "favorable", "adverse"):
            col = f"net_{name}"
            daily[f"{name}_pts"] = pd.Series(t[col].to_numpy()).groupby(t.entry_day.to_numpy()).sum().reindex(days, fill_value=0.0).to_numpy()
        daily["utc_day"] = pd.to_datetime(daily.utc_day * 86_400_000_000_000, utc=True).dt.date
        daily.to_csv(HERE / "temporal_daily.csv", index=False)
    else:
        res["decision_7_3"] = "UNRESOLVED (no taken positions)"
    keep = ["riz_id", "tf_minutes", "side", "t0_pos", "q", "b", "c", "M", "a", "d", "s", "p0", "u", "k", "ttc", "r",
            "cls", "code", "exit_bar", "E", "status", "net_identified", "net_favorable", "net_adverse"]
    out = s[keep].copy()
    out["q_utc"] = pd.to_datetime(ts[out.q.to_numpy()], utc=True)
    out.to_csv(HERE / "temporal_signals.csv", index=False)
    res = json.loads(json.dumps(res, default=lambda o: int(o) if isinstance(o, np.integer) else float(o)))
    (HERE / "temporal_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
