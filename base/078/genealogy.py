"""078A step 2 — informational genealogy of the canonical T0.

Answers one question and no other: WHICH information out of the history
available at p does the RIZ machine need in order for the T0 status of p to be
determined at all? The answer is measured on the frozen field, not quoted from
the glossary, wherever the field can answer it.

Reads the field read-only. Y is never defined, computed or approached. Volume
is never loaded. Nothing is written into data/ or into the field.

Run: python -B base/078/genealogy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
from calendar_utils import date_key, year_of  # noqa: E402

INSTRUMENT = "NQ"
TF = 54
MINUTE = 60_000_000_000
OUT = ROOT / "work" / "078a"
OUT.mkdir(parents=True, exist_ok=True)


def pct(a, q):
    return np.percentile(np.asarray(a, float), q)


def main() -> None:
    cell = ROOT / "data" / "field" / INSTRUMENT / f"cells/tf_{TF:04d}"
    passports = pq.read_table(cell / "passports.parquet")
    events = pq.read_table(cell / "events.parquet")
    market = ROOT / "data" / "market" / INSTRUMENT
    arr = {n: np.load(market / f"{n}.npy", mmap_mode="r")
           for n in ("open", "high", "low", "close", "close_ts_utc_ns", "session_id")}
    with np.load(market / "sessions.npz") as z:
        sess = {k: z[k].copy() for k in z.files}

    p = {name: np.asarray(passports[name]) for name in passports.schema.names}
    n = len(p["riz_id"])
    report: dict[str, object] = {
        "instrument": INSTRUMENT, "tf_minutes": TF,
        "corpus_id": str(p["corpus_id"][0]),
        "passport_rows": int(n),
        "tape_minutes": int(len(arr["close"])),
        "tape_from_utc": str(np.datetime64(int(arr["close_ts_utc_ns"][0]), "ns")),
        "tape_to_utc": str(np.datetime64(int(arr["close_ts_utc_ns"][-1]), "ns")),
    }

    t0_pos = p["t0_spine_pos"].astype(np.int64)
    report["distinct_t0_minutes"] = int(np.unique(t0_pos).size)
    report["distinct_session_days"] = int(np.unique(np.asarray(arr["session_id"])[t0_pos]).size)

    # --- layer 1: raw OHLC / time inputs -------------------------------------
    # Nothing to measure beyond the tape itself; recorded for the contract.

    # --- layer 2: native-TF aggregation --------------------------------------
    # dur = offset of T0 inside its native bar, in observed minutes, exactly as
    # 077 defines it. Reconstructed from the session anchor.
    sid = np.asarray(arr["session_id"])[t0_pos]
    open_ns = sess["session_open_utc_ns"][sid]
    # close label of a minute is the END of the minute; the minute that opens at
    # the session anchor has close label anchor + 1 minute.
    elapsed = (np.asarray(arr["close_ts_utc_ns"])[t0_pos] - open_ns) // MINUTE - 1
    dur = (elapsed % TF).astype(np.int64) + 1  # observed minutes of the native bar up to and including T0
    report["native_offset_dur"] = {
        "definition": "observed minutes of the session-aligned native bar up to and including T0",
        "min": int(dur.min()), "p25": float(pct(dur, 25)), "median": float(pct(dur, 50)),
        "p75": float(pct(dur, 75)), "max": int(dur.max()),
        "share_dur_eq_tf": float(np.mean(dur == TF)),
    }

    # --- layer 3: FVG / precursor birth history ------------------------------
    prec_pos = p["precursor_formed_spine_pos"].astype(np.int64)
    back_prec = t0_pos - prec_pos
    report["precursor_to_t0_minutes"] = {
        "min": int(back_prec.min()), "p05": float(pct(back_prec, 5)),
        "p25": float(pct(back_prec, 25)), "median": float(pct(back_prec, 50)),
        "p75": float(pct(back_prec, 75)), "p95": float(pct(back_prec, 95)),
        "max": int(back_prec.max()),
        "share_within_5": float(np.mean(back_prec <= 5)),
        "share_within_30": float(np.mean(back_prec <= 30)),
        "share_within_60": float(np.mean(back_prec <= 60)),
        "share_within_1440": float(np.mean(back_prec <= 1440)),
    }
    # the precursor itself is the close of C3 of a three-bar native pattern, so
    # the earliest tape minute the zone price depends on is ~2 native bars older
    back_c1 = back_prec + 2 * TF
    report["earliest_price_input_minutes"] = {
        "definition": "T0 minus the first minute of C1 of the native three-bar precursor",
        "median": float(pct(back_c1, 50)), "p95": float(pct(back_c1, 95)),
        "share_within_30": float(np.mean(back_c1 <= 30)),
        "share_within_60": float(np.mean(back_c1 <= 60)),
    }

    # --- layer 4: span / nx history ------------------------------------------
    ev_kind = np.asarray(events["event_kind"])
    ev_riz = np.asarray(events["riz_id"])
    ev_pos = np.asarray(events["market_spine_pos"]).astype(np.int64)
    riz_index = {r: i for i, r in enumerate(p["riz_id"])}
    spans_before = np.zeros(n, dtype=np.int64)
    first_span_pos = np.full(n, -1, dtype=np.int64)
    mask = ev_kind == "accepted_span"
    for r, pos in zip(ev_riz[mask], ev_pos[mask]):
        i = riz_index[r]
        if pos <= t0_pos[i]:
            spans_before[i] += 1
            if first_span_pos[i] < 0 or pos < first_span_pos[i]:
                first_span_pos[i] = pos
    have = first_span_pos >= 0
    back_span = t0_pos[have] - first_span_pos[have]
    report["span_history"] = {
        "t0_span_count_values": {str(int(v)): int(c) for v, c in
                                 zip(*np.unique(p["t0_span_count"], return_counts=True))},
        "accepted_spans_reported_at_or_before_t0": {
            str(int(v)): int(c) for v, c in zip(*np.unique(spans_before, return_counts=True))},
        "rows_with_an_accepted_span_at_or_before_t0": int(have.sum()),
        "first_accepted_span_to_t0_minutes": {
            "min": int(back_span.min()), "median": float(pct(back_span, 50)),
            "p95": float(pct(back_span, 95)), "max": int(back_span.max()),
            "share_within_30": float(np.mean(back_span <= 30)),
            "share_within_60": float(np.mean(back_span <= 60)),
        },
        "note": ("an accepted_span event is reported at native close, so its "
                 "market_spine_pos is the report address, not the crossing"),
    }

    # --- layer 5: alive / dead status ----------------------------------------
    report["alive_status_at_t0"] = {
        "definition": ("both boundaries alive and a non-breaker eligible tier are "
                       "required for Blue; the field stores only final_* and the "
                       "retirement events, so the status at T0 is reconstructed "
                       "from event timing"),
        "north_retired_at_or_before_t0": int(sum(
            1 for r, pos, k in zip(ev_riz, ev_pos, ev_kind)
            if k == "north_boundary_retired" and pos <= t0_pos[riz_index[r]])),
        "south_retired_at_or_before_t0": int(sum(
            1 for r, pos, k in zip(ev_riz, ev_pos, ev_kind)
            if k == "south_boundary_retired" and pos <= t0_pos[riz_index[r]])),
        "breaker_entry_at_or_before_t0": int(sum(
            1 for r, pos, k in zip(ev_riz, ev_pos, ev_kind)
            if k == "breaker_entry" and pos <= t0_pos[riz_index[r]])),
    }

    # --- layer 6: Blue / 2X appearance conditions ----------------------------
    kind = np.asarray(p["t0_kind"])
    report["t0_kind"] = {str(v): int(c) for v, c in zip(*np.unique(kind, return_counts=True))}
    report["t0_exit_side"] = {str(v): int(c) for v, c in
                              zip(*np.unique(np.asarray(p["t0_exit_side"]), return_counts=True))}

    # --- layer 7: RIZ boundaries and identity --------------------------------
    width = p["zone_width"].astype(float)
    # local ruler of 076: median (H-L) over the 60 contiguous minutes strictly
    # before the minute in question.
    hi = np.asarray(arr["high"]); lo = np.asarray(arr["low"])
    ts = np.asarray(arr["close_ts_utc_ns"])

    def ruler(pos: np.ndarray, k: int = 60) -> np.ndarray:
        out = np.full(len(pos), np.nan)
        for j, q in enumerate(pos):
            a = int(q) - k
            if a < 0:
                continue
            w = ts[a:q]
            if len(w) < k or np.any(np.diff(w) != MINUTE):
                continue
            v = float(np.median(hi[a:q] - lo[a:q]))
            if v > 0:
                out[j] = v
        return out

    u60 = ruler(t0_pos, 60)
    ok = np.isfinite(u60)
    report["zone_geometry"] = {
        "width_points": {"p05": float(pct(width, 5)), "median": float(pct(width, 50)),
                         "p95": float(pct(width, 95))},
        "width_over_prior_ruler60": {
            "n": int(ok.sum()),
            "p05": float(pct(width[ok] / u60[ok], 5)),
            "median": float(pct(width[ok] / u60[ok], 50)),
            "p95": float(pct(width[ok] / u60[ok], 95))},
    }

    # --- how deep the T0 predicate reaches, summarised -----------------------
    report["depth_summary"] = {
        "share_of_t0_whose_generating_price_history_fits_in_5_minutes":
            float(np.mean(back_c1 <= 5)),
        "share_fitting_in_30_minutes": float(np.mean(back_c1 <= 30)),
        "share_fitting_in_60_minutes": float(np.mean(back_c1 <= 60)),
        "median_depth_minutes": float(pct(back_c1, 50)),
    }

    # --- how much of that genealogy a fixed generic window could see ---------
    # Measured here so the compression contracts can state it BEFORE any
    # representation is compared. Nothing about G is decided from this.
    report["visible_to_a_generic_window"] = {
        "note": ("windows are counted inclusive of p, the S-09 prefix convention; "
                 "this is genealogy, not a representation"),
        "share_full_generating_price_history_inside_5": float(np.mean(back_c1 <= 5)),
        "share_full_generating_price_history_inside_30": float(np.mean(back_c1 <= 30)),
        "share_t0_native_bar_observed_part_inside_5": float(np.mean(dur <= 5)),
        "share_t0_native_bar_observed_part_inside_30": float(np.mean(dur <= 30)),
        "share_first_accepted_span_report_inside_5": float(np.mean(t0_pos - first_span_pos <= 5)),
        "share_first_accepted_span_report_inside_30": float(np.mean(t0_pos - first_span_pos <= 30)),
        "share_precursor_inside_30": float(np.mean(back_prec <= 30)),
    }

    # --- calendar spread of the canonical group ------------------------------
    yrs = year_of(np.asarray(arr["close_ts_utc_ns"])[t0_pos])
    report["t0_by_year"] = {str(int(v)): int(c) for v, c in zip(*np.unique(yrs, return_counts=True))}
    report["t0_distinct_dates"] = int(np.unique(
        date_key(np.asarray(arr["close_ts_utc_ns"])[t0_pos])).size)

    # --- multiplicity of the T0 minute (kept, never collapsed silently) ------
    order = np.argsort(t0_pos, kind="stable")
    uniq, starts, counts = np.unique(t0_pos[order], return_index=True, return_counts=True)
    multi = {}
    for u, s, c in zip(uniq, starts, counts):
        if c == 1:
            continue
        rows = order[s:s + c]
        multi[str(int(u))] = {
            "n_riz": int(c),
            "riz_ids": [str(x) for x in p["riz_id"][rows]],
            "distinct_zones": sorted({(float(a), float(b)) for a, b in
                                      zip(p["zone_top"][rows], p["zone_bottom"][rows])}),
            "distinct_exit_sides": sorted({str(x) for x in p["t0_exit_side"][rows]}),
            "utc": str(np.datetime64(int(np.asarray(arr["close_ts_utc_ns"])[int(u)]), "ns")),
        }
    report["t0_minute_multiplicity"] = {
        "distinct_minutes": int(uniq.size),
        "minutes_with_more_than_one_riz": len(multi),
        "max_riz_on_one_minute": int(counts.max()),
        "minutes_where_those_riz_are_more_than_one_distinct_zone":
            sum(1 for v in multi.values() if len(v["distinct_zones"]) > 1),
        "detail": multi,
    }

    path = OUT / "genealogy.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
