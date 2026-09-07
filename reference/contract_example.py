"""Bounded teaching experiment; no RIZ construction, trade, or edge claim.

From the repository root: python -B reference/contract_example.py
S is fixed at T0+2; Y is a later price contact within ten calendar minutes.
All NQ TF54 RIZ are inspected, including failures. No Volume is read.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from g3riz.query import Field

MINUTE = 60_000_000_000
APRIL = "riz_98a531c9575a718312f2f9819e525bda"


def recognize(high, low, close, boundary, up):
    """S on the first two post-T0 candles; never use the continuation."""
    if len(close) < 3:
        return False
    outside = np.asarray(low[1:3]) > boundary if up else np.asarray(high[1:3]) < boundary
    towards = close[2] < close[1] if up else close[2] > close[1]
    return bool(outside.all() and towards)


def label(high, low, ts, boundary):
    """Presence by a calendar deadline; gaps cannot establish absence."""
    deadline = int(ts[2]) + 10 * MINUTE
    future = np.flatnonzero((ts > ts[2]) & (ts <= deadline))
    hits = [int(i) for i in future if low[i] <= boundary <= high[i]]
    if hits:
        k = hits[0]
        continuous = bool(np.all(np.diff(ts[2:k + 1]) == MINUTE))
        return "yes", k if continuous else None
    observed = ts[(ts >= ts[2]) & (ts <= deadline)]
    complete = len(observed) == 11 and np.all(np.diff(observed) == MINUTE)
    return ("no" if complete else "unknown"), None


def run(repo):
    field = Field(repo, "NQ")
    rows = field.passports(tf=54).to_pylist()
    m = field.market
    counts = Counter(population=len(rows))
    ledger = []
    for row in rows:
        p = int(row["t0_spine_pos"])
        ts = np.asarray(m.close_ts_utc_ns[p:p + 13])
        if len(ts) < 3 or not np.all(np.diff(ts[:3]) == MINUTE):
            counts["prefix_unavailable_or_gapped"] += 1
            continue
        deletion = row["c1_deletion_spine_pos"]
        if deletion is not None and int(deletion) <= p + 2:
            counts["already_deleted_at_decision"] += 1
            continue
        h, l, c = (np.asarray(getattr(m, name)[p:p + 13]) for name in ("high", "low", "close"))
        up = row["t0_exit_side"] == "north"
        b = float(row["zone_top"] if up else row["zone_bottom"])
        signal = recognize(h, l, c, b, up)
        # Same decision with the suffix absent or contradictory: no label input.
        assert signal == recognize(h[:3], l[:3], c[:3], b, up)
        poisoned = [np.array(x, copy=True) for x in (h, l, c)]
        for x in poisoned:
            x[3:] = b
        assert signal == recognize(*poisoned, b, up)
        counts["prefix_checks"] += 1
        if not signal:
            counts["S_false"] += 1
            continue
        outcome, first = label(h, l, ts, b)
        counts[f"S_Y_{outcome}"] += 1
        minute_span = min(float(m.open[p]), float(m.close[p])) < row["zone_bottom"] and max(float(m.open[p]), float(m.close[p])) > row["zone_top"]
        absent_confirmation = row["native_blue_confirmation_ts_ns"] is None
        counts["would_lose_with_minute_T0_detector"] += int(not minute_span)
        counts["would_lose_with_future_confirmation_filter"] += int(absent_confirmation)
        ledger.append({"riz_id": row["riz_id"], "t0_spine_pos": p,
                       "decision_spine_pos": p + 2, "Y": outcome,
                       "first_contact_ordinal": first,
                       "minutes_from_recognition_to_contact": None if first is None else first - 2,
                       "minute_T0_body_spans": bool(minute_span),
                       "future_confirmation_absent_DIAGNOSTIC_ONLY": absent_confirmation})
    april = next(r for r in ledger if r["riz_id"] == APRIL)
    assert april["Y"] == "yes" and april["first_contact_ordinal"] == 3
    assert not april["minute_T0_body_spans"]
    assert april["future_confirmation_absent_DIAGNOSTIC_ONLY"]
    key_files = [Path(__file__), repo / "SOURCE_DATA.json",
                 repo / "data/field/NQ/cells/tf_0054/manifest.json",
                 repo / "data/field/NQ/cells/tf_0054/passports.parquet"]
    hashes = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in key_files}
    return {"kind": "teaching_calibration_not_edge_or_confirmation", "instrument": "NQ", "tf": 54,
            "S": "at T0+2, existing object, two consecutive full outside candles, second close towards boundary",
            "Y": "price touches its fixed exit boundary in ten calendar minutes after S; no survival filter on Y",
            "selection": "all stored NQ TF54 RIZ; rule chosen from an already seen April illustration",
            "counts": dict(counts), "april": april, "hashes": hashes, "ledger": ledger,
            "next_decision": "Recognition is testable. A predictive or trading claim still needs a named comparison and execution/path calculation; no such claim is made."}


if __name__ == "__main__":
    print(json.dumps(run(Path(__file__).resolve().parents[1]), ensure_ascii=False, indent=2))
