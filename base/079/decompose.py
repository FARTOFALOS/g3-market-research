"""079A step A — decompose the canonical T0 predicate, and prove the operator.

Two jobs, both before any support is computed:

  1. Split the pinned machine's Blue/2X condition into current-interaction,
     prior-object-history and eligibility parts, reading the pinned Pine source
     rather than intuition.
  2. Rebuild, from the minute tape alone, the address at which the machine's
     `is2xPreview` must have fired, and check it reproduces the stored
     `t0_spine_pos` and `t0_kind` exactly. Without that the H0 class cannot be
     placed on the same clock as H1, and the contrast would compare a forming
     native body against a closed one.

Reads the field read-only. No Y, no Volume, no future.

python -B base/079/decompose.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "079a"
OUT.mkdir(parents=True, exist_ok=True)
INSTRUMENT, TF = "NQ", 54
MINUTE = 60_000_000_000


class Bars:
    """Session-aligned native bars of one timeframe over the frozen minute tape."""

    def __init__(self, instrument: str, tf: int):
        m = ROOT / "data" / "market" / instrument
        self.open = np.load(m / "open.npy")
        self.high = np.load(m / "high.npy")
        self.low = np.load(m / "low.npy")
        self.close = np.load(m / "close.npy")
        self.ts = np.load(m / "close_ts_utc_ns.npy")
        self.session_id = np.load(m / "session_id.npy")
        with np.load(m / "sessions.npz") as z:
            self.sess = {k: z[k].copy() for k in z.files}
        self.tf = tf
        self.n = len(self.close)

    def bar_of(self, pos: np.ndarray):
        """(bar index within session, elapsed minutes into the bar incl. pos)."""
        pos = np.asarray(pos, dtype=np.int64)
        sid = self.session_id[pos]
        anchor = self.sess["session_open_utc_ns"][sid]
        elapsed = (self.ts[pos] - anchor) // MINUTE - 1     # 0-based minute of session
        return (elapsed // self.tf).astype(np.int64), (elapsed % self.tf).astype(np.int64) + 1

    def bar_minutes(self, pos: int) -> np.ndarray:
        """All tape positions belonging to the native bar that contains `pos`."""
        sid = int(self.session_id[pos])
        anchor = int(self.sess["session_open_utc_ns"][sid])
        k, _ = self.bar_of(np.array([pos]))
        start_ns = anchor + int(k[0]) * self.tf * MINUTE          # bar opens here
        lo = int(np.searchsorted(self.ts, start_ns + MINUTE, "left"))
        hi = int(np.searchsorted(self.ts, start_ns + (self.tf + 1) * MINUTE, "left"))
        return np.arange(lo, hi, dtype=np.int64)


def first_span_minute(bars: Bars, bar_pos: int, zt: float, zb: float):
    """First minute of that native bar at which the FORMING body spans the zone.

    The machine reads `pbmin = min(open, close)`, `pbmax = max(open, close)` of
    the DEVELOPING native bar, so `open` is the bar's own open and `close` is the
    last closed minute. Strict inequalities, exactly as the pinned source.
    Returns (position, minutes_into_bar, n_minutes_in_bar) or (-1, -1, n).
    """
    mins = bars.bar_minutes(bar_pos)
    if mins.size == 0:
        return -1, -1, 0
    bar_open = float(bars.open[mins[0]])
    closes = bars.close[mins]
    bmin = np.minimum(bar_open, closes)
    bmax = np.maximum(bar_open, closes)
    hit = np.flatnonzero((bmin < zb) & (bmax > zt))
    if hit.size == 0:
        return -1, -1, int(mins.size)
    j = int(hit[0])
    return int(mins[j]), j + 1, int(mins.size)


DECOMPOSITION = {
    "source": "reference/pine/RIZ_BLUE_v1.0.pine, lines 132-140 and 236-262",
    "is2xPreview": ("show_2x and prev_on and lv_tf and (tr == 1 or tr == 2) and "
                    "strict2x_dyn and spanNow and bar_index >= zCr + 2"),
    "is2xLive": "show_2x and (tr == 1 or tr == 2) and zNx >= 2 and strict2x_dyn",
    "conditions": [
        {"condition": "spanNow = pbmin < zb and pbmax > zt",
         "class": "CURRENT INTERACTION",
         "reading": ("the body of the native bar as it stands at this minute "
                     "strictly traverses the whole zone; strict on both sides")},
        {"condition": "lv_tf = not barstate.isconfirmed",
         "class": "CURRENT INTERACTION (observation phase)",
         "reading": ("the preview is a reading of a DEVELOPING native bar. At the "
                     "bar's last minute the bar is confirmed and is2xLive decides "
                     "instead. This is why t0_kind exists")},
        {"condition": "tr == 1 or tr == 2  (i.e. ac == true)",
         "class": "PRIOR OBJECT HISTORY",
         "reading": ("the object is already ACTIVATED: an accepted span exists "
                     "before this minute. tier is fixed at activation: 1 if both "
                     "sides were alive then, 2 otherwise")},
        {"condition": "bar_index >= zCr + 2",
         "class": "PRIOR OBJECT HISTORY",
         "reading": ("spacing: at least one whole native bar between accepted "
                     "spans. zCr is the bar index of the PREVIOUS accepted span, "
                     "so this condition does not exist before activation")},
        {"condition": "zNx >= 2  (is2xLive only)",
         "class": "PRIOR OBJECT HISTORY",
         "reading": "counted spans; the preview does not test it"},
        {"condition": "strict2x_dyn = nA and sA and nAp and sAp",
         "class": "ELIGIBILITY / ALIVE",
         "reading": ("both boundaries still alive, including the intrabar "
                     "provisional retirement nAp/sAp. History-dependent in "
                     "origin, but it is a state of the object at this minute")},
        {"condition": "not brkAny",
         "class": "ELIGIBILITY / ALIVE",
         "reading": "the current bar is not a breaker close through a live side"},
    ],
    "consequence_for_H0_H1": (
        "The only machine condition that separates the activating span from the "
        "Blue-making span is `ac` (a prior accepted span exists) together with "
        "the spacing rule it brings with it. `spanNow` is byte-identical in both. "
        "So the declared contrast is isolable IF and ONLY IF both classes are "
        "read on the same clock: the developing native body on the minute tape."),
}


def main() -> None:
    bars = Bars(INSTRUMENT, TF)
    cell = ROOT / "data" / "field" / INSTRUMENT / f"cells/tf_{TF:04d}"
    pp = pq.read_table(cell / "passports.parquet")
    ev = pq.read_table(cell / "events.parquet")

    t0 = np.asarray(pp["t0_spine_pos"]).astype(np.int64)
    zt = np.asarray(pp["zone_top"]).astype(float)
    zb = np.asarray(pp["zone_bottom"]).astype(float)
    kind = np.asarray(pp["t0_kind"])
    riz = np.asarray(pp["riz_id"])

    report = {"instrument": INSTRUMENT, "tf_minutes": TF,
              "predicate_decomposition": DECOMPOSITION}

    # ---- check 1: the native bar schedule ------------------------------
    k, dur = bars.bar_of(t0)
    last_minute = dur == TF
    agree = (last_minute == (kind == "native_confirmation"))
    report["check_native_schedule"] = {
        "claim": ("a reconstructed `dur == TF` must coincide with "
                  "t0_kind == native_confirmation, because the preview needs an "
                  "unconfirmed bar"),
        "rows": int(len(t0)),
        "agree": int(agree.sum()),
        "disagree": int((~agree).sum()),
        "dur_eq_TF": int(last_minute.sum()),
        "native_confirmation": int((kind == "native_confirmation").sum()),
    }

    # ---- check 2: rebuild the preview address ---------------------------
    got = np.full(len(t0), -1, dtype=np.int64)
    got_dur = np.full(len(t0), -1, dtype=np.int64)
    nmin = np.zeros(len(t0), dtype=np.int64)
    for i in range(len(t0)):
        got[i], got_dur[i], nmin[i] = first_span_minute(bars, int(t0[i]),
                                                        zt[i], zb[i])
    exact = got == t0
    report["check_preview_operator"] = {
        "claim": ("the first minute of the T0 native bar whose forming body "
                  "spans the zone must BE the stored t0_spine_pos"),
        "rows": int(len(t0)),
        "exact": int(exact.sum()),
        "share_exact": float(exact.mean()),
        "no_span_found": int((got < 0).sum()),
        "earlier_than_stored": int(((got >= 0) & (got < t0)).sum()),
        "later_than_stored": int(((got > t0)).sum()),
        "bars_with_missing_minutes": int((nmin < TF).sum()),
    }
    bad = np.flatnonzero(~exact)
    report["check_preview_operator"]["mismatch_examples"] = [
        {"riz_id": str(riz[i]), "t0_kind": str(kind[i]),
         "stored_pos": int(t0[i]), "rebuilt_pos": int(got[i]),
         "stored_utc": str(np.datetime64(int(bars.ts[t0[i]]), "ns")),
         "rebuilt_utc": (str(np.datetime64(int(bars.ts[got[i]]), "ns"))
                         if got[i] >= 0 else None),
         "minutes_in_bar": int(nmin[i]), "zone": [float(zt[i]), float(zb[i])]}
        for i in bad[:12]]

    # ---- inventory of the accepted-span events --------------------------
    kinds = np.asarray(ev["event_kind"])
    mask = kinds == "accepted_span"
    ev_riz = np.asarray(ev["riz_id"])[mask]
    ev_pos = np.asarray(ev["market_spine_pos"]).astype(np.int64)[mask]
    ev_bar = np.asarray(ev["native_bar_index"]).astype(np.int64)[mask]
    order = np.lexsort((ev_pos, ev_riz))
    ev_riz, ev_pos, ev_bar = ev_riz[order], ev_pos[order], ev_bar[order]
    first_of = {}
    per = {}
    for r, p_ in zip(ev_riz, ev_pos):
        per[r] = per.get(r, 0) + 1
        if r not in first_of:
            first_of[r] = int(p_)
    report["accepted_span_inventory"] = {
        "events": int(mask.sum()),
        "objects": int(len(per)),
        "spans_per_object": {str(v): int(sum(1 for x in per.values() if x == v))
                             for v in sorted(set(per.values()))},
        "objects_whose_first_span_is_before_t0": int(sum(
            1 for r, p_ in zip(riz, t0) if first_of.get(r, 10**18) < p_)),
        "note": ("an accepted_span event address is the NATIVE CLOSE report, not "
                 "the market crossing; the H0 clock has to be rebuilt the same "
                 "way T0's was"),
    }

    (OUT / "decompose.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k != "predicate_decomposition"},
                     indent=2, ensure_ascii=False, default=float))
    print(f"\nwritten: {OUT / 'decompose.json'}")


if __name__ == "__main__":
    main()
