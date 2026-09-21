"""Overlap-only alignment of lynx1231 contract minute bars against the local NQ spine.

Nothing after the local tape end (2026-05-04 04:38 UTC) is read for comparison.
Outputs: best timestamp offset, OHLC agreement per contract per day, and the local tape's
roll moment between NQH26 and NQM26 (which contract the local continuous series follows).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
import sys
INST = sys.argv[1].upper() if len(sys.argv) > 1 else "NQ"
root = REPO / "data" / "market" / INST
cts = np.load(root / "close_ts_utc_ns.npy", mmap_mode="r")
lo = int(np.searchsorted(cts, pd.Timestamp("2025-12-01", tz="UTC").value))
loc = pd.DataFrame(
    {
        "open_ts": pd.to_datetime(np.asarray(cts[lo:]) - 60_000_000_000, utc=True),
        "o": np.load(root / "open.npy", mmap_mode="r")[lo:],
        "h": np.load(root / "high.npy", mmap_mode="r")[lo:],
        "l": np.load(root / "low.npy", mmap_mode="r")[lo:],
        "c": np.load(root / "close.npy", mmap_mode="r")[lo:],
    }
).set_index("open_ts")
LOCAL_END = loc.index[-1]
print("local slice", loc.index[0], "..", LOCAL_END, len(loc))

lx = pq.read_table(REPO / "data/forward/_incoming/lynx1231_equity_index_minute.parquet").to_pandas()
lx = lx[lx.contract_code.str.startswith(INST)].copy()
# lynx integer timestamps are Chicago wall-clock (bar-open label) stored as if UTC: established on overlap
lx["t"] = (pd.to_datetime(lx.timestamp, unit="ms").dt.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC"))
lx = lx[lx.t.notna()]
lx = lx[lx.t <= LOCAL_END]  # overlap only

res = {}
for code, g in lx.groupby("contract_code"):
    g = g.set_index("t").sort_index()
    best = None
    for shift in (-2, -1, 0, 1, 2):
        gi = g.copy()
        gi.index = gi.index + pd.Timedelta(minutes=shift)
        j = loc.join(gi[["open", "high", "low", "close"]], how="inner")
        if len(j) == 0:
            continue
        exact = float(((j.h == j.high) & (j.l == j.low) & (j.c == j.close)).mean())
        if best is None or exact > best[1]:
            best = (shift, exact, len(j))
    res[code] = best
    print(code, "best shift(min)", best)

# daily agreement per contract at shift 0 -> which contract the local tape follows, day by day
rows = []
for code, g in lx.groupby("contract_code"):
    g = g.set_index("t").sort_index()
    j = loc.join(g[["open", "high", "low", "close"]], how="inner")
    j["exact"] = (j.h == j.high) & (j.l == j.low) & (j.c == j.close)
    j["close_eq"] = j.c == j.close
    j["open_eq"] = j.o == j.open
    j["absdiff"] = (j.c - j.close).abs()
    d = j.groupby(j.index.tz_convert("America/Chicago").date).agg(
        n=("exact", "size"), exact=("exact", "mean"), close_eq=("close_eq", "mean"), open_eq=("open_eq", "mean"), med_absdiff=("absdiff", "median")
    )
    d["contract"] = code
    rows.append(d)
daily = pd.concat(rows).reset_index(names="date")
piv = daily.pivot(index="date", columns="contract", values="exact").round(3)
pivn = daily.pivot(index="date", columns="contract", values="n")
print("\nDaily share of bars with exact H,L,C equality (local vs contract), Feb 20 .. end:")
print(piv.loc[pd.Timestamp("2026-02-20").date():].to_string())
print("\njoined bars per day:")
print(pivn.loc[pd.Timestamp("2026-03-05").date(): pd.Timestamp("2026-03-25").date()].to_string())
out = REPO / "work" / "forward-tape" / f"align_lynx_daily_{INST}.csv"
daily.to_csv(out, index=False)
print("saved", out)
