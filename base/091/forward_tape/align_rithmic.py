"""Three-way mechanical check. No operator outcome is computed anywhere here.

1. Resolve the Rithmic naive clock on the overlap with the local spine (<= local tape end).
2. local vs Rithmic-built bars on the overlap.
3. lynx contract bars vs Rithmic-built bars AFTER the local tape end: bar-equality only
   (integrity of the post-cutoff extension source), plus which contract Rithmic follows by day.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
INST = sys.argv[1].upper() if len(sys.argv) > 1 else "NQ"

root = REPO / "data" / "market" / INST
cts = np.load(root / "close_ts_utc_ns.npy", mmap_mode="r")
lo = int(np.searchsorted(cts, pd.Timestamp("2025-12-31", tz="UTC").value))
loc = pd.DataFrame(
    {
        "t": pd.to_datetime(np.asarray(cts[lo:]) - 60_000_000_000, utc=True),
        "o": np.load(root / "open.npy", mmap_mode="r")[lo:],
        "h": np.load(root / "high.npy", mmap_mode="r")[lo:],
        "l": np.load(root / "low.npy", mmap_mode="r")[lo:],
        "c": np.load(root / "close.npy", mmap_mode="r")[lo:],
    }
).set_index("t")
LOCAL_END = loc.index[-1]

rb = pd.read_parquet(REPO / f"data/forward/_incoming/rithmic_{INST.lower()}_1m_naive.parquet")


def eq(a, b):
    j = a.join(b, how="inner", rsuffix="_r")
    if len(j) == 0:
        return 0, 0.0
    return len(j), float(((j.h == j.high) & (j.l == j.low) & (j.c == j.close)).mean())


print("== 1. clock of Rithmic file (overlap only) ==")
best = None
for zone in ("UTC", "US/Eastern", "America/Chicago"):
    idx = rb.index.tz_localize(zone, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
    r = rb[idx.notna()].copy()
    r.index = idx[idx.notna()]
    r = r[r.index <= LOCAL_END]
    n, e = eq(loc, r[["open", "high", "low", "close"]])
    print(f"  {zone:16s} joined {n:6d}  exact H,L,C {e:.4f}")
    if best is None or e > best[1]:
        best = (zone, e)
ZONE = best[0]
idx = rb.index.tz_localize(ZONE, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
r = rb[idx.notna()].copy()
r.index = idx[idx.notna()]
print("  -> clock:", ZONE)

print("== 2. local vs Rithmic-built bars, overlap ==")
ov = loc.join(r[r.index <= LOCAL_END][["open", "high", "low", "close", "ticks"]], how="outer")
both = ov.dropna(subset=["c", "close"])
print("  local bars", int(ov.c.notna().sum()), " rithmic bars", int(ov.close.notna().sum()), " both", len(both))
for a, b in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close")):
    d = (both[a] - both[b]).abs()
    print(f"  {a}: equal {float((d == 0).mean()):.4f}  <=1tick {float((d <= 0.25).mean()):.4f}  max {d.max():.2f}")
both = both.assign(exact=(both.h == both.high) & (both.l == both.low) & (both.c == both.close))
byday = both.groupby(both.index.tz_convert("America/Chicago").date).exact.mean()
print("  days with exact share < 0.9:", [(str(k), round(v, 3)) for k, v in byday.items() if v < 0.9])

print("== 3. lynx contracts vs Rithmic-built bars AFTER local end (integrity only) ==")
lx = pq.read_table(REPO / "data/forward/_incoming/lynx1231_equity_index_minute.parquet").to_pandas()
lx = lx[lx.contract_code.str.startswith(INST)].copy()
lx["t"] = pd.to_datetime(lx.timestamp, unit="ms").dt.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
lx = lx[lx.t.notna()]
post = r[r.index > LOCAL_END]
rows = []
for code, g in lx.groupby("contract_code"):
    g = g.set_index("t").sort_index()
    g = g[g.index > LOCAL_END]
    j = post.join(g[["open", "high", "low", "close"]], how="inner", rsuffix="_x")
    if len(j) == 0:
        continue
    j["exact"] = (j.high == j.high_x) & (j.low == j.low_x) & (j.close == j.close_x)
    d = j.groupby(j.index.tz_convert("America/Chicago").date).agg(n=("exact", "size"), exact=("exact", "mean"))
    d["contract"] = code
    rows.append(d)
daily = pd.concat(rows).reset_index(names="date")
piv = daily.pivot(index="date", columns="contract", values="exact").round(3)
print(piv.to_string())
daily.to_csv(REPO / f"work/forward-tape/align_rithmic_post_{INST}.csv", index=False)
