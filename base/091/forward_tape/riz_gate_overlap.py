"""Event-reproduction gate on the OVERLAP only (guardrail AF).

Same minute grid, two vendors' bars of the same contract (local tape vs lynx contract bars).
The pinned RIZ machine (verbatim replay from work/line-097/gap_lens.py, validated against the field)
is run from an empty state on both, per native TF, and the objects are compared:
gaps born, first spans, Blue (2X) events -- identical bounds and identical native bar.
Window: from the local roll into the June contract (Sun 2026-03-15 session) to the local tape end.
Nothing after the local tape end is touched.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
INST = sys.argv[1].upper() if len(sys.argv) > 1 else "NQ"
CODE = {"NQ": "NQM26", "ES": "ESM26"}[INST]
MIN = 60_000_000_000

spec = importlib.util.spec_from_file_location("gap_lens", REPO / "work" / "line-097" / "gap_lens.py")
gl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gl)

root = REPO / "data" / "market" / INST
cts = np.load(root / "close_ts_utc_ns.npy", mmap_mode="r")
sid_all = np.load(root / "session_id.npy", mmap_mode="r")
z = np.load(root / "sessions.npz")
sopen = dict(zip(z["session_id"].tolist(), z["session_open_utc_ns"].tolist()))
lo = int(np.searchsorted(cts, pd.Timestamp("2026-03-15 21:00", tz="UTC").value))
ts = np.asarray(cts[lo:])
sid = np.asarray(sid_all[lo:])
A = {k: np.asarray(np.load(root / f"{k}.npy", mmap_mode="r")[lo:]) for k in ("open", "high", "low", "close")}

lx = pq.read_table(REPO / "data/forward/_incoming/lynx1231_equity_index_minute.parquet").to_pandas()
g = lx[lx.contract_code == CODE].copy()
g["close_ns"] = (pd.to_datetime(g.timestamp, unit="ms") + pd.Timedelta(minutes=1)).dt.tz_localize(
    "America/Chicago", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]").astype("int64")
g = g.set_index("close_ns")
common = np.isin(ts, g.index.to_numpy())
print(INST, CODE, "window minutes", len(ts), "on common grid", int(common.sum()))
ts, sid = ts[common], sid[common]
A = {k: v[common] for k, v in A.items()}
gg = g.loc[ts]
B = {k: gg[k].to_numpy() for k in ("open", "high", "low", "close")}
print("minute bars equal: O %.4f H %.4f L %.4f C %.4f" % tuple(float((A[k] == B[k]).mean()) for k in ("open", "high", "low", "close")))


def native(src, tf):
    k = ((ts - np.array([sopen[int(s)] for s in sid])) // MIN - 1) // tf
    df = pd.DataFrame(dict(key=sid.astype(np.int64) * 100000 + k, o=src["open"], h=src["high"], l=src["low"], c=src["close"]))
    a = df.groupby("key", sort=True).agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    return a.index.to_numpy(), a.o.to_numpy(), a.h.to_numpy(), a.l.to_numpy(), a.c.to_numpy()


rows = []
for tf in (1, 2, 3, 5, 10, 15, 30, 60):
    res = {}
    for name, src in (("local", A), ("ext", B)):
        key, o, h, l, c = native(src, tf)
        zt, zb, bull, birth, r0, fc, fkind, pen, clo, run, s1, blue, dead, cause = gl.machine(o, h, l, c)
        born = set(zip(np.round(zt, 2), np.round(zb, 2), key[birth]))
        sp = s1 >= 0
        span1 = set(zip(np.round(zt[sp], 2), np.round(zb[sp], 2), key[s1[sp]]))
        bl = blue >= 0
        blues = set(zip(np.round(zt[bl], 2), np.round(zb[bl], 2), key[blue[bl]]))
        res[name] = (born, span1, blues)
    out = {"tf": tf}
    for i, what in enumerate(("born", "span1", "blue")):
        a, b = res["local"][i], res["ext"][i]
        out[f"{what}_local"] = len(a)
        out[f"{what}_ext"] = len(b)
        out[f"{what}_same"] = len(a & b)
        out[f"{what}_share_of_local"] = round(len(a & b) / max(1, len(a)), 4)
    rows.append(out)
rep = pd.DataFrame(rows)
print(rep.to_string(index=False))
rep.to_csv(REPO / "work" / "forward-tape" / f"riz_gate_overlap_{INST}.csv", index=False)
