"""NSR1 pre-declaration diagnostic: where the machine records native confirmation relative to the close of the native
bar that holds T0. Machine clocks only; no price outcome after T0 is read."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from g3riz.query import Field
from g3riz.entry_check import native_grid
F = Field(ROOT, "NQ"); m = F.market
ts = np.asarray(m.close_ts_utc_ns)
cols = ["riz_id", "tf_minutes", "t0_ts_ns", "t0_spine_pos", "t0_kind", "t0_exit_side", "native_blue_confirmation_spine_pos",
        "c1_deletion_spine_pos", "blue_eligibility_end_spine_pos", "t0_native_bar_index", "native_blue_confirmation_native_bar_index"]
out = []
for tf in range(120, 241):
    P = F.passports(tf=tf, t0_start_ns=pd.Timestamp("2006-01-01", tz="UTC").value, t0_stop_ns=pd.Timestamp("2026-01-01", tz="UTC").value).select(cols).to_pandas()
    st, sp = native_grid(m, tf)
    j0 = np.searchsorted(sp, P.t0_spine_pos.to_numpy(), side="right")
    q = sp[j0] - 1
    P["q"] = q; P["bar_len"] = sp[j0] - st[j0]
    P["sched_min"] = (ts[q] - ts[st[j0]]) // 60_000_000_000 + 1
    out.append(P)
P = pd.concat(out, ignore_index=True)
et = pd.to_datetime(P.t0_ts_ns - 60_000_000_000, utc=True).dt.tz_convert("America/New_York")
P["mod"] = et.dt.hour * 60 + et.dt.minute
R = P[(P["mod"] >= 570) & (P["mod"] < 960)].copy()
c = R.native_blue_confirmation_spine_pos
R["conf_rel"] = np.where(c.isna(), "none", np.where(c == R.q, "at_q", np.where(c < R.q, "before_q", "after_q")))
R["del_before_q"] = R.c1_deletion_spine_pos.notna() & (R.c1_deletion_spine_pos <= R.q)
R["blue_end_before_q"] = R.blue_eligibility_end_spine_pos.notna() & (R.blue_eligibility_end_spine_pos <= R.q)
print("rows", len(R), "distinct T0 minutes", R.t0_spine_pos.nunique())
print(pd.crosstab(R.t0_kind, R.conf_rel))
print(pd.crosstab([R.t0_kind, R.conf_rel], R.del_before_q))
print(pd.crosstab([R.t0_kind, R.conf_rel], R.blue_end_before_q))
x = R[R.conf_rel == "after_q"]
if len(x):
    print("after_q: bar offset", (x.native_blue_confirmation_native_bar_index - x.t0_native_bar_index).value_counts().head())
x = R[R.conf_rel == "before_q"]
if len(x): print("before_q sample", x.head())
R.to_parquet(ROOT / "base/105/nsr1_out/diag_clock.parquet")
