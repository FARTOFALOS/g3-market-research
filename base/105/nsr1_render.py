"""NSR1 cold viewing: RIZ scenes at T0 on M1 and on the native TF, future closed. Prefix only; no outcome is read.

Sample: NQ, TF 120..240, T0 in 09:30-16:00 ET, 2021-2025, one row per T0 minute (random member), seed 20260924, 12 scenes.
Shown: M1 candles T0-120..T0; native TF: last 10 completed native bars + the forming bar up to T0 (partial); zone box,
exit boundary, T0 marker; text: TF, origin direction, exit side, t0_kind, span count, zone age, minutes elapsed / left
in the native bar. Not shown: confirmation, deletion, blue end, any later event.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.compute as pc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from g3riz.query import Field
from g3riz.entry_check import native_grid

F = Field(ROOT, "NQ"); m = F.market
o, h, l, c = (np.asarray(getattr(m, k)) for k in ("open", "high", "low", "close")); ts = np.asarray(m.close_ts_utc_ns)
rows = []
for tf in range(120, 241):
    t = F.passports(tf=tf, t0_start_ns=pd.Timestamp("2021-01-01", tz="UTC").value, t0_stop_ns=pd.Timestamp("2026-01-01", tz="UTC").value)
    if t.num_rows: rows.append(t.select(["riz_id", "tf_minutes", "zone_top", "zone_bottom", "bullish", "precursor_formed_ts_ns", "t0_ts_ns",
                                         "t0_spine_pos", "t0_kind", "t0_exit_side", "t0_close", "t0_span_count", "t0_native_bar_index"]).to_pandas())
P = pd.concat(rows, ignore_index=True)
et = pd.to_datetime(P.t0_ts_ns - 60_000_000_000, utc=True).dt.tz_convert("America/New_York")
P["mod"] = et.dt.hour * 60 + et.dt.minute
P = P[(P["mod"] >= 570) & (P["mod"] < 960)]
rng = np.random.default_rng(20260924)
P = P.sample(frac=1.0, random_state=20260924).drop_duplicates("t0_spine_pos")   # one random member per T0 minute
pick = P.sample(12, random_state=20260924).sort_values("t0_ts_ns")
print("population (distinct T0 minutes, TF 120-240, RTH, 2021-25):", len(P))
grids = {}
for n, r in enumerate(pick.itertuples(), 1):
    tf = int(r.tf_minutes)
    if tf not in grids: grids[tf] = native_grid(m, tf)
    st, sp = grids[tf]; j0 = int(np.searchsorted(st, r.t0_spine_pos, side="right") - 1); p0 = int(r.t0_spine_pos)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw=dict(width_ratios=[3, 2]))
    idx = np.arange(p0 - 120, p0 + 1)
    for x, i in enumerate(idx):
        col = "#2a9d8f" if c[i] >= o[i] else "#e76f51"
        a1.vlines(x, l[i], h[i], color=col, lw=0.8); a1.vlines(x, min(o[i], c[i]), max(o[i], c[i]), color=col, lw=3)
    bars = []
    for j in range(j0 - 10, j0 + 1):
        a, b = int(st[j]), int(sp[j]); b = min(b, p0 + 1) if j == j0 else b
        bars.append((o[a], h[a:b].max(), l[a:b].min(), c[b - 1], j == j0))
    for x, (bo, bh, bl, bc, forming) in enumerate(bars):
        col = "#2a9d8f" if bc >= bo else "#e76f51"
        a2.vlines(x, bl, bh, color=col, lw=1.2); a2.vlines(x, min(bo, bc), max(bo, bc), color=col, lw=9, alpha=0.45 if forming else 1)
    for ax, width in ((a1, len(idx)), (a2, len(bars))):
        ax.axhspan(r.zone_bottom, r.zone_top, color="#2962FF", alpha=0.18)
        eb = r.zone_top if r.t0_exit_side == "north" else r.zone_bottom
        ax.axhline(eb, color="#2962FF", lw=1.2)
    a1.axvline(len(idx) - 1, color="k", ls=":", lw=0.8)
    age_h = (r.t0_ts_ns - r.precursor_formed_ts_ns) / 3.6e12
    el = p0 - int(st[j0]) + 1; left = int(sp[j0]) - p0 - 1
    a1.set_title(f"#{n} M1, T0-120..T0   (T0 {pd.Timestamp(r.t0_ts_ns - 6e10, tz='UTC').tz_convert('America/New_York'):%Y-%m-%d %H:%M} ET)")
    a2.set_title(f"TF {tf}: 10 bars + forming (pale) | origin {'BISI' if r.bullish else 'SIBI'}, exit {r.t0_exit_side}\n"
                 f"{r.t0_kind}, spans at T0 {r.t0_span_count}, zone age {age_h:.1f} h, native bar {el} min in / {left} min left")
    fig.tight_layout(); fig.savefig(ROOT / f"base/105/nsr1_out/scenes/scene_{n:02d}.png", dpi=80); plt.close(fig)
pick.to_csv(ROOT / "base/105/nsr1_out/scenes/sample.csv", index=False)
print("rendered", len(pick))
