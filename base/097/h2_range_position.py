#!/usr/bin/env python3
"""097 H2, born from the 24-scene X-ray (classification there was post-hoc).

Reading: when T0 is the fresh tip of a run (nothing traded above b within the
viewing window) the price tended to fall back through; when T0 re-enters
territory already occupied by bodies (V-recovery) it tended to continue outward.
Prefix operator, no tuned number: the viewing window of the packet was 45 bars.
  pos45 = oriented position of close[T0] inside [min low, max high] of the
          45 bars BEFORE T0 (1 = fresh tip outward, 0 = far end)
  occ45 = share of those 45 closes that lie outside b (already-occupied territory)
Outcome: outward gross from open[k+1] held 30 / 60 bars, time exit only, same
session, contiguous minutes. One bet per (minute, side). Day-block t.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / 'work' / '097'
ROOT = OUT.parents[1]
MK = ROOT / 'data/market/NQ'
O = np.load(MK / 'open.npy'); H = np.load(MK / 'high.npy'); L = np.load(MK / 'low.npy')
C = np.load(MK / 'close.npy'); TS = np.load(MK / 'close_ts_utc_ns.npy'); SID = np.load(MK / 'session_id.npy')
MIN = 60_000_000_000
N = 45


def build():
    m = pd.read_parquet(OUT / 'moments_NQ.parquet')
    rows = []
    for r in m.itertuples():
        k, p = int(r.k), int(r.t0_spine_pos)
        s = 1.0 if r.side == 'north' else -1.0
        a = p - N
        if a < 0 or SID[a] != SID[p] or TS[p] - TS[a] != N * MIN:
            continue
        hi, lo = H[a:p].max(), L[a:p].min()
        hi, lo = max(hi, H[p]), min(lo, L[p])
        pos = (C[p] - lo) / (hi - lo) if hi > lo else np.nan
        if s < 0:
            pos = 1 - pos
        occ = float(np.mean(s * (C[a:p] - r.exit_boundary) > 0))
        out = {}
        for hz in (30, 60):
            e = k + 1 + hz
            if e >= len(C) or SID[e] != SID[k] or TS[e] - TS[k] != (hz + 1) * MIN:
                out[hz] = np.nan
            else:
                out[hz] = s * (C[e] - O[k + 1])
        rows.append(dict(k=k, side=r.side, sid=r.sid, year=r.year, wait=r.wait, w=r.w,
                         pos45=pos, occ45=occ, g30=out[30], g60=out[60]))
    d = pd.DataFrame(rows)
    d.to_parquet(OUT / 'h2_rows.parquet')
    return d


def line(d, col):
    x = d.dropna(subset=[col])
    day = x.groupby('sid')[col].agg(['sum', 'size'])
    m = x[col].mean()
    se = np.sqrt(((day['sum'] - m * day['size']) ** 2).sum()) / day['size'].sum()
    return len(x), m, m / se


if __name__ == '__main__':
    d = build()
    d = d[d.year <= 2025]
    print('moments with a clean 45-bar pre-T0 window:', len(d))
    for name, col, cuts in [('pos45 (1 = fresh tip)', 'pos45', [0, .5, .8, .95, 1.0001]),
                            ('occ45 (share of prior closes outside b)', 'occ45', [-.001, 0, .1, .4, 1.0001])]:
        print('\n' + name)
        print('%-16s %7s | %9s %6s | %9s %6s' % ('bin', 'n', 'g30', 't', 'g60', 't'))
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            x = d[(d[col] > lo) & (d[col] <= hi)] if lo >= 0 or col == 'occ45' else d
            if col == 'pos45':
                x = d[(d[col] >= lo) & (d[col] < hi)]
            n30, m30, t30 = line(x, 'g30'); n60, m60, t60 = line(x, 'g60')
            print('%-16s %7d | %+9.3f %6.2f | %+9.3f %6.2f' % ('(%.2f, %.2f]' % (lo, hi), n30, m30, t30, m60, t60))
    print('\nextreme contrast by year, g60: fresh tip (pos45>=0.95 & occ45==0) vs V-type (occ45>0.4)')
    for y in range(2021, 2026):
        a = d[(d.year == y) & (d.pos45 >= .95) & (d.occ45 == 0)]
        b = d[(d.year == y) & (d.occ45 > .4)]
        print(y, 'tip n=%d g60 %+.3f t %.2f | V n=%d g60 %+.3f t %.2f' % (
            (line(a, 'g60')) + (line(b, 'g60'))))
