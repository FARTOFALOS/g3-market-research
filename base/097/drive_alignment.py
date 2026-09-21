#!/usr/bin/env python3
"""098: what does RIZ add to the morning continuation?  Declared before the count.

Two worlds that 095 (0.566 against placebo 0.502) did not separate:
  W1  the NY open sets a direction that persists (S-07); RIZ ignitions simply occur
      more often along that direction, so their continuation is the drive seen
      through a selected place. Then INSIDE a stratum of alignment with the drive
      RIZ and an ordinary minute continue equally, and the whole 0.566 is the
      share of aligned ignitions times the drive.
  W2  the ignition itself keeps or amplifies continuation: inside the same
      alignment stratum RIZ continues more often than an ordinary minute.
The 095 placebo matched 5-bar momentum but not alignment with the drive.

Scene   095 slice as published: NQ, RIZ with TF >= 60, T0 after the 09:33 ET bar
        and within 120 minutes of the 09:30 open; entry open[T0+1]; side = away
        from own b; outcome = sign of (close of the 16:00 ET bar - entry) on that
        side, and the same in points. Per RIZ and per (minute, side).
Drive   S-07 side of the session, known at the close of the 09:33 bar: close
        above the middle of the high-low range of the last 30 bars -> up.
        aligned = RIZ side equals the drive side.
Placebo A  same ET minute on the nearest earlier session, same N/S side.
Placebo B  same minute as A, side = sign of the last 5-bar move (095 placebo).
Both placebos are split by alignment with THEIR session's drive.
Read    share aligned; P(cont) and mean points inside aligned / counter, RIZ
        against each placebo; session-day clustered errors; eras 2006-2018 and
        2019-2026 apart (both already exposed by 095).
Rule    W2 only if RIZ minus placebo inside a stratum has |t| >= 3 with one sign
        in both eras. Otherwise the RIZ-specific reading of 0.566 is not isolated.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[2] / 'work' / '098'
MK = ROOT / 'data/market/NQ'
O = np.load(MK / 'open.npy'); H = np.load(MK / 'high.npy'); L = np.load(MK / 'low.npy')
C = np.load(MK / 'close.npy'); TS = np.load(MK / 'close_ts_utc_ns.npy'); SID = np.load(MK / 'session_id.npy')
MIN = 60_000_000_000; DAY = 24 * 60 * MIN

et = pd.to_datetime(TS, utc=True).tz_convert('America/New_York')
MOD = (et.hour * 60 + et.minute).to_numpy()
pos = np.arange(len(TS))
i33 = pd.Series(np.where(MOD == 9 * 60 + 33, pos, -1)).groupby(SID).max()
i16 = pd.Series(np.where(MOD == 16 * 60, pos, -1)).groupby(SID).max()
drive = {}
for sid, i in i33.items():
    if i < 30 or TS[i] - TS[i - 29] != 29 * MIN:
        continue
    mid = (H[i - 29:i + 1].max() + L[i - 29:i + 1].min()) / 2
    drive[int(sid)] = 1.0 if C[i] > mid else -1.0


def outcome(k, side):
    """entry open[k+1], exit close of the 16:00 bar of the same session."""
    sid = int(SID[k]); e = int(i16.get(sid, -1))
    if sid not in drive or e <= k + 1 or TS[k + 1] - TS[k] != MIN:
        return None
    if not (9 * 60 + 33 < MOD[k] <= 11 * 60 + 30):
        return None
    g = side * (C[e] - O[k + 1])
    return dict(sid=sid, aligned=bool(side == drive[sid]), cont=float(g > 0), g=g,
                year=int(et[k].year), clock=int(MOD[k]))


def build():
    t = pq.read_table(ROOT / 'work/080a/index/film1_NQ.parquet',
                      columns=['riz_id', 'tf_minutes', 'side', 't0_spine_pos']).to_pandas()
    t = t[t.tf_minutes >= 60]
    rows = []
    for r in t.itertuples():
        k = int(r.t0_spine_pos); s = 1.0 if r.side == 'north' else -1.0
        a = outcome(k, s)
        if a is None:
            continue
        a.update(kind='RIZ', k=k, side=r.side); rows.append(a)
        kp = -1
        for j in range(1, 6):
            i = int(np.searchsorted(TS, TS[k] - j * DAY))
            if i < len(TS) and TS[i] == TS[k] - j * DAY:
                kp = i; break
        if kp < 5:
            continue
        b = outcome(kp, s)
        if b is not None:
            b.update(kind='PLC_A', k=kp, side=r.side); rows.append(b)
        m5 = np.sign(C[kp] - C[kp - 5])
        if m5 != 0 and TS[kp] - TS[kp - 5] == 5 * MIN:
            c = outcome(kp, float(m5))
            if c is not None:
                c.update(kind='PLC_B', k=kp, side='north' if m5 > 0 else 'south'); rows.append(c)
    d = pd.DataFrame(rows)
    d.to_parquet(OUT / 'rows.parquet')
    return d


def est(x, col):
    day = x.groupby('sid')[col].agg(['sum', 'size']); m = x[col].mean()
    se = np.sqrt(((day['sum'] - m * day['size']) ** 2).sum()) / day['size'].sum()
    return m, se


if __name__ == '__main__':
    import sys
    d = build() if 'run' in sys.argv else pd.read_parquet(OUT / 'rows.parquet')
    for era, lo, hi in [('2006-2018', 2006, 2018), ('2019-2026', 2019, 2026)]:
        e = d[(d.year >= lo) & (d.year <= hi)]
        print('\n=== %s  (per RIZ rows; sessions %d)' % (era, e.sid.nunique()))
        print('%-6s %7s | %-13s | %-23s | %-23s' % ('', 'n', 'share aligned', 'aligned: P(cont)  pts', 'counter: P(cont)  pts'))
        keep = {}
        for kind in ['RIZ', 'PLC_A', 'PLC_B']:
            x = e[e.kind == kind]
            sa, sse = est(x, 'aligned'.replace('aligned', 'al')) if False else est(x.assign(al=x.aligned.astype(float)), 'al')
            pa, pas = est(x[x.aligned], 'cont'); ga, gas = est(x[x.aligned], 'g')
            pc, pcs = est(x[~x.aligned], 'cont'); gc, gcs = est(x[~x.aligned], 'g')
            pu, pus = est(x, 'cont')
            keep[kind] = (pa, pas, pc, pcs, ga, gas, gc, gcs)
            print('%-6s %7d | %.3f ±%.3f  | %.3f ±%.3f  %+6.2f ±%.2f | %.3f ±%.3f  %+6.2f ±%.2f | all %.3f ±%.3f'
                  % (kind, len(x), sa, sse, pa, pas, ga, gas, pc, pcs, gc, gcs, pu, pus))
        for plc in ['PLC_A', 'PLC_B']:
            r, p = keep['RIZ'], keep[plc]
            print('   RIZ - %s: aligned dP %+.3f (t %.2f) dpts %+.2f (t %.2f) | counter dP %+.3f (t %.2f) dpts %+.2f (t %.2f)'
                  % (plc, r[0] - p[0], (r[0] - p[0]) / np.hypot(r[1], p[1]), r[4] - p[4], (r[4] - p[4]) / np.hypot(r[5], p[5]),
                     r[2] - p[2], (r[2] - p[2]) / np.hypot(r[3], p[3]), r[6] - p[6], (r[6] - p[6]) / np.hypot(r[7], p[7])))
