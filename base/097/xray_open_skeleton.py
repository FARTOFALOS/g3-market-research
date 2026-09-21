#!/usr/bin/env python3
"""Line 097, X-ray (NOT a test, no outcomes): does the opening move have a RIZ skeleton at all?

Honest constraint found before looking: the field holds only zones that reached T0 (second
body pass). A thin place made by the move and not yet re-passed is a latent candidate and is
NOT an object of the field. So at a cursor c inside the move the observable skeleton is:
zones whose precursor formed after 09:30 ET of this session AND whose T0 <= c.
"Belongs to the move": born inside the process window by a leg of the move.
  forming direction (bullish) x T0 exit side, both against the S-07 drive side known at 09:33:
  own-leg zone, T0 against the drive  = the move's own ground re-traded back by body (undoing)
  own-leg zone, T0 along the drive    = re-acceleration through own thin place after a slow pullback
  counter-leg zone, T0 along the drive = a pullback's ground taken back by the drive
  counter-leg zone, T0 against         = the pullback itself re-accelerating
Counts are physical events (T0 minute, exit side, forming direction), NQ 2021-2025, TF 1..40
(three native bars must complete inside the 124-minute S-07 holding interval).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
MK = ROOT / 'data/market/NQ'
H, L, C, TS, SID = (np.load(MK / f'{n}.npy') for n in ('high', 'low', 'close', 'close_ts_utc_ns', 'session_id'))
et = pd.to_datetime(TS, utc=True).tz_convert('America/New_York')
MOD = (et.hour * 60 + et.minute).to_numpy(); YR = et.year.to_numpy(); pos = np.arange(len(TS))
MIN = 60_000_000_000

k33 = pos[(MOD == 9 * 60 + 33) & (YR >= 2021) & (YR <= 2025)]
k33 = k33[(TS[k33] - TS[k33 - 29] == 29 * MIN)]
drive = {int(SID[m]): (1 if C[m] > (H[m - 29:m + 1].max() + L[m - 29:m + 1].min()) / 2 else -1) for m in k33}
p30 = {int(SID[m]): m - 3 for m in k33}                                   # bar closing 09:30
mr = {int(SID[m]): float(np.median(H[m - 3:m + 28] - L[m - 3:m + 28])) for m in k33}   # minute range 09:30-10:00, scale only

frames = []
for tf in range(1, 41):
    t = pq.read_table(ROOT / f'data/field/NQ/cells/tf_{tf:04d}/passports.parquet', columns=[
        'tf_minutes', 'zone_width', 'bullish', 'precursor_formed_spine_pos', 't0_spine_pos', 't0_exit_side',
        'c1_deletion_spine_pos', 'x3_t0_spine_pos']).to_pandas()
    frames.append(t)
z = pd.concat(frames)
z = z[(z.t0_spine_pos > 0) & (z.precursor_formed_spine_pos > 0)]
z['sid'] = SID[z.t0_spine_pos.to_numpy()]
z = z[z.sid.isin(drive.keys())].copy()
z['o'] = z.sid.map(p30); z['dr'] = z.sid.map(drive); z['mr'] = z.sid.map(mr)
z = z[(z.precursor_formed_spine_pos > z.o) & (z.t0_spine_pos <= z.o + 124) & (z.t0_spine_pos > z.o + 3)].copy()
z['leg'] = np.where(np.where(z.bullish, 1, -1) == z.dr, 'own', 'counter')
z['t0dir'] = np.where(np.where(z.t0_exit_side == 'north', 1, -1) == z.dr, 'along', 'against')
z['typ'] = z.leg + '-leg zone, T0 ' + z.t0dir
z['age'] = z.t0_spine_pos - z.precursor_formed_spine_pos
z['t_open'] = z.t0_spine_pos - z.o
z['wr'] = z.zone_width / z.mr
ev = z.sort_values('tf_minutes').drop_duplicates(['t0_spine_pos', 't0_exit_side', 'bullish'])
ns = len(drive)
print('sessions %d | RIZ born after 09:30 with T0 inside the S-07 hold: rows %d, physical events %d (%.1f per session)' % (ns, len(z), len(ev), len(ev) / ns))
print('\n%-36s %7s %9s | %-22s | %-20s | %-18s | sessions with >=1 by 10:03 / by 10:33' % ('type', 'events', 'per sess', 'T0 minutes from open', 'age birth->T0, bars', 'width / minute range'))
for typ, g in ev.groupby('typ'):
    s1 = g[g.t_open <= 33].sid.nunique() / ns; s2 = g[g.t_open <= 63].sid.nunique() / ns
    q = lambda c: ' / '.join('%.0f' % v for v in g[c].quantile([.25, .5, .75]))
    print('%-36s %7d %9.2f | p25/50/75 %-12s | %-20s | %-18s | %.3f / %.3f' % (
        typ, len(g), len(g) / ns, q('t_open'), q('age'), ' / '.join('%.2f' % v for v in g.wr.quantile([.25, .5, .75])), s1, s2))
print('\nTF of the zones (physical events): ' + ' '.join('%d:%d' % (a, b) for a, b in ev.tf_minutes.value_counts().sort_index().head(12).items()))
per = ev.groupby('sid').size().reindex(list(drive.keys()), fill_value=0)
print('events per session: p10 %.0f p50 %.0f p90 %.0f | sessions with none %.3f' % (per.quantile(.1), per.median(), per.quantile(.9), (per == 0).mean()))
und = ev[ev.typ == 'own-leg zone, T0 against']
first = und.groupby('sid').t_open.min()
print('first UNDOING event of the day: in %.3f of sessions; minutes from open p25/50/75 %s'
      % (len(first) / ns, ' / '.join('%.0f' % v for v in first.quantile([.25, .5, .75]))))
ev.to_parquet((Path(__file__).resolve().parents[2] / 'work' / 'line-097') / 'open_skeleton_events.parquet')
