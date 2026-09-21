#!/usr/bin/env python3
"""098 step 4: how rare is the 095 number in the WHOLE admissible space of conditions?

Declared before the count, 2026-09-21. Choosing "neighbouring" NQ slices and "random"
ES/YM slices after seeing step 3 would build the two hypotheses into the design. So
the whole relief is published instead, on one fixed grid, same frozen construction as
step 3 (event = a minute that is T0 of >=1 RIZ of the band, one side, side = direction
of the bar; control = minutes that are no T0 of any TF, matched on 10-minute clock bin
x quintile of m1 x quintile of m5; read = P(move to the 16:00 ET bar along the side)).
Grid   7 TF bands [1-4, 5-14, 15-29, 30-59, 60-119, 120-239, 240-1440]
     x 6 two-hour clock windows from 03:31 to 15:30 ET  = 42 cells
     x 2 eras x 3 instruments. The 095 slice = bands 60+ in the 09:31-11:30 window.
z      (P_event - P_matched) / session-clustered se of P_event.
Worlds
  extremum of an adaptive search: |z|>=2 in about 5 % of cells, both signs; the 095
      cells are in the tail together with unrelated cells; z of a cell in one era
      says nothing about its z in the other (correlation over cells about 0).
  structure: era-to-era correlation of z over the 42 cells clearly above 0
      (se of r about 0.155), or z moving monotonically along the TF axis in the same
      window in both eras. A functional dependence needs no slice boundary.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
MIN = 60_000_000_000
BANDS = [(1, 4), (5, 14), (15, 29), (30, 59), (60, 119), (120, 239), (240, 1440)]
W0 = 3 * 60 + 31


def run(inst):
    mk = ROOT / f'data/market/{inst}'
    O, Hh, Ll, C, TS, SID = (np.load(mk / f'{n}.npy') for n in ('open', 'high', 'low', 'close', 'close_ts_utc_ns', 'session_id'))
    et = pd.to_datetime(TS, utc=True).tz_convert('America/New_York')
    mod = (et.hour * 60 + et.minute).to_numpy(); yr = et.year.to_numpy(); n = len(TS); pos = np.arange(n)
    i16 = pd.Series(np.where(mod == 16 * 60, pos, -1)).groupby(SID).max().reindex(SID).to_numpy()
    mr = pd.Series(Hh - Ll).rolling(30).median().shift(1).to_numpy()
    k = pos[(mod >= W0) & (mod <= 15 * 60 + 30) & (pos >= 31) & (pos < n - 1)]
    k = k[(TS[k] - TS[k - 30] == 30 * MIN) & (TS[k + 1] - TS[k] == MIN) & (i16[k] > k + 1) & (mr[k] > 0)]
    d1 = C[k] - C[k - 1]; k = k[d1 != 0]; d1 = C[k] - C[k - 1]; s = np.sign(d1)
    df = pd.DataFrame(dict(k=k, sid=SID[k], era=(yr[k] >= 2019).astype(int), cw=(mod[k] - W0) // 120,
                           b10=((mod[k] - W0) % 120) // 10, m1=np.abs(d1) / mr[k],
                           m5=s * (C[k] - C[k - 5]) / mr[k], side=s,
                           y=(s * (C[i16[k]] - O[k + 1]) > 0).astype(float)))
    t = pq.read_table(ROOT / f'work/080a/index/film1_{inst}.parquet', columns=['tf_minutes', 'side', 't0_spine_pos']).to_pandas()
    t['sg'] = np.where(t.side == 'north', 1.0, -1.0)
    df['any'] = df.k.isin(t.t0_spine_pos.unique())
    out = []
    for bi, (lo, hi) in enumerate(BANDS):
        tb = t[(t.tf_minutes >= lo) & (t.tf_minutes <= hi)].drop_duplicates(['t0_spine_pos', 'side'])
        g = tb.groupby('t0_spine_pos').sg.agg(['sum', 'size']); one = g[g['sum'].abs() == g['size']]
        ev_side = df.k.map(pd.Series(np.sign(one['sum']), index=one.index))
        is_ev = ev_side.notna() & (ev_side == df.side)
        for era in (0, 1):
            for cw in range(6):
                m = (df.era == era) & (df.cw == cw)
                e = df[m & is_ev]; c = df[m & ~df['any']]
                if len(e) < 200:
                    out.append((bi, era, cw, len(e), np.nan, np.nan, np.nan)); continue
                q1 = np.unique(e.m1.quantile([.2, .4, .6, .8])); q5 = np.unique(e.m5.quantile([.2, .4, .6, .8]))
                ce = e.b10.to_numpy() * 25 + np.searchsorted(q1, e.m1) * 5 + np.searchsorted(q5, e.m5)
                cc = c.b10.to_numpy() * 25 + np.searchsorted(q1, c.m1) * 5 + np.searchsorted(q5, c.m5)
                cm = pd.Series(c.y.to_numpy()).groupby(cc).mean()
                pm = pd.Series(ce).map(cm)
                ok = pm.notna().to_numpy()
                ey = e.y.to_numpy()[ok]; sd = e.sid.to_numpy()[ok]
                p = ey.mean(); day = pd.DataFrame(dict(v=ey, s=sd)).groupby('s').v.agg(['sum', 'size'])
                se = np.sqrt(((day['sum'] - p * day['size']) ** 2).sum()) / day['size'].sum()
                out.append((bi, era, cw, int(ok.sum()), p, pm[ok].mean(), (p - pm[ok].mean()) / se))
    r = pd.DataFrame(out, columns=['band', 'era', 'cw', 'n', 'p', 'pm', 'z'])
    r['inst'] = inst
    r.to_parquet(ROOT / f'work/098/relief_{inst}.parquet')
    print(f'\n##### {inst}  z by TF band (rows) x clock window (cols: 03:31 05:31 07:31 09:31 11:31 13:31)')
    for era in (0, 1):
        print(' era', ['2006-2018', '2019-2026'][era])
        for bi, (lo, hi) in enumerate(BANDS):
            x = r[(r.band == bi) & (r.era == era)].sort_values('cw')
            print('   TF %4d-%-4d ' % (lo, hi) + ' '.join('%6.2f' % v if np.isfinite(v) else '   -- ' for v in x.z)
                  + '   | n ' + ' '.join('%5d' % v for v in x.n))
    z = r.pivot_table(index=['band', 'cw'], columns='era', values='z').dropna()
    zz = r.z.dropna()
    print(' cells with z: %d | share |z|>=2: %.3f (pos %.3f, neg %.3f) | mean z %+.2f sd %.2f'
          % (len(zz), (zz.abs() >= 2).mean(), (zz >= 2).mean(), (zz <= -2).mean(), zz.mean(), zz.std()))
    print(' era-to-era correlation of z over %d cells: %+.3f' % (len(z), np.corrcoef(z[0], z[1])[0, 1]))


if __name__ == '__main__':
    for inst in (sys.argv[1:] or ['NQ', 'ES', 'YM']):
        run(inst)
