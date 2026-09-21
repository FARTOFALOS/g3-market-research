#!/usr/bin/env python3
"""099: a composition of the two facts of the project that DO transfer. Declared before the count.

Changed role of RIZ (lens set 2026-09-21): not a signal of direction but the space in
which a direction chosen elsewhere has room or has none.
  fact 1  S-07: side by the close of 09:33 ET against the middle of the last 30 bars,
          hold 120 minutes; sign positive in candle units in three epochs on three
          indexes; its money lives in the tail days.
  fact 2  050/055: the count of LIVE RIZ boundaries within +-R_prev of price compresses
          the next hour's range and cuts the tail, not the typical bar; three indexes.
Prediction, direction fixed in advance: the denser the field at the decision minute,
the smaller the S-07 result in candles - mean and right tail - on all three indexes.
Density operator is 050's verbatim (live boundary: T0 <= m <= deletion or last
observed; band +-R_prev, R_prev = range of the previous 60 minutes), only the minute
is the 09:33 bar instead of a T0 minute.
S-07 without management: entry open of the next bar, exit close 120 bars later.
candle = median true range of the last 30 bars. Unit = session.
Rule  holds only if the mean in candles falls monotonically over density terciles on
      all three indexes and low-minus-high has |t| >= 3 on NQ with the same sign on ES, YM.
Also read: mean/sd inside terciles - if only the scale shrinks, the ratio stays and the
field gives sizing, not selection.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
MIN = 60_000_000_000


def run(inst):
    mk = ROOT / f'data/market/{inst}'
    O, H, L, C, TS, SID = (np.load(mk / f'{n}.npy') for n in ('open', 'high', 'low', 'close', 'close_ts_utc_ns', 'session_id'))
    et = pd.to_datetime(TS, utc=True).tz_convert('America/New_York')
    mod = (et.hour * 60 + et.minute).to_numpy(); yr = et.year.to_numpy(); pos = np.arange(len(TS))
    k = pos[mod == 9 * 60 + 33]
    k = k[(k > 61) & (k + 121 < len(TS))]
    k = k[(TS[k] - TS[k - 60] == 60 * MIN) & (TS[k + 121] - TS[k] == 121 * MIN)]
    z = pq.read_table(ROOT / f'work/080a/index/film1_{inst}.parquet', columns=[
        'zone_top', 'zone_bottom', 't0_spine_pos', 'c1_deletion_spine_pos', 'last_observed_spine_pos']).to_pandas()
    end = np.where(z.c1_deletion_spine_pos > 0, z.c1_deletion_spine_pos, z.last_observed_spine_pos).astype(np.int64)
    bp = np.concatenate([z.zone_top, z.zone_bottom]); bs = np.concatenate([z.t0_spine_pos, z.t0_spine_pos]); be = np.concatenate([end, end])
    o = np.argsort(bp); bp, bs, be = bp[o], bs[o], be[o]
    rows = []
    for m in k:
        rp = H[m - 60:m + 1].max() - L[m - 60:m + 1].min()
        a, b = np.searchsorted(bp, (C[m] - rp, C[m] + rp))
        dens = int(((bs[a:b] <= m) & (be[a:b] >= m)).sum())
        hi, lo = H[m - 29:m + 1].max(), L[m - 29:m + 1].min()
        side = 1.0 if C[m] > (hi + lo) / 2 else -1.0
        pc = C[m - 30:m]
        tr = np.maximum(H[m - 29:m + 1], pc) - np.minimum(L[m - 29:m + 1], pc)
        cand = np.median(tr)
        if cand <= 0 or rp <= 0:
            continue
        rows.append(dict(year=int(yr[m]), dens=dens, res=side * (C[m + 121] - O[m + 1]) / cand,
                         rn=(H[m + 1:m + 61].max() - L[m + 1:m + 61].min()) / rp))
    d = pd.DataFrame(rows)
    d['tc'] = pd.qcut(d.dens.rank(method='first'), 3, labels=False)
    print(f'\n##### {inst}: sessions {len(d)} | density p10/p50/p90 {d.dens.quantile(.1):.0f}/{d.dens.median():.0f}/{d.dens.quantile(.9):.0f}'
          f' | S-07 mean {d.res.mean():+.3f} candles (t {d.res.mean() / d.res.std() * np.sqrt(len(d)):.2f})')
    print(' tercile      n  dens p50 | R_next/R_prev p50 | mean cand      t |    sd  mean/sd |    p5    p95')
    for q, x in d.groupby('tc'):
        print(' %d        %5d  %7.0f | %17.3f | %+9.3f %6.2f | %5.1f  %+6.3f | %+6.1f %+6.1f' % (
            q + 1, len(x), x.dens.median(), x.rn.median(), x.res.mean(), x.res.mean() / x.res.std() * np.sqrt(len(x)),
            x.res.std(), x.res.mean() / x.res.std(), x.res.quantile(.05), x.res.quantile(.95)))
    a, b = d[d.tc == 0].res, d[d.tc == 2].res
    print(' low minus high density: %+.3f candles, t %.2f' % (a.mean() - b.mean(), (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))))
    for name, lo, hi in [('2006-2012', 2006, 2012), ('2013-2019', 2013, 2019), ('2020-2026', 2020, 2026)]:
        e = d[(d.year >= lo) & (d.year <= hi)].copy(); e['tc'] = pd.qcut(e.dens.rank(method='first'), 3, labels=False)
        print('   %s mean by tercile: ' % name + ' / '.join('%+.3f' % e[e.tc == q].res.mean() for q in range(3)))


if __name__ == '__main__':
    import sys
    for inst in (sys.argv[1:] or ['NQ', 'ES', 'YM']):
        run(inst)
