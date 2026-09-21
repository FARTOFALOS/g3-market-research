#!/usr/bin/env python3
"""098 step 3: erase the RIZ label. Is the 095 event more than "a push of this size at
this time"?  Declared before the count, 2026-09-21.

Event without the label   a minute of 09:34..11:30 ET that closed in direction side*
  (= sign of its own close-to-close move) after a made move of a given size:
  m1 = |C[k]-C[k-1]| / mr,  m5 = side* (C[k]-C[k-5]) / mr,  mr = median bar range of
  the 30 bars before k.  Everything else (level, zone, biography) is circumstance.
Two worlds   A: a strong push simply makes such a RIZ event likely; then an ordinary
  minute with the same clock, m1 and m5 continues the same.  B: crossing the zone adds
  something; then it does not.  Matching touches only the push and the clock, so it
  cannot delete what a level contributes AFTER the event.
Unit   one event per minute: several RIZ lit by one minute are one impulse a trader
  could see once and use once.  RIZ event = minute that is T0 of >=1 RIZ with TF>=60,
  all on one side, and that side equals side*.  Control = every other minute.
Match  cells = 10-minute clock bin x RIZ-quintile of m1 x RIZ-quintile of m5, per era;
  control averaged with the RIZ cell weights.
Read   P(move to the 16:00 bar is along side*), and the interval profile
  side*(C[k+h]-O[k+1])/mr at h = 5, 15, 30, 60, 120 (where the effect starts, how it
  decays).  Session bootstrap, 400 resamples.
Rule   world B only if RIZ minus matched control in P has |t| >= 3 in both eras.
New knowledge, not a re-description of the slice picked in 095: the same frozen
  comparison on ES and YM, which took no part in choosing the slice.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
MIN = 60_000_000_000
HZ = (5, 15, 30, 60, 120)


def run(inst, B=400, seed=20260921):
    mk = ROOT / f'data/market/{inst}'
    O, Hh, Ll, C, TS, SID = (np.load(mk / f'{n}.npy') for n in ('open', 'high', 'low', 'close', 'close_ts_utc_ns', 'session_id'))
    et = pd.to_datetime(TS, utc=True).tz_convert('America/New_York')
    mod = (et.hour * 60 + et.minute).to_numpy(); yr = et.year.to_numpy(); n = len(TS); pos = np.arange(n)
    i16 = pd.Series(np.where(mod == 16 * 60, pos, -1)).groupby(SID).max().reindex(SID).to_numpy()
    mr = pd.Series(Hh - Ll).rolling(30).median().shift(1).to_numpy()
    k = pos[(mod > 9 * 60 + 33) & (mod <= 11 * 60 + 30) & (pos >= 31) & (pos < n - 1)]
    k = k[(TS[k] - TS[k - 30] == 30 * MIN) & (TS[k + 1] - TS[k] == MIN) & (i16[k] > k + 1) & (mr[k] > 0)]
    d1 = C[k] - C[k - 1]; k = k[d1 != 0]; d1 = C[k] - C[k - 1]
    s = np.sign(d1)
    df = pd.DataFrame(dict(k=k, sid=SID[k], year=yr[k], clock=(mod[k] - (9 * 60 + 34)) // 10,
                           m1=np.abs(d1) / mr[k], m5=s * (C[k] - C[k - 5]) / mr[k],
                           p16=(s * (C[i16[k]] - O[k + 1]) > 0).astype(float)))
    for h in HZ:
        ok = (k + h <= i16[k]) & (TS[np.minimum(k + h, n - 1)] - TS[k] == h * MIN)
        df[f'r{h}'] = np.where(ok, s * (C[np.minimum(k + h, n - 1)] - O[k + 1]) / mr[k], np.nan)
    t = pq.read_table(ROOT / f'work/080a/index/film1_{inst}.parquet', columns=['tf_minutes', 'side', 't0_spine_pos']).to_pandas()
    t = t[t.tf_minutes >= 60].drop_duplicates(['t0_spine_pos', 'side'])
    t['sg'] = np.where(t.side == 'north', 1.0, -1.0)
    g = t.groupby('t0_spine_pos').sg.agg(['sum', 'size'])
    one = g[g['sum'].abs() == g['size']]
    rz = pd.Series(np.sign(one['sum']), index=one.index)
    df['riz_side'] = df.k.map(rz); df['any_riz'] = df.k.isin(g.index)
    df['side'] = s
    df['is_riz'] = df.riz_side.notna() & (df.riz_side == df.side)
    print(f'\n##### {inst}: window minutes {len(df)}, minutes that are a TF>=60 T0 {int(df.any_riz.sum())}, '
          f'of them side equals the bar direction {df.is_riz.sum() / max(1, df.riz_side.notna().sum()):.3f}')
    rng = np.random.default_rng(seed)
    for era, lo, hi in [('2006-2018', 2006, 2018), ('2019-2026', 2019, 2026)]:
        e = df[(df.year >= lo) & (df.year <= hi) & (df.is_riz | ~df.any_riz)].copy()
        r = e[e.is_riz]
        q1 = np.unique(r.m1.quantile([.2, .4, .6, .8])); q5 = np.unique(r.m5.quantile([.2, .4, .6, .8]))
        e['cell'] = e.clock.to_numpy() * 25 + np.searchsorted(q1, e.m1) * 5 + np.searchsorted(q5, e.m5)
        sc = pd.factorize(e.sid)[0]; ns = sc.max() + 1; nc = 12 * 25
        ir = e.is_riz.to_numpy(); cell = e.cell.to_numpy()
        print(f'=== {era}: RIZ events {ir.sum()} in {e[ir].sid.nunique()} sessions | control minutes {(~ir).sum()} '
              f'| RIZ push m1 p50 {r.m1.median():.2f} m5 p50 {r.m5.median():.2f} vs all minutes m1 {e[~ir].m1.median():.2f} m5 {e[~ir].m5.median():.2f}')
        print('%-5s | %8s %8s %8s | %8s %6s | unmatched control' % ('read', 'RIZ', 'matched', 'diff', 'se', 't'))
        for col in ['p16'] + [f'r{h}' for h in HZ]:
            y = e[col].to_numpy(); v = np.isfinite(y)
            def acc(mask, w):
                return np.bincount(sc[mask] * nc + cell[mask], weights=w, minlength=ns * nc).reshape(ns, nc)
            SR, NR = acc(v & ir, y[v & ir]), acc(v & ir, None)
            SC, NC = acc(v & ~ir, y[v & ~ir]), acc(v & ~ir, None)

            def stat(w):
                sr, nr, sc_, nc_ = w @ SR, w @ NR, w @ SC, w @ NC
                ok = (nr > 0) & (nc_ > 0)
                a = sr[ok].sum() / nr[ok].sum()
                b = (nr[ok] * sc_[ok] / nc_[ok]).sum() / nr[ok].sum()
                return a, b, sc_.sum() / nc_.sum()
            a, b, u = stat(np.ones(ns))
            bs = np.array([np.subtract(*stat(rng.multinomial(ns, np.ones(ns) / ns).astype(float))[:2]) for _ in range(B)])
            se = bs.std(ddof=1)
            print('%-5s | %8.4f %8.4f %+8.4f | %8.4f %6.2f | %8.4f' % (col, a, b, a - b, se, (a - b) / se, u))


if __name__ == '__main__':
    for inst in (sys.argv[1:] or ['NQ']):
        run(inst)
