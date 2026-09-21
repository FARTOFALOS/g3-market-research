#!/usr/bin/env python3
"""098 step 5: "only NQ" - a statement about the instrument or about the moments?

Declared before the count. At the event minutes of instrument A (T0 of a TF>=60 RIZ of
A's own field, 09:34..11:30 ET, side = direction of A's bar) read the move to the 16:00
bar along that side on A itself AND on the other two indexes at the same timestamp.
  NQ-specific price behaviour: NQ continues at its own events, ES and YM at the very
      same minutes do not (the spread continues).
  moments: all three continue at NQ's event minutes; then ES/YM fields simply do not
      pick those minutes, and the carrier is the choice of moments, not NQ's price.
  The three indexes share the day, so pure selection noise on NQ also shows up on ES
      and YM at NQ's minutes: this step separates "NQ price" from the rest, not
      "moments" from "noise".
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
MIN = 60_000_000_000
INST = ['NQ', 'ES', 'YM']
M = {}
for a in INST:
    mk = ROOT / f'data/market/{a}'
    O, C, TS, SID = (np.load(mk / f'{n}.npy') for n in ('open', 'close', 'close_ts_utc_ns', 'session_id'))
    et = pd.to_datetime(TS, utc=True).tz_convert('America/New_York')
    mod = (et.hour * 60 + et.minute).to_numpy(); pos = np.arange(len(TS))
    i16 = pd.Series(np.where(mod == 16 * 60, pos, -1)).groupby(SID).max().reindex(SID).to_numpy()
    M[a] = dict(O=O, C=C, TS=TS, SID=SID, mod=mod, i16=i16, yr=et.year.to_numpy())


def events(a):
    m = M[a]
    t = pq.read_table(ROOT / f'work/080a/index/film1_{a}.parquet', columns=['tf_minutes', 'side', 't0_spine_pos']).to_pandas()
    t = t[t.tf_minutes >= 60].drop_duplicates(['t0_spine_pos', 'side']); t['sg'] = np.where(t.side == 'north', 1.0, -1.0)
    g = t.groupby('t0_spine_pos').sg.agg(['sum', 'size']); one = g[g['sum'].abs() == g['size']]
    k = one.index.to_numpy(); s = np.sign(one['sum'].to_numpy())
    ok = (m['mod'][k] > 9 * 60 + 33) & (m['mod'][k] <= 11 * 60 + 30) & (k > 0)
    k, s = k[ok], s[ok]
    ok = np.sign(m['C'][k] - m['C'][k - 1]) == s
    return k[ok], s[ok]


def read(b, ts, s):
    m = M[b]; i = np.searchsorted(m['TS'], ts); i = np.minimum(i, len(m['TS']) - 2)
    ok = (m['TS'][i] == ts) & (m['TS'][i + 1] - m['TS'][i] == MIN) & (m['i16'][i] > i + 1)
    y = np.full(len(ts), np.nan)
    y[ok] = (s[ok] * (m['C'][m['i16'][i[ok]]] - m['O'][i[ok] + 1]) > 0).astype(float)
    return y


def est(y, sid):
    v = np.isfinite(y); d = pd.DataFrame(dict(v=y[v], s=sid[v])).groupby('s').v.agg(['sum', 'size']); p = y[v].mean()
    return p, np.sqrt(((d['sum'] - p * d['size']) ** 2).sum()) / d['size'].sum(), int(v.sum())


if __name__ == '__main__':
    for a in INST:
        k, s = events(a); ts = M[a]['TS'][k]; sid = M[a]['SID'][k]; yr = M[a]['yr'][k]
        for era, lo, hi in [('2006-2018', 2006, 2018), ('2019-2026', 2019, 2026)]:
            e = (yr >= lo) & (yr <= hi)
            cells = []
            for b in INST:
                p, se, n = est(read(b, ts[e], s[e]), sid[e]); cells.append('%s %.3f ±%.3f' % (b, p, se))
            print('events of %s %s (n %d): ' % (a, era, e.sum()) + ' | '.join(cells))
