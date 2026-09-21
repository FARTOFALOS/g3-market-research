#!/usr/bin/env python3
"""Line 097, X-ray without price outcomes: can the RIZ life cycle be a probe of anything but the
volatility clock?  Declared before looking (2026-09-22).

Null that would kill the probe idea at the root: the tape is a TIME-CHANGED self-similar noise.
Then a zone born at scale r0 (range of its C3 native bar) lives among bars of another size when
volatility moves, and every life statistic becomes a function of ONE number,
    rho = median range of the 10 native bars after birth / r0,
because spans are defined by native-bar bodies against a zone 0.2*r0 wide. Clock time, sessions and
the open may then enter ONLY through rho.
Check: life statistics by clock of birth, raw and inside terciles of rho. If the clock pattern
disappears inside rho strata, the probe reads the volatility curve and nothing else.
Life statistics (no price outcome): native bars birth -> T0; share of lives cut by the machine's
too_early_respan; share reaching x3; native bars T0 -> deletion.
NQ, T0 in 2021-2025, TF 1, 2, 3, 5 (native bar short enough for the clock to be resolved).
Selection stated: the field holds only zones that reached T0.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
MK = ROOT / 'data/market/NQ'
H = np.load(MK / 'high.npy'); L = np.load(MK / 'low.npy'); TS = np.load(MK / 'close_ts_utc_ns.npy')
et = pd.to_datetime(TS, utc=True).tz_convert('America/New_York'); MOD = (et.hour * 60 + et.minute).to_numpy()
BK = [('18:00-03:00 night', lambda c: (c >= 18 * 60) | (c < 3 * 60)), ('03:00-08:00 Europe', lambda c: (c >= 180) & (c < 480)),
      ('08:00-09:30 pre-open', lambda c: (c >= 480) & (c < 570)), ('09:30-10:30 open', lambda c: (c >= 570) & (c < 630)),
      ('10:30-15:00 day', lambda c: (c >= 630) & (c < 900)), ('15:00-17:00 close', lambda c: (c >= 900) & (c < 1020))]

rows = []
for tf in (1, 2, 3, 5):
    cell = ROOT / f'data/field/NQ/cells/tf_{tf:04d}'
    p = pq.read_table(cell / 'passports.parquet', columns=['riz_id', 'precursor_formed_spine_pos', 't0_ts_ns']).to_pandas()
    p = p[pd.to_datetime(p.t0_ts_ns).dt.year.between(2021, 2025) & (p.precursor_formed_spine_pos > 0)]
    e = pq.read_table(cell / 'events.parquet', columns=['riz_id', 'event_kind', 'native_bar_index', 'deletion_cause']).to_pandas()
    e = e[e.riz_id.isin(p.riz_id)]
    g = lambda k: e[e.event_kind == k].groupby('riz_id').native_bar_index.min()
    d = pd.DataFrame(dict(b0=g('precursor_formed'), t0=g('t0'), dl=g('deleted'))).join(p.set_index('riz_id').precursor_formed_spine_pos)
    d['x3'] = d.index.isin(e[e.event_kind == 'x3_t0'].riz_id)
    d['early'] = d.index.map(e[e.event_kind == 'deleted'].set_index('riz_id').deletion_cause) == 'too_early_respan'
    q = d.precursor_formed_spine_pos.to_numpy().astype(np.int64)
    r0 = np.array([H[a - tf + 1:a + 1].max() - L[a - tf + 1:a + 1].min() for a in q])
    nxt = np.array([np.median([H[a + 1 + j * tf:a + 1 + (j + 1) * tf].max() - L[a + 1 + j * tf:a + 1 + (j + 1) * tf].min() for j in range(10)]) for a in q])
    d['rho'] = nxt / np.where(r0 > 0, r0, np.nan); d['clock'] = MOD[q]; d['age'] = d.t0 - d.b0; d['post'] = d.dl - d.t0
    rows.append(d)
d = pd.concat(rows).dropna(subset=['rho'])
d['rt'] = pd.qcut(d.rho, 3, labels=['rho low', 'rho mid', 'rho high'])
print('objects %d | rho quartiles %.2f / %.2f / %.2f' % ((len(d),) + tuple(d.rho.quantile([.25, .5, .75]))))
print('\nRAW by clock of birth:          n   rho p50 | bars birth->T0 p50 | too-early | x3    | bars T0->death p50')
for nm, f in BK:
    x = d[f(d.clock)]
    print('  %-22s %6d   %5.2f   | %8.0f           | %.3f     | %.3f | %.0f' % (nm, len(x), x.rho.median(), x.age.median(), x.early.mean(), x.x3.mean(), x.post.median()))
print('\nBY rho alone:')
for rt, x in d.groupby('rt', observed=True):
    print('  %-22s %6d   %5.2f   | %8.0f           | %.3f     | %.3f | %.0f' % (rt, len(x), x.rho.median(), x.age.median(), x.early.mean(), x.x3.mean(), x.post.median()))
print('\nINSIDE rho terciles, by clock (bars birth->T0 p50 | too-early | x3):')
for rt, y in d.groupby('rt', observed=True):
    print(' ' + str(rt) + ': ' + ' ; '.join('%s %.0f|%.2f|%.2f' % (nm.split()[1], y[f(y.clock)].age.median(), y[f(y.clock)].early.mean(), y[f(y.clock)].x3.mean()) for nm, f in BK))
