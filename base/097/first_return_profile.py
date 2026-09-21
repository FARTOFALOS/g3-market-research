#!/usr/bin/env python3
"""Line 097: first price reading of the parent object. Declared before the count, 2026-09-22.

Parent unit: a fresh gap and the first bar that comes back to it (4-5 native bars). The census showed ONE continuous
return, not four states, so no separator is searched. Outcome language = a PROFILE: the move along the impulse after the
contact, as a function of where the contact bar closed (clo, in zone widths from the near edge), in own units (r0 = range
of the bar that closed the gap), at horizons of 1, 2, 5, 10, 20 native bars.
Two honest entries:
  M  market: open of the native bar after the contact bar (everything about the contact is known);
  E  edge:   resting order at the near edge of the gap, placeable at the birth close; the contact fills it
             (99.9 % of gaps are contacted within 1-2 bars, so the fill is not selected by the future).
What would make this one process and not a set of TF effects: the SAME function of clo on every TF (data collapse in own
units), not merely one sign.
Most dangerous world, named in advance: the profile is flat at zero for every clo on every TF - then the tree is real and
empty. Errors clustered by session. NQ, births 2021-2025. Round trip 1.00 pt is printed in r0 units per TF.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gap_lens import native_bars, ROOT, OUT

SID = np.load(ROOT / 'data/market/NQ/session_id.npy')
TSY = pd.to_datetime(np.load(ROOT / 'data/market/NQ/close_ts_utc_ns.npy')).year.to_numpy()
HZ = (1, 2, 5, 10, 20)
BINS = [(-1e9, -1.0, 'closed beyond the far edge'), (-1.0, 0.0, 'closed inside the gap'), (0.0, 1.0, 'closed just back, 0..1 w'),
        (1.0, 3.0, 'closed 1..3 w back'), (3.0, 1e9, 'closed > 3 w back')]


def tcl(v, sid):
    x = pd.DataFrame(dict(v=v, s=sid)).dropna(); d = x.groupby('s').v.agg(['sum', 'size']); m = x.v.mean()
    se = np.sqrt(((d['sum'] - m * d['size']) ** 2).sum()) / d['size'].sum()
    return m, m / se if se > 0 else np.nan


rows = []
for tf in (1, 5, 15, 60):
    o, h, l, c, end = native_bars('NQ', tf)
    g = pd.read_parquet(OUT / f'gaps_NQ_{tf}.parquet'); g = g[(g.fc >= 0) & (g.fc + 21 < len(o)) & (g.r0 >= 0.75) & ((g.zt - g.zb) >= 0.25)]   # a C3 bar of zero range makes own units meaningless
    g = g[(TSY[g.birth_pos.to_numpy()] >= 2021) & (TSY[g.birth_pos.to_numpy()] <= 2025)]
    fc = g.fc.to_numpy(); s = np.where(g.bull, 1.0, -1.0); r0 = g.r0.to_numpy()
    edge = np.where(g.bull, g.zt, g.zb)
    d = pd.DataFrame(dict(tf=tf, clo=g.clo.to_numpy(), sid=SID[end[fc]], r0=r0, w=(g.zt - g.zb).to_numpy() / r0,
                          run=g.run.to_numpy(), age=(g.fc - g.birth).to_numpy()))
    for k in HZ:
        d[f'M{k}'] = s * (c[fc + k] - o[fc + 1]) / r0
        d[f'E{k}'] = s * (c[fc + k] - edge) / r0
    d['E0'] = s * (c[fc] - edge) / r0
    rows.append(d)
    print('TF %2d: contacts %d | r0 median %.2f pt -> round trip 1.00 pt = %.3f r0 | zone width %.2f r0' % (tf, len(d), np.median(r0), 1 / np.median(r0), d.w.median()))
d = pd.concat(rows); d.to_parquet(OUT / 'first_return_profile.parquet')
print('ENTRY M (open of the bar after the contact): mean move along the impulse, r0 units (t by session); horizons %s' % (HZ,))
for lo, hi, nm in BINS:
    x = d[(d.clo >= lo) & (d.clo < hi)]
    print('  %-28s n %7d | ' % (nm, len(x)) + ' | '.join('%+.3f (%.1f)' % tcl(x[f'M{k}'], x.sid) for k in HZ))
print('  same function on every TF? horizon 10, by the same bins:')
for tf in (1, 5, 15, 60):
    y = d[d.tf == tf]
    print('     TF %2d: ' % tf + ' | '.join('%+.3f (%.1f)' % tcl(y[(y.clo >= lo) & (y.clo < hi)].M10, y[(y.clo >= lo) & (y.clo < hi)].sid) for lo, hi, _ in BINS))
print('ENTRY E (resting order at the near edge, placed at the birth close; binned ONLY by what is known before the fill):')
print('  all contacts: at the contact close %+.3f | ' % d.E0.mean() + ' | '.join('%+.3f (%.1f)' % tcl(d[f'E{k}'], d.sid) for k in HZ))
for nm, m in [('contact on the very next bar (age 1)', d.age == 1), ('contact later (age >= 2)', d.age >= 2),
              ('thin gap, w < 0.3 r0', d.w < .3), ('wide gap, w >= 0.8 r0', d.w >= .8)]:
    x = d[m]; print('  %-38s n %7d | close %+.3f | ' % (nm, len(x), x.E0.mean()) + ' | '.join('%+.3f (%.1f)' % tcl(x[f'E{k}'], x.sid) for k in HZ))
for tf in (1, 5, 15, 60):
    y = d[d.tf == tf]; print('     TF %2d all contacts: ' % tf + ' | '.join('%+.3f (%.1f)' % tcl(y[f'E{k}'], y.sid) for k in HZ))
