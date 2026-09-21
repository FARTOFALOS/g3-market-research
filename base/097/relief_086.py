#!/usr/bin/env python3
"""Line 097: 086 put beside S-07 - not to rescue it and not to close it, but to learn what separates a real non-random
place from an extremum found by a search. Declared before the count, 2026-09-22. Nothing in 086 is re-tuned: frozen event,
frozen fork, frozen medians of the four coordinates (cells086.json); only the READING is the one this line developed.

What S-07 has (measured in this line) and what a search extremum need not have:
  T  transfer: one sign on other indexes and in other epochs, in own units
  R  relief: on the whole declared map the place stands out and the map is not noise - the z of a cell in one period says
     something about its z in another period (correlation over cells clearly above 0)
  N  neighbourhood: the effect is graded along the place's OWN coordinates (S-07: smooth peak over decision minutes), so the
     cells one bit away share the sign; an extremum of a search is an isolated cell
Reading: per cell X = win - p0 toward b (086 sign convention; the candidate is the NEGATIVE side, continuation), z with
day-clustered error, on NQ 2006-2018, NQ 2020-2025 (where the cell was selected), ES and YM 2019-2025, NQ holdout.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'base/086'))
import fork086 as F

PER = ['NQ_development', 'NQ_search', 'ES_evaluation', 'YM_evaluation', 'NQ_holdout']
SEL = 1                                                     # u+ k- ttc- r-


def zday(x, day):
    d = pd.DataFrame(dict(x=x, d=day)).groupby('d').x.agg(['sum', 'size']); m = x.mean()
    se = np.sqrt(((d['sum'] - m * d['size']) ** 2).sum()) / d['size'].sum()
    return m, (m / se if se > 0 else np.nan), len(d)


if __name__ == '__main__':
    med = json.loads((ROOT / 'work/086/cells086.json').read_text(encoding='utf-8'))['medians']
    Z = {}; rows = {}
    for per in PER:
        g = F.geometry(per); ok = np.isfinite(g.ttc.to_numpy()); g = g[ok].reset_index(drop=True); g.attrs['instrument'] = F.PERIODS[per][0]
        code, _ = F.walk(g); xl, xh, gl, gh = F.per_film(g, code); x = (xl + xh) / 2; gg = (gl + gh) / 2
        cid = F.cell_ids(g, med); day = g.t0_day.to_numpy()
        z = np.full(16, np.nan); info = {}
        for c in range(16):
            m = cid == c
            if m.sum() >= 30:
                mx, zz, nd = zday(x[m], day[m]); z[c] = zz if nd >= 20 else np.nan
                info[c] = (int(m.sum()), nd, mx, zz, gg[m].mean())
        Z[per] = z; rows[per] = info
        mx, zz, nd = zday(x, day)
        print('%-15s films %6d days %4d | whole population X %+.4f (z %+.1f)' % (per, len(g), nd, mx, zz))
    print('\nz of X by cell (negative = continuation side wins more than its fair line). * = the cell selected in 086')
    print('%-16s ' % 'cell' + ' '.join('%15s' % p for p in PER))
    for c in range(16):
        print('%-16s ' % (F.cell_label(c) + (' *' if c == SEL else '')) + ' '.join(
            ('%+6.1f (n%5d)' % (rows[p][c][3], rows[p][c][0])) .rjust(15) if c in rows[p] and np.isfinite(Z[p][c]) else '--'.rjust(15) for p in PER))
    print('\nR  is the map more than noise? correlation of cell z between periods (cells with z in both):')
    for a, b in [('NQ_development', 'NQ_search'), ('NQ_search', 'ES_evaluation'), ('NQ_search', 'YM_evaluation'), ('NQ_development', 'ES_evaluation'), ('NQ_development', 'YM_evaluation')]:
        k = np.isfinite(Z[a]) & np.isfinite(Z[b]); k2 = k.copy(); k2[SEL] = False
        print('   %-15s vs %-15s r %+.2f over %d cells | without the selected cell %+.2f' % (a, b, np.corrcoef(Z[a][k], Z[b][k])[0, 1], k.sum(), np.corrcoef(Z[a][k2], Z[b][k2])[0, 1]))
    print('\nT  the selected cell itself: X (z) [G pts] by period')
    print('   ' + ' | '.join('%s %+.3f (%+.1f) [%+.1f]' % (p.split('_')[0] + '_' + p.split('_')[1][:4], rows[p][SEL][2], rows[p][SEL][3], rows[p][SEL][4]) if SEL in rows[p] else p + ' --' for p in PER))
    print('\nN  neighbours of the selected cell, one coordinate flipped: z by period')
    for bit, nm in enumerate(F.COORDS):
        c = SEL ^ (1 << bit)
        print('   flip %-3s -> %-16s ' % (nm, F.cell_label(c)) + ' '.join(('%+6.1f' % Z[p][c]).rjust(15) if np.isfinite(Z[p][c]) else '--'.rjust(15) for p in PER))
