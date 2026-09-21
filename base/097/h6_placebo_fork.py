#!/usr/bin/env python3
"""097 H6: is the fair line at the contact a property of the scene or of the question?

p0 = s/(d+s) is the optional-stopping identity of ANY driftless price, whatever
its volatility: it belongs to "no drift between these two levels on this time
scale", not to the levels. If the same forks placed at ordinary minutes sit on
the same line, then 097 found no scene-specific departure from the generic law
of the tape, and nothing about U, b or the contact.

Placebo  for every H4 moment: the bar with the same ET clock on the nearest
         earlier session (1..5 calendar days back), same side, same d and s in
         points laid around open[k'+1]; same tick-pass rule, same 16:00 cutoff.
Power    Wald: mean money of a stopped bet = drift per bar x mean bars in the
         bet. Printed: mean bars of the H4 fork and the drift it could pay for.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from race_money import cutoffs, O, H, L, C, TS, SID, MIN

OUT = Path(__file__).resolve().parents[2] / 'work' / '097'
TICK = 0.25
DAY = 24 * 60 * MIN

if __name__ == '__main__':
    x = pd.read_parquet(OUT / 'h4_rows.parquet'); x = x[x.year <= 2025]
    cut, _ = cutoffs()
    rows = []
    for r in x.itertuples():
        kp = -1
        for j in range(1, 6):
            i = int(np.searchsorted(TS, TS[int(r.k)] - j * DAY))
            if i < len(TS) and TS[i] == TS[int(r.k)] - j * DAY:
                kp = i; break
        if kp < 0:
            continue
        ce = int(cut[kp])
        if ce <= kp or TS[kp + 1] - TS[kp] != MIN:
            continue
        s_ = 1.0 if r.side == 'north' else -1.0
        e0 = O[kp + 1]
        seg = slice(kp + 1, ce + 1)
        ts = TS[seg]; brk = np.flatnonzero(np.diff(ts) != MIN); n = (brk[0] + 1) if brk.size else len(ts)
        hi = s_ * (H[seg][:n] - e0) if s_ > 0 else s_ * (L[seg][:n] - e0)
        lo = s_ * (L[seg][:n] - e0) if s_ > 0 else s_ * (H[seg][:n] - e0)
        iw = np.flatnonzero(hi >= r.d + TICK); il = np.flatnonzero(lo <= -r.s - TICK)
        jw = iw[0] if iw.size else 10 ** 9; jl = il[0] if il.size else 10 ** 9
        if jw == jl == 10 ** 9:
            kind, g, dur = 'time', s_ * (C[seg][n - 1] - e0), n
        elif jl <= jw:
            kind, g, dur = 'loss', min(-r.s - TICK, s_ * (O[seg][jl] - e0)), jl + 1
        else:
            kind, g, dur = 'win', r.d, jw + 1
        rows.append(dict(sid=int(SID[kp]), p0=r.p0, kind=kind, g=g, dur=dur))
    p = pd.DataFrame(rows)
    for name, y in [('SCENE  (H4)', x), ('PLACEBO same clock, ordinary minute', p)]:
        day = y.groupby('sid').g.agg(['sum', 'size']); m = y.g.mean()
        se = np.sqrt(((day['sum'] - m * day['size']) ** 2).sum()) / day['size'].sum()
        print('%-38s n %5d  win %.3f fair %.3f  G %+.3f (t %.2f)  bars mean %.1f p50 %.0f'
              % (name, len(y), (y.kind == 'win').mean(), y.p0.mean(), m, m / se, y.dur.mean(), y.dur.median()))
        q = pd.qcut(y.p0, 5, labels=False)
        print('   by p0 quintile win/fair: ' + ' · '.join(
            '%.3f/%.3f' % ((y[q == i].kind == 'win').mean(), y[q == i].p0.mean()) for i in range(5)))
    mt = x.dur.mean()
    print('\nWald: G = drift x bars. H4 fork lives %.1f bars on average; to pay the 1.00 pt round trip it'
          ' needs a drift of %.3f pt/bar; to show G at t=3 (se %.3f) it needs %.3f pt/bar.'
          % (mt, 1.0 / mt, 0.337, 3 * 0.337 / mt))
