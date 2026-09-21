#!/usr/bin/env python3
"""Line 097: what is the CLOCK of the decay of the opening response - the calendar minute or the amount of process already
realised?  Declared before the count, 2026-09-22 (lens of the human).

At equal calendar minute, location u and realised range rg, two sessions can be at different AGES of the process:
   stale = (t - t_peak) / t          share of the elapsed time since the side last made a new best close (0 = extending now)
   chop  = log(path length / rg)     how many times the path has already re-walked its own range (0 = one clean leg)
Neither is recoverable from (minute, u, rg), and neither is the order of extremes tested before.
If the clock of the decay is process-internal, then at a fixed minute the directional residual must be LOWER for the
older process: negative coefficients of stale and chop in   R/rg ~ deciles(u) + u + log(rg) + stale + chop.
Rule: one sign on NQ, ES, YM with |t| >= 3 on two, at more than one cursor. If it holds, the faster decay of the modern
epoch should show up as an older process at the same minute (printed: median chop and stale by epoch).
If both coefficients are zero, the calendar minute is the clock at this resolution, and the low-dimensional state survives
one more honest coordinate.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s07_position_state import paths, ols

CUR = (5, 10, 15, 20, 30, 45)
EP = ((2006, 2012), (2013, 2019), (2020, 2026))


def st(P, t):
    j = t - 1; seg = np.concatenate([np.zeros((len(P), 1)), P[:, :j + 1]], axis=1)
    pk, tr = seg.max(1), seg.min(1); rg = pk - tr; ok = rg > 0
    u = (P[:, j] - tr) / np.where(ok, rg, 1)
    stale = (t - seg.argmax(1)) / t
    pl = np.abs(np.diff(seg, axis=1)).sum(1); chop = np.log(np.where(ok, pl / np.where(ok, rg, 1), 1))
    return u, rg, stale, chop, P[:, -1] - P[:, j], ok


if __name__ == '__main__':
    S = {i: paths(i) for i in ['NQ', 'ES', 'YM']}
    for inst, (P, Y) in S.items():
        print(f'\n##### {inst}: coefficients at equal minute, location and realised range (t)')
        print('  min | mean R/rg: stale        chop          | right tail: stale   chop       | left tail: stale    chop')
        for t in CUR:
            u, rg, sl, ch, R, ok = st(P, t); u, rg, sl, ch, R = u[ok], rg[ok], sl[ok], ch[ok], R[ok]
            dec = pd.qcut(pd.Series(u).rank(method='first'), 10, labels=False).to_numpy()
            X = np.column_stack([np.eye(10)[dec], u, np.log(rg), sl, ch]); y = np.clip(R / rg, -20, 20)
            out = []
            for yy in (y, (y >= .5).astype(float), (y <= -.5).astype(float)):
                b, tt = ols(yy, X); out.append('%+.3f (%4.1f)  %+.3f (%4.1f)' % (b[-2], tt[-2], b[-1], tt[-1]))
            print('  %3d | ' % t + ' | '.join(out))
    print('\nage of the process at the same calendar minute, by epoch (three indexes pooled): median chop | median stale')
    for t in (10, 20, 30):
        row = []
        for a, b in EP:
            ch_, sl_ = [], []
            for inst, (P, Y) in S.items():
                m = (Y >= a) & (Y <= b); u, rg, sl, ch, R, ok = st(P[m], t); ch_.append(ch[ok]); sl_.append(sl[ok])
            row.append('%d-%d: %.3f | %.2f' % (a, b, np.median(np.concatenate(ch_)), np.median(np.concatenate(sl_))))
        print('  min %2d  ' % t + '   '.join(row))
