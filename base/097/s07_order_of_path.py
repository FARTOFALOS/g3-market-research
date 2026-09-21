#!/usr/bin/env python3
"""Line 097: does the ORDER of the path matter once location, realised range and the minute are equal?
Declared before the count, 2026-09-22 (lens of the human).

State in today's own units, prefix only, at cursor t (minutes of the S-07 position):
   rg  = peak - trough of closes so far (entry level included)       -> uncertainty coordinate
   u   = (mark - trough) / rg  in [0, 1]                              -> location inside today's own range
   ord = (t_peak - t_trough) / t  in [-1, 1]    +: trough first, then recovery;  -: peak first, then give-back
Two positions with equal (t, u, rg) differ by ord exactly as "fell deep and recovered" differs from "went far up and came back".
Outcome in the same units: R / rg (what is still to come, in today's realised ranges).
   mean, right tail 1[R/rg >= +0.5], left tail 1[R/rg <= -0.5], width |R|/rg.
Regression per cursor and index:  outcome ~ deciles(u) + u + log(rg) + ord   (HC errors, one row per session).
Order is a new honest dimension <=> one sign of the ord coefficient on NQ, ES, YM with |t| >= 3 on two, at more than one cursor.
If it is zero everywhere, the history of S-07 compresses to (minute, location, realised range).
Also printed, because a state must carry over:  (a) is rg a transferable unit - sd of R/rg by epoch and index at two cursors;
(b) "wide path -> wide future on both sides" without any tail threshold - mean |R| (candles) by tercile of rg;
(c) FORM across epochs - mean R/rg by location bin at minutes 10, 20, 30, 60, three indexes pooled, per epoch.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s07_position_state import paths, ols

CUR = (10, 15, 20, 30, 45, 60)
EP = ((2006, 2012), (2013, 2019), (2020, 2026))


def state(P, t):
    j = t - 1; seg = np.concatenate([np.zeros((len(P), 1)), P[:, :j + 1]], axis=1)
    pk, tr = seg.max(1), seg.min(1); rg = pk - tr
    tp, tt = seg.argmax(1), seg.argmin(1)
    ok = rg > 0
    u = np.where(ok, (P[:, j] - tr) / np.where(ok, rg, 1), np.nan)
    return u, rg, (tp - tt) / t, P[:, -1] - P[:, j], ok


if __name__ == '__main__':
    S = {}
    for inst in ['NQ', 'ES', 'YM']:
        P, Y = paths(inst); S[inst] = (P, Y)
        print(f'\n##### {inst}: sessions {len(P)} | coefficient of ORDER (t) at equal minute, location and realised range')
        print('  min | mean R/rg        | right tail >= +0.5 rg | left tail <= -0.5 rg | width |R|/rg')
        for t in CUR:
            u, rg, od, R, ok = state(P, t)
            u, rg, od, R = u[ok], rg[ok], od[ok], R[ok]
            dec = pd.qcut(pd.Series(u).rank(method='first'), 10, labels=False).to_numpy()
            X = np.column_stack([np.eye(10)[dec], u, np.log(rg), od]); y = R / rg
            out = []
            for yy in (y, (y >= .5).astype(float), (y <= -.5).astype(float), np.abs(y)):
                b, tt = ols(np.clip(yy, -20, 20), X); out.append('%+.3f (%5.1f)' % (b[-1], tt[-1]))
            print('  %3d | ' % t + '   | '.join(out))
    print('\n(a) is today\'s realised range a transferable unit? sd of R/rg at minute 15 | 30, by index and epoch')
    for inst, (P, Y) in S.items():
        row = []
        for a, b in EP:
            m = (Y >= a) & (Y <= b); cells = []
            for t in (15, 30):
                u, rg, od, R, ok = state(P[m], t); cells.append('%.2f' % np.std(np.clip(R[ok] / rg[ok], -20, 20)))
            row.append('%d-%d %s' % (a, b, ' | '.join(cells)))
        print('  %s: ' % inst + '   '.join(row))
    print('\n(b) wide path -> wide future, no tail threshold: mean |R| in candles by tercile of realised range at minute 20')
    for inst, (P, Y) in S.items():
        u, rg, od, R, ok = state(P, 20); q = pd.qcut(pd.Series(rg[ok]).rank(method='first'), 3, labels=False).to_numpy()
        print('  %s: ' % inst + ' / '.join('%.1f' % np.abs(R[ok])[q == i].mean() for i in range(3))
              + '   | mean R by the same terciles: ' + ' / '.join('%+.2f' % R[ok][q == i].mean() for i in range(3)))
    print('\n(c) FORM across epochs, three indexes pooled: mean R/rg by location in today\'s range (u bins 0-.2 .2-.4 .4-.6 .6-.8 .8-1)')
    for t in (10, 20, 30, 60):
        for a, b in EP:
            U, Rn = [], []
            for inst, (P, Y) in S.items():
                m = (Y >= a) & (Y <= b); u, rg, od, R, ok = state(P[m], t); U.append(u[ok]); Rn.append(np.clip(R[ok] / rg[ok], -20, 20))
            U, Rn = np.concatenate(U), np.concatenate(Rn)
            cells = []
            for lo in (0, .2, .4, .6, .8):
                k = (U >= lo) & (U < lo + .2 + (1e-9 if lo == .8 else 0)); v = Rn[k]
                cells.append('%+.3f (%4.1f)' % (v.mean(), v.mean() / (v.std() / np.sqrt(len(v)))))
            print('  min %2d  %d-%d: ' % (t, a, b) + ' | '.join(cells))
