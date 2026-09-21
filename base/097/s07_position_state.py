#!/usr/bin/env python3
"""Line 097: every minute of the S-07 position as a new decision (lens of 2026-09-22). Declared before the count.

Question 1 (sufficiency): is (minute, current mark) a sufficient state for the distribution of what is still to come,
  or does the path already travelled keep reproducible information about the future tail?
  Path, scale-free and prefix-only:  peak = best close so far (>= 0), trough = worst close so far (<= 0).
  Two positions at the same minute with the same mark differ exactly by these two numbers (how far it had been up, how far
  it had been down). Regression per cursor, one row per session:
      outcome ~ decile-of-mark dummies + mark + peak + trough        (HC errors)
  outcomes: remaining result R (to the normal exit), right tail 1[R >= +10 candles], left tail 1[R <= -10].
  Sufficient state <=> the peak and trough coefficients are zero on NQ, ES and YM. Path matters <=> one sign on the three
  indexes with |t| >= 3 on two, at more than one cursor.
Question 2 (the surface is not a stop table): what is read at a point is a DISTRIBUTION - mean, right tail, left tail -
  because a point with mean <= 0 may still own a rare large right tail at bounded left risk. Printed for NQ on fixed
  mark bins, the same on ES and YM only as the zero-crossing of the mean.
No stop is chosen here. Candles = median true range of the 30 bars before the decision. 2006-2026, gross.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MIN = 60_000_000_000
CUR = (5, 10, 15, 20, 30, 45, 60, 90)
BINS = [-1e9, -8, -4, -1, 1, 4, 8, 1e9]


def paths(inst):
    mk = ROOT / f'data/market/{inst}'
    O, H, L, C, TS = (np.load(mk / f'{n}.npy') for n in ('open', 'high', 'low', 'close', 'close_ts_utc_ns'))
    et = pd.to_datetime(TS, utc=True).tz_convert('America/New_York')
    mod = (et.hour * 60 + et.minute).to_numpy(); yr = et.year.to_numpy(); pos = np.arange(len(TS))
    k = pos[mod == 9 * 60 + 33]; k = k[(k > 61) & (k + 123 < len(TS))]
    k = k[(TS[k] - TS[k - 30] == 30 * MIN) & (TS[k + 122] - TS[k] == 122 * MIN)]
    P, Y = [], []
    for g in k:
        hi, lo = H[g - 29:g + 1].max(), L[g - 29:g + 1].min(); s = 1.0 if C[g] > (hi + lo) / 2 else -1.0
        pc = C[g - 30:g]; cand = np.median(np.maximum(H[g - 29:g + 1], pc) - np.minimum(L[g - 29:g + 1], pc))
        if cand <= 0:
            continue
        P.append(s * (C[g + 1:g + 122] - O[g + 1]) / cand); Y.append(yr[g])
    return np.array(P), np.array(Y)


def ols(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b
    Xi = np.linalg.pinv(X.T @ X); V = Xi @ (X.T * r ** 2) @ X @ Xi
    return b, b / np.sqrt(np.maximum(np.diag(V), 1e-18))


if __name__ == '__main__':
    store = {}
    for inst in ['NQ', 'ES', 'YM']:
        P, Y = paths(inst); store[inst] = P
        print(f'\n##### {inst}: sessions {len(P)} | is (minute, mark) a sufficient state? coefficients of PEAK and TROUGH (t), by cursor')
        print('  min | remaining: peak  trough      | right tail R>=+10: peak  trough | left tail R<=-10: peak  trough')
        for t in CUR:
            j = t - 1; mark = P[:, j]; peak = np.maximum(P[:, :j + 1].max(1), 0); tr = np.minimum(P[:, :j + 1].min(1), 0); R = P[:, -1] - mark
            dec = pd.qcut(pd.Series(mark).rank(method='first'), 10, labels=False).to_numpy()
            D = np.eye(10)[dec]
            X = np.column_stack([D, mark, peak, tr])
            out = []
            for y in (R, (R >= 10).astype(float), (R <= -10).astype(float)):
                b, tt = ols(y, X); out.append('%+.3f (%4.1f)  %+.3f (%4.1f)' % (b[-2], tt[-2], b[-1], tt[-1]))
            print('  %3d | ' % t + ' | '.join(out))
    P = store['NQ']
    print('\n##### NQ: what a point of the surface OWNS - mean remaining | P(R >= +10) | P(R <= -10) | n ; rows = minute, columns = mark bins (candles)')
    print('  min | ' + ' | '.join('%-22s' % ('%s..%s' % ('' if a < -1e8 else '%+d' % a, '' if b > 1e8 else '%+d' % b)) for a, b in zip(BINS[:-1], BINS[1:])))
    for t in CUR:
        j = t - 1; mark = P[:, j]; R = P[:, -1] - mark; cells = []
        for a, b in zip(BINS[:-1], BINS[1:]):
            m_ = (mark >= a) & (mark < b); r = R[m_]
            cells.append('%+5.2f %.2f %.2f %4d    ' % (r.mean(), (r >= 10).mean(), (r <= -10).mean(), m_.sum()) if m_.sum() >= 40 else '        --            ')
        print('  %3d | ' % t + ' | '.join(cells))
    print('\nmean remaining by mark bin at minute 15 and 30, three indexes (does the zero-crossing sit in the same place?):')
    for inst in ['NQ', 'ES', 'YM']:
        P = store[inst]
        for t in (15, 30):
            mark = P[:, t - 1]; R = P[:, -1] - mark
            print('  %s +%d: ' % (inst, t) + ' | '.join('%+.2f' % R[(mark >= a) & (mark < b)].mean() if ((mark >= a) & (mark < b)).sum() >= 40 else '  -- ' for a, b in zip(BINS[:-1], BINS[1:])))
