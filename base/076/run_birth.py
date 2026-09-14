import numpy as np
from pathlib import Path
import birth as B

RNG = np.random.default_rng(20260914)
BF = ['imp_size', 'imp_shape', 'imp_close_pos']
LOC = ['close_s', 'sigma']

D = {}
for i, tf in [('NQ', 54), ('ES', 54), ('YM', 54), ('NQ', 30)]:
    D[(i, tf)] = B.attach(i, tf)

def mk(rows, feats):
    X = np.array([[r[f] for f in feats] for r in rows], float)
    y = np.array([r['drift'] for r in rows], float)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    return X[ok], y[ok], ok

def ols(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0]

def pr(b, X):
    return np.column_stack([np.ones(len(X)), X]) @ b

tr = [r for r in D[('NQ', 54)] if r['date'] <= '2018-12-31']
Xb, yb, _ = mk(tr, LOC); Xe, ye, _ = mk(tr, LOC + BF)
bb, be = ols(Xb, yb), ols(Xe, ye)
print('trained on NQ tf54 2006-2018, n=%d / %d' % (len(yb), len(ye)))
print('birth coefficients (standardised scale of the raw features):')
for f, c in zip(LOC + BF, be[1:]):
    print('   %-14s %+.4f' % (f, c))
print()
print('%-12s %6s %11s %11s %12s %s' % ('transfer', 'n', 'MSE base', 'MSE ext',
                                       'delta MSE', '90% block CI'))
for k, d in D.items():
    te = [r for r in d if r['date'] >= '2019-01-01']
    if k == ('NQ', 54):
        lab = 'NQ tf54*'     # same object as training, reported not counted
    else:
        lab = '%s tf%d' % k
    if len(te) < 40:
        print('%-12s %6d  too few' % (lab, len(te))); continue
    Xtb, yt, ok1 = mk(te, LOC); Xte, yt2, ok2 = mk(te, LOC + BF)
    keep = ok1 & ok2
    sub = [r for r, m in zip(te, keep) if m]
    Xtb, yt, _ = mk(sub, LOC); Xte, _, _ = mk(sub, LOC + BF)
    eb = (yt - pr(bb, Xtb)) ** 2; ee = (yt - pr(be, Xte)) ** 2
    dd = eb - ee
    day = np.array([r['date'] for r in sub]); ud = np.unique(day)
    by = {u: dd[day == u] for u in ud}
    bs = np.array([np.concatenate([by[u] for u in RNG.choice(ud, len(ud))]).mean()
                   for _ in range(1000)])
    p5, p95 = np.percentile(bs, [5, 95])
    print('%-12s %6d %11.4f %11.4f %+12.4f [%+.4f, %+.4f]'
          % (lab, len(yt), eb.mean(), ee.mean(), dd.mean(), p5, p95))
print('\n* training object, reported but not part of the criterion')
