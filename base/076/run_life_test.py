import numpy as np
import life as L

RNG = np.random.default_rng(20260914)
BASE = ['close_s', 'sigma', 'imp_size', 'imp_shape', 'imp_close_pos']
EXT = BASE + L.LF

D = {}
for i, tf in [('NQ', 54), ('ES', 54), ('YM', 54), ('NQ', 30)]:
    D[(i, tf)] = L.attach(i, tf)

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
Xb, yb, _ = mk(tr, BASE); Xe, ye, _ = mk(tr, EXT)
bb, be = ols(Xb, yb), ols(Xe, ye)
print('trained on NQ tf54 2006-2018, n=%d' % len(yb))
print('life coefficients:')
for f, c in zip(EXT, be[1:]):
    if f in L.LF: print('   %-14s %+.4f' % (f, c))
print()
print('%-12s %6s %11s %11s %12s %s' % ('transfer', 'n', 'MSE base', 'MSE ext',
                                       'delta MSE', '90% block CI'))
for k, d in D.items():
    te = [r for r in d if r['date'] >= '2019-01-01']
    lab = 'NQ tf54*' if k == ('NQ', 54) else '%s tf%d' % k
    if len(te) < 40:
        print('%-12s %6d  too few' % (lab, len(te))); continue
    _, _, o1 = mk(te, BASE); _, _, o2 = mk(te, EXT)
    sub = [r for r, m in zip(te, o1 & o2) if m]
    Xtb, yt, _ = mk(sub, BASE); Xte, _, _ = mk(sub, EXT)
    dd = (yt - pr(bb, Xtb)) ** 2 - (yt - pr(be, Xte)) ** 2
    day = np.array([r['date'] for r in sub]); ud = np.unique(day)
    by = {u: dd[day == u] for u in ud}
    bs = np.array([np.concatenate([by[u] for u in RNG.choice(ud, len(ud))]).mean()
                   for _ in range(1000)])
    p5, p95 = np.percentile(bs, [5, 95])
    print('%-12s %6d %11.4f %11.4f %+12.4f [%+.4f, %+.4f]'
          % (lab, len(yt), ((yt - pr(bb, Xtb)) ** 2).mean(),
             ((yt - pr(be, Xte)) ** 2).mean(), dd.mean(), p5, p95))
print('\n* training object, reported but not part of the criterion')
