"""One joint test instead of thirteen median splits.

Question: is there any observable state available before the decision, NOT a
function of the current coordinate and local sigma, that reproducibly changes
the law of future movement?

Baseline  : drift ~ close_s + sigma          (the local state, and nothing else)
Extended  : + every prefix and external coordinate available at T0'
Metric    : out-of-sample mean squared error on a frozen transfer, paired
            difference, day-block bootstrap. Trained on NQ tf54 2006-2018,
            transferred unchanged to ES/YM tf54 and NQ tf30, 2019-2026.

Power is reported first, so that a null is interpretable rather than assumed.
"""
import numpy as np
from pathlib import Path
import run_external as R           # builds D with prefix + external coords

D = R.D
RNG = np.random.default_rng(20260914)

LOCAL = ['close_s', 'sigma']
EXTRA = ['runs_rate', 'exit_share', 'far_share', 'inside_share', 'plen',
         'depth_s', 'sig_ratio', 'field_zones', 'cross_mean', 'cross_absdiff',
         'session_min', 'session_pos']


def frame(rows, feats, target):
    X = np.array([[r[f] for f in feats] for r in rows], float)
    y = np.array([r[target] for r in rows], float)
    d = np.array([r.get('date', '') for r in rows])
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    return X[ok], y[ok], ok


def ols(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def pred(b, X):
    return np.column_stack([np.ones(len(X)), X]) @ b


for target in ['drift', 'ac1']:
    print('\n' + '=' * 86)
    print('TARGET %s' % target)
    print('=' * 86)
    ref = D[('NQ', 54)]
    y_all = np.array([r[target] for r in ref], float)
    y_all = y_all[np.isfinite(y_all)]
    sd = y_all.std()
    print('per-unit sd on NQ tf54: %.4f' % sd)
    for n in [250, 500, 1000, 2000]:
        # smallest mean shift detectable at 80% power, two-sided 5%, median split
        mde = 2.8 * sd * np.sqrt(2.0 / (n / 2))
        print('   n=%-5d  smallest detectable shift between halves: %.3f  (%.2f sd)'
              % (n, mde, mde / sd))

    # frozen transfer
    tr = [r for r in D[('NQ', 54)] if r.get('date', '9') <= '2018-12-31']
    if len(tr) < 60:
        tr = D[('NQ', 54)]
        note = 'no date field on rows: trained on all of NQ tf54 (labelled)'
    else:
        note = 'trained on NQ tf54 2006-2018'
    Xb, yb, _ = frame(tr, LOCAL, target)
    Xe, ye, _ = frame(tr, LOCAL + EXTRA, target)
    bb, be = ols(Xb, yb), ols(Xe, ye)
    print('\n%s ; n_train baseline=%d extended=%d ; features %d vs %d'
          % (note, len(yb), len(ye), len(LOCAL), len(LOCAL) + len(EXTRA)))
    print('%-12s %6s %12s %12s %14s %s'
          % ('transfer', 'n', 'MSE base', 'MSE ext', 'delta MSE', '90% block CI'))
    for k, d in D.items():
        te = [r for r in d if r.get('date', '0') >= '2019-01-01'] or d
        Xtb, ytb, okb = frame(te, LOCAL, target)
        Xte, yte, oke = frame(te, LOCAL + EXTRA, target)
        keep = okb & oke
        Xtb, _, _ = frame([r for r, m in zip(te, keep) if m], LOCAL, target)
        Xte, yt, _ = frame([r for r, m in zip(te, keep) if m], LOCAL + EXTRA, target)
        if len(yt) < 40:
            print('%-12s %6d  too few' % ('%s tf%d' % k, len(yt))); continue
        eb = (yt - pred(bb, Xtb)) ** 2
        ee = (yt - pred(be, Xte)) ** 2
        dd = eb - ee                       # positive means extended is better
        day = np.array([r['date'] for r, m in zip(te, keep) if m])
        ud = np.unique(day); by = {u: dd[day == u] for u in ud}
        bs = np.array([np.concatenate([by[u] for u in RNG.choice(ud, len(ud))]).mean()
                       for _ in range(1000)])
        p5, p95 = np.percentile(bs, [5, 95])
        print('%-12s %6d %12.4f %12.4f %+14.4f [%+.4f, %+.4f]'
              % ('%s tf%d' % k, len(yt), eb.mean(), ee.mean(), dd.mean(), p5, p95))
