import numpy as np
import life as L

D = {}
for i, tf in [('NQ', 54), ('ES', 54), ('YM', 54), ('NQ', 30)]:
    D[(i, tf)] = L.attach(i, tf)
    print('%s tf%d n=%d with life attached' % (i, tf, len(D[(i, tf)])), flush=True)

def r2(y, X):
    A = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    res = y - A @ b; ss = ((y - y.mean()) ** 2).sum()
    return 1.0 - (res ** 2).sum() / ss if ss > 0 else np.nan

CTRL = L.CUR + L.BF
print('\n1. IS L RECOVERABLE FROM CURRENT STATE + BIRTH?')
print('   R2 of each life coordinate on close_s, sigma, width, and the three birth coords')
print('%-14s %s' % ('life coord', ' '.join('%-12s' % ('%s tf%d' % k) for k in D)))
for f in L.LF:
    cells = []
    for k, d in D.items():
        y = np.array([x[f] for x in d], float)
        X = np.array([[x[c] for c in CTRL] for x in d], float)
        ok = np.isfinite(y) & np.isfinite(X).all(1)
        cells.append('%.4f' % r2(y[ok], X[ok]) if ok.sum() > 30 else '--')
    print('%-14s %s' % (f, ' '.join('%-12s' % c for c in cells)))

print('\n2. SUPPORT: L VARIATION AT COMPARABLE CURRENT STATE')
ref = D[('NQ', 54)]
CS = np.percentile([x['close_s'] for x in ref], [20, 40, 60, 80])
SG = np.percentile([x['sigma'] for x in ref], [33.3, 66.7])
print('%-14s %s' % ('life coord', ' '.join('%-16s' % ('%s tf%d' % k) for k in D)))
for f in L.LF:
    cells = []
    for k, d in D.items():
        v = np.array([x[f] for x in d], float)
        a = np.searchsorted(CS, [x['close_s'] for x in d])
        b = np.searchsorted(SG, [x['sigma'] for x in d])
        tot = np.nanstd(v); w, n = [], []
        for i in range(5):
            for j in range(3):
                m = (a == i) & (b == j) & np.isfinite(v)
                if m.sum() >= 10: w.append(np.nanstd(v[m])); n.append(m.sum())
        cells.append('%.3f / %.3f = %.2f' % (np.average(w, weights=n), tot,
                                             np.average(w, weights=n) / tot) if w else '--')
    print('%-14s %s' % (f, ' '.join('%-16s' % c for c in cells)))

print('\n   spread of each life coordinate on NQ tf54:')
for f in L.LF + ['win_len']:
    v = np.array([x[f] for x in ref], float)
    print('      %-12s p5=%.3f med=%.3f p95=%.3f' % (f, *np.nanpercentile(v, [5, 50, 95])))
