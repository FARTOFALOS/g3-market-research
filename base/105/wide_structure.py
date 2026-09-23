"""D3: структура дальнейшего пути широкой сессии — порядок пробоя, вынос, возврат."""
import numpy as np, pandas as pd
from tape import load_minutes
cal = pd.read_csv('calendar_nq.csv').set_index('date')
df = load_minutes('2005-12-01', '2026-01-01')
M0 = 570; TT = list(range(630, 901, 30))
objs = []
for d, g in df.groupby('date'):
    if d not in cal.index or cal.at[d, 'status'] != 'regular': continue
    w = g[(g['mod'] >= M0) & (g['mod'] < 960)]
    if len(w) == 390: objs.append((d, w))
Rs = pd.Series({d: w.h.max() - w.l.min() for d, w in objs}).sort_index(); N = Rs.rolling(20).median().shift(1)
rows = []
for d, w in objs:
    if pd.isna(N.get(d)): continue
    o = w.o.to_numpy(); h = w.h.to_numpy(); l = w.l.to_numpy(); c = w.c.to_numpy()
    rh = np.maximum.accumulate(h); rl = np.minimum.accumulate(l)
    for t in TT:
        k = t - M0 - 1
        H, L = rh[k], rl[k]; R = H - L; e = o[k + 1]; rho = R / N[d]
        if not (L < e < H) or not (0.6 <= rho <= 0.9 or rho > 1.2): continue
        iH = np.nonzero(h[:k + 1] >= H)[0][0]; iL = np.nonzero(l[:k + 1] <= L)[0][0]
        last = 'H' if iH > iL else 'L'
        fu = np.nonzero(h[k + 1:] > H)[0]; fd = np.nonzero(l[k + 1:] < L)[0]
        tu = fu[0] if len(fu) else 10**6; td = fd[0] if len(fd) else 10**6
        rec = dict(date=d, t=t, wide=rho > 1.2, last=last, p0=(e - L) / R, R=R)
        if tu == td == 10**6: rec.update(first='none')
        elif tu == td: rec.update(first='both')
        else:
            side = 'H' if tu < td else 'L'; j = k + 1 + min(tu, td)
            rest_h = h[j:]; rest_l = l[j:]; rest_c = c[j:]
            ext = (rest_h.max() - H) / R if side == 'H' else (L - rest_l.min()) / R
            back = bool((rest_c < H).any()) if side == 'H' else bool((rest_c > L).any())
            rec.update(first=side, wait=min(tu, td) + 1, ext=ext, back=back, end_beyond=((c[-1] - H) if side == 'H' else (L - c[-1])) / R)
        rows.append(rec)
r = pd.DataFrame(rows)
r['ep'] = pd.cut(r.date // 10000, [2005, 2012, 2019, 2025], labels=['2006-12', '2013-19', '2020-25'])
r['pos'] = pd.cut(r.p0, [0, 1 / 3, 2 / 3, 1], labels=['low third', 'mid', 'high third'])
r.to_csv('wide_structure.csv', index=False)
def fork(g):
    ok = g[g['first'].isin(['H', 'L'])]
    y = (ok['first'] == 'H').astype(float); dY = y.mean() - ok.p0.mean(); se = np.sqrt((ok.p0 * (1 - ok.p0)).sum()) / len(ok)
    return pd.Series({'n': len(ok), 'none': round((g['first'] == 'none').mean(), 2), 'Y': round(y.mean(), 3), 'p0': round(ok.p0.mean(), 3), 'dY': round(dY, 3), 'z': round(dY / se, 1)})
pd.set_option('display.width', 250)
print('order of first break vs fair line, by width / last-set extreme / epoch:')
print(r.groupby(['wide', 'last', 'ep'], observed=True).apply(fork, include_groups=False).unstack('ep')[['dY', 'z', 'n']].to_string())
print('\nby width / position / epoch:')
print(r.groupby(['wide', 'pos', 'ep'], observed=True).apply(fork, include_groups=False).unstack('ep')[['dY', 'z', 'n']].to_string())
b = r[r['first'].isin(['H', 'L'])]
print('\nafter the first break: median wait (min), median extension beyond (share of R_t), share returning inside, median close beyond (share of R_t)')
print(b.groupby(['wide', 'first', 'ep'], observed=True).agg(n=('ext', 'size'), wait=('wait', 'median'), ext=('ext', 'median'), back=('back', 'mean'), end=('end_beyond', 'median')).round(3).unstack('ep').to_string())
