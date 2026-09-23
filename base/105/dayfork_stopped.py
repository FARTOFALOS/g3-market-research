"""D2/D3 заново: вилка дня с цензурой по закрытию (исход = коснувшаяся граница или close), E = 0 без сноса."""
import sys, numpy as np, pandas as pd
from tape import load_minutes
M0 = int(sys.argv[1]) if len(sys.argv) > 1 else 570
cal = pd.read_csv('calendar_nq.csv').set_index('date')
df = load_minutes('2005-12-01', '2026-01-01')
TT = list(range(630, 901, 30))
objs = []
for d, g in df.groupby('date'):
    if d not in cal.index or cal.at[d, 'status'] != 'regular': continue
    w = g[(g['mod'] >= M0) & (g['mod'] < 960)]
    if len(w) == 960 - M0: objs.append((d, w))
Rs = pd.Series({d: w.h.max() - w.l.min() for d, w in objs}).sort_index(); N = Rs.rolling(20).median().shift(1)
rows = []
for d, w in objs:
    if pd.isna(N.get(d)): continue
    o = w.o.to_numpy(); h = w.h.to_numpy(); l = w.l.to_numpy(); c = w.c.to_numpy()
    rh = np.maximum.accumulate(h); rl = np.minimum.accumulate(l)
    for t in TT:
        k = t - M0 - 1
        H, L = rh[k], rl[k]; R = H - L; e = o[k + 1]
        if not (L < e < H): continue
        iH = np.nonzero(h[:k + 1] >= H)[0][0]; iL = np.nonzero(l[:k + 1] <= L)[0][0]
        fu = np.nonzero(h[k + 1:] > H)[0]; fd = np.nonzero(l[k + 1:] < L)[0]
        tu = fu[0] if len(fu) else 10**6; td = fd[0] if len(fd) else 10**6
        if tu == td and tu < 10**6: s_lo, s_hi = (L - e) / R, (H - e) / R      # оба в одной минуте — границы
        elif tu < td: s_lo = s_hi = (H - e) / R
        elif td < tu: s_lo = s_hi = (L - e) / R
        else: s_lo = s_hi = (c[-1] - e) / R
        rows.append(dict(date=d, t=t, rho=R / N[d], last='H' if iH > iL else 'L', p0=(e - L) / R, R=R, s_lo=s_lo, s_hi=s_hi,
                         cens=int(tu == td == 10**6)))
r = pd.DataFrame(rows)
r['ep'] = pd.cut(r.date // 10000, [2005, 2012, 2019, 2025], labels=['2006-12', '2013-19', '2020-25'])
r['pos'] = pd.cut(r.p0, [0, 1 / 3, 2 / 3, 1], labels=['low', 'mid', 'high'])
r['rhob'] = pd.cut(r.rho, [0, 0.6, 0.9, 1.2, 100], labels=['<.6', '.6-.9', '.9-1.2', '>1.2'])
def st(g):
    s = (g.s_lo + g.s_hi) / 2; se = s.std() / np.sqrt(len(s))
    return pd.Series({'n': len(g), 'S': round(s.mean(), 4), 'z': round(s.mean() / se, 1), 'bounds': round((g.s_hi - g.s_lo).mean(), 4), 'pts': round((s * g.R).mean(), 2)})
pd.set_option('display.width', 250)
print(f'object from {M0//60:02d}:{M0%60:02d}; stopped-fork mean S = E[(exit - entry)/R], 0 without drift')
print('\nALL:\n', r.groupby('ep', observed=True).apply(st, include_groups=False).to_string())
for fac in ['rhob', 'last', 'pos']:
    print(f'\nby {fac}:\n', r.groupby([fac, 'ep'], observed=True).apply(st, include_groups=False)[['S', 'z', 'n']].unstack('ep').to_string())
print('\nwidth x last:\n', r.groupby(['rhob', 'last', 'ep'], observed=True).apply(st, include_groups=False)[['S', 'z']].unstack('ep').to_string())
