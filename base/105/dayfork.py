"""D2: собственная вилка дня H_t против L_t по состояниям (возраст последнего экстремума, ρ_t)."""
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
    if len(w) != 960 - M0: continue
    objs.append((d, w))
Rs = pd.Series({d: w.h.max() - w.l.min() for d, w in objs}).sort_index()
N = Rs.rolling(20).median().shift(1)
rows = []
for d, w in objs:
    if pd.isna(N.get(d)): continue
    m = w['mod'].to_numpy(); o = w.o.to_numpy(); h = w.h.to_numpy(); l = w.l.to_numpy()
    rh = np.maximum.accumulate(h); rl = np.minimum.accumulate(l)
    for t in TT:
        k = t - M0 - 1                       # индекс минуты, закрывающейся в t
        if k + 1 >= len(m): continue
        H, L = rh[k], rl[k]; e = o[k + 1]
        if not (L < e < H): continue
        newext = np.nonzero((h[:k + 1] >= rh[:k + 1]) | (l[:k + 1] <= rl[:k + 1]))[0]
        age = k - newext[-1]
        fu = h[k + 1:] > H; fd = l[k + 1:] < L
        iu = np.argmax(fu) if fu.any() else 10**6; idd = np.argmax(fd) if fd.any() else 10**6
        y = np.nan if iu == idd else float(iu < idd)
        rows.append(dict(date=d, t=t, age=age, rho=(H - L) / N[d], p0=(e - L) / (H - L), y=y, R=H - L,
                         cens=int(iu == idd == 10**6), both=int(iu == idd and iu < 10**6)))
r = pd.DataFrame(rows)
r['ep'] = pd.cut(r.date // 10000, [2005, 2012, 2019, 2025], labels=['2006-12', '2013-19', '2020-25'])
r['ageb'] = pd.cut(r.age, [-1, 14, 59, 119, 10**4], labels=['<15', '15-60', '60-120', '>120'])
r['rhob'] = pd.cut(r.rho, [0, 0.6, 0.9, 1.2, 100], labels=['<.6', '.6-.9', '.9-1.2', '>1.2'])
r.to_csv(f'dayfork_{M0}.csv', index=False)
def summ(g):
    ok = g.dropna(subset=['y'])
    if len(ok) < 40: return pd.Series({'n': len(ok)})
    dY = ok.y.mean() - ok.p0.mean(); se = np.sqrt((ok.p0 * (1 - ok.p0)).sum()) / len(ok)
    return pd.Series({'n': len(ok), 'cens': round(g.cens.mean(), 2), 'dY': round(dY, 3), 'z': round(dY / se, 1),
                      'pts': round((ok.R * (ok.y - ok.p0)).mean(), 2)})
pd.set_option('display.width', 250)
print(f'object from {M0//60:02d}:{M0%60:02d}; points', len(r))
print('\nALL by epoch:\n', r.groupby('ep', observed=True).apply(summ, include_groups=False).to_string())
for fac in ['ageb', 'rhob']:
    t = r.groupby([fac, 'ep'], observed=True).apply(summ, include_groups=False)
    print(f'\nby {fac}:\n', t.unstack('ep')[['dY', 'z', 'n']].to_string())
t = r.groupby(['ageb', 'rhob', 'ep'], observed=True).apply(summ, include_groups=False)
print('\nage x rho (dY, z) by epoch:\n', t[['dY', 'z']].unstack('ep').to_string())
