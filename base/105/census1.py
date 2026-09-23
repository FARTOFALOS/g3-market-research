import numpy as np, pandas as pd
from tape import load_minutes, window
df = load_minutes('2019-12-31', '2026-07-11')
w = window(df)
w = w[w.date >= 20200101].copy()
w['yr'] = w.date // 10000
w['hr'] = w['mod'] // 60
w['rng'] = w.h - w.l
t = w.pivot_table(index='hr', columns='yr', values='rng', aggfunc='median')
print('median 1-min range, pts (rows=hour ET, cols=year)')
print(t.round(1).to_string())
# first passage +-20 from close of a minute, within same date window, up to 120 min
res = []
for d, g in w.groupby('date'):
    c = g.c.to_numpy(); h = g.h.to_numpy(); l = g.l.to_numpy(); m = g['mod'].to_numpy()
    n = len(c)
    for i in range(0, n - 1, 5):
        e = c[i]
        hh = h[i+1:i+121]; ll = l[i+1:i+121]
        up = np.nonzero(hh >= e + 20)[0]; dn = np.nonzero(ll <= e - 20)[0]
        tu = up[0] if len(up) else 999; td = dn[0] if len(dn) else 999
        res.append((d // 10000, m[i] // 60, min(tu, td)))
r = pd.DataFrame(res, columns=['yr', 'hr', 'tfp'])
r['hit'] = r.tfp < 999
print('\nmedian minutes until price first moves 20 pts either way from a minute close (999=not in 120)')
print(r.pivot_table(index='hr', columns='yr', values='tfp', aggfunc='median').to_string())
print('\nshare of entries where +-20 not reached within 120 min')
print((1 - r.pivot_table(index='hr', columns='yr', values='hit', aggfunc='mean')).round(2).to_string())
