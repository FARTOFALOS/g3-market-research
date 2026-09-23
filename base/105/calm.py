"""Вопрос 4: снос и путь после относительного затишья."""
import numpy as np, pandas as pd
from tape import load_minutes
PTS = list(range(120, 506, 5)) + list(range(600, 896, 5))
def build(inst, y0, y1):
    df = load_minutes(f'{y0-1}-11-01', '2026-07-11' if y1 == 2026 else f'{y1}-12-31', inst).reset_index(drop=True)
    df['u'] = (df.h - df.l).rolling(30).median()
    sub = df[df['mod'].isin(PTS)]
    U = sub.pivot_table(index='date', columns='mod', values='u')
    N = U.rolling(20, min_periods=15).median().shift(1)
    rows = []
    for d, g in df.groupby('date'):
        if d // 10000 < y0 or d not in N.index: continue
        m = g['mod'].to_numpy(); o = g.o.to_numpy(); c = g.c.to_numpy(); h = g.h.to_numpy(); l = g.l.to_numpy(); u = g.u.to_numpy()
        pos = {mm: k for k, mm in enumerate(m)}
        for t in PTS:
            k = pos.get(t)
            if k is None or k + 1 >= len(m) or m[k + 1] != t + 1 or not (u[k] > 0): continue
            nrm = N.at[d, t] if t in N.columns else np.nan
            if not (nrm > 0): continue
            e = o[k + 1]; rec = [d, 'NY' if t >= 600 else 'LDN', u[k] / nrm, u[k]]
            for s in [30, 60, 120]:
                b = pos.get(t + s)
                rec.append(np.nan if b is None or t + s > 959 else c[b] - e)
            end = min(k + 60, len(m) - 1)
            seg = slice(k + 1, end + 1)
            rec += [(h[seg].max() - e) / u[k], (e - l[seg].min()) / u[k]]
            rows.append(rec)
    return pd.DataFrame(rows, columns=['date', 'blk', 'r', 'u', 'm30', 'm60', 'm120', 'mfe60', 'mae60'])
def cl_se(x, d):
    x = x.dropna(); dd = d.loc[x.index]
    s = (x - x.mean()).groupby(dd).sum()
    return x.mean(), np.sqrt((s ** 2).sum()) / len(x)
res = []
for inst, y0, y1 in [('NQ', 2020, 2026), ('NQ', 2013, 2019), ('ES', 2020, 2026)]:
    r = build(inst, y0, y1); r.to_parquet(f'calm_{inst}_{y0}.parquet')
    for blk, g in r.groupby('blk'):
        g = g.copy(); g['q'] = pd.qcut(g.r.rank(method='first'), 5, labels=['Q1calm', 'Q2', 'Q3', 'Q4', 'Q5agit'])
        for q, gg in list(g.groupby('q', observed=True)) + [('ALL', g)]:
            rec = dict(cell=f'{inst} {y0}-{y1 % 100}', blk=blk, q=q, n=len(gg), r=round(gg.r.median(), 2), u=round(gg.u.median(), 2))
            for s in ['m30', 'm60', 'm120']:
                mu, se = cl_se(gg[s], gg.date); rec[s] = round(mu, 2); rec['z' + s[1:]] = round(mu / se, 1)
                rec[s + '_ph'] = round(mu / int(s[1:]) * 60, 2)
            rec['mfe-mae60_u'] = round((gg.mfe60 - gg.mae60).mean(), 3)
            res.append(rec)
o = pd.DataFrame(res); pd.set_option('display.width', 250); print(o.to_string(index=False)); o.to_csv('calm.csv', index=False)
