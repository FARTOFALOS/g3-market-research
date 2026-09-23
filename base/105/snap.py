"""Вопрос 3b: условный ход после пятиминутного рывка."""
import numpy as np, pandas as pd
from tape import load_minutes
NX = [5, 10, 15, 30]
BINS = [-np.inf, -4, -2, -1, 1, 2, 4, np.inf]
def build(inst, y0, y1):
    df = load_minutes(f'{y0-1}-12-20', f'{y1}-12-31' if y1 < 2026 else '2026-07-11', inst).reset_index(drop=True)
    df['u'] = (df.h - df.l).rolling(30).median()
    rows = []
    for d, g in df.groupby('date'):
        if d // 10000 < y0: continue
        m = g['mod'].to_numpy(); c = g.c.to_numpy(); u = g.u.to_numpy(); o = g.o.to_numpy()
        pos = {mm: k for k, mm in enumerate(m)}
        for t in list(range(125, 511, 5)) + list(range(600, 931, 5)):
            k = pos.get(t); a = pos.get(t - 5)
            if k is None or a is None or not (u[k] > 0) or k + 1 >= len(m) or m[k + 1] != t + 1: continue
            e = o[k + 1]  # вход по open следующей минуты
            rec = [d, 'NY' if t >= 600 else 'LDN', (c[k] - c[a]) / u[k], c[k] - c[a], u[k]]
            for s in NX:
                b = pos.get(t + 1 + s)
                rec.append(np.nan if b is None or t + 1 + s > 960 else c[b] - e)
            rows.append(rec)
    return pd.DataFrame(rows, columns=['date', 'blk', 'p_u', 'p_pt', 'u'] + [f'n{s}' for s in NX])
res = []
for inst, y0, y1 in [('NQ', 2020, 2026), ('NQ', 2013, 2019), ('ES', 2020, 2026)]:
    r = build(inst, y0, y1)
    r['bin'] = pd.cut(r.p_u, BINS)
    for (blk, b), g in r.groupby(['blk', 'bin'], observed=True):
        rec = dict(cell=f'{inst} {y0}-{y1 % 100}', blk=blk, bin=str(b), n=len(g), prev_pt=round(g.p_pt.mean(), 1))
        for s in NX:
            x = g[f'n{s}'].dropna(); dd = g.loc[x.index, 'date']
            cl = (x - x.mean()).groupby(dd).sum()
            se = np.sqrt((cl ** 2).sum()) / len(x)
            rec[f'n{s}_pt'] = round(x.mean(), 2); rec[f'z{s}'] = round(x.mean() / se, 1)
        res.append(rec)
o = pd.DataFrame(res); pd.set_option('display.width', 250); print(o.to_string(index=False)); o.to_csv('snap.csv', index=False)
