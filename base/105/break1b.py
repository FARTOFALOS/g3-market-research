"""Вопрос 1b: разделяет ли prefix-признак минуты пробоя продолжение и возврат (терцили внутри клетки)."""
import numpy as np, pandas as pd
from break1 import scenes
F = ['beyond_u', 'bar_u', 'run15_u', 'touches60', 'age', 'width_u', 'expand']
rows = []
cells = {}
for inst in ['NQ', 'ES']:
    s = scenes(inst); s.to_csv(f'break1_{inst}.csv', index=False)
    s = s[s.status == 'ok'].copy()
    s['ep'] = np.where(s.date >= 20200101, '2020-26', '2013-19')
    for ep, g in s.groupby('ep'):
        cells[(inst, ep)] = g
for f in F:
    for k in [2, 3]:
        rec = {'feat': f, 'k': k}
        for (inst, ep), g in cells.items():
            g = g[g[f'y{k}'].isin(['cont', 'ret'])].copy()
            g['q'] = pd.qcut(g[f].rank(method='first'), 3, labels=[0, 1, 2])
            y = (g[f'y{k}'] == 'cont').astype(float)
            m = y.groupby(g.q, observed=True).mean(); n = y.groupby(g.q, observed=True).size()
            d = m[2] - m[0]; se = np.sqrt(m[2] * (1 - m[2]) / n[2] + m[0] * (1 - m[0]) / n[0])
            tag = f'{inst}{ep[2:4]}'
            rec[tag + '_lo'] = round(m[0], 3); rec[tag + '_hi'] = round(m[2], 3); rec[tag + '_z'] = round(d / se, 1)
            if (inst, ep) == ('NQ', '2020-26'):
                ce = (0.75 / (2 * k * g.u)).mean(); rec['costeq'] = round(ce, 3)
        zs = [rec[t + '_z'] for t in ['NQ13', 'NQ20', 'ES13', 'ES20']]
        rec['same_sign'] = all(z > 0 for z in zs) or all(z < 0 for z in zs)
        ext = max(abs(rec['NQ20_lo'] - 0.5), abs(rec['NQ20_hi'] - 0.5))
        rec['PASS'] = rec['same_sign'] and abs(rec['NQ20_z']) > 2 and ext > rec['costeq']
        rows.append(rec)
r = pd.DataFrame(rows); pd.set_option('display.width', 300)
print(r.to_string(index=False)); r.to_csv('break1b_summary.csv', index=False)
