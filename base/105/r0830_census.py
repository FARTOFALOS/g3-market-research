import numpy as np, pandas as pd
from tape import load_minutes
for inst, y0, y1 in [('NQ', 2020, 2026), ('NQ', 2013, 2019), ('ES', 2020, 2026)]:
    df = load_minutes(f'{y0}-01-01', '2026-07-11' if y1 == 2026 else f'{y1}-12-31', inst)
    rows = []
    for d, g in df.groupby('date'):
        g = g.set_index('mod')
        if not all(m in g.index for m in range(450, 691)): continue
        pre = g.loc[480:509]; u = (pre.h - pre.l).median()
        if not (u > 0): continue
        b = g.loc[510]; nb = g.loc[511]
        rows.append(dict(date=d, u=u, r_u=(b.h - b.l) / u, body_u=abs(b.c - b.o) / u, r1_u=(nb.h - nb.l) / u,
                         r_prev_u=(g.loc[509].h - g.loc[509].l) / u, r_0930_u=(g.loc[570].h - g.loc[570].l) / u))
    r = pd.DataFrame(rows); r.to_csv(f'r0830_{inst}_{y0}.csv', index=False)
    edges = [0, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 1000]
    h = pd.cut(r.r_u, edges).value_counts().sort_index()
    hp = pd.cut(r.r_prev_u, edges).value_counts().sort_index()
    print(f'\n{inst} {y0}-{y1}: days {len(r)}; median u_pre {r.u.median():.2f} pt')
    print(pd.DataFrame({'08:30 bar range/u': h, '08:29 bar (control)': hp}).to_string())
