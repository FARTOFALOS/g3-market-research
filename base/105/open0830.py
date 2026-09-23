"""Шаг 3: сообщает ли эпизод 08:30 к 09:29 направление открытия."""
import numpy as np, pandas as pd
from tape import load_minutes
def run(inst, y0, y1):
    ep = pd.read_csv(f'x0830_{inst}_{y0}.csv'); ep = ep[~ep.whip].set_index('date')
    df = load_minutes(f'{y0}-01-01', '2026-07-11' if y1 == 2026 else f'{y1}-12-31', inst)
    medAu = ep.A_u.median()
    out = []
    for d, g in df.groupby('date'):
        g = g.set_index('mod')
        if not all(m in g.index for m in range(450, 691)): continue
        pre = g.loc[480:509]; u = (pre.h - pre.l).median()
        if not (u > 0): continue
        b = g.loc[510]
        if d in ep.index:
            e = ep.loc[d]; s = int(e.s); O = b.o; A = e.A; kind = 'episode'
        elif (b.h - b.l) / u < 2:
            s = 1 if g.at[569, 'c'] >= b.o else -1; O = b.o; A = medAu * u; kind = 'control'
        else: continue
        q = s * (g.at[569, 'c'] - O) / A
        e0 = g.at[570, 'o']
        mins = list(range(570, 691)); h = g.loc[mins, 'h'].to_numpy(); l = g.loc[mins, 'l'].to_numpy()
        rec = dict(date=d, kind=kind, s=s, q=q, A=A)
        for k in [1, 2]:
            fu = (h >= e0 + s * k * A) if s > 0 else (l <= e0 + s * k * A)
            fd = (l <= e0 - s * k * A) if s > 0 else (h >= e0 - s * k * A)
            iu = np.argmax(fu) if fu.any() else 10**6; idd = np.argmax(fd) if fd.any() else 10**6
            rec[f'y{k}'] = np.nan if iu == idd else float(iu < idd)
        rec['m60'] = s * (g.at[629, 'c'] - e0)
        out.append(rec)
    return pd.DataFrame(out)
rows = []
for inst, y0, y1 in [('NQ', 2020, 2026), ('NQ', 2013, 2019), ('ES', 2020, 2026)]:
    r = run(inst, y0, y1)
    r['qb'] = pd.cut(r.q, [-np.inf, 0, 1, np.inf], labels=['q<0 (beyond O)', '0..1', 'q>1 (beyond X)'])
    for (kind, qb), g in list(r.groupby(['kind', 'qb'], observed=True)) + [((k, 'ALL'), gg) for k, gg in r.groupby('kind')]:
        rec = dict(cell=f'{inst} {y0}-{y1 % 100}', kind=kind, q=qb, n=len(g))
        for k in [1, 2]:
            y = g[f'y{k}'].dropna(); rec[f'Y{k}'] = round(y.mean(), 3); rec[f'z{k}'] = round((y.mean() - 0.5) / np.sqrt(0.25 / len(y)), 1)
        rec['m60_pt'] = round(g.m60.mean(), 1); rec['m60_A'] = round((g.m60 / g.A).mean(), 2)
        rows.append(rec)
o = pd.DataFrame(rows); pd.set_option('display.width', 220); print(o.to_string(index=False)); o.to_csv('open0830.csv', index=False)
