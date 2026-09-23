"""Вопрос 3: наклон next_q на prev_q по часам и группам лет (LOG.md)."""
import numpy as np, pandas as pd
from tape import load_minutes
QS = [5, 15, 30, 60]
GR = [(2013, 2016), (2017, 2019), (2020, 2021), (2022, 2023), (2024, 2026)]

def build(inst):
    df = load_minutes('2012-12-20', '2026-07-11', inst).reset_index(drop=True)
    df['u'] = (df.h - df.l).rolling(30).median()
    rows = []
    for d, g in df.groupby('date'):
        if d < 20130101: continue
        m = g['mod'].to_numpy(); c = g.c.to_numpy(); u = g.u.to_numpy()
        pos = {mm: k for k, mm in enumerate(m)}
        for t in range(135, 901, 15):
            k = pos.get(t)
            if k is None or not (u[k] > 0): continue
            for q in QS:
                a = pos.get(t - q); b = pos.get(t + q)
                if a is None or b is None or t + q > 960: continue
                rows.append((d, t // 60, q, (c[k] - c[a]) / u[k], (c[b] - c[k]) / u[k], (c[b] - c[k]), (c[k] - c[a])))
    return pd.DataFrame(rows, columns=['date', 'hr', 'q', 'p', 'n', 'n_pt', 'p_pt'])

def beta(g):
    x = g.p.to_numpy(); y = g.n.to_numpy()
    x = x - x.mean(); y = y - y.mean()
    b = (x * y).sum() / (x * x).sum()
    e = y - b * x
    cl = pd.DataFrame({'d': g.date.to_numpy(), 's': x * e}).groupby('d').s.sum().to_numpy()
    se = np.sqrt((cl ** 2).sum()) / (x * x).sum()
    return b, se

out = []
for inst in ['NQ', 'ES']:
    r = build(inst); r.to_parquet(f'vr_{inst}.parquet')
    r['yr'] = r.date // 10000
    for (lo, hi) in GR:
        gg = r[(r.yr >= lo) & (r.yr <= hi)]
        for (hr, q), g in gg.groupby(['hr', 'q']):
            b, se = beta(g)
            out.append((inst, f'{lo}-{hi % 100:02d}', hr, q, len(g), b, se))
o = pd.DataFrame(out, columns=['inst', 'grp', 'hr', 'q', 'n', 'beta', 'se'])
o.to_csv('vr_map.csv', index=False)
for inst in ['NQ', 'ES']:
    for q in QS:
        t = o[(o.inst == inst) & (o.q == q)].pivot(index='hr', columns='grp', values='beta').round(3)
        z = o[(o.inst == inst) & (o.q == q)].pivot(index='hr', columns='grp', values='se')
        print(f'\n{inst} q={q}: beta (z in brackets)')
        print((t.astype(str) + ' (' + (t / z).round(1).astype(str) + ')').to_string())
