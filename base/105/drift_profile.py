"""Профиль условного сноса после узнавания без правила выхода: m(s) = E[сторона·(close(вход+s) − вход)],
s = 5…120 мин, не позже 16:00. Одна сцена = один день."""
import numpy as np, pandas as pd
from tape import load_minutes
S = [5, 15, 30, 60, 90, 120]
for inst in ['NQ', 'ES']:
    df = load_minutes('2012-12-01', '2026-07-11', inst).reset_index(drop=True)
    c = df.c.to_numpy(); mod = df['mod'].to_numpy(); date = df.date.to_numpy()
    key = pd.Series(np.arange(len(df)), index=pd.MultiIndex.from_arrays([date, mod]))
    for q in ['break1', 'break2']:
        s = pd.read_csv(f'{q}_{inst}.csv'); s = s[s.status == 'ok'].copy()
        s['ep'] = np.where(s.date >= 20200101, '2020-26', '2013-19')
        for s_ in S:
            v = []
            for _, r in s.iterrows():
                i = key.get((int(r.date), int(r['mod']) + 1 + s_))
                v.append(np.nan if i is None or r['mod'] + 1 + s_ >= 960 else r.side * (c[i] - r.entry))
            s[f'm{s_}'] = v
        for ep, g in s.groupby('ep'):
            line = f'{inst} {q} {ep} n={len(g)}: '
            for s_ in S:
                x = g[f'm{s_}'].dropna(); xu = (x / g.loc[x.index, 'u'])
                line += f' {s_}m {x.mean():+.2f}pt ({xu.mean():+.2f}u, t {x.mean()/x.std()*np.sqrt(len(x)):+.1f})'
            print(line)
