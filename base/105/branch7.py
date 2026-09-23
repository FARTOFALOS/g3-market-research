"""Вопрос 7: ветви первого пробоя — удержался N минут / вернулся внутрь."""
import numpy as np, pandas as pd
from tape import load_minutes
S = [5, 15, 30, 60, 120]
def run(inst, y0, y1):
    df = load_minutes(f'{y0-1}-12-20', '2026-07-11' if y1 == 2026 else f'{y1}-12-31', inst).reset_index(drop=True)
    c = df.c.to_numpy(); o = df.o.to_numpy(); mod = df['mod'].to_numpy(); date = df.date.to_numpy()
    key = pd.Series(np.arange(len(df)), index=pd.MultiIndex.from_arrays([date, mod]))
    b = pd.read_csv(f'break1_{inst}.csv'); b = b[(b.status == 'ok') & (b.date // 10000 >= y0) & (b.date // 10000 <= y1)]
    out = []
    for _, r in b.iterrows():
        D = int(r.date); i = key.get((D, int(r['mod'])))
        if i is None: continue
        sd = int(r.side); lvl = r.H if sd > 0 else r.L
        inside = lambda j: (c[j] < r.H) if sd > 0 else (c[j] > r.L)
        # первый возврат внутрь
        j = i + 1; ret = None
        while j < len(c) and date[j] == D and mod[j] < 960:
            if mod[j] != mod[j - 1] + 1: break
            if inside(j): ret = j; break
            j += 1
        def prof(q, tag, N=None):
            if q + 1 >= len(c) or date[q + 1] != D or mod[q + 1] >= 960: return
            e = o[q + 1]; rec = dict(date=D, tag=tag, N=N, mod=mod[q])
            for s in S:
                k = key.get((D, int(mod[q]) + 1 + s))
                rec[f'm{s}'] = sd * (c[k] - e) if k is not None and mod[q] + 1 + s <= 959 else np.nan
            out.append(rec)
        for N in [15, 30]:
            q = i + N
            if q < len(c) and date[q] == D and mod[q] == mod[i] + N and mod[q] < 959:
                prof(q, 'all_at_i+N', N)
                if ret is None or ret > q: prof(q, 'held', N)
        if ret is not None: prof(ret, 'returned')
    return pd.DataFrame(out)
rows = []
for inst, y0, y1 in [('NQ', 2020, 2026), ('NQ', 2013, 2019), ('ES', 2020, 2026)]:
    p = run(inst, y0, y1)
    for (tag, N), g in p.groupby(['tag', p.N.fillna(0)]):
        rec = dict(cell=f'{inst} {y0}-{y1 % 100}', branch=tag, N=int(N), n=len(g))
        for s in S:
            x = g[f'm{s}'].dropna()
            rec[f'm{s}'] = round(x.mean(), 2); rec[f'z{s}'] = round(x.mean() / x.std() * np.sqrt(len(x)), 1)
        rows.append(rec)
o = pd.DataFrame(rows); pd.set_option('display.width', 250); print(o.to_string(index=False)); o.to_csv('branch7.csv', index=False)
