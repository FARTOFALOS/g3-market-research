"""Что сообщает срабатывание предела 17 пт в состоянии S-07: ход после стопа и после нового закрытия на стороне S-07."""
import numpy as np, pandas as pd
from tape import load_minutes
def run(inst, y0, y1):
    df = load_minutes(f'{y0-1}-12-28', '2026-07-11' if y1 == 2026 else f'{y1}-12-31', inst)
    out = []
    for d, g in df.groupby('date'):
        if d // 10000 < y0: continue
        g = g.set_index('mod')
        if not all(m in g.index for m in range(543, 694)): continue
        pre = g.loc[543:572]; mid = (pre.h.max() + pre.l.min()) / 2
        side = 1 if g.at[572, 'c'] > mid else -1
        e0 = g.at[573, 'o']; end = g.at[692, 'c']
        rec = dict(date=d, uncond=side * (end - e0))
        stop_j = None
        for j in range(573, 693):
            if side * ((g.at[j, 'l'] if side > 0 else g.at[j, 'h']) - (e0 - side * 17)) <= 0: stop_j = j; break
        rec['stopped'] = stop_j is not None
        if stop_j is not None and stop_j + 1 <= 692:
            e1 = g.at[stop_j + 1, 'o']; rec['after_stop'] = side * (end - e1)
            for j in range(stop_j + 1, 692):
                if side * (g.at[j, 'c'] - mid) > 0:
                    rec['after_reconfirm'] = side * (end - g.at[j + 1, 'o']); rec['reconf_min'] = j - 573; break
        out.append(rec)
    return pd.DataFrame(out)
def m(x):
    x = x.dropna(); return f'{x.mean():+6.2f} pt (t {x.mean()/x.std()*np.sqrt(len(x)):+.1f}, n {len(x)})'
for inst, y0, y1 in [('NQ', 2020, 2026), ('NQ', 2013, 2019), ('ES', 2020, 2026)]:
    r = run(inst, y0, y1)
    print(f'{inst} {y0}-{y1}: all days from 09:34 {m(r.uncond)} | stopped share {r.stopped.mean():.2f}')
    print(f'   days not stopped, from 09:34      {m(r[~r.stopped].uncond)}')
    print(f'   days stopped, from 09:34          {m(r[r.stopped].uncond)}')
    print(f'   after first stop, to 11:33        {m(r.after_stop)}')
    print(f'   after re-close on S-07 side       {m(r.get("after_reconfirm", pd.Series(dtype=float)))}')
