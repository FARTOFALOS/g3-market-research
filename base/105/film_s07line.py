"""Рентген v0-3 (LOG.md): перестаёт ли идея S-07 быть той же после закрытия за линией стороны M."""
import numpy as np, pandas as pd
from trade import Tape, epoch
from v0_batch1 import s07_side

T = Tape()
rows = []
for D, ix in T.days.items():
    sg, i33 = s07_side(T, D, ix)
    if sg == 0:
        continue
    i1133 = T.at(D, 693)
    if i1133 < 0 or i1133 - i33 != 120:      # непрерывные минуты 09:33…11:33
        continue
    b = np.arange(i33 - 29, i33 + 1)
    M = (T.h[b].max() + T.l[b].min()) / 2
    path = np.arange(i33 + 1, i1133 + 1)      # 09:34 … 11:33
    fin = T.c[i1133]
    cross = np.nonzero(sg * (T.c[path[:-30]] - M) < 0)[0]   # закрытие за M до 11:03
    tx = cross[0] if len(cross) else -1
    # ход по σ от open каждой минуты до close 11:33 (для контроля)
    mv = sg * (fin - T.o[path])
    rows.append(dict(date=D, ep=epoch(D), sg=sg, dist=sg * (T.o[i33 + 1] - M), tx=tx,
                     after=(sg * (fin - T.o[path[tx + 1]])) if tx >= 0 else np.nan, mv=mv))
df = pd.DataFrame(rows)
out = []
for ep, g in df.groupby('ep'):
    if ep == 'other':
        continue
    MV = np.vstack(g.mv.to_numpy())                 # дни × минуты
    ctrl_by_min = MV.mean(0)
    e = g[g.tx >= 0]
    ctrl = np.array([ctrl_by_min[t + 1] for t in e.tx])
    add = e.after.to_numpy() - ctrl
    out.append(dict(ep=ep, days=len(g), s07_hold=round(MV[:, 0].mean(), 2),
                    dist_med=round(g.dist.median(), 1), crossed=round(len(e) / len(g), 2),
                    cross_min_med=int(np.median(e.tx)), after=round(e.after.mean(), 2), ctrl=round(ctrl.mean(), 2),
                    add=round(add.mean(), 2), z=round(add.mean() / add.std() * np.sqrt(len(add)), 2),
                    notcross_hold=round(MV[g.tx.to_numpy() < 0, 0].mean(), 2)))
print(pd.DataFrame(out).to_string(index=False))
df.drop(columns='mv').to_csv('film_s07line.csv', index=False)
