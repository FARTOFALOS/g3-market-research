"""OP1 (LOG.md, объявлено до счёта): ранний откат против стороны S-07 после 09:33.
Узнавание: close 09:33 (сторона σ S-07 v1). Действие: ПРОТИВ σ на open 09:34; цель X = 3, 5, 8, 12 пт (лимит с
проходом тика); защита 18,5 пт (свойство политики); время 15 и 30 мин. Та же сделка ПО σ — контроль. Честная доля
(разрешившиеся) p0 = 18,5/(X + 0,25 + 18,5). Минутная свеча, содержащая и цель, и защиту, — защита (консервативно);
отдельно показывается доля таких свечей. По эпохам и годам 2020–25.
"""
import numpy as np, pandas as pd
from trade import Tape, simulate, epoch
from v0_batch1 import s07_side

pd.set_option('display.width', 250)
T = Tape()
base = []
for D, ix in T.days.items():
    if D > 20251231:
        continue
    sg, i33 = s07_side(T, D, ix)
    if sg == 0 or i33 + 1 >= len(T.o) or T.mod[i33 + 1] != 574:
        continue
    base.append(dict(date=D, irec=i33, sg=sg, e=T.o[i33 + 1]))
B = pd.DataFrame(base)
rows = []
for X in (3.0, 5.0, 8.0, 12.0):
    for dirn in ('against', 'with'):
        sd = -B.sg if dirn == 'against' else B.sg
        sig = pd.DataFrame(dict(date=B.date, irec=B.irec, side=sd, stop=B.e - sd * 18.5, target=B.e + sd * X))
        for tm in (15, 30):
            s = simulate(T, sig, tmax=tm)
            s = s[s.xtype > 0]
            res = s.xtype.isin([1, 2])
            j = s.jexit.to_numpy().astype(int)
            both = ((s.xtype == 1) & ((s.side > 0) & (T.h[j] >= s.target + 0.25) | (s.side < 0) & (T.l[j] <= s.target - 0.25))).mean()
            for ep, g in s.groupby('epoch'):
                r = g.xtype.isin([1, 2])
                rows.append(dict(X=X, dir=dirn, tm=tm, ep=ep, n=len(g), win=(g.xtype == 2).mean(), win_res=(g[r].xtype == 2).mean(),
                                 p0=18.5 / (X + 0.25 + 18.5), net=g.net.mean(), t=g.net.mean() / g.net.std() * np.sqrt(len(g)),
                                 both=both))
R = pd.DataFrame(rows)
R['dY'] = R.win_res - R.p0
print(R[R.ep != '2026'].round(3).to_string(index=False))
