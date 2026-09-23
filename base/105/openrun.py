"""OP2 (LOG.md, объявлено до счёта): «чистые пробеги» на открытии — отражение находки OP1.
OP1: в 2020–25 из open 09:34 малая цель X против защиты 18,5 берётся реже честной в ОБЕ стороны (по σ 0,79, против 0,76
при 0,85; разрешилось 99 %) — пробег без отката чаще, чем у мартингала. Отражение: близкая отмена S и дальняя цель G —
выигрыш должен быть выше честной S/(S+G) в любую сторону.
Сделки из open 09:34 (узнавание — close 09:33): по σ и против σ; S = 3, 5, 8 пт; G = 12, 18,5, 25 пт; время 30 мин.
Та же геометрия из open 10:30, 12:00, 14:00 (контроль места: свойство открытия или любой минуты). Минутная свеча с целью
и отменой одновременно — отмена. Эпохи; в ранних много неразрешённых — показывается доля разрешившихся.
"""
import numpy as np, pandas as pd
from trade import Tape, simulate
from v0_batch1 import s07_side

pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
T = Tape()
rows = []
for D, ix in T.days.items():
    if D > 20251231:
        continue
    sg, i33 = s07_side(T, D, ix)
    if sg == 0:
        continue
    for mm in (573, 629, 719, 839):
        ir = T.at(D, mm)
        if ir < 0 or ir + 1 >= len(T.o) or T.mod[ir + 1] != mm + 1:
            continue
        rows.append(dict(date=D, irec=ir, sg=sg, at=mm + 1, e=T.o[ir + 1]))
B = pd.DataFrame(rows)
out = []
for S_ in (3.0, 5.0, 8.0):
    for G in (12.0, 18.5, 25.0):
        for dirn in ('with', 'against'):
            sd = B.sg if dirn == 'with' else -B.sg
            sig = pd.DataFrame(dict(date=B.date, irec=B.irec, side=sd, stop=B.e - sd * S_, target=B.e + sd * G))
            s = simulate(T, sig, tmax=30)
            s['at'] = B['at'].values
            s = s[s.xtype > 0]
            for (at, ep), g in s.groupby(['at', 'epoch']):
                if ep == '2026':
                    continue
                r = g.xtype.isin([1, 2])
                out.append(dict(S=S_, G=G, dir=dirn, at=at, ep=ep, n=len(g), resolved=r.mean(), win_res=(g[r].xtype == 2).mean(),
                                p0=S_ / (S_ + G + 0.25), net=g.net.mean(), t=g.net.mean() / g.net.std() * np.sqrt(len(g))))
R = pd.DataFrame(out); R['dY'] = R.win_res - R.p0
R.to_csv('op2.csv', index=False)
x = R[(R.ep == '2020-25')]
print(x.pivot_table(index=['S', 'G', 'dir'], columns='at', values=['dY', 'net']).round(3).to_string())
print(R[(R['at'] == 574)].pivot_table(index=['S', 'G', 'dir'], columns='ep', values=['dY', 'net', 'resolved']).round(3).to_string())
