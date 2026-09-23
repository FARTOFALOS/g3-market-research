"""HR1 (LOG.md, объявлено до счёта): отношения двух последних закрытых часовых свечей (часы по :00 ET) → следующий час.
Часовая свеча h (закрыта в hh:59) против h−1: UP_BOS — close выше high(h−1); DN_BOS — close ниже low(h−1);
SWEEP_H — high выше high(h−1), close обратно ниже него (вынос вверх с возвратом); SWEEP_L — зеркально; INSIDE —
внутри h−1; OUTSIDE — перекрывает h−1 с обеих сторон; прочее — NONE. Узнавание: close hh:59; вход open (hh+1):00.
Сторона: BOS — по пробою; SWEEP — против выноса; INSIDE/OUTSIDE — по телу часа h. Защита 18,5 (свойство политики),
выход close (hh+1):59 (60 мин). Часы узнавания 02:59…14:59. Показать ход следующего часа без защиты и net, по эпохам и
по часу; t по дням.
"""
import numpy as np, pandas as pd
from trade import Tape, simulate

pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
T = Tape()
rows = []
for D in sorted(T.days):
    if D > 20251231:
        break
    ix = T.days[D]; m = T.mod[ix]
    bars = {}
    for hr in range(2, 16):
        b = ix[(m >= hr * 60) & (m < hr * 60 + 60)]
        if len(b) == 60:
            bars[hr] = (T.o[b[0]], T.h[b].max(), T.l[b].min(), T.c[b[-1]], b[-1])
    for hr in range(3, 15):
        if hr not in bars or hr - 1 not in bars or hr + 1 not in bars:
            continue
        o, h, l, c, il = bars[hr]; _, hp, lp, _, _ = bars[hr - 1]
        if c > hp: ty, sd = 'UP_BOS', 1
        elif c < lp: ty, sd = 'DN_BOS', -1
        elif h > hp and c <= hp and l >= lp: ty, sd = 'SWEEP_H', -1
        elif l < lp and c >= lp and h <= hp: ty, sd = 'SWEEP_L', 1
        elif h <= hp and l >= lp: ty, sd = 'INSIDE', (1 if c > o else -1)
        elif h > hp and l < lp: ty, sd = 'OUTSIDE', (1 if c > o else -1)
        else: continue
        e = T.o[il + 1]; iend = bars[hr + 1][4]
        rows.append(dict(date=D, irec=il, side=sd, stop=e - sd * 18.5, target=np.nan, iend=iend, ty=ty, hr=hr,
                         gross=sd * (T.c[iend] - e)))
S = pd.DataFrame(rows)
s = simulate(T, S, tmax=60)
s = s[s.xtype > 0].copy()
s['bos_sw'] = s.ty.map(lambda t: 'BOS' if 'BOS' in t else ('SWEEP' if 'SWEEP' in t else t))
g = s.groupby(['bos_sw', 'epoch']).agg(n=('net', 'size'), gross=('gross', 'mean'), net=('net', 'mean'),
                                       t=('gross', lambda x: x.mean() / x.std() * np.sqrt(len(x)))).round(2).unstack('epoch')
print(g.to_string())
g2 = s[s.epoch.isin(['2013-19', '2020-25'])].groupby(['bos_sw', 'hr', 'epoch']).gross.mean().round(2).unstack('epoch')
print(g2.to_string())
s.to_csv('HR1_trades.csv', index=False)
