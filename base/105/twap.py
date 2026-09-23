"""TW1 (LOG.md, объявлено до счёта): средняя по времени цена сессии (TWAP = среднее close минут с 09:30 до t) как
естественная линия утреннего тренда. Сторона σ — S-07 v1 (09:33). С 09:45 до 11:30 первая минута, чей low (σ вверх)
коснулся TWAP[t−1] (среднее по закрытым минутам до неё), а close остался выше TWAP[t−1]; если раньше close ниже TWAP —
день ушёл под свою среднюю, сцены нет. Вход open следующей; отмена — TWAP[t−1] − 1u − тик (день ниже своей средней =
идея тренда кончилась); остаток — продолжение тренда: время 120 (TW1) и выход по первому close ниже текущей TWAP
(TW1x — ведение по линии; исполнение open следующей минуты, отмена-предел остаётся). Контроль: та же сделка в
случайную минуту 09:45–11:30 с той же отменой по расстоянию (TW1r). Эпохи.
"""
import numpy as np, pandas as pd
from trade import Tape, simulate, report, TICK, COST, SLIP
from v0_batch1 import s07_side

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
T = Tape()
rng = np.random.default_rng(7)
rows, rnd, tw_map = [], [], {}
for D, ix in T.days.items():
    if D > 20251231:
        continue
    sg, i33 = s07_side(T, D, ix)
    if sg == 0:
        continue
    m = T.mod[ix]
    ny = ix[(m >= 570) & (m <= 810)]
    if len(ny) < 200 or T.mod[ny[0]] != 570:
        continue
    tw = np.cumsum(T.c[ny]) / np.arange(1, len(ny) + 1)
    tw_map[D] = (ny, tw)
    for q in range(15, len(ny)):
        i = ny[q]
        if T.mod[i] > 690:
            break
        prev_tw = tw[q - 1]
        if sg * (T.c[i] - prev_tw) < 0:
            break
        touch = (T.l[i] <= prev_tw) if sg > 0 else (T.h[i] >= prev_tw)
        if touch:
            u = T.u[i]
            rows.append(dict(date=D, irec=int(i), side=sg, stop=prev_tw - sg * (u + TICK), target=np.nan))
            q2 = rng.integers(15, min(len(ny) - 1, 120))
            j = ny[q2]
            rnd.append(dict(date=D, irec=int(j), side=sg, stop=T.o[j + 1] - sg * (abs(T.o[i + 1] - (prev_tw - sg * (u + TICK)))), target=np.nan))
            break
S = pd.DataFrame(rows); R = pd.DataFrame(rnd)
s = simulate(T, S, tmax=120); s.to_csv('TW1_trades.csv', index=False)
r = simulate(T, R, tmax=120)
# TW1x: выход по первому закрытию за TWAP
x = s.copy()
for k, t in x[x.xtype > 0].iterrows():
    ny, tw = tw_map[t.date]
    pos = np.searchsorted(ny, t.irec)
    for q in range(pos + 1, len(ny) - 1):
        j = ny[q]
        if j > t.jexit:
            break
        if t.side * (T.c[j] - tw[q]) < 0:
            x.loc[k, 'exit'] = T.o[j + 1]; x.loc[k, 'jexit'] = j + 1; x.loc[k, 'xtype'] = 3
            x.loc[k, 'net'] = t.side * (T.o[j + 1] - t.entry) - COST
            break
out = report(s, 'TW1') + report(x, 'TW1x') + report(r, 'TW1r')
print(pd.DataFrame(out)[['v', 'epoch', 'signals', 'fits', 'risk_med', 'net_pt', 't', 'win', 'avg_w', 'avg_l', 'hold', 'stop']].to_string(index=False))
