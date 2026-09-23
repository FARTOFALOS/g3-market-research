"""NB1 (LOG.md, объявлено до счёта): отбой от границы нормы в день, который ещё внутри нормы (пара к V7m).
Сцена: NY 10:00–14:59, close минуты внутри нормы (LB(k) < close < UB(k)), день ещё ни разу не закрывался за границей
с 10:00. Узнавание: первая минута на сторону, чей high (low) дошёл до зоны границы — не дальше 1u от UB(k) (LB(k)),
а close остался внутри. Действие: против границы на open следующей. Отмена: UB(k) + 1u (LB(k) − 1u) на момент узнавания
— день вышел за норму, идея «день внутри нормы» кончилась (там же рождается V7m). Остаток — возврат к центру: цель —
середина между открытием 09:30 и границей со стороны отбоя... проще и без выбора: цель = open 09:30 (центр нормы
дня). Время 120. Одна на сторону в день. Показать net, struct, долю целей против честной p0, по эпохам.
"""
import numpy as np, pandas as pd
from trade import Tape, simulate, report, TICK
from v0_s18 import bands

pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
T = Tape()
info, UB, LB = bands(T)
rows = []
for D, K, up, dn in info:
    if D > 20251231:
        continue
    i930 = K[1]
    if i930 < 0:
        continue
    O = T.o[i930]
    done = {1: False, -1: False}
    exited = False
    for k in range(30, 330):
        i = K[k]
        if i < 0 or not np.isfinite(up[k]):
            continue
        if T.c[i] > up[k] or T.c[i] < dn[k]:
            exited = True
        if exited:
            break
        u = T.u[i]
        for sd in (-1, 1):       # sd — сторона сделки: −1 отбой вниз от верхней границы
            if done[sd]:
                continue
            if sd < 0 and T.h[i] >= up[k] - u and T.c[i] < up[k]:
                rows.append(dict(date=D, irec=int(i), side=-1, stop=up[k] + u, target=O, k=k)); done[sd] = True
            if sd > 0 and T.l[i] <= dn[k] + u and T.c[i] > dn[k]:
                rows.append(dict(date=D, irec=int(i), side=1, stop=dn[k] - u, target=O, k=k)); done[sd] = True
S = pd.DataFrame(rows)
s = simulate(T, S, tmax=120)
s.to_csv('NB1_trades.csv', index=False)
out = report(s, 'NB1')
print(pd.DataFrame(out)[['v', 'epoch', 'signals', 'fits', 'risk_med', 'net_pt', 't', 'win', 'avg_w', 'avg_l', 'hold', 'stop', 'target', 'time']].to_string(index=False))
f = s[s.fits & (s.epoch != '2026')].copy()
f['struct'] = f.net + 0.75 + 0.5 * (f.xtype == 1)
f['p0'] = f.risk / (f.risk + f.side * (f.target - f.entry)); f['res'] = f.xtype.isin([1, 2]); f['hit'] = (f.xtype == 2).astype(float)
print(f.groupby('epoch').apply(lambda x: pd.Series(dict(n=len(x), struct=x.struct.mean(), dY=(x[x.res].hit - x[x.res].p0).mean(),
      p0=x[x.res].p0.mean())), include_groups=False).round(3).to_string())
