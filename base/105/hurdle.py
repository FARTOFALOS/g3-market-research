"""Порог сноса: деньги сделки = снос × время в сделке (Вальд). Скобка ±17 пт (лимит 20 минус расход и запас на
проскальзывание — защитный предел по деньгам, не рыночный стоп), не дольше 120 мин и 16:00.
mu* = расход 0,75 пт / среднее время в скобке, в пунктах за час. Диагностика, не правило."""
import numpy as np, pandas as pd
from tape import load_minutes, window
A = 17.0
w = window(load_minutes('2019-12-31', '2026-07-11'))
w = w[w.date >= 20200101]
res = []
for d, g in w.groupby('date'):
    c = g.c.to_numpy(); h = g.h.to_numpy(); l = g.l.to_numpy(); m = g['mod'].to_numpy()
    for i in range(0, len(c) - 1, 5):
        e = c[i]; hh = h[i+1:i+121]; ll = l[i+1:i+121]
        hit = np.nonzero((hh >= e + A) | (ll <= e - A))[0]
        t = hit[0] + 1 if len(hit) else len(hh)
        res.append((d // 10000, m[i] // 60, t))
r = pd.DataFrame(res, columns=['yr', 'hr', 't'])
T = r.pivot_table(index='hr', columns='yr', values='t', aggfunc='mean')
print('mean minutes in a +-17 bracket (cap 120 / 16:00)'); print(T.round(0).to_string())
print('\nrequired conditional drift, pts per hour, to pay 0.75 pt cost'); print((0.75 / T * 60).round(1).to_string())
