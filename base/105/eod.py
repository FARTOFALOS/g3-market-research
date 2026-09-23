"""EOD1 (LOG.md, объявлено до счёта): ход дня → последние 30 и 60 минут (внутридневной моментум в закрытие).
Узнавание: close 15:29 (для 30 мин) и close 14:59 (для 60 мин). Сторона: знак (A) хода от вчерашнего close NY до
узнавания; (B) хода от open 09:30 до узнавания. Вход open следующей минуты, выход close 15:59 (последняя минута
обычного дня; короткие дни исключаются), защита 18,5 пт (свойство политики). Разрезы — |ход дня| в долях нормы
(медиана |хода| тех же часов за 20 сессий): < 0,5, 0,5–1, 1–2, > 2. По эпохам. Показать: ход без защиты (пт), net с
защитой и расходом, t.
"""
import numpy as np, pandas as pd
from trade import Tape, simulate, epoch

pd.set_option('display.width', 250)
T = Tape()
rows = []; hist = {('A', 929): [], ('B', 929): [], ('A', 899): [], ('B', 899): []}
prevC = None
for D in sorted(T.days):
    if D > 20251231:
        break
    ix = T.days[D]
    i930 = T.at(D, 570); i1529 = T.at(D, 929); i1459 = T.at(D, 899); i1559 = T.at(D, 959)
    ok = i930 >= 0 and i1559 >= 0 and prevC is not None
    if ok:
        for mm, ir in ((929, i1529), (899, i1459)):
            if ir < 0 or i1559 - ir != 959 - mm:
                continue
            for kind, ref in (('A', prevC), ('B', T.o[i930])):
                mv = T.c[ir] - ref
                h = hist[(kind, mm)]
                norm = np.median(np.abs(h[-20:])) if len(h) >= 20 else np.nan
                h.append(mv)
                if not np.isfinite(norm) or mv == 0:
                    continue
                sd = 1 if mv > 0 else -1
                e = T.o[ir + 1]
                rows.append(dict(date=D, irec=ir, side=sd, stop=e - sd * 18.5, target=np.nan, iend=i1559, kind=kind, at=mm,
                                 rel=abs(mv) / norm, gross=sd * (T.c[i1559] - e)))
    i_last = ix[-1]
    prevC = T.c[i_last] if T.mod[i_last] >= 955 or T.cal.loc[D, 'status'] == 'short' else None
S = pd.DataFrame(rows)
s = simulate(T, S, tmax=61)
s = s[s.xtype > 0].copy()
s['rb'] = pd.cut(s.rel, [0, 0.5, 1, 2, 1e9], labels=['<0.5', '0.5-1', '1-2', '>2'])
g = s.groupby(['kind', 'at', 'epoch']).agg(n=('net', 'size'), gross=('gross', 'mean'), net=('net', 'mean'),
                                          t=('net', lambda x: x.mean() / x.std() * np.sqrt(len(x)))).round(2)
print(g.unstack('epoch').to_string())
g2 = s[s.epoch != '2026'].groupby(['kind', 'at', 'rb', 'epoch'], observed=True).agg(n=('net', 'size'), gross=('gross', 'mean'),
                                                                                   net=('net', 'mean')).round(2)
print(g2.unstack('epoch').to_string())
s.to_csv('EOD1_trades.csv', index=False)
