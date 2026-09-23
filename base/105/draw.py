"""DL1 (LOG.md, объявлено до счёта): «сессия забирает ближний экстремум предыдущей» — притяжение уровней.
Моменты: 02:00 (цель — ближний край ASIA), 09:30 (ближний край LON 02:00–09:29), 09:30 (ближний край вчерашней
сессии PDH/PDL), все — только если цена ВНУТРИ диапазона уровня. Вход open следующей минуты после минуты-момента
(02:00 → 02:01, 09:30 → 09:31) в сторону ближнего края; цель — край (проход на тик); защита — симметричная
18,5 пт (свойство политики); время 120. Честная линия p0 = 18,5/(d + 18,5), d — расстояние до цели на входе.
Показать долю целей против p0 и net по эпохам и по корзинам d.
"""
import numpy as np, pandas as pd
from trade import Tape, simulate, TICK

pd.set_option('display.width', 260)
T = Tape()
dates = sorted(D for D in T.days if D <= 20251231)
rows = []; prev_ny = None
for D in dates:
    ix = T.days[D]; m = T.mod[ix]
    Dp = int((pd.Timestamp(str(D)) - pd.Timedelta(days=1)).strftime('%Y%m%d'))
    i200 = T.at(D, 120); i930 = T.at(D, 570)
    if Dp in T.date_start and i200 >= 0:
        a0 = T.date_start[Dp]; a1 = T.date_start[D]
        pa = np.arange(a0, a1); pa = pa[T.mod[pa] >= 1080]
        da = np.arange(a1, ix[0]); da = da[T.mod[da] < 120]
        asia = np.concatenate([pa, da])
        if len(asia) >= 400 and T.mod[asia[0]] == 1080:
            rows.append(('ASIA@02', D, i200, T.h[asia].max(), T.l[asia].min()))
    lon = ix[(m >= 120) & (m <= 569)]
    if i930 >= 0 and len(lon) >= 440:
        rows.append(('LON@0930', D, i930, T.h[lon].max(), T.l[lon].min()))
    if i930 >= 0 and prev_ny is not None:
        rows.append(('PD@0930', D, i930, prev_ny[0], prev_ny[1]))
    ny = ix[m >= 570]
    prev_ny = (T.h[ny].max(), T.l[ny].min()) if len(ny) >= 300 else None
sig = []
for nm, D, i, H, L in rows:
    e = T.o[i + 1] if i + 1 < len(T.o) else np.nan
    if not (L < e < H):
        continue
    up = H - e; dn = e - L
    sd = 1 if up <= dn else -1
    tg = H if sd > 0 else L
    sig.append(dict(date=D, irec=i, side=sd, stop=e - sd * 18.5, target=tg, src=nm, d=min(up, dn)))
S = pd.DataFrame(sig)
s = simulate(T, S, tmax=120)
s = s[s.xtype > 0].copy()
s['hit'] = (s.xtype == 2).astype(float); s['res'] = s.xtype.isin([1, 2])
s['p0'] = 18.5 / (s.d + 18.5)
s['db'] = pd.cut(s.d, [0, 10, 25, 50, 100, 1e9], labels=['<10', '10-25', '25-50', '50-100', '>100'])
g = s.groupby(['src', 'epoch']).apply(lambda x: pd.Series(dict(n=len(x), resolved=x.res.mean(), hit=x.hit.mean(),
        hit_res=x[x.res].hit.mean(), p0=x[x.res].p0.mean(), net=x.net.mean(), t=x.net.mean() / x.net.std() * np.sqrt(len(x)))), include_groups=False)
print(g.round(3).to_string())
g2 = s[s.epoch.isin(['2013-19', '2020-25'])].groupby(['src', 'db', 'epoch'], observed=True).apply(
    lambda x: pd.Series(dict(n=len(x), dY=(x[x.res].hit - x[x.res].p0).mean(), net=x.net.mean())), include_groups=False)
print(g2.round(3).unstack('epoch').to_string())
s.to_csv('DL1_trades.csv', index=False)
