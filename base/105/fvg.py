"""FV1 (LOG.md, объявлено до счёта): минутный разрыв толчка (три свечи, low[k] > high[k−2] — вверх; зеркально вниз)
→ первый возврат в разрыв → удержание → продолжение. Узнавание: первая минута в 30 мин после k, чей low ≤ low[k]
(вход в разрыв) и close > high[k−2] (нижний край разрыва удержан закрытием); если раньше close ≤ high[k−2] — разрыв
закрыт, сцены нет. Вход open следующей; отмена — high[k−2] − тик (разрыв закрыт = неторгованная ликвидность отдана);
цель — максимум с k до узнавания + тик прохода (снятие ликвидности толчка); время 60.
Окна узнавания (линза трейдера): LON 02:00–04:59, NYAM 09:30–09:59, SB1 10:00–10:59, MID 11:00–13:59, SB2 14:00–14:59,
прочее. Показать по окнам и эпохам: n, доля целей против честной p0 = риск/(риск + до цели), net, struct.
"""
import numpy as np, pandas as pd
from numba import njit
from trade import Tape, simulate, TICK

pd.set_option('display.width', 260)


@njit(cache=True)
def scan(o, h, l, c, mod, dt, inwin):
    n = len(o)
    I = []; S = []; ST = []; TG = []; K = []
    for k in range(2, n - 32):
        if not inwin[k] or dt[k - 2] != dt[k] or mod[k] - mod[k - 2] != 2:
            continue
        for s in (1, -1):
            if s > 0:
                if not (l[k] > h[k - 2]):
                    continue
                top = l[k]; bot = h[k - 2]
            else:
                if not (h[k] < l[k - 2]):
                    continue
                top = h[k]; bot = l[k - 2]
            ext = h[k] if s > 0 else l[k]
            for j in range(k + 1, k + 31):
                if dt[j] != dt[k] or mod[j] - mod[k] != j - k or not inwin[j]:
                    break
                if s > 0:
                    if c[j] <= bot:
                        break
                    if l[j] <= top:
                        I.append(j); S.append(1); ST.append(bot - 0.25); TG.append(ext); K.append(k); break
                    ext = max(ext, h[j])
                else:
                    if c[j] >= bot:
                        break
                    if h[j] >= top:
                        I.append(j); S.append(-1); ST.append(bot + 0.25); TG.append(ext); K.append(k); break
                    ext = min(ext, l[j])
    return np.array(I), np.array(S), np.array(ST), np.array(TG), np.array(K)


T = Tape()
inwin = np.zeros(len(T.o), bool)
for D, ix in T.days.items():
    if D <= 20251231:
        inwin[ix[T.mod[ix] <= 950]] = True
I, Sd, ST, TG, K = scan(T.o, T.h, T.l, T.c, T.mod, T.date, inwin)
sig = pd.DataFrame(dict(date=T.date[I], irec=I, side=Sd, stop=ST, target=TG))
mm = T.mod[I]
sig['win'] = np.select([mm < 300, mm < 570, mm < 600, mm < 660, mm < 840, mm < 900], ['LON', 'pre', 'NYAM', 'SB1', 'MID', 'SB2'], 'late')
s = simulate(T, sig, tmax=60)
s = s[s.xtype > 0].copy()
f = s[s.fits].copy()
f['struct'] = f.net + 0.75 + 0.5 * (f.xtype == 1)
f['p0'] = f.risk / (f.risk + f.side * (f.target - f.entry))
f['res'] = f.xtype.isin([1, 2]); f['hit'] = (f.xtype == 2).astype(float)
g = f[f.epoch != '2026'].groupby(['win', 'epoch']).apply(lambda x: pd.Series(dict(
    n=len(x), dY=(x[x.res].hit - x[x.res].p0).mean(), net=x.net.mean(), struct=x.struct.mean(),
    t=x.net.mean() / x.net.std() * np.sqrt(len(x)), risk=x.risk.median())), include_groups=False)
print(g.round(3).unstack('epoch').to_string())
f.to_parquet('FV1_trades.parquet')
