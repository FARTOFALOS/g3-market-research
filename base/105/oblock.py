"""OB1 (LOG.md, объявлено до счёта): блок ордеров — последняя встречная минутная свеча перед импульсом.
Импульс вверх: три минуты подряд с close выше high предыдущей, суммарный ход ≥ 4u; блок — последняя красная минута
(close < open) перед первой из трёх (не дальше 5 минут назад); зона блока — [low, open] этой свечи. После импульса
первое касание зоны (low ≤ open блока) в 60 мин, если раньше цена не закрывалась ниже low блока; узнавание — минута
касания закрылась выше low блока (зона удержана закрытием). Вход open следующей; отмена — low блока − тик; цель —
максимум импульса + тик; время 60. Зеркально вниз. Окна pre / NY; эпохи; доля целей против честной p0, net.
"""
import numpy as np, pandas as pd
from numba import njit
from trade import Tape, simulate

pd.set_option('display.width', 250)


@njit(cache=True)
def scan(o, h, l, c, mod, dt, u, inwin):
    n = len(o); I = []; S = []; ST = []; TG = []
    k = 8
    while k < n - 62:
        hit = False
        if inwin[k] and dt[k - 7] == dt[k] and mod[k] - mod[k - 7] == 7:
            for s in (1, -1):
                ok = True
                for j in range(k - 2, k + 1):
                    if (s > 0 and not c[j] > h[j - 1]) or (s < 0 and not c[j] < l[j - 1]):
                        ok = False; break
                if not ok or s * (c[k] - o[k - 2]) < 4 * u[k - 3]:
                    continue
                ob = -1
                for j in range(k - 3, k - 8, -1):
                    if (s > 0 and c[j] < o[j]) or (s < 0 and c[j] > o[j]):
                        ob = j; break
                if ob < 0:
                    continue
                zo = o[ob]; zl = l[ob] if s > 0 else h[ob]
                top = h[k] if s > 0 else l[k]
                for j in range(k + 1, k + 61):
                    if dt[j] != dt[k] or mod[j] - mod[k] != j - k or not inwin[j]:
                        break
                    if (s > 0 and c[j] < zl) or (s < 0 and c[j] > zl):
                        break
                    top = max(top, h[j]) if s > 0 else min(top, l[j])
                    if (s > 0 and l[j] <= zo) or (s < 0 and h[j] >= zo):
                        if (s > 0 and c[j] > zl) or (s < 0 and c[j] < zl):
                            I.append(j); S.append(s); ST.append(zl - s * 0.25); TG.append(top); hit = True
                        break
                if hit:
                    break
        k += 3 if hit else 1
    return np.array(I), np.array(S), np.array(ST), np.array(TG)


T = Tape()
inwin = np.zeros(len(T.o), bool)
for D, ix in T.days.items():
    if D <= 20251231:
        inwin[ix[T.mod[ix] <= 930]] = True
u = np.nan_to_num(T.u, nan=1e9)
I, Sd, ST, TG = scan(T.o, T.h, T.l, T.c, T.mod, T.date, u, inwin)
sig = pd.DataFrame(dict(date=T.date[I], irec=I, side=Sd, stop=ST, target=TG))
sig['win'] = np.where(T.mod[I] < 570, 'pre', 'NY')
s = simulate(T, sig, tmax=60)
s = s[(s.xtype > 0) & (s.epoch != '2026')].copy()
s['p0'] = s.risk / (s.risk + s.side * (s.target - s.entry)); s['res'] = s.xtype.isin([1, 2]); s['hit'] = (s.xtype == 2).astype(float)
f = s[s.fits]
print(f.groupby(['win', 'epoch']).apply(lambda x: pd.Series(dict(n=len(x), hit_res=x[x.res].hit.mean(), p0=x[x.res].p0.mean(),
      net=x.net.mean(), t=x.net.mean() / x.net.std() * np.sqrt(len(x)), risk=x.risk.median())), include_groups=False).round(3).to_string())
