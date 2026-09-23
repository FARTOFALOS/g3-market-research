"""ZZ1 (LOG.md, объявлено до счёта): представление колен — импульс → откат в зону 50–61,8 % → разворотная свеча → по
импульсу. Колено вверх: от минимума A до максимума B (B − A ≥ 6u(B), ≤ 60 мин), после B откат: первая минута, чей low
вошёл в зону [B − 0,618(B − A), B − 0,5(B − A)] при том, что после B цена не выходила ниже B − 0,786(B − A) и выше B;
узнавание — первая следующая минута (в 20 мин), закрывшаяся выше high предыдущей (разворот); если раньше low ниже
0,786 — сцены нет. Вход open следующей; отмена — B − 0,786(B − A) − тик (Фибо-граница «откат стал разворотом»); цель —
B + тик прохода (возврат к вершине импульса); время 60. A — минимум за 60 мин до B; B — первый максимум, после которого
5 минут нет нового максимума (подтверждение вершины к минуте B+5). Зеркально вниз. Окна: pre / NY. По эпохам: доля целей
против честной p0, net.
"""
import numpy as np, pandas as pd
from numba import njit
from trade import Tape, simulate, TICK

pd.set_option('display.width', 250)


@njit(cache=True)
def scan(o, h, l, c, mod, dt, u, inwin):
    n = len(o)
    I = []; S = []; ST = []; TG = []
    last_b = -100
    for b in range(60, n - 30):
        if not inwin[b] or b - last_b < 5:
            continue
        for s in (1, -1):
            # вершина B подтверждена 5 минутами без нового экстремума
            ok = True
            for j in range(b + 1, b + 6):
                if dt[j] != dt[b] or mod[j] - mod[b] != j - b:
                    ok = False; break
                if (s > 0 and h[j] > h[b]) or (s < 0 and l[j] < l[b]):
                    ok = False; break
            if not ok:
                continue
            # вершина — экстремум последних 60 минут
            is_ext = True
            A = l[b] if s > 0 else h[b]
            for j in range(b - 60, b):
                if dt[j] != dt[b]:
                    is_ext = False; break
                if (s > 0 and h[j] >= h[b]) or (s < 0 and l[j] <= l[b]):
                    is_ext = False; break
                A = min(A, l[j]) if s > 0 else max(A, h[j])
            if not is_ext:
                continue
            B = h[b] if s > 0 else l[b]
            L = s * (B - A)
            if L < 6 * u[b]:
                continue
            z_in = B - s * 0.5 * L; z_out = B - s * 0.618 * L; inv = B - s * 0.786 * L
            entered = False
            for j in range(b + 1, b + 41):
                if dt[j] != dt[b] or mod[j] - mod[b] != j - b or not inwin[j]:
                    break
                if (s > 0 and (l[j] <= inv or h[j] > B)) or (s < 0 and (h[j] >= inv or l[j] < B)):
                    break
                if not entered:
                    if (s > 0 and l[j] <= z_in) or (s < 0 and h[j] >= z_in):
                        entered = True
                    continue
                if (s > 0 and c[j] > h[j - 1]) or (s < 0 and c[j] < l[j - 1]):
                    I.append(j); S.append(s); ST.append(inv - s * 0.25); TG.append(B)
                    last_b = b
                    break
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
      resolved=x.res.mean(), net=x.net.mean(), t=x.net.mean() / x.net.std() * np.sqrt(len(x)), risk=x.risk.median())), include_groups=False).round(3).to_string())
print('fits share', s.groupby('epoch').fits.mean().round(2).to_dict())
