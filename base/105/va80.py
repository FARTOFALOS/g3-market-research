"""VA1 (LOG.md, объявлено до счёта): «правило 80 %» профиля рынка — новое представление: время у цены (без объёма).
Зона стоимости вчерашней сессии 09:30–закрытие: каждая минута добавляет 1 во все ценовые корзины между своими low и
high (ширина корзины — 1/200 размаха вчерашней сессии); POC — корзина с наибольшим временем; зона — от POC наращивается
в сторону большего соседнего времени до 70 % общего времени (VAL, VAH).
Сцена: открытие 09:30 вне зоны (выше VAH или ниже VAL). Получасовые скобки 09:30, 10:00, …, 15:00. Узнавание —
закрытие второй подряд скобки, закрывшейся внутри зоны (принятие); первая пара до 14:30. Действие — к противоположному
краю зоны на open следующей минуты. Цель — противоположный край (VAL при открытии выше). Отмена — край входа ± 1u
(цена снова вне зоны = принятие отвергнуто). Время 120. Вариант VA1c: вместо отмены-края — защитный предел 18,5.
Показать: доля целей против честной p0, net, долю влезающих, по эпохам.
"""
import numpy as np, pandas as pd
from trade import Tape, simulate, TICK

pd.set_option('display.width', 250)
T = Tape()


def value_area(ix):
    lo, hi = T.l[ix].min(), T.h[ix].max()
    w = (hi - lo) / 200 if hi > lo else 0.25
    nb = int(np.ceil((hi - lo) / w)) + 1
    cnt = np.zeros(nb)
    a = ((T.l[ix] - lo) / w).astype(int); b = ((T.h[ix] - lo) / w).astype(int)
    for x, y in zip(a, b):
        cnt[x:y + 1] += 1
    poc = int(np.argmax(cnt)); tot = cnt.sum(); L = R = poc; acc = cnt[poc]
    while acc < 0.7 * tot:
        up = cnt[R + 1] if R + 1 < nb else -1; dn = cnt[L - 1] if L - 1 >= 0 else -1
        if up >= dn:
            R += 1; acc += cnt[R]
        else:
            L -= 1; acc += cnt[L]
    return lo + L * w, lo + (R + 1) * w


rows = []
prev = None
for D in sorted(T.days):
    if D > 20251231:
        break
    ix = T.days[D]; m = T.mod[ix]
    ny = ix[m >= 570]
    if prev is not None and len(ny) >= 300 and T.mod[ny[0]] == 570:
        VAL, VAH = prev
        O = T.o[ny[0]]
        if O > VAH or O < VAL:
            sd = -1 if O > VAH else 1
            inside_prev = False
            for b0 in range(570, 900, 30):
                br = ix[(m >= b0) & (m < b0 + 30)]
                if len(br) < 25:
                    inside_prev = False; continue
                cl = T.c[br[-1]]
                ins = VAL < cl < VAH
                if ins and inside_prev:
                    i = br[-1]; u = T.u[i]
                    edge = VAH if sd < 0 else VAL
                    rows.append(dict(date=D, irec=i, side=sd, stop=edge - sd * u, target=(VAL if sd < 0 else VAH), b=b0))
                    break
                inside_prev = ins
    prev = value_area(ny) if len(ny) >= 300 else None
S = pd.DataFrame(rows)
out = []
for nm, sig in [('VA1', S), ('VA1c', S.assign(stop=np.nan))]:
    if nm == 'VA1c':
        e = T.o[S.irec.to_numpy() + 1]
        sig = S.assign(stop=e - S.side * 18.5)
    s = simulate(T, sig, tmax=120)
    s = s[s.xtype > 0].copy()
    s['p0'] = s.risk / (s.risk + s.side * (s.target - s.entry))
    s['res'] = s.xtype.isin([1, 2]); s['hit'] = (s.xtype == 2).astype(float)
    for ep, g in s.groupby('epoch'):
        if ep == '2026':
            continue
        f = g[g.fits]
        out.append(dict(v=nm, ep=ep, n=len(g), fits=round(g.fits.mean(), 2), hit=round(f.hit.mean(), 3),
                        hit_res=round(f[f.res].hit.mean(), 3), p0=round(f[f.res].p0.mean(), 3), net=round(f.net.mean(), 2),
                        t=round(f.net.mean() / f.net.std() * np.sqrt(len(f)), 2), dist=round((f.side * (f.target - f.entry)).median(), 1)))
print(pd.DataFrame(out).to_string(index=False))
