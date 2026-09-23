"""V12/V13 (LOG.md, объявлено до счёта).
V12 выход за диапазон первого часа (IB = H/L 09:30–10:29): с 10:30 до 14:59 первое закрытие минуты за краем IB →
по выходу на open следующей; отмена — край IB ∓ тик (цена вернулась в IB = идея «день расширяется» кончилась);
остаток — расширение дня, время 120. Вариант V12f — против выхода, если в 10 мин закрытие обратно внутри (отмена —
крайняя точка выноса ± тик, цель — середина IB).
V13 разворот 10:00: ход 09:30→10:00 (close 09:59 − open 09:30) больше 1,5 нормы этого хода (медиана |ход| 20 сессий);
первая минута 10:00–10:29, закрывшаяся ниже low предыдущей минуты (для хода вверх) → против хода; отмена — максимум
утра ± тик; остаток — отдача утреннего хода (цель — середина 09:30→экстремум), время 120.
"""
import numpy as np, pandas as pd
from trade import Tape, simulate, report, TICK

pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
T = Tape()
r12, r12f, r13 = [], [], []
hist = []
for D in sorted(T.days):
    ix = T.days[D]; m = T.mod[ix]
    ib = ix[(m >= 570) & (m <= 629)]
    if len(ib) != 60:
        continue
    IH, IL = T.h[ib].max(), T.l[ib].min()
    sc = ix[(m >= 630) & (m <= 899)]
    for sd in (1, -1):
        br = np.nonzero(T.c[sc] > IH)[0] if sd > 0 else np.nonzero(T.c[sc] < IL)[0]
        if not len(br):
            continue
        i = sc[br[0]]; lvl = IH if sd > 0 else IL
        r12.append(dict(date=D, irec=i, side=sd, stop=lvl - sd * TICK, target=np.nan))
        X = T.h[i] if sd > 0 else T.l[i]
        for j in sc[br[0] + 1: br[0] + 11]:
            if T.mod[j] - T.mod[i] > 10:
                break
            X = max(X, T.h[j]) if sd > 0 else min(X, T.l[j])
            if sd * (T.c[j] - lvl) < 0:
                r12f.append(dict(date=D, irec=j, side=-sd, stop=X + sd * TICK, target=(IH + IL) / 2)); break
    # V13
    am = ix[(m >= 570) & (m <= 599)]
    if len(am) != 30:
        continue
    mv = T.c[am[-1]] - T.o[am[0]]
    norm = np.median(np.abs(hist[-20:])) if len(hist) >= 20 else np.nan
    hist.append(mv)
    if not np.isfinite(norm) or abs(mv) <= 1.5 * norm:
        continue
    sd = -1 if mv > 0 else 1
    ext = T.h[am].max() if mv > 0 else T.l[am].min()
    sc2 = ix[(m >= 600) & (m <= 629)]
    for j in sc2:
        ext = max(ext, T.h[j]) if mv > 0 else min(ext, T.l[j])
        pj = j - 1
        if T.mod[pj] != T.mod[j] - 1:
            continue
        if (mv > 0 and T.c[j] < T.l[pj]) or (mv < 0 and T.c[j] > T.h[pj]):
            tg = (T.o[am[0]] + ext) / 2
            r13.append(dict(date=D, irec=j, side=sd, stop=ext - sd * TICK, target=tg)); break
out = []
for nm, rows, tm in [('V12', r12, 120), ('V12f', r12f, 120), ('V13', r13, 120)]:
    s = simulate(T, pd.DataFrame(rows), tmax=tm); s.to_csv(f'{nm}_trades.csv', index=False)
    out += report(s, nm)
    f = s[s.fits]; f = f.assign(struct=f.net + 0.75 + 0.5 * (f.xtype == 1))
    print(nm, f.groupby('epoch').agg(n=('net', 'size'), struct=('struct', 'mean')).round(2).to_dict())
print(pd.DataFrame(out).to_string(index=False))
