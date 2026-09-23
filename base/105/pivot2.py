"""PV2 (LOG.md, объявлено до счёта): «первый пивот после T0», где T0 — событие с измеренным креном.
T0 = V7m (первая минута выхода дня за норму, 10:00–15:30) и T0 = 09:33 по стороне S-07. После T0 — первый пивот
отката по стороне (low[p] < low[p−1], ≤ low[p+1], ≤ low[p+2]; подтверждение на close p+2; откат ≥ 1u от максимума с T0;
для V7m пивот должен остаться за границей нормы, для S-07 — выше середины M), окно поиска 30 мин.
Вход open p+3 по стороне; отмена — пивот ∓ тик; остаток — продолжение крена: время 5 / 30 / 120 мин.
Сравнение — сам T0-вход той же ветви (V7m с отменой-границей; S07 с защитой 18,5) — что даёт вход на пивоте.
"""
import numpy as np, pandas as pd
from trade import Tape, simulate, report, TICK
from v0_s18 import bands
from v0_batch1 import s07_side

pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
T = Tape()
info, UB, LB = bands(T)
v7m = pd.read_csv('V7m_trades.csv')


def first_pivot(i0, sd, D, guard):
    ix = T.days[D]; last = ix[-1]
    HH = T.h[i0] if sd > 0 else T.l[i0]
    for p in range(i0 + 1, min(i0 + 31, last - 2)):
        if T.mod[p + 2] - T.mod[i0] != p + 2 - i0:
            return None
        if sd > 0:
            if T.l[p] <= guard(p):
                return None
            ok = T.l[p] < T.l[p - 1] and T.l[p] <= T.l[p + 1] and T.l[p] <= T.l[p + 2] and HH - T.l[p] >= T.u[i0]
        else:
            if T.h[p] >= guard(p):
                return None
            ok = T.h[p] > T.h[p - 1] and T.h[p] >= T.h[p + 1] and T.h[p] >= T.h[p + 2] and T.h[p] - HH >= T.u[i0]
        if ok:
            return p
        HH = max(HH, T.h[p]) if sd > 0 else min(HH, T.l[p])
    return None


if __name__ == '__main__':
    rows = []
    for r in v7m.itertuples():
        sd = r.side
        g = (lambda p, sd=sd: UB[p]) if sd > 0 else (lambda p, sd=sd: LB[p])
        p = first_pivot(r.irec, sd, r.date, g)
        if p is not None:
            rows.append(dict(date=r.date, irec=p + 2, side=sd, stop=(T.l[p] if sd > 0 else T.h[p]) - sd * TICK, target=np.nan, src='V7m'))
    for D, ix in T.days.items():
        sg, i33 = s07_side(T, D, ix)
        if sg == 0:
            continue
        b = np.arange(i33 - 29, i33 + 1); M = (T.h[b].max() + T.l[b].min()) / 2
        p = first_pivot(i33, sg, D, lambda q, M=M: M)
        if p is not None:
            rows.append(dict(date=D, irec=p + 2, side=sg, stop=(T.l[p] if sg > 0 else T.h[p]) - sg * TICK, target=np.nan, src='S07'))
    S = pd.DataFrame(rows)
    out = []
    for src in ('V7m', 'S07'):
        for tm in (5, 30, 120):
            s = simulate(T, S[S.src == src], tmax=tm)
            if src == 'S07':
                s = s[T.mod[s.irec] < 693]
            s.to_csv(f'PV2_{src}_{tm}.csv', index=False)
            out += report(s, f'PV2 {src} t{tm}')
            f = s[s.fits]; f = f.assign(struct=f.net + 0.75 + 0.5 * (f.xtype == 1))
            print(src, tm, f.groupby('epoch').struct.mean().round(2).to_dict())
    print(pd.DataFrame(out)[['v', 'epoch', 'signals', 'fits', 'risk_med', 'net_pt', 't', 'win', 'avg_w', 'avg_l', 'hold', 'stop']].to_string(index=False))
