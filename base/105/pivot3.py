"""PV3 (LOG.md, объявлено до счёта): «T0 → первый пивот → продолжение» для якорей с измеренным утренним креном.
T0: (a) IB — первое закрытие за краем диапазона 09:30–10:29 (10:30–14:59); (b) OR — первое закрытие за краем
09:30–09:44 (09:45–11:59); (c) BOX — закрытие за краем 30-минутной коробки в 09:30–10:59 (первое на сторону в день).
Первый пивот отката по стороне в 30 мин (как PV2), пивот не должен уйти обратно за пробитый край (для BOX — за
середину коробки). Вход open p+3, отмена — пивот ∓ тик, время 120 (и 30). Правило пивота и окна — как в PV2, без подбора.
"""
import numpy as np, pandas as pd
from trade import Tape, simulate, report, TICK
from pivot2 import first_pivot, T

pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
rows = []
for D in sorted(T.days):
    if D > 20251231:
        continue
    ix = T.days[D]; m = T.mod[ix]
    ib = ix[(m >= 570) & (m <= 629)]
    orr = ix[(m >= 570) & (m <= 584)]
    anchors = []
    if len(ib) == 60:
        IH, IL = T.h[ib].max(), T.l[ib].min()
        anchors.append(('IB', ix[(m >= 630) & (m <= 899)], IH, IL, IH, IL))
    if len(orr) == 15:
        OH, OL = T.h[orr].max(), T.l[orr].min()
        anchors.append(('OR', ix[(m >= 585) & (m <= 719)], OH, OL, OH, OL))
    for nm, sc, H, L, gH, gL in anchors:
        for sd in (1, -1):
            br = np.nonzero(T.c[sc] > H)[0] if sd > 0 else np.nonzero(T.c[sc] < L)[0]
            if not len(br):
                continue
            i0 = sc[br[0]]; guard = gH if sd > 0 else gL
            p = first_pivot(i0, sd, D, lambda q, g=guard: g)
            if p is not None:
                rows.append(dict(date=D, irec=p + 2, side=sd, stop=(T.l[p] if sd > 0 else T.h[p]) - sd * TICK, target=np.nan, src=nm))
    # BOX
    op = ix[(m >= 570) & (m <= 659)]
    done = {1: False, -1: False}
    for i in op:
        if i - 31 < 0 or T.date[i - 30] != D or T.mod[i] - T.mod[i - 30] != 30:
            continue
        H = T.h[i - 30:i].max(); L = T.l[i - 30:i].min()
        if not (L <= T.c[i - 1] <= H):
            continue
        for sd in (1, -1):
            if done[sd]:
                continue
            if (sd > 0 and T.c[i] > H) or (sd < 0 and T.c[i] < L):
                done[sd] = True
                p = first_pivot(i, sd, D, lambda q, g=(H + L) / 2: g)
                if p is not None:
                    rows.append(dict(date=D, irec=p + 2, side=sd, stop=(T.l[p] if sd > 0 else T.h[p]) - sd * TICK, target=np.nan, src='BOX'))
S = pd.DataFrame(rows)
out = []
for src in ('IB', 'OR', 'BOX'):
    for tm in (30, 120):
        s = simulate(T, S[S.src == src], tmax=tm)
        s.to_csv(f'PV3_{src}_{tm}.csv', index=False)
        out += report(s, f'PV3 {src} t{tm}')
        f = s[s.fits]; f = f.assign(struct=f.net + 0.75 + 0.5 * (f.xtype == 1))
        print(src, tm, f.groupby('epoch').struct.mean().round(2).to_dict())
print(pd.DataFrame(out)[['v', 'epoch', 'signals', 'fits', 'risk_med', 'net_pt', 't', 'win', 'avg_w', 'avg_l', 'hold', 'stop']].to_string(index=False))
