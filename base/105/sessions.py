"""SE1 (LOG.md, объявлено до счёта): сессионные уровни как якоря T0 → первый пивот → продолжение (линза трейдера).
Уровни: ASIA = H/L 18:00 предыдущей даты … 01:59; LON = H/L 02:00…09:29; PDH/PDL = H/L вчерашней сессии 09:30–закрытие.
Якоря: A→L — первое закрытие 02:00–07:59 за краем ASIA; L→N — первое закрытие 09:30–11:59 за краем LON;
P→N — первое закрытие 09:30–13:59 за вчерашним H/L.
Прочтения: CONT — первый пивот отката по стороне пробоя (пивот не уходит за пробитый уровень), вход на подтверждении,
отмена — пивот; SWEEP — если в 15 мин после пробоя закрытие обратно внутри → сторона против выноса, первый пивот по
ней (не выше крайней точки выноса), вход на подтверждении, отмена — пивот. Время 120 (не позже закрытия).
Правило пивота — как в PV2 (откат ≥ 1u, окно 30 мин). Показать net/struct по эпохам и час T0 (линза).
"""
import numpy as np, pandas as pd
from trade import Tape, simulate, report, TICK
from pivot2 import first_pivot, T

pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
dates = sorted(D for D in T.days if D <= 20251231)
prev_ny = None
rows = []
for D in dates:
    ix = T.days[D]; m = T.mod[ix]
    Dp = int((pd.Timestamp(str(D)) - pd.Timedelta(days=1)).strftime('%Y%m%d'))
    # Азия: бары даты Dp с 18:00 и бары D до 02:00
    asia = np.array([], dtype=np.int64)
    if Dp in T.date_start:
        a0 = T.date_start[Dp]; a1 = T.date_start[D]
        pa = np.arange(a0, a1); pa = pa[T.mod[pa] >= 1080]
        da = np.arange(a1, ix[0]); da = da[T.mod[da] < 120]
        asia = np.concatenate([pa, da])
    lon = ix[(m >= 120) & (m <= 569)]
    ny = ix[m >= 570]
    levels = []
    if len(asia) >= 400 and T.mod[asia[0]] == 1080:
        levels.append(('A->L', T.h[asia].max(), T.l[asia].min(), ix[(m >= 120) & (m <= 479)]))
    if len(lon) >= 440:
        levels.append(('L->N', T.h[lon].max(), T.l[lon].min(), ix[(m >= 570) & (m <= 719)]))
    if prev_ny is not None:
        levels.append(('P->N', prev_ny[0], prev_ny[1], ix[(m >= 570) & (m <= 839)]))
    for nm, H, L, sc in levels:
        for sd in (1, -1):
            br = np.nonzero(T.c[sc] > H)[0] if sd > 0 else np.nonzero(T.c[sc] < L)[0]
            if not len(br):
                continue
            i0 = sc[br[0]]; lvl = H if sd > 0 else L
            p = first_pivot(i0, sd, D, lambda q, g=lvl: g)
            if p is not None:
                rows.append(dict(date=D, irec=p + 2, side=sd, stop=(T.l[p] if sd > 0 else T.h[p]) - sd * TICK, target=np.nan,
                                 src=nm, mode='CONT', t0=int(T.mod[i0])))
            # SWEEP
            X = T.h[i0] if sd > 0 else T.l[i0]
            for j in range(i0 + 1, min(i0 + 16, ix[-1])):
                if T.date[j] != D or T.mod[j] - T.mod[i0] > 15:
                    break
                X = max(X, T.h[j]) if sd > 0 else min(X, T.l[j])
                if sd * (T.c[j] - lvl) < 0:
                    p = first_pivot(j, -sd, D, lambda q, g=X: g)
                    if p is not None:
                        rows.append(dict(date=D, irec=p + 2, side=-sd, stop=(T.l[p] if -sd > 0 else T.h[p]) + sd * TICK,
                                         target=np.nan, src=nm, mode='SWEEP', t0=int(T.mod[i0])))
                    break
    prev_ny = (T.h[ny].max(), T.l[ny].min()) if len(ny) >= 300 else None
S = pd.DataFrame(rows)
s = simulate(T, S, tmax=120)
s.to_csv('SE1_trades.csv', index=False)
f = s[s.fits].copy(); f['struct'] = f.net + 0.75 + 0.5 * (f.xtype == 1)
f['hr'] = (f.t0 // 60).astype(int)
g = f.groupby(['src', 'mode', 'epoch']).agg(n=('net', 'size'), net=('net', 'mean'), struct=('struct', 'mean'),
                                           t=('net', lambda x: x.mean() / x.std() * np.sqrt(len(x))), win=('net', lambda x: (x > 0).mean()),
                                           risk=('risk', 'median')).round(2)
print(g.to_string())
print('\nлинза: час T0, 2020-25 и 2013-19, struct')
h = f[f.epoch.isin(['2013-19', '2020-25'])].groupby(['src', 'mode', 'hr', 'epoch']).struct.agg(['size', 'mean']).round(2).unstack('epoch')
print(h.to_string())
print('fits share', s[s.xtype > 0].groupby('epoch').fits.mean().round(2).to_dict())
