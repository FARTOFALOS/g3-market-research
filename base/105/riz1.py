"""RZ1 (LOG.md, объявлено до счёта): «после RIZ первый пивот → продолжение» (исходная интенция трейдера) на индексе
Film-1 G3 (work/080a/index/film1_NQ.parquet; поле не меняется, только читается). T0 — минута зажигания RIZ (t0_ts_ns),
сторона north → продолжение вверх (прочь от зоны), south — вниз. На одну минуту T0 и сторону — один фильм со старшим ТФ.
Первый пивот отката по стороне в 30 мин после T0 (правило PV2), откат не касается exit_boundary (касание = конец Film-1,
сцена продолжения исчезла). Вход на подтверждении пивота; отмена — пивот ∓ тик; время 120. Вариант RZ1z: отмена — край
зоны ∓ тик (конец Film-1). Разрезы: ТФ (1 / 2–15 / 16–59 / ≥ 60), минуты от открытия NY (T0 в 09:30–11:29 против прочего).
Эпохи 2006–12 / 2013–19 / 2020–25.
"""
import numpy as np, pandas as pd
from trade import Tape, simulate, report, TICK
from pivot2 import first_pivot, T

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
F = pd.read_parquet('../080a/index/film1_NQ.parquet', columns=['tf_minutes', 'side', 'exit_boundary', 't0_ts_ns'])
F['i'] = np.searchsorted(T.ts, F.t0_ts_ns.to_numpy())
F = F[(F.i < len(T.ts))]
F = F[T.ts[F.i.to_numpy()] == F.t0_ts_ns.to_numpy()]
F['sd'] = np.where(F.side == 'north', 1, -1)
F = F.sort_values('tf_minutes').groupby(['i', 'sd']).tail(1)
F['date'] = T.date[F.i.to_numpy()]; F['mod'] = T.mod[F.i.to_numpy()]
F = F[(F.date <= 20251231) & (F['mod'] >= 120) & (F['mod'] <= 900)]
F = F[[d in T.days for d in F.date]]
print('T0 минут×сторон', len(F))
rows = []
for r in F.itertuples():
    eb = r.exit_boundary
    p = first_pivot(int(r.i), r.sd, r.date, lambda q, g=eb: g)
    if p is not None:
        piv = T.l[p] if r.sd > 0 else T.h[p]
        rows.append(dict(date=r.date, irec=p + 2, side=r.sd, stop=piv - r.sd * TICK, target=np.nan, tf=r.tf_minutes,
                         open_win=570 <= r.mod < 690, eb=eb))
S = pd.DataFrame(rows)
S['tfb'] = pd.cut(S.tf, [0, 1, 15, 59, 10**6], labels=['1', '2-15', '16-59', '60+'])
out = []
for nm, sig in [('RZ1', S), ('RZ1z', S.assign(stop=S.eb - S.side * TICK))]:
    s = simulate(T, sig, tmax=120)
    s = s[(s.xtype > 0) & (s.epoch != '2026')].copy()
    s['struct'] = s.net + 0.75 + 0.5 * (s.xtype == 1)
    f = s[s.fits]
    g = f.groupby(['tfb', 'open_win', 'epoch'], observed=True).agg(n=('net', 'size'), net=('net', 'mean'), struct=('struct', 'mean'),
        t=('net', lambda x: x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 else np.nan), win=('net', lambda x: (x > 0).mean())).round(2)
    print(nm, 'fits share', s.groupby('epoch').fits.mean().round(2).to_dict())
    print(g.unstack('epoch').to_string())
