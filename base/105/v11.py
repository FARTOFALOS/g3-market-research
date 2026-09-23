"""V11 (LOG.md): второе испытание экстремума дня → по пробою. Объявлено до счёта."""
import numpy as np, pandas as pd
from trade import Tape, simulate, report, TICK
from grammar import build

pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
T = Tape()
S = build(T)
t8 = S[(S.trig == 9) & S.win.isin(['mid', 'late'])].copy()
a = t8.copy(); a['side'] = -a.side
i = a.irec.to_numpy()
a['stop'] = np.where(a.side > 0, T.l[i] - TICK, T.h[i] + TICK)
sa = simulate(T, a, tmax=60); sa.to_csv('V11a_trades.csv', index=False)
rows = []
for r in t8.itertuples():
    sd = -r.side; lvl = r.stop - sd * TICK   # экстремум дня (стоп T8 = экстремум ± тик)
    ext = T.l[r.irec] if sd > 0 else T.h[r.irec]
    for j in range(r.irec + 1, r.irec + 16):
        if T.date[j] != r.date or T.mod[j] != T.mod[j - 1] + 1 or T.mod[j] > 930:
            break
        ext = min(ext, T.l[j]) if sd > 0 else max(ext, T.h[j])
        if sd * (T.c[j] - lvl) > 0:
            rows.append(dict(date=r.date, irec=j, side=sd, stop=ext - sd * TICK, target=np.nan, win=r.win)); break
sb = simulate(T, pd.DataFrame(rows), tmax=60); sb.to_csv('V11b_trades.csv', index=False)
out = report(sa, 'V11a') + report(sb, 'V11b')
print(pd.DataFrame(out).to_string(index=False))
for nm, x in [('V11a', sa), ('V11b', sb)]:
    f = x[x.fits]; f = f.assign(struct=f.net + 0.75 + 0.5 * (f.xtype == 1))
    print(nm, f.groupby(['epoch', 'win']).agg(n=('net', 'size'), net=('net', 'mean'), struct=('struct', 'mean')).round(2).to_string())
