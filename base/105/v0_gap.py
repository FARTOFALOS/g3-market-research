"""v0-4 (LOG.md): гэп открытия, цель — вчерашний close, отмена — экстремум минуты 09:30. Объявлено до счёта."""
import numpy as np, pandas as pd
from trade import Tape, simulate, report, TICK, epoch

pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
T = Tape()
dates = sorted(T.days)
rows = []
prev_close = None
for D in dates:
    ix = T.days[D]
    i930 = T.at(D, 570)
    if i930 >= 0 and prev_close is not None:
        G = T.o[i930] - prev_close
        if G != 0:
            sd = -1 if G > 0 else 1                  # сторона сделки — против гэпа
            red = T.c[i930] < T.o[i930]; green = T.c[i930] > T.o[i930]
            against = (G > 0 and red) or (G < 0 and green)
            ext = T.h[i930] if G > 0 else T.l[i930]
            rows.append(dict(date=D, irec=i930, side=sd, stop=ext - sd * TICK, target=prev_close, against=against,
                             gap=G, gap_u=abs(G) / T.u[i930 - 1]))
    ny = ix[T.mod[ix] >= 570]
    prev_close = T.c[ix[-1]] if len(ny) and T.mod[ny[0]] == 570 else None
sig = pd.DataFrame(rows)
allr = []
s = simulate(T, sig[sig.against]); s.to_csv('V9_trades.csv', index=False); allr += report(s, 'V9')
s2 = simulate(T, sig); s2.to_csv('V9b_trades.csv', index=False); allr += report(s2, 'V9b')
print(pd.DataFrame(allr).to_string(index=False))
for lab, x in [('V9', s), ('V9b', s2)]:
    f = x[x.fits].copy()
    f['gq'] = f.groupby('epoch').gap_u.transform(lambda v: pd.qcut(v.rank(method='first'), 3, labels=['small', 'mid', 'big']))
    f['struct'] = f.net + 0.75 + 0.5 * (f.xtype == 1)
    print(lab, '\n', f.groupby(['epoch', 'gq'], observed=True).agg(n=('net', 'size'), net=('net', 'mean'), struct=('struct', 'mean'),
          tgt=('xtype', lambda v: (v == 2).mean()), gap_pt=('gap', lambda v: v.abs().median()), risk=('risk', 'median')).round(2).to_string())
