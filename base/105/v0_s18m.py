"""v0-5 (LOG.md): первая минута выхода дня за норму, отмена — граница. Объявлено до счёта."""
import numpy as np, pandas as pd
from trade import Tape, simulate, report, TICK
from v0_s18 import bands

pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
T = Tape()
info, UB, LB = bands(T)
rows = []
for D, K, up, dn in info:
    for sd in (1, -1):
        for k in range(30, 361):
            i = K[k]
            if i < 0 or not np.isfinite(up[k]):
                continue
            b = up[k] if sd > 0 else dn[k]
            if sd * (T.c[i] - b) > 0:
                rows.append(dict(date=D, irec=int(i), side=sd, stop=b - sd * TICK, target=np.nan, k=k))
                break
sig = pd.DataFrame(rows)
s = simulate(T, sig); s.to_csv('V7m_trades.csv', index=False)
f = s[s.fits].copy(); f['struct'] = f.net + 0.75 + 0.5 * (f.xtype == 1)
print(pd.DataFrame(report(s, 'V7m')).to_string(index=False))
print(f.groupby(['epoch', 'side']).agg(n=('net', 'size'), net=('net', 'mean'), struct=('struct', 'mean'), risk=('risk', 'median')).round(2).to_string())
