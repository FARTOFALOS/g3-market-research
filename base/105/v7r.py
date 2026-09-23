"""C1 (LOG.md): все свежие выходы цены за норму дня (повторные попытки по новому наблюдению)."""
import numpy as np, pandas as pd
from trade import Tape, simulate, TICK
from v0_s18 import bands

T = Tape()
info, UB, LB = bands(T)
rows = []
for D, K, up, dn in info:
    zp = None
    for k in range(29, 361):
        i = K[k]
        if i < 0 or not np.isfinite(up[k]):
            zp = None; continue
        p = T.c[i]
        z = 1 if p > up[k] else (-1 if p < dn[k] else 0)
        if k >= 30 and z != 0 and zp == 0:
            b = up[k] if z > 0 else dn[k]
            rows.append(dict(date=D, irec=int(i), side=z, stop=b - z * TICK, target=np.nan, k=k))
        zp = z
sig = pd.DataFrame(rows)
s = simulate(T, sig); s.to_csv('V7r_trades.csv', index=False)
f = s[s.fits]; f = f.assign(struct=f.net + 0.75 + 0.5 * (f.xtype == 1))
print(f.groupby('epoch').agg(n=('net', 'size'), per_day=('date', lambda d: len(d) / d.nunique()), net=('net', 'mean'),
      struct=('struct', 'mean'), win=('net', lambda x: (x > 0).mean()), risk=('risk', 'median')).round(2).to_string())
