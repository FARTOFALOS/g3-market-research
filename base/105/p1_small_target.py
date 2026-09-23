"""P1 (LOG.md): доля маленьких целей при защите 18,5 — против честной вилки. Объявлено до счёта."""
import numpy as np, pandas as pd
from numba import njit
from trade import Tape, epoch

@njit(cache=True)
def run(o, h, l, c, mod, dt, starts, X, S, tick, tmax, closem):
    n = len(starts); res = np.zeros((n, 2), np.int8); pnl = np.zeros((n, 2))
    for k in range(n):
        i = starts[k]; e = o[i]
        for sd in range(2):
            s = 1 if sd == 0 else -1
            tg = e + s * X; st = e - s * S
            out = 0; px = np.nan
            j = i
            while True:
                if s > 0:
                    if l[j] <= st: out = -1; px = min(o[j], st) - 0.5; break
                    if h[j] >= tg + tick: out = 1; px = max(o[j], tg); break
                else:
                    if h[j] >= st: out = -1; px = max(o[j], st) + 0.5; break
                    if l[j] <= tg - tick: out = 1; px = min(o[j], tg); break
                if j - i + 1 >= tmax or j + 1 >= len(o) or dt[j + 1] != dt[j] or mod[j + 1] != mod[j] + 1 or mod[j + 1] >= closem[j + 1]:
                    px = c[j]; out = 0; break
                j += 1
            res[k, sd] = out; pnl[k, sd] = s * (px - e) - 0.75
    return res, pnl

T = Tape()
idx = np.concatenate([ix for D, ix in T.days.items() if 20130101 <= D <= 20251231])
idx = idx[idx + 1 < len(T.o)]
st = idx + 1
st = st[(T.date[st] == T.date[idx]) & (T.mod[st] == T.mod[idx] + 1) & (T.mod[st] >= 120)]
rows = []
for X in [1.0, 2.0, 3.0, 5.0, 8.0]:
    res, pnl = run(T.o, T.h, T.l, T.c, T.mod, T.date, st.astype(np.int64), X, 18.5, 0.25, 120, T.close_mod)
    df = pd.DataFrame(dict(ep=np.where(T.date[st] >= 20200101, '2020-25', '2013-19'),
                           ny=np.where(T.mod[st] >= 570, 'NY', 'pre')))
    for sd, nm in [(0, 'long'), (1, 'short')]:
        df['out'] = res[:, sd]; df['pnl'] = pnl[:, sd]
        for (ep, ny), g in df.groupby(['ep', 'ny']):
            rows.append(dict(X=X, side=nm, ep=ep, hours=ny, n=len(g), tgt=round((g.out == 1).mean(), 4), stop=round((g.out == -1).mean(), 4),
                             fair=round(18.5 / (X + 0.25 + 18.5), 4), net=round(g.pnl.mean(), 3)))
r = pd.DataFrame(rows); r['delta'] = (r.tgt + (r.stop == 0) * 0 - r.fair * (r.tgt + r.stop)).round(4)
print(r.to_string(index=False))
