"""Рентген V7m: чем 8 % победителей отличаются от стопнутых в минуту узнавания (объявлено в LOG перед счётом)."""
import numpy as np, pandas as pd
from trade import Tape
from v0_batch1 import s07_side

pd.set_option('display.width', 260)
T = Tape()
s = pd.read_csv('V7m_trades.csv'); f = s[s.fits].copy()
f['W'] = (f.net > 0).astype(int)
f['struct'] = f.net + 0.75 + 0.5 * (f.xtype == 1)
feat = []
for r in f.itertuples():
    ix = T.days[r.date]; i = r.irec; u = T.u[i]
    ny = ix[(T.mod[ix] >= 570) & (ix <= i)]
    hi, lo = T.h[ny].max(), T.l[ny].min()
    sg, _ = s07_side(T, r.date, ix)
    feat.append(dict(risk_u=r.risk / u, bar_u=(T.h[i] - T.l[i]) / u,
                     s07=int(sg == r.side), run30=r.side * (T.c[i] - T.c[i - 30]) / u,
                     edge=int((T.h[i] >= hi) if r.side > 0 else (T.l[i] <= lo)),
                     rng_u=(hi - lo) / u))
F = pd.concat([f.reset_index(drop=True), pd.DataFrame(feat)], axis=1)
F.to_csv('film_v7m.csv', index=False)
rows = []
for c in ['risk_u', 'bar_u', 'k', 'run30', 'rng_u']:
    F['q'] = F.groupby('epoch')[c].transform(lambda v: pd.qcut(v.rank(method='first'), 3, labels=['lo', 'mid', 'hi']))
    for (ep, q), g in F[F.epoch != '2026'].groupby(['epoch', 'q'], observed=True):
        rows.append(dict(feat=c, ep=ep, q=q, n=len(g), W=round(g.W.mean(), 3), struct=round(g.struct.mean(), 2)))
for c in ['s07', 'edge']:
    for (ep, q), g in F[F.epoch != '2026'].groupby(['epoch', c]):
        rows.append(dict(feat=c, ep=ep, q=q, n=len(g), W=round(g.W.mean(), 3), struct=round(g.struct.mean(), 2)))
r = pd.DataFrame(rows)
print(r.pivot_table(index=['feat', 'q'], columns='ep', values=['W', 'struct']).round(2).to_string())
