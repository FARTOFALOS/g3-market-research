"""M1 (LOG.md): вчерашний день → ход сегодняшнего утра/дня. Объявлено до счёта."""
import numpy as np, pandas as pd
from trade import Tape, epoch

pd.set_option('display.width', 250)
T = Tape()
S = []
for D, ix in T.days.items():
    m = T.mod[ix]; ny = ix[m >= 570]
    if len(ny) < 300 or T.mod[ny[0]] != 570:
        continue
    i1130 = T.at(D, 689)
    S.append(dict(date=D, O=T.o[ny[0]], H=T.h[ny].max(), L=T.l[ny].min(), C=T.c[ny[-1]], o931=T.o[ny[1]] if T.mod[ny[1]] == 571 else np.nan,
                  c1130=T.c[i1130] if i1130 >= 0 else np.nan))
S = pd.DataFrame(S)
S['R'] = S.H - S.L
S['norm'] = S.R.rolling(20).median().shift(1)
p = S.shift(1); pp = S.shift(2)
S['open_loc'] = np.select([S.O > p.H, S.O >= (p.H + p.L) / 2, S.O >= p.L], ['aboveH', 'upperHalf', 'lowerHalf'], 'belowL')
cl = (p.C - p.L) / p.R
S['yclose'] = np.select([cl >= 0.8, cl <= 0.2], ['top20', 'bot20'], 'mid')
rr = p.R / S.norm.shift(1)
S['yrange'] = np.select([rr < 0.7, rr > 1.3], ['narrow', 'wide'], 'normal')
S['inside'] = np.where((p.H <= pp.H) & (p.L >= pp.L), 'inside', 'no')
S['ydir'] = np.where(p.C > p.O, 'yUp', 'yDown')
S['m_am'] = S.c1130 - S.o931; S['m_day'] = S.C - S.o931
S['ep'] = S.date.map(epoch)
S = S[(S.ep != 'other') & (S.ep != '2026')].dropna(subset=['m_am', 'm_day', 'norm'])
rows = []
for f in ['open_loc', 'yclose', 'yrange', 'inside', 'ydir']:
    for (lev, ep), g in S.groupby([f, 'ep']):
        for y in ['m_am', 'm_day']:
            rows.append(dict(feat=f, level=lev, ep=ep, out=y, n=len(g), mean=g[y].mean(), t=g[y].mean() / g[y].std() * np.sqrt(len(g))))
R = pd.DataFrame(rows)
W = R.pivot_table(index=['feat', 'level', 'out'], columns='ep', values=['mean', 't', 'n']).round(2)
print(W.to_string())
