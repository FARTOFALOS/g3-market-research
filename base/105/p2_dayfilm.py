"""P2 (LOG.md): рентген полных дней политики C0 (замок), 2020–25."""
import numpy as np, pandas as pd
from trade import Tape

pd.set_option('display.width', 250)
T = Tape()
tk = pd.read_csv('C0_lock.csv')
rows = []
for D, g in tk.groupby('date'):
    ix = T.days[D]; ny = ix[T.mod[ix] >= 570]
    if len(ny) < 300:
        continue
    H, L = T.h[ny].max(), T.l[ny].min(); C = T.c[ny[-1]]; O = T.o[ny[0]]
    tH = T.mod[ny[np.argmax(T.h[ny])]]; tL = T.mod[ny[np.argmin(T.l[ny])]]
    last = g.iloc[-1]; after = ix[ix > last.jexit]
    s = last.side
    fav_after = s * ((T.h[after].max() if s > 0 else T.l[after].min()) - T.c[int(last.jexit)]) if len(after) else 0.0
    adv_after = s * (T.c[int(last.jexit)] - (T.l[after].min() if s > 0 else T.h[after].max())) if len(after) else 0.0
    first = g.iloc[0]
    rows.append(dict(date=D, net=g.net.sum(), n=len(g), first_branch=first.branch, first_net=first.net,
                     first_side_right=np.sign(C - T.o[int(first.irec) + 1]) == first.side,
                     loc=(C - L) / (H - L), rng=H - L, day_dir=np.sign(C - O),
                     last_exit_min=int(T.mod[int(last.jexit)]), fav_after=fav_after, adv_after=adv_after,
                     ext_late=max(tH, tL) >= 720))
R = pd.DataFrame(rows)
R['cls'] = np.where(R.net >= 0, 'WIN', 'LOSS')
R['type'] = np.select([R['loc'] >= 0.8, R['loc'] <= 0.2], ['closeHigh', 'closeLow'], 'mid')
print(R.groupby('cls').agg(days=('net', 'size'), net=('net', 'mean'), trades=('n', 'mean'), rng=('rng', 'median'),
      first_right=('first_side_right', 'mean'), last_exit=('last_exit_min', 'median'), fav_after=('fav_after', 'median'),
      adv_after=('adv_after', 'median'), ext_late=('ext_late', 'mean')).round(2).to_string())
print(pd.crosstab(R.cls, R.type, normalize='index').round(2).to_string())
print(pd.crosstab(R.cls, R.first_branch).to_string())
L = R[R.cls == 'LOSS']
print('LOSS days: net distribution', L.net.describe().round(1).to_dict())
print('LOSS: first trade in day-direction share', L.first_side_right.mean().round(3), ' WIN:', R[R.cls == "WIN"].first_side_right.mean().round(3))
R.to_csv('p2_dayfilm.csv', index=False)
