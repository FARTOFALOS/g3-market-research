"""R1 (LOG.md): что происходит после стопа утренней ставки S07; есть ли узнаваемая ветвь восстановления."""
import numpy as np, pandas as pd
from trade import Tape, epoch, TICK
from composite import s07_signals
from trade import simulate
from v0_s18 import bands

pd.set_option('display.width', 250)
T = Tape()
s = simulate(T, s07_signals(T))
st = s[(s.xtype == 1) & (s.date >= 20130101) & (s.date <= 20251231)].copy()
info, UB, LB = bands(T)
rows = []
for r in st.itertuples():
    ix = T.days[r.date]; tau = int(r.jexit); sg = r.side
    end = min(tau + 120, ix[-1])
    path = np.arange(tau + 1, end + 1)
    if len(path) < 30:
        continue
    e = T.o[tau + 1]
    # вилка +20 / −18,5 от open после стопа по σ и против σ
    def fork(side):
        fav = side * ((T.h[path] if side > 0 else T.l[path]) - e); adv = side * (e - (T.l[path] if side > 0 else T.h[path]))
        a = np.nonzero(fav >= 20.25)[0]; b = np.nonzero(adv >= 18.5)[0]
        ta = a[0] if len(a) else 10**6; tb = b[0] if len(b) else 10**6
        return 1 if ta < tb else (0 if tb <= ta and tb < 10**6 else np.nan)
    ny = ix[(T.mod[ix] >= 570) & (ix <= tau)]
    newext = (T.l[tau] <= T.l[ny].min()) if sg > 0 else (T.h[tau] >= T.h[ny].max())
    later = ix[ix > tau]
    ub = UB[later]; lb = LB[later]
    ex_s = np.nonzero(sg * (T.c[later] - np.where(sg > 0, ub, lb)) > 0)[0]
    ex_o = np.nonzero(-sg * (T.c[later] - np.where(sg > 0, lb, ub)) > 0)[0]
    rows.append(dict(date=r.date, ep=epoch(r.date), tau_min=int(T.mod[tau]), early=T.mod[tau] < 600, newext=bool(newext),
                     win_s=fork(sg), win_o=fork(-sg), mv_close=sg * (T.c[ix[-1]] - e),
                     first_exit=('sigma' if (len(ex_s) and (not len(ex_o) or ex_s[0] < ex_o[0])) else ('against' if len(ex_o) else 'none'))))
R = pd.DataFrame(rows)
print('days', R.groupby('ep').size().to_dict())
print(R.groupby('ep')[['win_s', 'win_o', 'mv_close']].mean().round(3).to_string())
for f in ['early', 'newext', 'first_exit']:
    print(R.groupby(['ep', f])[['win_s', 'win_o', 'mv_close']].agg(['mean', 'size']).round(3).to_string())
R.to_csv('r1_recovery.csv', index=False)
