"""OB-1 (LOG.md): наблюдаема ли ошибка политики F0 в точке решения 09:33. Шаг (a): набор DP1 + одномерный рентген."""
import numpy as np, pandas as pd
from trade import Tape, simulate, epoch
from composite import s07_signals, run
from v0_batch1 import s07_side

pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
T = Tape()
cols = ['date', 'irec', 'side', 'stop', 'target', 'iend', 'branch', 'entry', 'exit', 'xtype', 'jexit', 'risk', 'plan_loss', 'net', 'fits']
s07 = simulate(T, s07_signals(T)); s07['branch'] = 'S07'
pv = pd.read_csv('PV2_V7m_120.csv'); pv['branch'] = 'PV2_V7m'; pv['iend'] = -1
lib = pd.concat([s07[cols], pv[cols]], ignore_index=True)
F0 = run(T, lib, lock=True, min_lim=3.0, y0=20060101, y1=20251231)
F0.to_csv('F0_policy.csv', index=False)
dayF0 = F0.groupby('date').net.sum()

prev = None; rows = []; hist_u = []
for D in sorted(T.days):
    if D > 20251231:
        break
    ix = T.days[D]; m = T.mod[ix]
    ny = ix[m >= 570]
    sg, i33 = s07_side(T, D, ix)
    rec = None
    if sg != 0 and prev is not None and T.at(D, 570) >= 0:
        i930 = T.at(D, 570)
        b = np.arange(i33 - 29, i33 + 1)
        M = (T.h[b].max() + T.l[b].min()) / 2
        u7 = np.median(T.h[b] - T.l[b])
        pre = ix[(m >= 480) & (m <= 569)]
        u_pre = np.median(T.h[pre] - T.l[pre]) if len(pre) >= 60 else np.nan
        lon = ix[(m >= 120) & (m <= 569)]
        Dp = int((pd.Timestamp(str(D)) - pd.Timedelta(days=1)).strftime('%Y%m%d'))
        asia = np.array([], dtype=np.int64)
        if Dp in T.date_start:
            a0 = T.date_start[Dp]; a1 = T.date_start[D]
            pa = np.arange(a0, a1); pa = pa[T.mod[pa] >= 1080]
            da = np.arange(a1, ix[0]); da = da[T.mod[da] < 120]
            asia = np.concatenate([pa, da])
        PH, PL, PC, PR, Pdir, Ploc = prev
        O = T.o[i930]; c33 = T.c[i33]
        first4 = np.arange(i930, i33 + 1)
        rec = dict(date=D, sg=sg, u7=u7, u_pre=u_pre, dM_pt=sg * (c33 - M), dM_u=sg * (c33 - M) / u7,
                   mv4=sg * (c33 - O), rng4=T.h[first4].max() - T.l[first4].min(),
                   gap=sg * (O - PC), open_loc=('above' if O > PH else 'below' if O < PL else 'inside'),
                   open_loc_sg=('with' if (O > PH and sg > 0) or (O < PL and sg < 0) else 'against' if (O > PH or O < PL) else 'inside'),
                   lon_rng=(T.h[lon].max() - T.l[lon].min()) if len(lon) >= 300 else np.nan,
                   asia_rng=(T.h[asia].max() - T.l[asia].min()) if len(asia) >= 300 else np.nan,
                   y_rng=PR, y_dir=sg * Pdir, y_loc=Ploc, dow=pd.Timestamp(str(D)).dayofweek)
        hist_u.append(u7)
        rec['u7_rel'] = u7 / np.median(hist_u[-21:-1]) if len(hist_u) > 20 else np.nan
    if len(ny) >= 300 and T.mod[ny[0]] == 570:
        H, L, C, O2 = T.h[ny].max(), T.l[ny].min(), T.c[ny[-1]], T.o[ny[0]]
        prev = (H, L, C, H - L, np.sign(C - O2), (C - L) / (H - L) if H > L else 0.5)
    else:
        prev = None
    if rec:
        rows.append(rec)
X = pd.DataFrame(rows)
s7 = s07[s07.xtype > 0].set_index('date')
X = X.join(s7[['net', 'xtype', 'entry', 'iend']].rename(columns={'net': 's07_net'}), on='date')
X['s07_stop'] = (X.xtype == 1).astype(float)
X['hold'] = [r.sg * (T.c[int(r.iend)] - r.entry) if r.iend == r.iend else np.nan for r in X.itertuples()]
X['day_F0'] = X.date.map(dayF0)
X['ep'] = X.date.map(epoch)
X = X.dropna(subset=['s07_net'])
X.to_parquet('obs1_dp1.parquet')
print('DP1 days', len(X), X.groupby('ep').size().to_dict())
print('F0 day<0 share by epoch', X.groupby('ep').day_F0.apply(lambda v: (v < 0).mean()).round(3).to_dict())
# одномерный рентген ценности действия
for f, q in [('u7', 5), ('u7_rel', 5), ('dM_pt', 5), ('dM_u', 5), ('mv4', 5), ('rng4', 5), ('gap', 5), ('lon_rng', 5), ('y_rng', 5)]:
    X['b'] = X.groupby('ep')[f].transform(lambda v: pd.qcut(v.rank(method='first'), q, labels=False))
    g = X.groupby(['b', 'ep']).agg(net=('s07_net', 'mean'), stop=('s07_stop', 'mean'), hold=('hold', 'mean'),
                                   dayneg=('day_F0', lambda v: (v < 0).mean())).round(2).unstack('ep')
    print('\n==', f, '(квинтили внутри эпохи)'); print(g[['net', 'stop', 'dayneg']].to_string())
for f in ['open_loc_sg', 'dow']:
    g = X.groupby([f, 'ep']).agg(n=('s07_net', 'size'), net=('s07_net', 'mean'), stop=('s07_stop', 'mean'),
                                 dayneg=('day_F0', lambda v: (v < 0).mean())).round(2).unstack('ep')
    print('\n==', f); print(g.to_string())
