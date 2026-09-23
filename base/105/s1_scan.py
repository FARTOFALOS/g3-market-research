"""S1 (LOG.md): поиск сильного крена в простых условиях (час × признак), три эпохи, t по дневным суммам."""
import numpy as np, pandas as pd
from trade import Tape, epoch

pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
T = Tape()
o, h, l, c, u, mod, dt = T.o, T.h, T.l, T.c, T.u, T.mod, T.date
H = 30
rows = []
for D, ix in T.days.items():
    if D > 20251231 or len(ix) < 300:
        continue
    m = mod[ix]
    # непрерывность: для исхода нужен бар t+H той же даты подряд
    hh = np.maximum.accumulate(h[ix]); ll = np.minimum.accumulate(l[ix])
    for k in range(5, len(ix) - H - 1):
        i = ix[k]
        if m[k] > 930 or ix[k + 1 + H - 1] - i != H or m[k + H] - m[k] != H:
            continue
        rows.append((D, m[k], i, hh[k], ll[k], hh[k - 1], ll[k - 1], ix[k - 5]))
A = np.array(rows, dtype=np.float64)
D = A[:, 0].astype(np.int64); M = A[:, 1].astype(int); I = A[:, 2].astype(np.int64)
HH, LL, HHp, LLp, I5 = A[:, 3], A[:, 4], A[:, 5], A[:, 6], A[:, 7].astype(np.int64)
uu = u[I]
fwd = c[I + H] - o[I + 1]
df = pd.DataFrame(dict(date=D, ep=[epoch(x) for x in D], fwd=fwd, fwd_u=fwd / uu))
edges = [120, 240, 360, 480, 570, 600, 660, 720, 780, 840, 900, 960]
df['hour'] = np.digitize(M, edges) - 1
rng1 = h[I] - l[I]; up1 = c[I] >= o[I]
df['candle'] = np.where(up1, np.where(rng1 >= 2 * uu, 'UP_big', 'up'), np.where(rng1 >= 2 * uu, 'DN_big', 'dn'))
mv5 = (c[I] - c[I5]) / uu
df['move5'] = pd.cut(mv5, [-np.inf, -4, -1.5, 1.5, 4, np.inf], labels=['<-4', '-4..-1.5', 'flat', '1.5..4', '>4']).astype(str)
pos = (c[I] - LL) / np.maximum(HH - LL, 1e-9)
df['pos'] = pd.cut(pos, [-0.01, 0.1, 0.5, 0.9, 1.01], labels=['bot10', 'lower', 'upper', 'top10']).astype(str)
df['newext'] = np.where(h[I] > HHp, 'newHigh', np.where(l[I] < LLp, 'newLow', 'none'))
df = df[np.isfinite(df.fwd_u) & (df.ep != 'other')]
out = []
for feat in ['candle', 'move5', 'pos', 'newext']:
    g = df.groupby(['hour', feat, 'ep', 'date']).fwd.agg(['sum', 'size']).reset_index()
    cell = g.groupby(['hour', feat, 'ep']).agg(n=('size', 'sum'), days=('sum', 'size'), S=('sum', 'sum'), SS=('sum', lambda x: (x ** 2).sum())).reset_index()
    cell['mean'] = cell.S / cell.n
    # t по дневным суммам: среднее дневной суммы / разброс * sqrt(дней)
    cell['dmean'] = cell.S / cell.days
    cell['dsd'] = np.sqrt(np.maximum(cell.SS / cell.days - cell.dmean ** 2, 1e-12))
    cell['t'] = cell.dmean / cell.dsd * np.sqrt(cell.days)
    w = cell.pivot_table(index=['hour', feat], columns='ep', values=['mean', 't', 'n'])
    w.columns = [f'{a}_{b}' for a, b in w.columns]
    w = w.reset_index().rename(columns={feat: 'level'}); w['feat'] = feat
    out.append(w)
R = pd.concat(out, ignore_index=True)
eps = ['2006-12', '2013-19', '2020-25']
R['sign_all'] = np.sign(R[[f'mean_{e}' for e in eps]]).abs().sum(1).eq(3) & (np.sign(R[[f'mean_{e}' for e in eps]]).sum(1).abs() == 3)
R['t_min'] = R[[f't_{e}' for e in eps]].abs().min(1)
R.to_csv('s1_scan.csv', index=False)
sel = R[R.sign_all & (R.t_min >= 2)].sort_values('t_min', ascending=False)
cols = ['feat', 'hour', 'level'] + [f'mean_{e}' for e in eps] + [f't_{e}' for e in eps] + ['n_2020-25']
print('cells', len(R), 'consistent t>=2 all epochs:', len(sel))
print(sel[cols].round(2).to_string(index=False))
print('\nlargest |mean 2020-25| with same sign all epochs:')
print(R[R.sign_all].reindex(R[R.sign_all]['mean_2020-25'].abs().sort_values(ascending=False).index)[cols].head(15).round(2).to_string(index=False))
