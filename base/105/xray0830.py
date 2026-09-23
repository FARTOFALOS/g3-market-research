"""Рентген эпизодов 08:30: завершённые пути до 10:30 в единицах реакции A, классы по будущему."""
import sys, numpy as np, pandas as pd
from tape import load_minutes
TS = [1, 2, 3, 5, 10, 15, 20, 30, 45, 59, 60, 75, 90, 120]
def episodes(inst, y0, y1, thr=4.0):
    df = load_minutes(f'{y0}-01-01', '2026-07-11' if y1 == 2026 else f'{y1}-12-31', inst)
    out = []
    for d, g in df.groupby('date'):
        g = g.set_index('mod')
        if not all(m in g.index for m in range(450, 691)): continue
        pre = g.loc[480:509]; u = (pre.h - pre.l).median()
        b = g.loc[510]
        if not (u > 0) or (b.h - b.l) / u < thr: continue
        O = b.o; body = b.c - b.o
        whip = abs(body) < 0.25 * (b.h - b.l)
        s = 1 if body > 0 else -1
        X = b.h if s > 0 else b.l; A = s * (X - O)
        if A <= 0: continue
        c = g.loc[511:630, 'c'].to_numpy(); h = g.loc[511:630, 'h'].to_numpy(); l = g.loc[511:630, 'l'].to_numpy()
        p = s * (c - O) / A                                   # путь закрытий в A, минуты 1..120 после 08:30-бара
        ext = (h - X) * s > 0 if s > 0 else (X - l) > 0      # новый экстремум за X
        back = (l <= O) if s > 0 else (h >= O)                # возврат к началу реакции
        rec = dict(date=d, s=s, whip=whip, A=A, A_u=A / u, u=u, r_u=(b.h - b.l) / u, close_loc=s * (b.c - O) / A,
                   pre30=s * (g.at[509, 'c'] - g.at[480, 'o']) / A, P1029=p[119] if len(p) >= 120 else np.nan, P0929=p[58])
        for t in TS: rec[f'p{t}'] = p[t - 1]
        rec['t_ext'] = int(np.argmax(ext)) + 1 if ext.any() else 999
        rec['t_back'] = int(np.argmax(back)) + 1 if back.any() else 999
        out.append(rec)
    return pd.DataFrame(out)
def cls(P):
    return np.where(P >= 2, 'CONT', np.where(P <= -1, 'REV', 'MID'))
for inst, y0, y1 in [('NQ', 2020, 2026), ('NQ', 2013, 2019), ('ES', 2020, 2026)]:
    e = episodes(inst, y0, y1); e.to_csv(f'x0830_{inst}_{y0}.csv', index=False)
    e['cls'] = cls(e.P1029)
    ne = e[~e.whip]
    print(f'\n=== {inst} {y0}-{y1}: episodes {len(e)} (whip {e.whip.sum()}), median A {(ne.A).median():.1f} pt = {ne.A_u.median():.1f}u; close_loc median {ne.close_loc.median():.2f}')
    print('class shares at 10:29 (non-whip):', ne.cls.value_counts(normalize=True).round(3).to_dict(), '| at 09:29:', pd.Series(cls(ne.P0929)).value_counts(normalize=True).round(3).to_dict())
    tab = ne.groupby('cls')[[f'p{t}' for t in TS]].median().round(2)
    tab.columns = [f'+{t}m' for t in TS]; print('median path in A by class:\n' + tab.to_string())
    for t in [1, 3, 5, 10, 30]:
        ne_ = ne.assign(ext_by=ne.t_ext <= t, back_by=ne.t_back <= t)
        print(f' by +{t}m: new extreme beyond X / touched O —', {k: (round(g.ext_by.mean(), 2), round(g.back_by.mean(), 2)) for k, g in ne_.groupby('cls')})
