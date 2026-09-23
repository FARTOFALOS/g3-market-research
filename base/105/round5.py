"""Вопрос 5: первое касание круглого уровня против сдвинутых сеток."""
import numpy as np, pandas as pd
from tape import load_minutes
S = [5, 15, 30, 60]; K = [1, 2, 4]
def events(inst, y0, y1, step, offs):
    df = load_minutes(f'{y0-1}-12-20', '2026-07-11' if y1 == 2026 else f'{y1}-12-31', inst).reset_index(drop=True)
    df['u'] = (df.h - df.l).rolling(30).median()
    out = []
    for d, g in df.groupby('date'):
        if d // 10000 < y0: continue
        g = g[(g['mod'] >= 120) & (g['mod'] < 960)]
        if len(g) < 600 or g['mod'].iloc[0] != 120: continue
        o = g.o.to_numpy(); h = g.h.to_numpy(); l = g.l.to_numpy(); c = g.c.to_numpy(); u = g.u.to_numpy(); m = g['mod'].to_numpy()
        p0 = o[0]; hmax = np.maximum.accumulate(h); lmin = np.minimum.accumulate(l)
        for off in offs:
            base = np.floor((p0 - off) / step) * step + off
            ups = base + step * np.arange(1, 40); dns = base - step * np.arange(0, 40)
            for L, sgn in [(x, 1) for x in ups if x <= h.max()] + [(x, -1) for x in dns if x >= l.min()]:
                j = np.argmax(hmax >= L) if sgn > 0 else np.argmax(lmin <= L)
                if j == 0 or j + 1 >= len(m) or m[j + 1] != m[j] + 1 or not (u[j] > 0): continue
                e = o[j + 1]
                thr = int((c[j] - L) * sgn > 0)
                rec = dict(date=d, off=off, round=int(off == 0), sgn=sgn, thr=thr, mod=m[j], u=u[j], dist0=abs(L - p0) / u[j])
                for s in S:
                    b = j + 1 + s
                    rec[f'm{s}'] = sgn * (c[b] - e) if b < len(m) and m[b] == m[j + 1] + s else np.nan
                fav = sgn * (h[j + 1:j + 121] - e) if sgn > 0 else sgn * (l[j + 1:j + 121] - e)
                adv = -sgn * (l[j + 1:j + 121] - e) if sgn > 0 else -sgn * (h[j + 1:j + 121] - e)
                for k in K:
                    a = np.nonzero(fav >= k * u[j])[0]; bb = np.nonzero(adv >= k * u[j])[0]
                    ta = a[0] if len(a) else 9999; tb = bb[0] if len(bb) else 9999
                    rec[f'y{k}'] = np.nan if ta == tb else float(ta < tb)
                out.append(rec)
    return pd.DataFrame(out)
res = []
for inst, y0, y1, step, offs in [('NQ', 2020, 2026, 100, [0, 13, 37, 63, 87]), ('NQ', 2013, 2019, 100, [0, 13, 37, 63, 87]),
                                  ('ES', 2020, 2026, 50, [0, 7, 18, 32, 43])]:
    ev = events(inst, y0, y1, step, offs); ev.to_parquet(f'round5_{inst}_{y0}.parquet')
    for thr, g in ev.groupby('thr'):
        rnd = g[g['round'] == 1]; ctl = g[g['round'] == 0]
        rec = dict(cell=f'{inst} {y0}-{y1 % 100}', kind='прошёл' if thr else 'удержал', n_round=len(rnd), n_ctl=len(ctl))
        for s in S:
            a = rnd[f'm{s}'].dropna(); b = ctl[f'm{s}'].dropna()
            # SE разности с кластером по дню для круглых
            cl = (a - a.mean()).groupby(rnd.loc[a.index, 'date']).sum(); se = np.sqrt((cl ** 2).sum()) / len(a)
            rec[f'd{s}_pt'] = round(a.mean() - b.mean(), 2); rec[f'z{s}'] = round((a.mean() - b.mean()) / se, 1)
        for k in K:
            rec[f'Y{k}_r'] = round(rnd[f'y{k}'].mean(), 3); rec[f'Y{k}_c'] = round(ctl[f'y{k}'].mean(), 3)
        res.append(rec)
o = pd.DataFrame(res); pd.set_option('display.width', 250); print(o.to_string(index=False)); o.to_csv('round5.csv', index=False)
