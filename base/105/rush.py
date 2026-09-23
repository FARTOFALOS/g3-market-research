"""Вопрос 6: трафарет пути после вертикального 5-минутного рывка."""
import numpy as np, pandas as pd
from tape import load_minutes
TH = [3, 4, 6]
def scenes(inst, y0, y1):
    df = load_minutes(f'{y0-1}-12-20', '2026-07-11' if y1 == 2026 else f'{y1}-12-31', inst).reset_index(drop=True)
    df['u'] = (df.h - df.l).rolling(30).median()
    out = []
    for d, g in df.groupby('date'):
        if d // 10000 < y0: continue
        m = g['mod'].to_numpy(); o = g.o.to_numpy(); h = g.h.to_numpy(); l = g.l.to_numpy(); c = g.c.to_numpy(); u = g.u.to_numpy()
        pos = {mm: k for k, mm in enumerate(m)}
        for th in TH:
            last = {1: -999, -1: -999}
            for k in range(len(m)):
                t = m[k]
                if t < 120 or t > 950 or not (u[k] > 0): continue
                a = pos.get(t - 5)
                if a is None: continue
                mv = (c[k] - c[a]) / u[k]
                for sgn in (1, -1):
                    if sgn * mv >= th:
                        if t - last[sgn] > 30:
                            if k + 1 < len(m) and m[k + 1] == t + 1:
                                e = o[k + 1]; rec = dict(date=d, th=th, sgn=sgn, mod=t, u=u[k], rush_u=mv * sgn)
                                ext = h[a:k + 1].max() if sgn > 0 else l[a:k + 1].min()   # экстремум рывка к узнаванию
                                rec['ext_gap_u'] = (ext - e) * sgn / u[k]                  # насколько вход ниже экстремума
                                seg = np.arange(k + 1, min(k + 61, len(m)))
                                seg = seg[m[seg] - (t + 1) < 60]
                                fad = -sgn * (c[seg] - e) / u[k]                            # результат ставки ПРОТИВ рывка
                                for s in [5, 10, 15, 30, 60]:
                                    j = pos.get(t + 1 + s)
                                    rec[f'f{s}'] = -sgn * (c[j] - e) / u[k] if j is not None and t + 1 + s <= 959 else np.nan
                                ext_after = (h[seg].max() - e) if sgn > 0 else (e - l[seg].min())
                                back = (e - l[seg].min()) if sgn > 0 else (h[seg].max() - e)
                                rec['extend_u'] = ext_after / u[k]; rec['retrace_u'] = back / u[k]
                                ex_idx = np.argmax(h[seg]) if sgn > 0 else np.argmin(l[seg])
                                rec['t_ext'] = int(m[seg][ex_idx] - t)
                                # продление экстремума до 30-й минуты
                                s30 = seg[m[seg] - t <= 30]
                                rec['extend30_u'] = ((h[s30].max() - e) if sgn > 0 else (e - l[s30].min())) / u[k]
                                out.append(rec)
                        last[sgn] = t
    return pd.DataFrame(out)
def blk(t): return 'LDN' if t < 570 else ('OPEN' if t < 600 else 'NY')
rows = []
for inst, y0, y1 in [('NQ', 2020, 2026), ('NQ', 2013, 2019), ('ES', 2020, 2026)]:
    s = scenes(inst, y0, y1); s.to_parquet(f'rush_{inst}_{y0}.parquet'); s['blk'] = s['mod'].map(blk)
    for (th, sgn, b), g in s.groupby(['th', 'sgn', 'blk']):
        rec = dict(cell=f'{inst} {y0}-{y1 % 100}', th=th, dir='UP' if sgn > 0 else 'DN', blk=b, n=len(g), per_yr=round(len(g) / (y1 - y0 + (0.53 if y1 == 2026 else 1)), 1),
                   u_pt=round(g.u.median(), 1))
        for sx in [5, 15, 30, 60]:
            x = g[f'f{sx}'].dropna(); cl = (x - x.mean()).groupby(g.loc[x.index, 'date']).sum(); se = np.sqrt((cl ** 2).sum()) / len(x)
            rec[f'fade{sx}_u'] = round(x.mean(), 2); rec[f'z{sx}'] = round(x.mean() / se, 1)
        rec['fade30_pt'] = round((g.f30 * g.u).mean(), 1)
        rec['ext30_med_u'] = round(g.extend30_u.median(), 2); rec['ret60_med_u'] = round(g.retrace_u.median(), 2)
        rows.append(rec)
o = pd.DataFrame(rows); pd.set_option('display.width', 260); print(o.to_string(index=False)); o.to_csv('rush.csv', index=False)
