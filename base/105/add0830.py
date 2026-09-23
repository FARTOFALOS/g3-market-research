"""Шаг 4: добавка эпизода 08:30 сверх стороны S-07 (разность разностей с плацебо)."""
import numpy as np, pandas as pd
from tape import load_minutes
K = [4, 8, 16]
def run(inst, y0, y1):
    df = load_minutes(f'{y0}-01-01', '2026-07-11' if y1 == 2026 else f'{y1}-12-31', inst)
    out = []
    for d, g in df.groupby('date'):
        g = g.set_index('mod')
        if not all(m in g.index for m in range(450, 694)): continue
        pre = g.loc[480:509]; u = (pre.h - pre.l).median()
        if not (u > 0): continue
        b = g.loc[510]; r = (b.h - b.l) / u
        if r >= 4: grp = 'E'
        elif r < 2: grp = 'P'
        else: continue
        if b.c == b.o: continue
        s = 1 if b.c > b.o else -1
        w = g.loc[543:572]; mid = (w.h.max() + w.l.min()) / 2; u7 = (w.h - w.l).median()
        sig = 1 if g.at[572, 'c'] > mid else -1
        e = g.at[573, 'o']; mins = list(range(573, 693))
        h = g.loc[mins, 'h'].to_numpy(); l = g.loc[mins, 'l'].to_numpy()
        rec = dict(date=d, grp=grp + ('+' if s == sig else '-'), u7=u7)
        for k in K:
            fu = (h >= e + sig * k * u7) if sig > 0 else (l <= e + sig * k * u7)
            fd = (l <= e - sig * k * u7) if sig > 0 else (h >= e - sig * k * u7)
            iu = np.argmax(fu) if fu.any() else 10**6; idd = np.argmax(fd) if fd.any() else 10**6
            rec[f'y{k}'] = np.nan if iu == idd else float(iu < idd)
        rec['m60'] = sig * (g.at[632, 'c'] - e) / u7; rec['m120'] = sig * (g.at[692, 'c'] - e) / u7
        out.append(rec)
    return pd.DataFrame(out)
rows = []
for inst, y0, y1 in [('NQ', 2020, 2026), ('NQ', 2013, 2019), ('ES', 2020, 2026)]:
    r = run(inst, y0, y1)
    stat = {}
    for gname, g in r.groupby('grp'):
        rec = dict(cell=f'{inst} {y0}-{y1 % 100}', grp=gname, n=len(g))
        for k in K:
            y = g[f'y{k}'].dropna(); rec[f'Y{k}'] = round(y.mean(), 3); stat[(gname, k)] = (y.mean(), y.var() / len(y))
        for m in ['m60', 'm120']:
            rec[m + '_u7'] = round(g[m].mean(), 2); stat[(gname, m)] = (g[m].mean(), g[m].var() / len(g))
        rows.append(rec)
    rec = dict(cell=f'{inst} {y0}-{y1 % 100}', grp='ADD=(E+-E-)-(P+-P-)', n=None)
    for key in [4, 8, 16, 'm60', 'm120']:
        dd = stat[('E+', key)][0] - stat[('E-', key)][0] - stat[('P+', key)][0] + stat[('P-', key)][0]
        se = np.sqrt(sum(stat[(gg, key)][1] for gg in ['E+', 'E-', 'P+', 'P-']))
        name = f'Y{key}' if isinstance(key, int) else key + '_u7'
        rec[name] = f'{dd:+.3f} (z {dd / se:+.1f})'
    rows.append(rec)
o = pd.DataFrame(rows); pd.set_option('display.width', 220); print(o.to_string(index=False)); o.to_csv('add0830.csv', index=False)
