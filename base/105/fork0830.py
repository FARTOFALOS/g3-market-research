"""Шаг 2: вилка X против O по ходу развития эпизода 08:30."""
import numpy as np, pandas as pd
from tape import load_minutes
TT = [1, 2, 3, 5, 10, 15, 20, 30, 45]
def run(inst, y0, y1):
    ep = pd.read_csv(f'x0830_{inst}_{y0}.csv'); ep = ep[~ep.whip]
    df = load_minutes(f'{y0}-01-01', '2026-07-11' if y1 == 2026 else f'{y1}-12-31', inst)
    out = []
    for d, g in df.groupby('date'):
        if d not in set(ep.date): continue
        e = ep[ep.date == d].iloc[0]; g = g.set_index('mod')
        b = g.loc[510]; O = b.o; s = int(e.s); X = b.h if s > 0 else b.l; A = e.A
        mins = list(range(511, 691))
        h = g.loc[mins, 'h'].to_numpy(); l = g.loc[mins, 'l'].to_numpy(); o = g.loc[mins, 'o'].to_numpy(); c = g.loc[mins, 'c'].to_numpy()
        up = (h >= X) if s > 0 else (l <= X)       # коснулся X (по ходу реакции)
        dn = (l <= O) if s > 0 else (h >= O)       # коснулся O
        for t in TT:
            k = t - 1                               # индекс минуты t после реакции; вход open k+1
            if k + 1 >= len(mins): continue
            pc = s * (c[k] - O) / A
            if not (0 < pc < 1): continue
            ext_before = bool(up[:k + 1].any()); back_before = bool(dn[:k + 1].any())
            e_ = o[k + 1]; p0 = s * (e_ - O) / A
            if not (0 < p0 < 1): continue
            fu = up[k + 1:]; fd = dn[k + 1:]
            iu = np.argmax(fu) if fu.any() else 10**6; idd = np.argmax(fd) if fd.any() else 10**6
            if iu == idd == 10**6: y = np.nan; res = 'cens'
            elif iu == idd: y = np.nan; res = 'both'
            else: y = float(iu < idd); res = 'ok'
            out.append(dict(date=d, t=t, p0=p0, y=y, res=res, ext_before=ext_before, back_before=back_before, A=A))
    return pd.DataFrame(out)
rows = []
for inst, y0, y1 in [('NQ', 2020, 2026), ('NQ', 2013, 2019), ('ES', 2020, 2026)]:
    f = run(inst, y0, y1); f.to_csv(f'fork0830_{inst}_{y0}.csv', index=False)
    for t, g in f.groupby('t'):
        ok = g[g.res == 'ok']
        for lab, gg in [('all', ok), ('ext_before', ok[ok.ext_before]), ('no_ext_before', ok[~ok.ext_before])]:
            if len(gg) < 15: continue
            dY = gg.y.mean() - gg.p0.mean(); se = np.sqrt((gg.p0 * (1 - gg.p0)).sum()) / len(gg)
            rows.append(dict(cell=f'{inst} {y0}-{y1 % 100}', t=t, layer=lab, n=len(gg), cens=int((g.res != 'ok').sum()),
                             p0=round(gg.p0.mean(), 3), Y=round(gg.y.mean(), 3), dY=round(dY, 3), z=round(dY / se, 1),
                             pts=round((gg.A * (gg.y - gg.p0)).mean(), 2)))
o = pd.DataFrame(rows); pd.set_option('display.width', 220)
print(o[o.layer == 'all'].pivot_table(index='t', columns='cell', values=['dY', 'z', 'n', 'pts'], aggfunc='first').to_string())
print(o[o.layer != 'all'].pivot_table(index=['layer', 't'], columns='cell', values=['dY', 'z', 'n'], aggfunc='first').to_string())
o.to_csv('fork0830.csv', index=False)
