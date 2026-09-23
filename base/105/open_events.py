"""Rewind открытия: события в пути после хода против, ход после них против контроля той же минуты."""
import numpy as np, pandas as pd
from tape import load_minutes
def run(inst, y0, y1):
    df = load_minutes(f'{y0}-01-01', '2026-07-11' if y1 == 2026 else f'{y1}-12-31', inst)
    days = []
    for d, g in df.groupby('date'):
        g = g.set_index('mod')
        if not all(m in g.index for m in range(543, 694)): continue
        w = g.loc[543:572]; mid = (w.h.max() + w.l.min()) / 2; u7 = (w.h - w.l).median()
        if not (u7 > 0): continue
        sig = 1 if g.at[572, 'c'] > mid else -1
        mins = list(range(573, 693))
        days.append(dict(date=d, sig=sig, u7=u7, e=g.at[573, 'o'], o=g.loc[mins, 'o'].to_numpy(), h=g.loc[mins, 'h'].to_numpy(),
                         l=g.loc[mins, 'l'].to_numpy(), c=g.loc[mins, 'c'].to_numpy(), end=g.at[692, 'c']))
    # контроль: средний ход по σ от open минуты j+1 до 11:33, по всем дням, для каждой j
    ctrl = np.nanmean([[d['sig'] * (d['end'] - d['o'][j + 1]) if j + 1 < 120 else np.nan for j in range(119)] for d in days], axis=0)
    ctrl_u = np.nanmean([[d['sig'] * (d['end'] - d['o'][j + 1]) / d['u7'] if j + 1 < 120 else np.nan for j in range(119)] for d in days], axis=0)
    out = []
    for d in days:
        s, u, e = d['sig'], d['u7'], d['e']
        adv = -s * ((d['l'] if s > 0 else d['h']) - e) / u          # ход против по минутам (u7)
        close_side = s * (d['c'] - e)                                 # закрытие по σ-сторону от цены 09:34
        for k in [1, 2, 4]:
            hit = np.nonzero(adv >= k)[0]
            if not len(hit): continue
            j0 = hit[0]
            for ev, N in [('H', 5), ('H', 10), ('R', 0)]:
                j = None
                if ev == 'H':
                    run_max = np.maximum.accumulate(adv)
                    for jj in range(j0, 118):
                        # экстремум против не обновлялся N минут
                        last_new = np.nonzero(adv[:jj + 1] >= run_max[jj])[0][-1]
                        if jj - last_new >= N: j = jj; break
                else:
                    r = np.nonzero(close_side[j0:118] > 0)[0]
                    if len(r): j = j0 + r[0]
                if j is None or j + 1 >= 119: continue
                ent = d['o'][j + 1]
                inval = (d['l'][:j + 1].min() if s > 0 else d['h'][:j + 1].max())
                out.append(dict(date=d['date'], k=k, ev=f'{ev}{N}' if ev == 'H' else 'R', j=j + 1,
                                rem=s * (d['end'] - ent), rem_u=s * (d['end'] - ent) / u, ctrl=ctrl[j], ctrl_u=ctrl_u[j],
                                inval_pt=s * (ent - inval)))
    return pd.DataFrame(out), ctrl
rows = []
for inst, y0, y1 in [('NQ', 2020, 2026), ('NQ', 2013, 2019), ('ES', 2020, 2026)]:
    r, ctrl = run(inst, y0, y1)
    print(f'{inst} {y0}-{y1}: control drift to 11:33 from minute 1/10/30/60: {ctrl[0]:.2f} / {ctrl[9]:.2f} / {ctrl[29]:.2f} / {ctrl[59]:.2f} pt')
    for (k, ev), g in r.groupby(['k', 'ev']):
        dd = g.rem - g.ctrl; ddu = g.rem_u - g.ctrl_u
        rows.append(dict(cell=f'{inst} {y0}-{y1 % 100}', k=k, ev=ev, n=len(g), minute_med=int(g.j.median()),
                         rem_pt=round(g.rem.mean(), 2), ctrl_pt=round(g.ctrl.mean(), 2), add_pt=round(dd.mean(), 2),
                         z=round(dd.mean() / dd.std() * np.sqrt(len(dd)), 1), add_u7=round(ddu.mean(), 2),
                         inval_pt_med=round(g.inval_pt.median(), 1), inval_le20=round((g.inval_pt <= 19).mean(), 2)))
o = pd.DataFrame(rows); pd.set_option('display.width', 220); print(o.to_string(index=False)); o.to_csv('open_events.csv', index=False)
