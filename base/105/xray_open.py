"""Рентген открытия NY по стороне S-07: завершённые пути до 11:33 в u7, классы по итогу."""
import numpy as np, pandas as pd
from tape import load_minutes
TS = [1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 119]
def run(inst, y0, y1):
    df = load_minutes(f'{y0}-01-01', '2026-07-11' if y1 == 2026 else f'{y1}-12-31', inst)
    out = []
    for d, g in df.groupby('date'):
        g = g.set_index('mod')
        if not all(m in g.index for m in range(543, 694)): continue
        w = g.loc[543:572]; mid = (w.h.max() + w.l.min()) / 2; u7 = (w.h - w.l).median()
        if not (u7 > 0): continue
        sig = 1 if g.at[572, 'c'] > mid else -1
        e = g.at[573, 'o']; mins = range(573, 693)
        c = g.loc[mins, 'c'].to_numpy(); h = g.loc[mins, 'h'].to_numpy(); l = g.loc[mins, 'l'].to_numpy()
        fav = sig * ((h if sig > 0 else l) - e) / u7; adv = -sig * ((l if sig > 0 else h) - e) / u7
        p = sig * (c - e) / u7
        imax = int(np.argmax(fav))
        rec = dict(date=d, sig=sig, u7=u7, end=p[-1], mfe=fav.max(), t_mfe=imax + 1,
                   mae_before_mfe=adv[:imax + 1].max(), t_mae=int(np.argmax(adv[:imax + 1])) + 1, mae=adv.max())
        for t in TS: rec[f'p{t}'] = p[t - 1]
        out.append(rec)
    r = pd.DataFrame(out); r['cls'] = np.where(r.end >= 8, 'WIN', np.where(r.end <= -8, 'LOSS', 'FLAT'))
    return r
for inst, y0, y1 in [('NQ', 2020, 2026), ('NQ', 2013, 2019), ('ES', 2020, 2026)]:
    r = run(inst, y0, y1); r.to_csv(f'xopen_{inst}_{y0}.csv', index=False)
    print(f'\n=== {inst} {y0}-{y1}: days {len(r)}, median u7 {r.u7.median():.2f} pt; classes', r.cls.value_counts(normalize=True).round(3).to_dict())
    t = r.groupby('cls')[[f'p{x}' for x in TS]].median().round(2); t.columns = [f'+{x}' for x in TS]; print(t.to_string())
    w = r[r.cls == 'WIN']
    print('WIN: MAE before peak, u7 quantiles', w.mae_before_mfe.quantile([.25, .5, .75, .9]).round(1).to_dict(),
          '| minute of that MAE quantiles', w.t_mae.quantile([.25, .5, .75, .9]).to_dict(), '| peak minute median', w.t_mfe.median())
    for c_ in ['LOSS', 'FLAT']:
        x = r[r.cls == c_]; print(f'{c_}: MAE quantiles', x.mae.quantile([.25, .5, .75]).round(1).to_dict())
