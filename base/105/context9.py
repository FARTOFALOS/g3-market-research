"""Вопрос 9: контекст на 02:00 → ход 02:00–06:00 и 02:00–09:30."""
import numpy as np, pandas as pd
from tape import load_minutes
def build(inst, y0, y1):
    df = load_minutes(f'{y0-1}-11-01', '2026-07-11' if y1 == 2026 else f'{y1}-12-31', inst)
    df['u'] = (df.h - df.l).rolling(30).median()
    g = df.groupby('date')
    rows = {}
    for d, x in g:
        x = x.set_index('mod')
        r = {}
        if 570 in x.index and 959 in x.index: r['ny'] = x.at[959, 'c'] - x.at[570, 'o']; r['u_ny'] = x.loc[570:959, 'u'].median()
        if 120 in x.index: r['o02'] = x.at[120, 'o']; r['u02'] = x.at[120, 'u']
        if 359 in x.index: r['c06'] = x.at[359, 'c']
        if 569 in x.index: r['c0930'] = x.at[569, 'c']
        if 119 in x.index: r['c0159'] = x.at[119, 'c']
        if 1080 in x.index: r['o18'] = x.at[1080, 'o']
        rows[d] = r
    t = pd.DataFrame.from_dict(rows, orient='index').sort_index()
    dates = t.index.to_numpy()
    t['prev_ny'] = np.nan; t['night'] = np.nan
    ny_dates = t.index[t.ny.notna()]
    prev = None
    for d in t.index:
        if prev is not None: t.at[d, 'prev_ny'] = t.at[prev, 'ny']; t.at[d, 'prev_u'] = t.at[prev, 'u_ny']
        if not np.isnan(t.at[d, 'ny']) if 'ny' in t else False: prev = d
    # ночь: open 18:00 предыдущей календарной даты → close 01:59 этой
    cal = pd.to_datetime(t.index.astype(str))
    t['cal'] = cal
    o18 = t['o18'].copy(); o18.index = cal
    t['night'] = t['c0159'].to_numpy() - o18.reindex(cal - pd.Timedelta(days=1)).to_numpy()
    t = t[(t.index // 10000 >= y0)]
    t['r06'] = t.c06 - t.o02; t['r0930'] = t.c0930 - t.o02
    # стык месяцев по датам с сессией NY
    nyd = pd.Series(t.index[t.ny.notna()])
    ym = nyd // 100
    last = set(nyd[ym != ym.shift(-1)]); first3 = set(nyd.groupby(ym).head(3))
    t['tom'] = [d in last or d in first3 for d in t.index]
    t['mon'] = pd.to_datetime(t.index.astype(str)).dayofweek == 0
    return t
def st(x):
    x = x.dropna(); return (round(x.mean(), 2), round(x.mean() / x.std() * np.sqrt(len(x)), 1), len(x)) if len(x) > 30 else (np.nan, np.nan, len(x))
out = []
for inst, y0, y1 in [('NQ', 2020, 2026), ('NQ', 2013, 2019), ('ES', 2020, 2026)]:
    t = build(inst, y0, y1)
    cell = f'{inst} {y0}-{y1 % 100}'
    for h in ['r06', 'r0930']:
        out.append((cell, 'ALL', h) + st(t[h]))
        for fam, col, scale in [('prevNY', 'prev_ny', 'prev_u'), ('night', 'night', 'u02')]:
            z = t[col] / t[scale]; sgn = np.sign(t[col])
            q = pd.qcut(z.abs().rank(method='first'), 3, labels=['small', 'mid', 'big'])
            for lab in ['small', 'mid', 'big']:
                msk = q == lab
                out.append((cell, f'{fam} cont {lab}', h) + st((sgn * t[h])[msk]))
        out.append((cell, 'TOM', h) + st(t.loc[t.tom, h])); out.append((cell, 'nonTOM', h) + st(t.loc[~t.tom, h]))
        out.append((cell, 'Monday', h) + st(t.loc[t.mon, h]))
o = pd.DataFrame(out, columns=['cell', 'state', 'horizon', 'mean_pt', 't', 'n'])
p = o.pivot_table(index=['horizon', 'state'], columns='cell', values=['mean_pt', 't'], aggfunc='first')
pd.set_option('display.width', 250); print(p.to_string())
