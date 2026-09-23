"""D1: рентген завершённых дней NQ 2006–2025 — формы дня и когда они различимы."""
import sys, numpy as np, pandas as pd
from tape import load_minutes
INST = sys.argv[1] if len(sys.argv) > 1 else 'NQ'
M0 = int(sys.argv[2]) if len(sys.argv) > 2 else 120
TT = [600, 660, 720, 780, 840]          # 10:00 11:00 12:00 13:00 14:00 (минута открытия бара)
df = load_minutes('2005-12-01', '2026-01-01', INST)
rows = []
for d, g in df.groupby('date'):
    w = g[(g['mod'] >= M0) & (g['mod'] < 960)]
    if len(w) != 960 - M0: continue          # объект известен только при всех минутах
    m = w['mod'].to_numpy(); h = w.h.to_numpy(); l = w.l.to_numpy(); c = w.c.to_numpy()
    H, L = h.max(), l.min(); R = H - L
    if R <= 0: continue
    iH, iL = int(np.argmax(h)), int(np.argmin(l))   # первое достижение окончательного экстремума
    last_is_H = iH > iL
    Elast = H if last_is_H else L
    rec = dict(date=d, R=R, C=c[-1], tH=m[iH], tL=m[iL], last_H=last_is_H, dlast=abs(c[-1] - Elast) / R)
    rh = np.maximum.accumulate(h); rl = np.minimum.accumulate(l)
    for t in TT:
        k = np.searchsorted(m, t, side='right') - 1
        if k < 0: continue
        Rt = rh[k] - rl[k]
        newext = np.nonzero((h[:k + 1] >= rh[:k + 1]) | (l[:k + 1] <= rl[:k + 1]))[0]
        age = m[k] - m[newext[-1]]
        pos = (c[k] - rl[k]) / Rt if Rt > 0 else 0.5
        rec[f'R_{t}'] = Rt; rec[f'age_{t}'] = age; rec[f'edge_{t}'] = min(pos, 1 - pos)
        rec[f'done_{t}'] = int(max(iH, iL) <= k)      # оба окончательных экстремума уже стоят к t
    rows.append(rec)
r = pd.DataFrame(rows).sort_values('date')
r['N'] = r.R.rolling(20).median().shift(1)
r = r.dropna(subset=['N'])
r['rho'] = r.R / r.N
r['cls'] = np.where(r.dlast <= 0.2, 'TREND', np.where(r.dlast >= 0.6, 'FADE', 'RANGE'))
r['ep'] = pd.cut(r.date // 10000, [2005, 2012, 2019, 2025], labels=['2006-12', '2013-19', '2020-25'])
for t in TT: r[f'rho_{t}'] = r[f'R_{t}'] / r.N
r.to_csv(f'dayx_{INST}_{M0}.csv', index=False)
pd.set_option('display.width', 220)
print(f'{INST}, object from {M0//60:02d}:{M0%60:02d} (strict: all minutes); known objects', len(r)); print('known per epoch:', r.ep.value_counts().sort_index().to_dict())
print('class shares by epoch:\n', pd.crosstab(r.ep, r.cls, normalize='index').round(3).to_string())
print('final range / norm (median) by class & epoch:\n', r.pivot_table(index='ep', columns='cls', values='rho', aggfunc='median', observed=True).round(2).to_string())
print('share of days with BOTH final extremes set by t:\n', r.groupby('ep', observed=True)[[f'done_{t}' for t in TT]].mean().round(3).to_string())
for t in TT:
    s = r.groupby(['ep', 'cls'], observed=True).agg(rho=(f'rho_{t}', 'median'), age=(f'age_{t}', 'median'), edge=(f'edge_{t}', 'median'), done=(f'done_{t}', 'mean')).round(2)
    print(f'\n-- at {t//60:02d}:00 -- medians by class: range-so-far/N, minutes since last new extreme, distance to nearer edge (share of range), share with final extremes already set')
    print(s.unstack('cls').to_string())
