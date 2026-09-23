"""Различение несдвинутой склейки и сдвинутой (back-adjusted) серии по самой ленте.
(1) скачок пятница→воскресенье в неделю экспирации против обычных выходных, по годам;
(2) фаза скопления дневных экстремумов 02:00–16:00: доля на целом пункте (.00) и на xx00/xx50, по кварталам."""
import numpy as np, pandas as pd
from tape import load_minutes
df = load_minutes('2006-01-01', '2026-07-11').reset_index(drop=True)
ts = df.ts.to_numpy()
gap = np.r_[False, np.diff(ts) > 30 * 3600e9]          # первый бар после выходных
idx = np.nonzero(gap)[0]
et = pd.to_datetime(ts[idx] - 60_000_000_000, utc=True).tz_convert('America/New_York').tz_localize(None).normalize()
w = pd.DataFrame({'sun': et, 'jump': df.o.to_numpy()[idx] - df.c.to_numpy()[idx - 1], 'px': df.c.to_numpy()[idx - 1]})
def tf(y, m): return pd.Timestamp(y, m, 1) + pd.offsets.WeekOfMonth(week=2, weekday=4)
rolls = {tf(y, m) - pd.Timedelta(days=5) for y in range(2006, 2027) for m in (3, 6, 9, 12)}
w['roll'] = w.sun.isin(rolls); w['yr'] = w.sun.dt.year; w['rel_bp'] = w.jump / w.px * 1e4
t = w.groupby(['yr', 'roll']).rel_bp.agg(['count', 'mean']).unstack()
print('weekend gap, bp of price: count / mean (roll=True — Sunday of expiration week)')
print(t.round(1).to_string())
# (2) фаза скопления
d = df[(df['mod'] >= 120) & (df['mod'] < 960)]
ext = d.groupby('date').agg(h=('h', 'max'), l=('l', 'min')).reset_index()
ext['q'] = pd.to_datetime(ext.date.astype(str)).dt.to_period('Q')
v = pd.concat([ext[['q', 'h']].rename(columns={'h': 'p'}), ext[['q', 'l']].rename(columns={'l': 'p'})])
v['frac00'] = (v.p % 1 == 0); v['frac50'] = (v.p % 1 == 0.5); v['frac25'] = (v.p % 1 == 0.25); v['frac75'] = (v.p % 1 == 0.75)
v['yr'] = v.q.dt.year
print('\nshare of daily 02-16 extremes by fractional part (uniform = 0.25), by year')
print(v.groupby('yr')[['frac00', 'frac25', 'frac50', 'frac75']].mean().round(3).to_string())
q = v.groupby('q')[['frac00', 'frac25', 'frac50', 'frac75']].mean()
peak = q.idxmax(axis=1)
print('\nquarters by peak phase:', peak.value_counts().to_dict())
print('quarters where peak is not .00:', [str(k) for k, x in peak.items() if x != 'frac00'])
