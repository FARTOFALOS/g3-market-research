"""Какая prefix-мерка описывает шум БУДУЩИХ 30 минут (где будет жить сделка)?
L30 = средний размах 30 последних закрытых минут (включая до 02:00);
S10 = средний размах того же получаса по часам в 10 предыдущих датах.
Цель = средний размах следующих 30 минут этой даты. Диагностика, не правило."""
import numpy as np, pandas as pd
from tape import load_minutes
df = load_minutes('2019-11-01', '2026-07-11')
df['rng'] = df.h - df.l
df['L30'] = df['rng'].rolling(30).mean().shift(0)          # включая текущую закрытую минуту
M = df[(df['mod'] >= 90) & (df['mod'] < 990)].pivot_table(index='date', columns='mod', values='rng')
M = M.reindex(columns=range(90, 990))
A = M.to_numpy()
# forward 30-min mean from minute m (bars m+1..m+30 после решения на закрытии m)
fw = pd.DataFrame(A).T.rolling(30, min_periods=20).mean().shift(-30).T.to_numpy()
fwdf = pd.DataFrame(fw, index=M.index, columns=M.columns)
S10 = fwdf.rolling(10, min_periods=7).mean().shift(1)       # те же часы, предыдущие даты
L = df.set_index(['date', 'mod'])['L30']
rows = []
for m in range(120, 930, 5):
    s = pd.DataFrame({'fut': fwdf[m], 's10': S10[m]})
    s['l30'] = L.xs(m, level='mod').reindex(s.index)
    s = s[s.index >= 20200101].dropna()
    s = s[(s > 0).all(axis=1)]; s = np.log(s)
    X1 = np.c_[np.ones(len(s)), s.l30]; X2 = np.c_[np.ones(len(s)), s.s10]; X3 = np.c_[np.ones(len(s)), s.l30, s.s10]
    def r2(X):
        b = np.linalg.lstsq(X, s.fut, rcond=None)[0]; e = s.fut - X @ b
        return 1 - e.var() / s.fut.var(), b
    a1, _ = r2(X1); a2, _ = r2(X2); a3, b3 = r2(X3)
    rows.append((m // 60, m, len(s), a1, a2, a3, b3[1], b3[2]))
r = pd.DataFrame(rows, columns=['hr', 'm', 'n', 'R2_L30', 'R2_S10', 'R2_both', 'w_L30', 'w_S10'])
print(r.groupby('hr')[['n', 'R2_L30', 'R2_S10', 'R2_both', 'w_L30', 'w_S10']].mean().round(2).to_string())
print('\nminutes around the NY open')
print(r[(r.m >= 540) & (r.m <= 590)].round(2).to_string(index=False))
