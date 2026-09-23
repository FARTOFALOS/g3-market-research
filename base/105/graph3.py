"""GR1 (LOG.md): трафарет графа отношений трёх последних минутных свечей → ход следующих 3 и 10 минут.
Объявлено до счёта. Узор минуты k: для пар (k−2→k−1) и (k−1→k) код закрытия относительно предыдущей свечи
(выше её high = 2, выше её close = 1, ниже её close = −1, ниже её low = −2; равенство close → −1/1 по телу) ×
обновлён ли high × обновлён ли low; плюс знак тела свечи k. Зеркальный узор (цена со знаком минус) склеивается:
крен = (ход после узора − ход после зеркала)/2, так общий рост индекса вычитается. Окна: pre 02:00–09:29,
open 09:30–10:59, day 11:00–15:45. Исход — close[k+m] − open[k+1], m = 3 и 10, в пт; t по дневным суммам.
Отбор: один знак во всех трёх эпохах и |t| ≥ 2 в каждой, |крен| в 2020–25 > 0,75 пт (m = 3) или > 1 пт (m = 10).
"""
import numpy as np, pandas as pd
from trade import Tape, epoch

pd.set_option('display.width', 260); pd.set_option('display.max_rows', 200)
T = Tape()
o, h, l, c, mod, dt = T.o, T.h, T.l, T.c, T.mod, T.date
n = len(o)
k = np.arange(2, n - 11)
ok = (dt[k - 2] == dt[k]) & (mod[k] - mod[k - 2] == 2) & (dt[k + 10] == dt[k]) & (mod[k + 10] - mod[k] == 10)
ok &= (mod[k] >= 120) & (mod[k] <= 945) & (dt[k] <= 20251231)
k = k[ok]


def pair(a, b, sgn):
    """код пары a→b в системе знака sgn (sgn=−1 — зеркало)."""
    ca, cb, ha, hb, la, lb, ob = sgn * c[a], sgn * c[b], sgn * np.where(sgn > 0, h[a], l[a]), sgn * np.where(sgn > 0, h[b], l[b]), \
        sgn * np.where(sgn > 0, l[a], h[a]), sgn * np.where(sgn > 0, l[b], h[b]), sgn * o[b]
    code = np.where(cb > ha, 2, np.where(cb > ca, 1, np.where(cb < la, -2, np.where(cb < ca, -1, np.where(cb >= ob, 1, -1)))))
    nh = hb > ha; nl = lb < la
    return (code + 2) * 4 + nh * 2 + nl


def pattern(sgn):
    p1 = pair(k - 2, k - 1, sgn); p2 = pair(k - 1, k, sgn)
    body = (sgn * (c[k] - o[k]) > 0).astype(int)
    return p1 * 40 + p2 * 2 + body


P = pattern(1); M = pattern(-1)
win = np.select([mod[k] < 570, mod[k] < 660], [0, 1], 2)
ep = np.array([epoch(x) for x in dt[k]])
f3 = c[k + 3] - o[k + 1]; f10 = c[k + 10] - o[k + 1]
base = pd.DataFrame(dict(date=dt[k], ep=ep, win=win, f3=f3, f10=f10))
# склейка: узор со знаком +, зеркальный узор со знаком − (в системе узора ход берётся с минусом)
A = base.assign(pat=P, s=1)
B = base.assign(pat=M, s=-1)
AB = pd.concat([A, B], ignore_index=True)
AB['y3'] = AB.s * AB.f3; AB['y10'] = AB.s * AB.f10
d = AB.groupby(['win', 'pat', 'ep', 'date']).agg(y3=('y3', 'sum'), y10=('y10', 'sum'), n=('y3', 'size')).reset_index()
g = d.groupby(['win', 'pat', 'ep']).agg(n=('n', 'sum'), days=('n', 'size'), s3=('y3', 'sum'), s10=('y10', 'sum'),
                                          sd3=('y3', 'std'), sd10=('y10', 'std')).reset_index()
g['m3'] = g.s3 / g.n; g['m10'] = g.s10 / g.n
g['t3'] = (g.s3 / g.days) / g.sd3 * np.sqrt(g.days); g['t10'] = (g.s10 / g.days) / g.sd10 * np.sqrt(g.days)
W = g.pivot_table(index=['win', 'pat'], columns='ep', values=['n', 'm3', 'm10', 't3', 't10'])
W.columns = [f'{a}_{b}' for a, b in W.columns]
W = W.reset_index()
eps = ['2006-12', '2013-19', '2020-25']
for m in ('3', '10'):
    sg = np.sign(W[[f'm{m}_{e}' for e in eps]])
    W[f'cons{m}'] = (sg.abs().sum(axis=1) == 3) & (sg.sum(axis=1).abs() == 3) & (W[[f't{m}_{e}' for e in eps]].abs().min(axis=1) >= 2)
W.to_csv('graph3.csv', index=False)
print('patterns x windows', len(W), ' min n 2020-25 filter 300')
Wf = W[W['n_2020-25'] >= 300]
for m, thr in (('3', 0.75), ('10', 1.0)):
    sel = Wf[Wf[f'cons{m}'] & (Wf[f'm{m}_2020-25'].abs() > thr)]
    print(f'm={m}: consistent |t|>=2 all epochs:', int(Wf[f'cons{m}'].sum()), ' of which |lean 2020-25| >', thr, ':', len(sel))
    cols = ['win', 'pat', 'n_2020-25'] + [f'm{m}_{e}' for e in eps] + [f't{m}_{e}' for e in eps]
    print(Wf[Wf[f'cons{m}']].sort_values(f'm{m}_2020-25', key=abs, ascending=False)[cols].head(15).round(2).to_string(index=False))
    print('max |lean 2020-25| among all with n>=300:', round(Wf[f'm{m}_2020-25'].abs().max(), 2))
