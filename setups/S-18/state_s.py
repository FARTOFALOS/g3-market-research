"""S-18 / отдельный вопрос — является ли «остались за нормой» возможностью участия само по себе.

ОБЪЯВЛЕНО ДО СЧЁТА (2026-09-21). v0 заморожена и здесь не меняется; ни задержанный вход, ни подтверждение,
ни новый выход версией не становятся. Вопрос человека: есть ли возможность для наблюдателя, который прежде
не участвовал и впервые посмотрел на рынок на отметке «остались за нормой» (S), — без присвоения уже
заработанного прежней позицией; что именно в этом состоянии сохраняется; какое решение исследование
позволит принять; какое наблюдение разрушит объяснение.

Моя гипотеза, которую проверяю на разрушение: S — состояние, которое платит само; свежий вход на отметке S по
текущей цене с выходом v0 имеет положительное ожидание, не зависящее от того, сколько отметок состояние уже
длится.

Самый опасный для неё контрпример выбран один: СХЛОПЫВАНИЕ ПРИ СЕССИОННОЙ ЕДИНИЦЕ И НОВОЙ ЦЕНЕ. Моё
свидетельство в пользу S (84 % валового, t 3,6) взвешено интервалами: длинная сессия вносит до одиннадцати
S-интервалов. Если у наблюдателя с одним входом на сессию, считающего путь от своей цены, экономики нет, то S —
название удерживаемой части траектории, а деньги S — многократный счёт длинных дней.

Части:
  a. Свежий наблюдатель, один вход на сессию, выход как у v0 (первая отметка, где состояние не равно стороне
     входа, иначе закрытие), расход $15:
       A1 — вход на первой отметке S сессии; A2 — вход на случайной отметке S сессии (равновероятно среди её
       S-отметок; 200 посевов, сообщаю среднее и разброс по посевам); R1 — эталон: вход на первой отметке F той же
       сессии (что сделала бы v0 своим первым входом), один вход.
     Единица — сессия. Сравнение A1 с R1 на одних сессиях отвечает, является ли первоначальный вход ценой
     доступа к редким оплачивающим дням: доля итога десяти лучших дней R1, сохранённая у A1.
  b. Состояние или траектория: ход следующего интервала и ход до закрытия по возрасту S (1, 2, 3, 4+ отметок
     подряд), t по сессионным суммам.
  c. Что сохраняется: на отметках S — трети положения за границей (в нормах этого часа) × знак продвижения за
     последний интервал -> ход следующего интервала. Растёт с положением — предмет в отношении цены к норме;
     зависит от знака продвижения — предмет в недавнем ходе; ровно — предмет в самом состоянии.
  d. Концентрация: сколько сессий несут половину денег S-интервалов.

Какое решение это позволит принять (названо заранее):
  - «входить позже допустимо» — если A2 после расходов положительна при t >= 2 в 2020-2026, валовое того же знака в
    двух ранних эпохах, и профиль по возрасту не сводится к возрасту 1;
  - «это знание о ведении, не новый момент входа» — если интервальный ход S положителен, но A1/A2 при сессионной
    единице не держатся;
  - «первоначальный вход — цена доступа» — если A1 теряет существенную часть лучших дней R1.
  В любом случае v0 остаётся как есть.
"""
import numpy as np
import pandas as pd

import anatomy
from run import HERE, POINT, COST

pd.set_option('display.width', 300, 'display.max_columns', 60)
RNG = np.random.default_rng(1802)


def fresh_trade(g, j, ins):
    """Вход на отметке j (позиция по стороне z[j]), выход по правилу v0. Возвращает net $, валовое б.п."""
    z, e = g.z.to_numpy(), g.e.to_numpy()
    d = z[j]
    out = g.cl.iloc[0]
    for m in range(j + 1, len(z)):
        if z[m] != d:
            out = e[m]
            break
    pts = d * (out - e[j])
    return pts * POINT[ins] - COST[ins], pts / g.open.iloc[0] * 1e4


def part_a(M, ins):
    rows = []
    for s, g in M.groupby('s'):
        g = g.sort_values('k').reset_index(drop=True)
        iS = np.where(g.kind == 'S')[0]
        iF = np.where(g.kind == 'F')[0]
        rec = {'s': s, 'date': g.date.iloc[0], 'epoch': g.epoch.iloc[0], 'a1': 0.0, 'r1': 0.0, 'a1_bp': 0.0, 'r1_bp': 0.0,
               'hasS': len(iS) > 0, 'hasF': len(iF) > 0, 'a2': np.nan}
        if len(iF):
            rec['r1'], rec['r1_bp'] = fresh_trade(g, iF[0], ins)
        if len(iS):
            rec['a1'], rec['a1_bp'] = fresh_trade(g, iS[0], ins)
            vals = np.array([fresh_trade(g, j, ins)[0] for j in iS])
            rec['a2_all'] = vals
        rows.append(rec)
    D = pd.DataFrame(rows)
    out = []
    for ep, g in list(D.groupby('epoch')) + [('2024-04+', D[D.date >= '2024-04-01'])]:
        n = len(g)
        for name, col in (('R1 первый F (эталон)', 'r1'), ('A1 первый S', 'a1')):
            x = g[col]
            entered = g.hasF if col == 'r1' else g.hasS
            out.append({'ins': ins, 'эпоха': ep, 'вход': name, 'сессий': n, 'с входом': round(entered.mean(), 3),
                        'средняя $/сессию': round(x.mean(), 1), 't': round(x.mean() / x.std() * np.sqrt(n), 2),
                        'валовое б.п./сессию': round(g[col + '_bp'].mean(), 2),
                        'итог $': int(x.sum()), 'худший день $': int(x.min()),
                        'без лучших 5% $': int(x[x < x.quantile(.95)].sum())})
        gs = g[g.hasS]
        means, ts = [], []
        for _ in range(200):
            pick = np.array([v[RNG.integers(len(v))] for v in gs.a2_all])
            x = np.r_[pick, np.zeros(n - len(gs))]
            means.append(x.mean())
            ts.append(x.mean() / x.std() * np.sqrt(n))
        out.append({'ins': ins, 'эпоха': ep, 'вход': 'A2 случайный S', 'сессий': n, 'с входом': round(g.hasS.mean(), 3),
                    'средняя $/сессию': round(float(np.mean(means)), 1), 't': round(float(np.mean(ts)), 2),
                    'валовое б.п./сессию': np.nan, 'итог $': int(np.mean(means) * n),
                    'худший день $': np.nan, 'без лучших 5% $': np.nan,
                    'разброс средней по посевам': round(float(np.std(means)), 1)})
    T = pd.DataFrame(out)
    # цена доступа: десять лучших дней R1 в 2020-2026
    g = D[D.epoch == '2020-2026']
    top = g.nlargest(10, 'r1')
    acc = {'десять лучших дней R1, $': int(top.r1.sum()), 'они же у A1, $': int(top.a1.sum()),
           'сохранено': round(float(top.a1.sum() / top.r1.sum()), 3),
           'дней из десяти, где A1 вообще вошла': int(top.hasS.sum()),
           'сессий с F без S (вход R1 есть, A1 нет)': int((g.hasF & ~g.hasS).sum()),
           'их итог у R1, $': int(g[g.hasF & ~g.hasS].r1.sum())}
    return T, acc, D


def part_b(M, ins):
    S = M[M.z != 0].copy()
    # возраст: номер отметки подряд в текущем состоянии (F = 0, первая S = 1, ...)
    age = []
    for s, g in S.groupby('s', sort=False):
        a, prev_k, prev_z = [], None, None
        cur = 0
        for k, z, kind in zip(g.k, g.z, g.kind):
            cur = 0 if kind == 'F' else cur + 1
            a.append(cur)
        age.extend(a)
    S['age'] = age
    S['ageb'] = np.where(S.age >= 4, '4+', S.age.astype(str))
    out = []
    for (ep, ab), g in S.groupby(['epoch', 'ageb']):
        a, ta = anatomy.sess_t(g, 'r_next')
        b, tb = anatomy.sess_t(g, 'r_close')
        out.append({'ins': ins, 'эпоха': ep, 'возраст (0 = свежий выход)': ab, 'отметок': len(g),
                    'след. интервал, б.п.': a, 't': ta, 'до закрытия, б.п.': b, 't ': tb})
    return pd.DataFrame(out), S


def part_c(S, ins):
    Q = S[S.kind == 'S'].copy()
    bound = np.where(Q.z > 0, Q.up, Q.dn)
    Q['x'] = Q.z * (Q.p - bound) / (Q.open * Q.norm)             # сколько норм этого часа за границей
    prev_p = S.groupby('s').p.shift(1).reindex(Q.index)
    Q['g'] = np.sign(Q.z * (Q.p - prev_p))
    out = []
    for ep, g in Q.groupby('epoch'):
        g = g.copy()
        g['xt'] = pd.qcut(g.x, 3, labels=['близко к границе', 'средне', 'далеко за границей'])
        for (xt, gg), h in g.groupby(['xt', 'g'], observed=True):
            if gg == 0:
                continue
            a, ta = anatomy.sess_t(h, 'r_next')
            out.append({'ins': ins, 'эпоха': ep, 'положение': xt, 'последний интервал': 'продвинулась' if gg > 0 else 'откатила',
                        'отметок': len(h), 'медиана x': round(h.x.median(), 2), 'след. интервал, б.п.': a, 't': ta})
    return pd.DataFrame(out)


def part_d(S, ins):
    g = S[(S.kind == 'S') & (S.epoch == '2020-2026')]
    per = (g.pts_next * POINT[ins]).groupby(g.s).sum().sort_values(ascending=False)
    cum = per.cumsum()
    half = int((cum < per.sum() / 2).sum()) + 1
    return {'сессий с S-интервалами': len(per), 'деньги S-интервалов $': int(per.sum()),
            'сессий, несущих половину': half, 'доля лучших 5 % сессий': round(float(per.head(max(1, len(per) // 20)).sum() / per.sum()), 3),
            'сессий в плюсе': round(float((per > 0).mean()), 3)}


def main():
    for ins in ('NQ', 'ES'):
        M = anatomy.states(ins)
        print(f'\n################ {ins}')
        T, acc, D = part_a(M, ins)
        print('--- a. свежий наблюдатель, один вход на сессию'); print(T.to_string(index=False))
        print('    цена доступа (2020-2026):', acc)
        B, S = part_b(M, ins)
        print('--- b. по возрасту состояния'); print(B.to_string(index=False))
        Cc = part_c(S, ins)
        print('--- c. что сохраняется'); print(Cc.to_string(index=False))
        print('--- d. концентрация (2020-2026):', part_d(S, ins))
        T.to_csv(HERE / f'state_s_a_{ins}.csv', index=False)
        B.to_csv(HERE / f'state_s_b_{ins}.csv', index=False)
        Cc.to_csv(HERE / f'state_s_c_{ins}.csv', index=False)


if __name__ == '__main__':
    main()
