"""S-18 / состояние S, дополнение к state_s.py — ДОБАВЛЕНО ПОСЛЕ результата A2, так и читать.

A2 (случайная S-отметка внутри сессии, сессии равновесны) оказалась отрицательной во всех эпохах на NQ и ES при
положительной A1. Прежде чем толковать, нужна вторая проекция того же вопроса, физически более естественная:
  A3 — наблюдатель смотрит на рынок в ОДНУ случайную из двенадцати отметок сессии и входит, только если видит S.
       Его ожидание на сессию = (сумма свежих входов по всем S-отметкам сессии) / 12. Здесь длинная сессия весит
       больше, потому что застать в ней S вероятнее. A2 и A3 не усредняются: это разные наблюдатели.
  E  — свежий вход по возрасту состояния на момент входа (1, 2, 3, 4+), выход v0, после расходов, на вход;
       t по сессионным суммам. Отвечает, оправдан ли вход «сколько бы состояние уже ни длилось».
  Z  — чем кончается S-отметка: доля следующих отметок с возвратом внутрь по возрасту, и ход интервала отдельно
       для «остались» и «вернулись» — чтобы видеть, из чего складывается отрицательная A2.
"""
import numpy as np
import pandas as pd

import anatomy
from run import HERE, POINT, COST
from state_s import fresh_trade

pd.set_option('display.width', 300, 'display.max_columns', 60)


def main():
    for ins in ('NQ', 'ES'):
        M = anatomy.states(ins)
        rows, ent = [], []
        for s, g in M.groupby('s'):
            g = g.sort_values('k').reset_index(drop=True)
            age, cur = [], 0
            for kind, z in zip(g.kind, g.z):
                cur = 0 if kind == 'F' else (cur + 1 if z != 0 else 0)
                age.append(cur)
            tot = 0.0
            for j in np.where(g.kind == 'S')[0]:
                net, bp = fresh_trade(g, j, ins)
                tot += net
                nz = g.z.iloc[j + 1] if j + 1 < len(g) else np.nan
                ent.append({'s': s, 'epoch': g.epoch.iloc[0], 'date': g.date.iloc[0], 'age': age[j], 'net': net,
                            'r_next': g.r_next.iloc[j], 'back': (nz == 0) if nz == nz else np.nan,
                            'stay': (nz == g.z.iloc[j]) if nz == nz else np.nan})
            rows.append({'s': s, 'epoch': g.epoch.iloc[0], 'date': g.date.iloc[0], 'a3': tot / 12.0, 'nS': int((g.kind == 'S').sum())})
        D, E = pd.DataFrame(rows), pd.DataFrame(ent)
        E['ageb'] = np.where(E.age >= 4, '4+', E.age.astype(str))
        print(f'\n################ {ins}')
        out = []
        for ep, g in list(D.groupby('epoch')) + [('2024-04+', D[D.date >= '2024-04-01'])]:
            x = g.a3
            out.append({'эпоха': ep, 'сессий': len(g), 'A3 средняя $/сессию': round(x.mean(), 1),
                        't': round(x.mean() / x.std() * np.sqrt(len(g)), 2), 'итог $': int(x.sum()),
                        'без лучших 5% $': int(x[x < x.quantile(.95)].sum())})
        print('--- A3: наблюдатель в случайное время'); print(pd.DataFrame(out).to_string(index=False))
        out = []
        for (ep, ab), g in E.groupby(['epoch', 'ageb']):
            per = g.groupby('s').net.sum()
            out.append({'эпоха': ep, 'возраст при входе': ab, 'входов': len(g), 'средняя $ на вход': round(g.net.mean(), 1),
                        't (сессии)': round(per.mean() / per.std() * np.sqrt(len(per)), 2),
                        'медиана $': round(g.net.median(), 0), 'доля прибыльных': round((g.net > 0).mean(), 3),
                        'без лучших 5% входов $': int(g.net[g.net < g.net.quantile(.95)].sum())})
        T = pd.DataFrame(out)
        print('--- E: свежий вход по возрасту состояния'); print(T.to_string(index=False))
        T.to_csv(HERE / f'state_s_e_{ins}.csv', index=False)
        out = []
        for (ep, ab), g in E[E.epoch == '2020-2026'].groupby(['epoch', 'ageb']):
            h = g.dropna(subset=['back'])
            out.append({'возраст': ab, 'вернулись на след. отметке': round(h.back.astype(float).mean(), 3),
                        'ход, если остались, б.п.': round(h[h.stay == True].r_next.mean(), 1),
                        'ход, если вернулись, б.п.': round(h[h.back == True].r_next.mean(), 1)})
        print('--- Z (2020-2026): чем кончается S-отметка'); print(pd.DataFrame(out).to_string(index=False))


if __name__ == '__main__':
    main()
