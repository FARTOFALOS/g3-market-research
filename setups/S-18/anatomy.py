"""S-18 — анатомия замороженной v0: из чего состоит её результат и что значит слабость последних лет.

ОБЪЯВЛЕНО ДО СЧЁТА (2026-09-21), после слова человека: S-18 не снимать; два вопроса — (1) что именно
конструкция позволяет делать иначе: распознавать необычное развитие дня, подключаться позднее или
сохранять участие, пока цена за нормой; (2) что означает слабость последних лет и какой вопрос о ней
действительно меняет решение. Правило v0 не меняется. Новых фильтров и перебора нет.

Поправка описания, найденная при сверке с первоисточником (раздел 3 и рис. 5): базовая модель авторов
выходит на ПРОТИВОПОЛОЖНОЙ границе области либо в закрытие (SPY: Sharpe 0,61); выход по возврату за
БЛИЖНЮЮ текущую границу — их первое улучшение (рис. 5a); итоговая модель добавляет VWAP и размер по
волатильности (Sharpe 1,33). v0 = вариант рис. 5a без VWAP и без размера по волатильности. Прежняя
формулировка «как у источника в базовом виде» неточна; правило и числа от этого не меняются.

Состояние на отметке j (12 отметок 10:00…15:30): z = +1 выше верха, −1 ниже низа, 0 внутри.
Виды отметок (до первой отметки z считается 0):
  F — свежий выход: z != 0 и z отличается от предыдущего (включая переворот); сторона d = z;
  S — остаёмся за нормой: z != 0 и равен предыдущему; d = z;
  R — возврат: z = 0, предыдущее != 0; d = прежняя сторона (что отдаёт или спасает выход v0);
  I — внутри и были внутри: d = знак (close отметки − open сессии) — контроль: несёт ли сторону дня
      простой знак хода без требования необычности.
Исходы в б.п. от open сессии, по цене исполнения (open следующей минуты): ход до следующей отметки и
ход до закрытия. Единица — отметка внутри сессии; t считается по сессионным суммам, чтобы отметки одного
дня не выдавались за независимые.

Части ответа:
  A. Таблица F / S / R / I по эпохам, NQ и ES.
  B. Точное разложение валового v0: каждый удержанный интервал начинается отметкой F или S, поэтому
     валовое = после F + после S без остатка; то же по времени начала интервала (до 11:30 / 12:00–13:30 /
     14:00 и позже). Сверка с band_daily обязательна.
  C. Выход, заданный самим источником иначе (противоположная граница): та же лента, те же входы —
     цена и польза выхода v0 в деньгах, просадке и худшем дне. Это не кандидат и не замена v0.
  D. Слабость последних лет: блоки 2020-01…2024-03 против 2024-04…2026-05 — частота свежих выходов,
     что с ними происходит на следующей отметке (остались / вернулись / перевернулись), размер хода в
     каждом случае, цена ложного выхода; разности с интервалами сессионного бутстрепа. Плюс контекст:
     место среднего позднего блока среди всех окон той же длины в прежней истории (контекст, не
     оправдание) и разрешение: какую разность средних блок такой длины вообще отличает.
Что меняет решение — названо заранее: если интервалы разностей частей включают ноль, имеющееся
свидетельство не различает объяснения, и решение о кандидате меняет только накопительная последующая
проверка замороженной v0, а не разбор причин.
"""
import numpy as np
import pandas as pd

import run as band
from run import HERE, KMAX, POINT, COST

pd.set_option('display.width', 300, 'display.max_columns', 60)
MARKS = np.arange(30, 361, 30)
RNG = np.random.default_rng(18)


def states(ins, look=14):
    cal, K, O, C = band.sessions(ins)
    S = len(cal)
    op = O[:, 1]
    cl = np.array([C[s, min(K[s], KMAX)] for s in range(S)])
    move = np.abs(C / op[:, None] - 1.0)
    rows = []
    for s in range(look, S):
        prev = cl[s - 1]
        if not (np.isfinite(op[s]) and np.isfinite(prev) and np.isfinite(cl[s])):
            continue
        hist = move[s - look:s]
        cnt = np.isfinite(hist).sum(axis=0)
        with np.errstate(invalid='ignore', all='ignore'):
            norm = np.where(cnt >= 10, np.nanmean(hist, axis=0), np.nan)
        up = max(op[s], prev) * (1 + norm)
        dn = min(op[s], prev) * (1 - norm)
        marks = [k for k in MARKS if k + 1 <= K[s] - 1]
        p = C[s, marks]
        e = O[s, [k + 1 for k in marks]]
        if not (np.isfinite(p).all() and np.isfinite(e).all() and np.isfinite(up[marks]).all()):
            continue
        z = np.where(p > up[marks], 1, np.where(p < dn[marks], -1, 0))
        nxt = np.r_[e[1:], cl[s]]
        zp = np.r_[0, z[:-1]]
        for j, k in enumerate(marks):
            kind = ('F' if z[j] != 0 and z[j] != zp[j] else 'S' if z[j] != 0 else 'R' if zp[j] != 0 else 'I')
            d = z[j] if z[j] != 0 else (zp[j] if zp[j] != 0 else int(np.sign(p[j] - op[s])))
            rows.append({'date': pd.Timestamp(cal.date.iloc[s]), 's': s, 'k': k, 'kind': kind, 'z': int(z[j]),
                         'zp': int(zp[j]), 'd': int(d), 'flip': bool(z[j] != 0 and zp[j] == -z[j]),
                         'r_next': d * (nxt[j] - e[j]) / op[s] * 1e4,
                         'r_close': d * (cl[s] - e[j]) / op[s] * 1e4,
                         'pts_next': d * (nxt[j] - e[j]), 'raw_next': nxt[j] - e[j], 'open': op[s],
                         'p': p[j], 'e': e[j], 'cl': cl[s], 'up': up[k], 'dn': dn[k], 'norm': norm[k]})
    M = pd.DataFrame(rows)
    M['year'] = M.date.dt.year
    M['epoch'] = band.epoch(M.year.values)
    # состояние на следующей отметке — для части D
    M['z_next'] = M.groupby('s').z.shift(-1)
    return M


def sess_t(g, col):
    """Среднее по отметкам и t по сессионным суммам."""
    per = g.groupby('s')[col].sum()
    n = len(per)
    t = per.mean() / per.std() * np.sqrt(n) if n > 1 and per.std() > 0 else np.nan
    return round(g[col].mean(), 2), round(t, 2)


def part_a(M, ins):
    out = []
    for ep, g in M.groupby('epoch'):
        ns = g.s.nunique()
        for kind in ('F', 'S', 'R', 'I'):
            q = g[g.kind == kind]
            a, ta = sess_t(q, 'r_next')
            b, tb = sess_t(q, 'r_close')
            out.append({'ins': ins, 'эпоха': ep, 'вид': kind, 'отметок': len(q), 'на сессию': round(len(q) / ns, 2),
                        'до след. отметки, б.п.': a, 't': ta, 'до закрытия, б.п.': b, 't ': tb})
    return pd.DataFrame(out)


def part_b(M, ins):
    held = M[M.z != 0].copy()
    held['usd'] = held.pts_next * POINT[ins]
    held['время'] = np.where(held.k <= 120, 'до 11:30', np.where(held.k <= 240, '12:00-13:30', '14:00+'))
    out = []
    for ep, g in held.groupby('epoch'):
        tot = g.usd.sum()
        row = {'ins': ins, 'эпоха': ep, 'валовое $': int(tot)}
        for kind in ('F', 'S'):
            row[f'после {kind} $'] = int(g.usd[g.kind == kind].sum())
            row[f'интервалов {kind}'] = int((g.kind == kind).sum())
        for w in ('до 11:30', '12:00-13:30', '14:00+'):
            row[w + ' $'] = int(g.usd[g['время'] == w].sum())
        out.append(row)
    return pd.DataFrame(out), held


def part_c(M, ins):
    """Выход источника: держать до противоположной границы или закрытия."""
    rows = []
    for s, g in M.groupby('s'):
        g = g.sort_values('k')
        z = g.z.to_numpy()
        p = np.zeros(len(z), int)
        cur = 0
        for j in range(len(z)):
            if z[j] != 0:
                cur = z[j]
            p[j] = cur
        raw = g.raw_next.to_numpy()                  # ход рынка на интервале, пункты
        gross = float((p * raw).sum()) * POINT[ins]
        seg = int((np.diff(np.r_[0, p]) != 0).sum())
        z_seg = int((g.kind == 'F').sum())           # каждый свежий выход открывает один оборот v0
        g0 = float((z * raw).sum()) * POINT[ins]
        rows.append({'s': s, 'date': g.date.iloc[0], 'year': g.year.iloc[0], 'epoch': g.epoch.iloc[0],
                     'opp': gross - COST[ins] * seg, 'v0': g0 - COST[ins] * z_seg, 'v0_gross': g0})
    D = pd.DataFrame(rows)
    out = []
    for ep, g in list(D.groupby('epoch')) + [('2024-04+', D[D.date >= '2024-04-01'])]:
        for name in ('v0', 'opp'):
            x = g[name]
            eq = x.cumsum()
            out.append({'ins': ins, 'эпоха': ep, 'выход': 'ближняя граница (v0)' if name == 'v0' else 'противоположная (база источника)',
                        'средняя $': round(x.mean(), 1), 't': round(x.mean() / x.std() * np.sqrt(len(x)), 2),
                        'итог $': int(x.sum()), 'просадка $': int((eq.cummax() - eq).max()),
                        'худший день $': int(x.min()), 'без лучших 5% $': int(x[x < x.quantile(.95)].sum())})
    return pd.DataFrame(out), D


def part_d(M, D):
    """Каждая величина — отношение сессионных сумм; бутстреп пересэмплирует сессии (2000 повторов)."""
    F = M.kind == 'F'
    has = M.z_next.notna()
    stay = F & has & (M.z_next == M.z)
    back = F & has & (M.z_next == 0)
    one = pd.Series(1.0, index=M.index)
    defs = {
        'свежих выходов на сессию': (F.astype(float), None),
        'доля F, оставшихся за нормой на след. отметке': (stay.astype(float), (F & has).astype(float)),
        'доля F, вернувшихся на след. отметке': (back.astype(float), (F & has).astype(float)),
        'ход до след. отметки после F, б.п.': (M.r_next.where(F, 0.0), F.astype(float)),
        'ход после F, если вернулись (цена ложного выхода), б.п.': (M.r_next.where(back, 0.0), back.astype(float)),
        'ход после F, если остались, б.п.': (M.r_next.where(stay, 0.0), stay.astype(float)),
        'ход до след. отметки после S, б.п.': (M.r_next.where(M.kind == 'S', 0.0), (M.kind == 'S').astype(float)),
        'ход до закрытия после F, б.п.': (M.r_close.where(F, 0.0), F.astype(float)),
        'ход до закрытия после R в прежнюю сторону, б.п.': (M.r_close.where(M.kind == 'R', 0.0), (M.kind == 'R').astype(float)),
        'валовое v0 на сессию, б.п.': (M.r_next.where(M.z != 0, 0.0), None),
    }
    first = M.groupby('s').date.first()
    e_idx = first[(first >= '2020-01-01') & (first < '2024-04-01')].index
    l_idx = first[first >= '2024-04-01'].index
    out = []
    for name, (num, den) in defs.items():
        ns = num.groupby(M.s).sum()
        ds = den.groupby(M.s).sum() if den is not None else pd.Series(1.0, index=ns.index)
        def val(idx):
            return ns.loc[idx].sum() / ds.loc[idx].sum()
        ne, de, nl, dl = ns.loc[e_idx].to_numpy(), ds.loc[e_idx].to_numpy(), ns.loc[l_idx].to_numpy(), ds.loc[l_idx].to_numpy()
        diffs = []
        for _ in range(2000):
            i = RNG.integers(0, len(ne), len(ne))
            j = RNG.integers(0, len(nl), len(nl))
            diffs.append(nl[j].sum() / dl[j].sum() - ne[i].sum() / de[i].sum())
        lo, hi = np.nanquantile(diffs, [.025, .975])
        ve, vl = val(e_idx), val(l_idx)
        out.append({'величина': name, '2020-01…2024-03': round(ve, 3), '2024-04…2026-05': round(vl, 3),
                    'разность': round(vl - ve, 3), '95% низ': round(lo, 3), '95% верх': round(hi, 3),
                    'ноль внутри': bool(lo <= 0 <= hi)})
    T = pd.DataFrame(out)
    # контекст: место позднего блока среди окон той же длины в прежней истории (валовое б.п. на сессию)
    per = M.r_next.where(M.z != 0, 0.0).groupby(M.s).sum()
    dates = first.reindex(per.index)
    hist = per[dates < '2024-04-01'].to_numpy()
    lt = per[dates >= '2024-04-01'].to_numpy()
    n = len(lt)
    roll = pd.Series(hist).rolling(n).mean().dropna().to_numpy()
    hist20 = per[(dates >= '2020-01-01') & (dates < '2024-04-01')].to_numpy()
    roll20 = pd.Series(hist20).rolling(n).mean().dropna().to_numpy()
    se = np.sqrt(hist20.var(ddof=1) / len(hist20) + lt.var(ddof=1) / n)
    ctx = {'поздний блок, сессий': n, 'среднее позднего, б.п.': round(float(lt.mean()), 2),
           'среднее 2020-01…2024-03, б.п.': round(float(hist20.mean()), 2),
           'место позднего среди окон той же длины 2006…2024-03': round(float((roll < lt.mean()).mean()), 3),
           'место позднего среди окон той же длины 2020…2024-03': round(float((roll20 < lt.mean()).mean()), 3),
           'независимых окон такой длины во всей прежней истории': int(len(hist) // n),
           'ст. ошибка разности средних, б.п.': round(float(se), 2),
           'разность, различимая при t=2, б.п.': round(float(2 * se), 2),
           'наблюдённая разность, б.п.': round(float(lt.mean() - hist20.mean()), 2)}
    # последний короткий кусок отдельно
    last = per[dates >= '2025-11-01'].to_numpy()
    rl = pd.Series(hist).rolling(len(last)).mean().dropna().to_numpy()
    ctx['кусок 2025-11+: сессий'] = int(len(last))
    ctx['кусок 2025-11+: среднее, б.п.'] = round(float(last.mean()), 2)
    ctx['кусок 2025-11+: место среди окон той же длины 2006…2024-03'] = round(float((rl < last.mean()).mean()), 3)
    return T, ctx


def part_f_by_time(M, ins):
    """Добавлено после частей A–C (часть D тогда ещё не считалась): свежие выходы по времени дня."""
    F = M[M.kind == 'F'].copy()
    F['время'] = np.where(F.k <= 120, 'до 11:30', np.where(F.k <= 240, '12:00-13:30', '14:00+'))
    out = []
    for (ep, w), g in F.groupby(['epoch', 'время']):
        a, ta = sess_t(g, 'r_next')
        b, tb = sess_t(g, 'r_close')
        out.append({'ins': ins, 'эпоха': ep, 'время': w, 'свежих выходов': len(g),
                    'до след. отметки, б.п.': a, 't': ta, 'до закрытия, б.п.': b, 't ': tb})
    return pd.DataFrame(out)


def main():
    res = {}
    for ins in ('NQ', 'ES'):
        M = states(ins)
        A = part_a(M, ins)
        B, held = part_b(M, ins)
        Cc, D = part_c(M, ins)
        print(f'\n################ {ins}')
        print('--- A. виды отметок'); print(A.to_string(index=False))
        print('--- B. точное разложение валового v0'); print(B.to_string(index=False))
        if ins == 'NQ':
            daily = pd.read_csv(HERE / 'band_daily_NQ.csv', parse_dates=['date'])
            j = D.merge(daily[daily.status == 'ok'][['date', 'gross']], on='date')
            print('сверка с band_daily: общих сессий', len(j), '; макс. расхождение валового $',
                  round(float((j.v0_gross - j.gross).abs().max()), 6),
                  '; сессий v0 всего', int((daily.status == 'ok').sum()))
        print('--- C. выход v0 против выхода базы источника'); print(Cc.to_string(index=False))
        Ft = part_f_by_time(M, ins)
        print('--- свежие выходы по времени дня'); print(Ft.to_string(index=False))
        Ft.to_csv(HERE / f'anatomy_F_time_{ins}.csv', index=False)
        A.to_csv(HERE / f'anatomy_A_{ins}.csv', index=False)
        B.to_csv(HERE / f'anatomy_B_{ins}.csv', index=False)
        Cc.to_csv(HERE / f'anatomy_C_{ins}.csv', index=False)
        if ins == 'NQ':
            T, ctx = part_d(M, D)
            print('--- D. слабость последних лет'); print(T.to_string(index=False))
            for k, v in ctx.items():
                print(f'   {k}: {v}')
            T.to_csv(HERE / 'anatomy_D_NQ.csv', index=False)
            pd.Series(ctx).to_json(HERE / 'anatomy_D_context.json', force_ascii=False, indent=1)


if __name__ == '__main__':
    main()
