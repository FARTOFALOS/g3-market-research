#!/usr/bin/env python3
"""Счетоводство по находкам `ordinal_search.py`. Здесь ничего не подбирается.

Условия зафиксированы дословно по итогу перебора на полном языке порядковых
отношений. Перебор шёл по прореженной популяции (шаг объявлен); здесь те же
условия меряются на ВСЕЙ подходящей популяции — это пересчёт зафиксированного
правила, а не новый подбор. Отдельно считается резервный кусок, который при
поиске не открывался, разбор состава, зеркальная сторона, вклад по дням и
кварталам.

Язык условий — только порядок точек O/H/L/C: какие точки выше, ниже или на
уровне других. Длин, расстояний, ATR и Volume здесь нет.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calendar_utils import year_of, quarter_of, date_key  # noqa: E402

from candidate_check import POINT, COST                            # noqa: E402
from minute_edge import load, MIN                                  # noqa: E402
from minute_candidates import COST_MODEL                           # noqa: E402

BACK = 9


def relations(m, k):
    """Именованные порядковые отношения. Все закрыты к концу минуты k."""
    O, H, L, C = m['open'], m['high'], m['low'], m['close']
    return {
        'C[-9] > H[-3]': C[k - 9] > H[k - 3],
        'L[-2] > O[0]': L[k - 2] > O[k],
        'L[-9] > H[-1]': L[k - 9] > H[k - 1],
        'C[-7] < H[-3]': C[k - 7] < H[k - 3],
        'H[-6] > O[-1]': H[k - 6] > O[k - 1],
        'O[-5] < L[-3]': O[k - 5] < L[k - 3],
    }


CANDIDATES = {
    'O-1': {
        'name': 'спад девяти минут и низ двух минут назад выше открытия решающей',
        'atoms': ['C[-9] > H[-3]', 'L[-2] > O[0]'],
        'side': 1,
        'holds': [1, 3, 5],
        'main_hold': 1,
        'relations_plainly': 'закрытие девятой минуты назад выше максимума третьей '
                             'минуты назад; низ второй минуты назад выше открытия '
                             'решающей минуты',
        'sequence': 'за девять минут цена оказалась ниже, чем была, и к решающей '
                    'минуте продолжала опускаться: точка, бывшая низом два шага '
                    'назад, теперь выше текущего открытия',
        'hypothesis': 'после затяжного одностороннего спада ближайшая минута чаще '
                      'идёт вверх — это отскок, а не разворот; смысл приписан ПОСЛЕ '
                      'восстановления отношений и в имени не заложен',
        'weakens_it': 'если тот же подъём следует и без второго отношения, длина '
                      'спада ни при чём; разбор состава ниже это и проверяет',
        'riz_link': 'прямой связи с RIZ нет; поле RIZ не читалось'},
    'O-2': {
        'name': 'провал ниже всего окна и последующее возвращение к максимуму',
        'atoms': ['L[-9] > H[-1]', 'C[-7] < H[-3]'],
        'side': 1,
        'holds': [1, 5, 15],
        'main_hold': 5,
        'relations_plainly': 'низ девятой минуты назад выше максимума прошлой '
                             'минуты; закрытие седьмой минуты назад ниже максимума '
                             'третьей',
        'sequence': 'весь участок ушёл ниже начала окна, при этом внутри участка '
                    'был подъём к третьей минуте назад',
        'hypothesis': 'глубокий спад с промежуточным подъёмом внутри — цена чаще '
                      'идёт вверх на горизонте пяти минут',
        'weakens_it': 'исчезновение подъёма при удлинении или укорочении горизонта',
        'riz_link': 'прямой связи с RIZ нет'},
    'O-3': {
        'name': 'подъём начала окна и провал середины',
        'atoms': ['H[-6] > O[-1]', 'O[-5] < L[-3]'],
        'side': 1,
        'holds': [5, 15],
        'main_hold': 15,
        'relations_plainly': 'максимум шестой минуты назад выше открытия прошлой '
                             'минуты; открытие пятой минуты назад ниже минимума '
                             'третьей',
        'sequence': 'внутри окна был подъём от пятой к третьей минуте назад, а к '
                    'решающей минуте цена вернулась ниже того максимума',
        'hypothesis': 'подъём внутри окна возобновляется на горизонте пятнадцати '
                      'минут',
        'weakens_it': 'отсутствие подъёма на коротких горизонтах — он и наблюдается',
        'riz_link': 'прямой связи с RIZ нет'}}


def mask_of(rel, cand):
    sel = np.ones(len(next(iter(rel.values()))), bool)
    for nm in cand['atoms']:
        sel &= rel[nm]
    return sel


def stream(m, k, sel, side, instrument, hold):
    ts, O = m['close_ts_utc_ns'], m['open']
    pv = POINT[instrument]
    busy, out = -1, []
    for i in k[sel]:
        t_in = int(ts[i + 1])
        if t_in <= busy:
            continue
        out.append((t_in, float(side * (O[i + 1 + hold] - O[i + 1]) * pv)))
        busy = int(ts[i + hold])
    return out


def summarise(tr, cost):
    if not tr:
        return {'n': 0}
    g = np.array([t[1] for t in tr])
    ts = np.array([t[0] for t in tr])
    day = date_key(ts)
    year = year_of(ts)
    quarter = quarter_of(ts)
    net = g - cost
    eq = np.cumsum(net)
    sem = float(g.std(ddof=1) / np.sqrt(len(g)))
    byg = {int(y): float(g[year == y].mean()) for y in np.unique(year)}
    byq = {int(q): float(net[quarter == q].mean()) for q in np.unique(quarter)}
    days = np.unique(day)
    by_day_g = np.array([g[day == d].sum() for d in days])
    order = np.argsort(-by_day_g)
    keep5 = ~np.isin(day, days[order[:5]])
    return {'n': len(g), 'mean_gross': float(g.mean()), 'sem_gross': sem,
            't_gross': float(g.mean() / sem),
            'median_gross': float(np.median(g)),
            'share_positive': float((g > 0).mean()),
            'mean_net_at_15': float(net.mean()),
            'total_net_at_15': float(net.sum()),
            'breakeven_cost_usd': float(g.mean()),
            'drawdown_net_at_15': float(np.max(np.maximum.accumulate(eq) - eq)),
            'by_year_mean_gross': byg,
            'positive_years_gross': int(sum(v > 0 for v in byg.values())),
            'years': len(byg),
            'quarters_positive_net': int(sum(v > 0 for v in byq.values())),
            'quarters': len(byq),
            'worst_quarter_net': float(min(byq.values())),
            'best_quarter_net': float(max(byq.values())),
            'days': int(len(days)),
            'top1_day_share_of_gross': float(by_day_g[order[0]] / g.sum())
            if g.sum() else None,
            'top5_days_share_of_gross': float(by_day_g[order[:5]].sum() / g.sum())
            if g.sum() else None,
            'mean_gross_drop_best_5_days': float(g[keep5].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instrument', default='NQ')
    ap.add_argument('--start', default='2020-01-01')
    ap.add_argument('--stop', default='2025-11-01')
    ap.add_argument('--reserve-start', default='2025-11-01')
    ap.add_argument('--reserve-stop', default='2026-05-05')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cost = COST[a.instrument]
    books = {}
    for tag, s0, s1 in (('поиск', a.start, a.stop),
                        ('резерв', a.reserve_start, a.reserve_stop)):
        m, k, _ = load(a.instrument, s0, s1)
        k = k[k >= BACK]
        books[tag] = (m, k, relations(m, k))
    res = {'implementation_sha256': hashlib.sha256(
               Path(__file__).read_bytes()).hexdigest(),
           'language': 'только порядок точек O/H/L/C; длин и расстояний нет',
           'note': 'условия зафиксированы по итогу перебора; здесь пересчёт на '
                   'полной популяции без прореживания и без подбора',
           'cost_model': COST_MODEL, 'candidates': {}}
    for key, cand in CANDIDATES.items():
        entry = {q: v for q, v in cand.items() if q != 'holds'}
        entry['periods'] = {}
        for tag in ('поиск', 'резерв'):
            m, k, rel = books[tag]
            sel = mask_of(rel, cand)
            block = {'decision_minutes': int(len(k)), 'films': int(sel.sum())}
            for h in cand['holds']:
                block[f'hold_{h}'] = summarise(
                    stream(m, k, sel, cand['side'], a.instrument, h), cost)
            entry['periods'][tag] = block
        m, k, rel = books['поиск']
        entry['dissection'] = {}
        for nm in cand['atoms']:
            s = summarise(stream(m, k, rel[nm], cand['side'], a.instrument,
                                 cand['main_hold']), cost)
            entry['dissection'][nm] = {'n': s['n'], 'mean_gross': s['mean_gross'],
                                       't_gross': s['t_gross']}
        sel = mask_of(rel, cand)
        s = summarise(stream(m, k, sel, -cand['side'], a.instrument,
                             cand['main_hold']), cost)
        entry['mirror_side_mean_gross'] = s['mean_gross']
        res['candidates'][key] = entry
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1,
                                      default=str), encoding='utf-8')
    brief = {}
    for key, c in res['candidates'].items():
        brief[key] = {t: {h: {'n': b[h]['n'], 'g': round(b[h]['mean_gross'], 2),
                              't': round(b[h]['t_gross'], 2),
                              'net': round(b[h]['mean_net_at_15'], 2),
                              'yr+': b[h]['positive_years_gross'],
                              'q+': f"{b[h]['quarters_positive_net']}/{b[h]['quarters']}"}
                          for h in b if h.startswith('hold_')}
                      for t, b in c['periods'].items()}
        brief[key]['состав'] = {q: (v['n'], round(v['mean_gross'], 2),
                                    round(v['t_gross'], 2))
                                for q, v in c['dissection'].items()}
    print(json.dumps(brief, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
