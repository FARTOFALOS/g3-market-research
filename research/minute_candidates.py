#!/usr/bin/env python3
"""Счетоводство по кандидатам, выжившим в переборе `minute_edge.py`.

Поиск закончен в `minute_edge.py`; здесь ничего не подбирается. Условия
зафиксированы дословно по итогу перебора, и для каждого считается исполнимый
поток: одна позиция одновременно, вход `open k+1`, выход `open k+1+hold`,
расход по объявленной модели, разбор состава, зеркальная сторона, устойчивость
по календарю и проверка на резервном куске, который при поиске не открывался.

МОДЕЛЬ РАСХОДА, НАЗВАННАЯ ПОСОСТАВНО
====================================
Исследовательское допущение репозитория: $15 за полный оборот NQ =
$5 комиссии + 0,5 пункта совокупного проскальзывания ($10 при $20 за пункт).
Происхождение частей:

- **комиссия** $5 за оборот — суммарная величина, в которой брокерская
  составляющая и биржевые сборы CME НЕ разделены. У розничного брокера
  выставляется единый тариф за сторону, внутрь которого уже входят биржевой и
  клиринговый сборы; складывать их повторно нельзя. Фактический тариф этого
  счёта **неизвестен** и остаётся неизвестным.
- **проскальзывание** 0,5 пункта за оборот — допущение об исполнении рыночным
  приказом на `open` минуты; отдельного замера книги здесь нет.
- **модель исполнения** — вход и выход по `open` минуты, без частичных
  исполнений и без отказа в заполнении.

Поэтому для каждого потока печатается ТРИ величины: валовое, чистое при $15 и
**предельный средний расход**, при котором средняя сделка остаётся
положительной. Сценарии расхода названы явно, фактический тариф не выдуман.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidate_check import POINT, COST                            # noqa: E402
from minute_edge import (BACK, AHEAD, MIN, ROOT, load,             # noqa: E402
                         session_stage)

COST_MODEL = {
    'total_per_turn_usd': 15.0,
    'commission_usd': 5.0,
    'commission_note': 'единый тариф за оборот; брокерская и биржевая части не '
                       'разделены и повторно не складываются; фактический тариф '
                       'счёта неизвестен',
    'slippage_points': 0.5,
    'slippage_usd': 10.0,
    'execution': 'рыночный приказ по open минуты, полное заполнение',
    'source': 'setups/S-05 (line 62); константа в research/candidate_check.py'}


# ------------------------------------------------------- условия кандидатов --

def parts(m, k, sess_start):
    """Именованные части условий. Всё известно на закрытии минуты k."""
    O, H, L, C = m['open'], m['high'], m['low'], m['close']
    hs = np.max(np.stack([H[k - i] for i in range(1, BACK + 1)]), axis=0)
    top = np.maximum(O[k], C[k])
    bot = np.minimum(O[k], C[k])
    stage = session_stage(sess_start, k)
    return {
        'C[0]>maxH[-4..-1]': C[k] > hs,
        'upper_wick>body[0]': (H[k] - top) > np.abs(C[k] - O[k]),
        'lower_wick>body[0]': (bot - L[k]) > np.abs(C[k] - O[k]),
        'C[-1]>C[-4]': C[k - 1] > C[k - 4],
        'C[0]<L[-4]': C[k] < L[k - 4],
        'сессия: первые 5 минут': stage['сессия: первые 5 минут'],
        'сессия: последние 30 минут': stage['сессия: последние 30 минут']}


CANDIDATES = {
    'K-1': {
        'name': 'отвергнутый толчок за четырёхминутный максимум',
        'condition': ['C[0]>maxH[-4..-1]', 'upper_wick>body[0]'],
        'negate': [],
        'side': -1,
        'holds': [1, 3, 5, 10, 15],
        'reading': 'минута закрылась выше максимума предыдущих четырёх, но '
                   'верхняя тень длиннее тела: толчок наверх состоялся и был '
                   'отвергнут внутри той же минуты',
        'expects': 'снос обратно вниз в ближайшие минуты',
        'riz_link': 'прямой связи с RIZ нет'},
    'K-2': {
        'name': 'первые пять минут сессии без подъёма закрытий',
        'condition': ['сессия: первые 5 минут'],
        'negate': ['C[-1]>C[-4]'],
        'side': -1,
        'holds': [5, 15, 30],
        'reading': 'минута из первых пяти минут основной сессии, в которой '
                   'закрытие предыдущей минуты не выше закрытия четырёхминутной '
                   'давности',
        'expects': 'снос вниз в первые минуты после открытия',
        'riz_link': 'прямой связи с RIZ нет; пересекается по территории с '
                    'утренней дверью S-07/S-09 и самостоятельным подтверждением '
                    'её не является'},
    'K-3': {
        'name': 'последние тридцать минут без провала под четырёхминутный минимум',
        'condition': ['сессия: последние 30 минут'],
        'negate': ['C[0]<L[-4]'],
        'side': 1,
        'holds': [15, 30],
        'reading': 'минута из последних тридцати минут сессии, закрытие которой '
                   'не ушло под минимум четырёхминутной давности',
        'expects': 'дрейф вверх к закрытию сессии',
        'riz_link': 'прямой связи с RIZ нет'}}


def mask_of(p, cand):
    sel = np.ones(len(next(iter(p.values()))), bool)
    for nm in cand['condition']:
        sel &= p[nm]
    for nm in cand['negate']:
        sel &= ~p[nm]
    return sel


# ------------------------------------------------------------------ поток ----

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


def summarise(trades, cost):
    if not trades:
        return {'n': 0}
    g = np.array([t[1] for t in trades])
    tsx = np.array([t[0] for t in trades])
    year = (tsx // MIN) // int(365.2425 * 1440) + 1970
    net = g - cost
    eq = np.cumsum(net)
    sem = float(g.std(ddof=1) / np.sqrt(len(g)))
    by = {int(y): float(net[year == y].sum()) for y in np.unique(year)}
    byg = {int(y): float(g[year == y].mean()) for y in np.unique(year)}
    return {'n': len(g), 'mean_gross': float(g.mean()), 'sem_gross': sem,
            't_gross': float(g.mean() / sem) if sem else None,
            'median_gross': float(np.median(g)),
            'share_positive': float((g > 0).mean()),
            'mean_net_at_15': float(net.mean()),
            'total_net_at_15': float(net.sum()),
            'breakeven_cost_usd': float(g.mean()),
            'drawdown_net_at_15': float(np.max(np.maximum.accumulate(eq) - eq)),
            'by_year_net_at_15': by, 'by_year_mean_gross': byg,
            'positive_years_gross': int(sum(v > 0 for v in byg.values())),
            'years': len(by),
            'gross_without_top1pct': float(
                np.sort(g)[:-max(1, len(g) // 100)].mean())}


def dissect(m, k, p, cand, instrument, hold):
    """Разбор состава: что даёт каждая часть по отдельности."""
    out = {}
    names = cand['condition'] + ['не ' + n for n in cand['negate']]
    for nm in names:
        if nm.startswith('не '):
            sel = ~p[nm[3:]]
        else:
            sel = p[nm]
        tr = stream(m, k, sel, cand['side'], instrument, hold)
        s = summarise(tr, COST[instrument])
        out[nm] = {'n': s['n'], 'mean_gross': s.get('mean_gross'),
                   't_gross': s.get('t_gross')}
    return out


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
    res = {'implementation_sha256': hashlib.sha256(
               Path(__file__).read_bytes()).hexdigest(),
           'cost_model': COST_MODEL,
           'note': 'поиск закончен в minute_edge.py; здесь ничего не подбирается',
           'candidates': {}}

    books = {}
    for tag, (s0, s1) in (('поиск', (a.start, a.stop)),
                          ('резерв', (a.reserve_start, a.reserve_stop))):
        m, k, ss = load(a.instrument, s0, s1)
        books[tag] = (m, k, parts(m, k, ss), len(k))

    for key, cand in CANDIDATES.items():
        entry = {k2: v for k2, v in cand.items() if k2 != 'holds'}
        entry['holds'] = {}
        for tag in ('поиск', 'резерв'):
            m, k, p, nmin = books[tag]
            sel = mask_of(p, cand)
            block = {'decision_minutes': nmin, 'films': int(sel.sum())}
            for hold in cand['holds']:
                block[f'hold_{hold}'] = summarise(
                    stream(m, k, sel, cand['side'], a.instrument, hold), cost)
                block[f'hold_{hold}_mirror_mean_gross'] = -block[
                    f'hold_{hold}']['mean_gross'] if block[f'hold_{hold}']['n'] else None
            entry['holds'][tag] = block
        m, k, p, _ = books['поиск']
        entry['dissection_at_main_hold'] = dissect(
            m, k, p, cand, a.instrument, cand['holds'][len(cand['holds']) // 2])
        res['candidates'][key] = entry

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1,
                                      default=str), encoding='utf-8')
    brief = {}
    for key, c in res['candidates'].items():
        brief[key] = {}
        for tag in ('поиск', 'резерв'):
            b = c['holds'][tag]
            brief[key][tag] = {h: {'n': b[h]['n'],
                                   'g': round(b[h].get('mean_gross', 0), 2),
                                   't': round(b[h].get('t_gross') or 0, 2),
                                   'net15': round(b[h].get('mean_net_at_15', 0), 2),
                                   'yrs+': b[h].get('positive_years_gross')}
                               for h in b if h.startswith('hold_')
                               and not h.endswith('mirror_mean_gross')}
    print(json.dumps(brief, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
