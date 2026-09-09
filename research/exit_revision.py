#!/usr/bin/env python3
"""Ревизия ведения позиции: выход, завершающий конкретное ожидание.

ЧТО ОЖИДАЕТСЯ И ЧЕМ ЭТО КОНЧАЕТСЯ — СФОРМУЛИРОВАНО НА СВЕЧАХ ДО СЧЁТА
====================================================================
Конструкция юга говорит: за первые пять минут последний новый максимум пришёл
РАНЬШЕ последнего нового минимума, то есть окно шло вниз. Север зеркален.

* **Что ожидается.** Ход продолжится за уже достигнутый край: цена уйдёт ниже
  минимума первых пяти минут (для юга) и закрепится там закрытием.
* **Реализация ожидания** — закрытие ниже `minL[1..5]` (юг) / выше `maxH[1..5]`
  (север). Это наблюдаемое событие означает, что ожидаемая часть движения
  состоялась: ход вышел за край, известный на минуте решения.
* **Изменение сцены (ошибка идеи)** — закрытие выше `maxH[1..5]` (юг) / ниже
  `minL[1..5]` (север). Это не «противоположное движение вообще»: закрытие за
  противоположным краем ОТМЕНЯЕТ сам порядок экстремумов, которым конструкция
  и определена. Ожидание после этого больше не то, с которым входили.
* **Неизвестно** — пока не наступило ни того, ни другого, ожидание открыто.
  Просадка внутри диапазона первых пяти минут ошибкой идеи НЕ является: она
  обычная часть пути.
* **Защитное ограничение убытка** держится отдельно и деньгами и называется
  ограничением убытка, а не ошибкой идеи. Значение перенесено из прежней
  версии без пересчёта: пересчитывать его под новое ведение значило бы
  подгонять.

ВРЕМЯ СЧИТАЕТСЯ ОТ РЕШЕНИЯ
==========================
Исходные номера свечей и T0 сохраняются. Рядом идёт локальный отсчёт: решение
на минуте 5, минута 6 — это k=1 ожидания. Медианы наступления сроком не
являются: они описывают только дошедшие фильмы и молчат о тех, где событие не
случилось никогда. Срок назначается только если у него есть наблюдаемое
основание на наложении, и тогда называется доля, которую он отсекает, включая
позднее успешные.

ИСПОЛНЕНИЕ
==========
Событие по закрытию доступно только к концу своей минуты, поэтому выход —
по `open` следующей минуты. Закрытие свечи не используется как выход по её
более ранней выгодной цене. Внутриминутный порядок неизвестен: если в одну
минуту достижим и защитный предел, и событие, разрешается ПРОТИВ сделки —
сначала предел.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calendar_utils import year_of, quarter_of, date_key  # noqa: E402

from relational_stencil import Corpus                      # noqa: E402
from candidate_check import CURSOR, HORIZON, POINT, COST, sides, tape  # noqa: E402

MIN = 60_000_000_000
CARRIED_LIMIT = 28.475     # перенесён из отвергнутой версии, не пересчитан


def decisions(inputs, instrument):
    """Та же популяция, распознавание и сторона, что в отвергнутой версии."""
    anchors, side = [], []
    for path in inputs:
        c = Corpus.load(path)
        s, _, _ = sides(c)
        anchors.append(np.array([int(u.split(':')[1]) for u in c.unit_ids]))
        side.append(s)
    anchors = np.concatenate(anchors)
    side = np.concatenate(side)
    keep = np.unique(anchors, return_index=True)[1]
    return anchors[keep], side[keep]


def paths(anchors, side, m):
    """Разобранные пути ожидания. Денег здесь нет."""
    ts = m['close_ts_utc_ns']
    pos = np.searchsorted(ts, anchors)
    out = []
    for i in np.argsort(anchors):
        if side[i] == 0:
            continue
        p = int(pos[i])
        want = ts[p] + np.arange(0, HORIZON + 1) * MIN
        at = np.searchsorted(ts, want)
        if at[-1] >= len(ts) or not np.all(ts[at] == want):
            continue
        hi, lo, cl, op = (m['high'][at], m['low'][at], m['close'][at], m['open'][at])
        prefix = slice(1, CURSOR + 1)
        top, bottom = float(hi[prefix].max()), float(lo[prefix].min())
        s = int(side[i])
        realise_level, error_level = (bottom, top) if s < 0 else (top, bottom)
        fut = np.arange(CURSOR + 1, HORIZON + 1)
        c_fut = cl[fut]
        realised = (c_fut < realise_level) if s < 0 else (c_fut > realise_level)
        errored = (c_fut > error_level) if s < 0 else (c_fut < error_level)
        k_real = int(np.argmax(realised)) + 1 if realised.any() else None
        k_err = int(np.argmax(errored)) + 1 if errored.any() else None
        out.append({'ts': int(ts[at[CURSOR + 1]]), 'index': p, 'side': s,
                    'top': top, 'bottom': bottom,
                    'k_realise': k_real, 'k_error': k_err,
                    'at': at, 'hi': hi, 'lo': lo, 'cl': cl, 'op': op})
    return out


def overlay(rows):
    """По каждой минуте ожидания: реализовано, сцена изменилась, открыто."""
    n = len(rows)
    table = []
    for k in range(1, HORIZON - CURSOR + 1):
        realised = err = still = 0
        for r in rows:
            kr, ke = r['k_realise'], r['k_error']
            first_r = kr if kr is not None and kr <= k else None
            first_e = ke if ke is not None and ke <= k else None
            if first_r is None and first_e is None:
                still += 1
            elif first_e is None or (first_r is not None and first_r <= first_e):
                realised += 1
            else:
                err += 1
        table.append({'k': k, 'realised': realised, 'scene_changed': err,
                      'still_open': still, 'observation_insufficient': 0,
                      'share_realised': realised / n, 'share_changed': err / n,
                      'share_open': still / n})
    return table


def hazards(rows, table):
    """Что ещё остаётся ждать в наблюдаемом состоянии, а не где кончались удачи."""
    out = []
    for k in range(1, HORIZON - CURSOR + 1):
        open_before = [r for r in rows
                       if (r['k_realise'] is None or r['k_realise'] >= k)
                       and (r['k_error'] is None or r['k_error'] >= k)]
        n = len(open_before)
        if n == 0:
            break
        r_now = sum(1 for r in open_before if r['k_realise'] == k
                    and (r['k_error'] is None or r['k_error'] > k))
        e_now = sum(1 for r in open_before if r['k_error'] == k
                    and (r['k_realise'] is None or r['k_realise'] > k))
        ever_r = sum(1 for r in open_before
                     if r['k_realise'] is not None
                     and (r['k_error'] is None or r['k_realise'] <= r['k_error']))
        out.append({'k': k, 'still_open_at_k': n,
                    'hazard_realise': r_now / n, 'hazard_change': e_now / n,
                    'eventually_realises_from_here': ever_r / n})
    return out


def execute(rows, mode, instrument='NQ', limit=CARRIED_LIMIT, deadline=None):
    """mode: 'budget' — прежнее ведение; 'expectation' — новое."""
    cost = COST[instrument]
    busy_until = -1
    trades = []
    for r in rows:
        if r['ts'] <= busy_until:
            continue
        s, at = r['side'], r['at']
        hi, lo, cl, op = r['hi'], r['lo'], r['cl'], r['op']
        entry = float(op[CURSOR + 1])
        stop = entry - s * limit
        realise_level, error_level = ((r['bottom'], r['top']) if s < 0
                                      else (r['top'], r['bottom']))
        exit_price, why, k_exit = None, None, None
        for k in range(1, HORIZON - CURSOR + 1):
            j = CURSOR + k
            # внутриминутный порядок неизвестен: предел разрешается против сделки
            if (s > 0 and lo[j] <= stop) or (s < 0 and hi[j] >= stop):
                px = stop
                if (s > 0 and op[j] < stop) or (s < 0 and op[j] > stop):
                    px = float(op[j])
                exit_price, why, k_exit = px, 'предел', k
                break
            if mode == 'expectation':
                c = float(cl[j])
                hit_r = c < realise_level if s < 0 else c > realise_level
                hit_e = c > error_level if s < 0 else c < error_level
                if hit_r or hit_e:
                    if j + 1 <= HORIZON:
                        exit_price = float(op[j + 1])
                    else:
                        exit_price = c
                    why = 'реализация' if hit_r else 'сцена изменилась'
                    k_exit = k
                    break
                if deadline is not None and k >= deadline:
                    exit_price = float(op[j + 1]) if j + 1 <= HORIZON else c
                    why, k_exit = 'срок ожидания', k
                    break
        if exit_price is None:
            exit_price, why, k_exit = float(cl[HORIZON]), 'граница наблюдения', HORIZON - CURSOR
        gross = s * (exit_price - entry) * POINT[instrument]
        trades.append({'ts': r['ts'], 'side': s, 'entry': entry, 'exit': exit_price,
                       'why': why, 'k_exit': k_exit, 'net': gross - cost,
                       'k_realise': r['k_realise'], 'k_error': r['k_error']})
        busy_until = int(r['ts'] + (HORIZON - CURSOR - 1) * MIN)
    return trades


def block(trades):
    if not trades:
        return {'n': 0}
    net = np.array([t['net'] for t in trades])
    ts = np.array([t['ts'] for t in trades])
    year = year_of(ts)
    by_year = {int(y): float(net[year == y].sum()) for y in np.unique(year)}
    eq = np.cumsum(net)
    reasons = {}
    for t in trades:
        reasons[t['why']] = reasons.get(t['why'], 0) + 1
    return {'n': len(trades), 'mean_net': float(net.mean()),
            'total_net': float(net.sum()),
            'positive_years': int(sum(v > 0 for v in by_year.values())),
            'years': len(by_year), 'by_year': by_year,
            'drawdown': float(np.max(np.maximum.accumulate(eq) - eq)),
            'worst': float(net.min()), 'best': float(net.max()),
            'median_k_exit': float(np.median([t['k_exit'] for t in trades])),
            'exit_reasons': reasons}


def compare(old, new):
    """Какие позиции изменились и почему."""
    a = {t['ts']: t for t in old}
    b = {t['ts']: t for t in new}
    shared = sorted(set(a) & set(b))
    buckets = {'раньше реализовали ожидание': [],
               'раньше признали изменение сцены': [],
               'отказались от дальнейшего ожидания': [],
               'преждевременно вышли перед поздним движением': [],
               'без изменения': []}
    for ts in shared:
        x, y = a[ts], b[ts]
        if abs(x['net'] - y['net']) < 1e-9 and x['why'] == y['why']:
            buckets['без изменения'].append(0.0)
            continue
        delta = y['net'] - x['net']
        if y['why'] == 'реализация':
            key = ('раньше реализовали ожидание' if delta >= 0
                   else 'преждевременно вышли перед поздним движением')
        elif y['why'] == 'сцена изменилась':
            key = ('раньше признали изменение сцены' if delta >= 0
                   else 'преждевременно вышли перед поздним движением')
        elif y['why'] == 'срок ожидания':
            key = 'отказались от дальнейшего ожидания'
        else:
            key = 'без изменения'
        buckets[key].append(delta)
    return {'shared_decisions': len(shared),
            'buckets': {k: {'n': len(v), 'sum_delta': float(np.sum(v)) if v else 0.0,
                            'mean_delta': float(np.mean(v)) if v else None}
                        for k, v in buckets.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--instrument', default='NQ')
    ap.add_argument('--deadline', type=int, default=None,
                    help='назначать только при наблюдаемом основании')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    m = tape(a.instrument)
    anchors, side = decisions(a.inputs, a.instrument)
    rows = paths(anchors, side, m)
    table = overlay(rows)
    haz = hazards(rows, table)
    old = execute(rows, 'budget', a.instrument)
    new = execute(rows, 'expectation', a.instrument, deadline=a.deadline)
    result = {
        'expectation': {
            'what_is_expected': 'ход продолжится за край первых пяти минут',
            'realisation': 'закрытие за краем в сторону конструкции',
            'scene_change': 'закрытие за противоположным краем — отменяет сам '
                            'порядок экстремумов, которым конструкция определена',
            'not_an_idea_error': 'просадка внутри диапазона первых пяти минут',
            'protective_limit_points': CARRIED_LIMIT,
            'limit_status': 'ограничение убытка, перенесено без пересчёта'},
        'population_of_expectation': len(rows),
        'overlay_by_minute_after_decision': table,
        'hazards_in_open_group': haz,
        'previous_version': block(old),
        'revised_version': block(new),
        'what_changed': compare(old, new),
        'execution_rules': [
            'событие по закрытию доступно к концу минуты — выход по open следующей',
            'внутриминутный порядок неизвестен: предел разрешается против сделки',
            'популяция, распознавание, сторона, вход и разрешение пересечений '
            'сохранены из отвергнутой версии'],
        'deadline_applied': a.deadline}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=1),
                           encoding='utf-8')
    print(json.dumps({'population': len(rows),
                      'previous': {k: result['previous_version'][k] for k in
                                   ('n', 'mean_net', 'total_net', 'positive_years',
                                    'drawdown')},
                      'revised': {k: result['revised_version'][k] for k in
                                  ('n', 'mean_net', 'total_net', 'positive_years',
                                   'drawdown', 'exit_reasons', 'median_k_exit')},
                      'what_changed': result['what_changed']},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
