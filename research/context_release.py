#!/usr/bin/env python3
"""Что связано с результатом S-13: свечная история, контекст события или их сумма.

ЧТО ЗДЕСЬ ИСПРАВЛЕНО И УСТАНОВЛЕНО ДО СРАВНЕНИЙ
===============================================
1. **Календарная атрибуция.** Год считался как `день // 365 + 1970`. Это не
   календарное преобразование: граница года уползала почти на две недели, и
   конец декабря относился к следующему году. На территории 2020-2025 неверную
   подпись получали 2,87% минут. Исправлено в `research/calendar_utils.py`
   (часовой пояс атрибуции объявлен: America/New_York). Разделение выборок по
   дате использовало точную дату и ошибкой не затронуто.

2. **Что означает метка минуты.** Проверено на данных, а не принято на слово:
   средний размах свечи по минуте ET за 1 494 дня даёт всплеск на метке **08:31**
   (23,40 против фона 5,7), а не на 08:30 (7,01); у 14:00 ET то же — всплеск на
   метке 14:01. Значит метка минуты есть **конец интервала**: свеча с меткой
   08:31 покрывает 08:30:00-08:31:00 и содержит релиз, назначенный на 08:30 ET.

3. **Следствие для примера S-13 и для исполнения.** У примера T0 = 2022-07-13
   12:20 UTC, решающая минута T0+10 имеет метку 12:30 UTC = 08:30 ET, то есть
   покрывает 08:29:00-08:30:00 — минуту НЕПОСРЕДСТВЕННО ПЕРЕД официальным
   релизом CPI (08:30 ET). Вход по `open` следующей свечи есть цена в момент
   08:30:00 — ровно момент публикации. Модель исполнения «рыночный приказ по
   `open` с проскальзыванием 0,5 пункта» к такому входу неприменима, и это
   считается отдельно.

КЛАСС СОБЫТИЙ И СОПОСТАВЛЕНИЕ ОБЪЯВЛЕНЫ ДО СРАВНЕНИЯ
====================================================
Контекст определяется сведениями, доступными к решению: **расписанием слотов по
часам**, а не знанием исхода. Слоты объявлены заранее и по прибыли не
подбирались: 08:30, 10:00 и 14:00 ET — стандартные времена публикаций США
(макростатистика BLS/BEA/Census; решения FOMC). Окно сопоставления —
входная свеча покрывает слот, то есть её метка равна слоту плюс одна минута.
Вариант с расширением до ±2 минут объявлен здесь же как проверка устойчивости
подписи, а не как подбор.

**Покрытие календаря остаётся неизвестным.** Какие именно даты несли реальную
публикацию, здесь не устанавливается: слот присутствует каждый торговый день.
Поэтому «контекст публикации» означает «вход попадает в назначенный слот», а не
«публикация состоялась».

**Единица события.** Несколько T0 вокруг одного слота одного дня принадлежат
одному событию. Поддержка показывается тремя числами: фильмы, торговые дни,
события (дата + слот).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calendar_utils import (year_of, quarter_of, date_key,           # noqa: E402
                            minute_of_day)
from candidate_check import POINT, COST                              # noqa: E402
from compact_ordinal import build, compact_events, TFS, BACK, MIN    # noqa: E402

SLOTS = (8 * 60 + 30, 10 * 60, 14 * 60)      # ET, объявлено до сравнения
CURSOR = 10
ENTRY = CURSOR + 1


def construction(corpus, cursor=CURSOR):
    """Зафиксированная конструкция S-13. Ничего не подбирается."""
    ev = compact_events(corpus, cursor)
    col = {int(o): i for i, o in enumerate(corpus.ordinals)}
    L = corpus.ohlc[:, :, 2]
    arg = ev[f'argH_first[-{BACK}..{cursor}]'].values == cursor
    low = L[:, col[-6]] < L[:, col[-5]]
    return {'полная конструкция': arg & low,
            'только обновление экстремума': arg,
            'только отношение низов': low,
            'без распознавания': np.ones(corpus.n, bool)}


def contexts(t0, tol):
    """Контекст по доступному расписанию слотов. Вход — свеча T0+ENTRY."""
    entry_ts = t0 + ENTRY * MIN
    mod = minute_of_day(entry_ts)
    near = np.zeros(len(t0), bool)
    slot_id = np.full(len(t0), -1)
    for s in SLOTS:
        hit = np.abs(mod - (s + 1)) <= tol
        slot_id[hit & (slot_id < 0)] = s
        near |= hit
    return {'слот публикации': near, 'вне слотов': ~near}, slot_id, entry_ts


def summarise(tr, cost, slot_id=None):
    if not tr:
        return {'n': 0}
    g = np.array([t[1] for t in tr])
    ts = np.array([t[0] for t in tr])
    key = np.array([t[2] for t in tr])
    net = g - cost
    eq = np.cumsum(net)
    sem = float(g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else 0.0
    yr = year_of(ts)
    return {'n': int(len(g)), 'days': int(len(np.unique(date_key(ts)))),
            'events': int(len(np.unique(key))),
            'mean_gross': float(g.mean()), 'sem_gross': sem,
            't_gross': float(g.mean() / sem) if sem else None,
            'median_gross': float(np.median(g)),
            'share_positive': float((g > 0).mean()),
            'mean_net_at_15': float(net.mean()),
            'breakeven_cost_usd': float(g.mean()),
            'drawdown_net_at_15': float(np.max(np.maximum.accumulate(eq) - eq)),
            'by_year_mean_gross': {int(y): float(g[yr == y].mean())
                                   for y in np.unique(yr)},
            'by_year_n': {int(y): int((yr == y).sum()) for y in np.unique(yr)},
            'positive_years': int(sum(g[yr == y].mean() > 0
                                      for y in np.unique(yr))),
            'years': int(len(np.unique(yr)))}


def stream(corpus, t0, sel, side, hold, instrument, slot_id, extra_slip=0.0):
    """Исполнимый поток. Одна позиция. Для слотов допускается иное исполнение."""
    col = {int(o): i for i, o in enumerate(corpus.ordinals)}
    O = corpus.ohlc[:, :, 0]
    p = POINT[instrument]
    busy, tr = -1, []
    for i in np.argsort(t0):
        if not sel[i]:
            continue
        a, b = col[ENTRY], col[ENTRY + hold]
        if not (np.isfinite(O[i, a]) and np.isfinite(O[i, b])):
            continue
        t_in = int(t0[i]) + ENTRY * MIN
        if t_in <= busy:
            continue
        g = float(side * (O[i, b] - O[i, a]) * p)
        if extra_slip and slot_id[i] >= 0:
            g -= extra_slip
        day = int(date_key([t_in])[0])
        tr.append((t_in, g, day * 10000 + int(max(slot_id[i], 0))))
        busy = int(t0[i]) + (ENTRY + hold) * MIN
    return tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instrument', default='NQ')
    ap.add_argument('--start', default='2020-01-01')
    ap.add_argument('--stop', default='2025-11-01')
    ap.add_argument('--split', default='2024-01-01')
    ap.add_argument('--holds', nargs='+', type=int, default=[1, 5, 15])
    ap.add_argument('--tol', type=int, default=0)
    ap.add_argument('--slip-scenarios', nargs='+', type=float,
                    default=[0.0, 200.0, 500.0])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    corpus, t0, side_col, coverage = build(a.instrument, a.start, a.stop, TFS)
    cost = COST[a.instrument]
    forms = construction(corpus)
    ctx, slot_id, entry_ts = contexts(t0, a.tol)
    split = np.datetime64(a.split, 'ns').astype('int64')
    train = t0 < split

    res = {'implementation_sha256': hashlib.sha256(
               Path(__file__).read_bytes()).hexdigest(),
           'calendar_fix': {'what': 'год считался делением на 365; заменён '
                                    'календарным преобразованием',
                            'timezone': 'America/New_York',
                            'mislabelled_minutes_share_2020_2025': 0.0287,
                            'affects': 'только годовые и квартальные подписи; '
                                       'разделение выборок использовало дату'},
           'minute_label': {'established_on_data': 'метка минуты = КОНЕЦ интервала',
                            'evidence': 'средний размах по минуте ET за 1494 дня: '
                                        'всплеск на метке 08:31 (23,40) против '
                                        '08:30 (7,01) и фона 5,7; у 14:00 ET '
                                        'всплеск на метке 14:01',
                            'consequence': 'вход по open свечи с меткой слот+1 '
                                           'есть цена В МОМЕНТ публикации'},
           'territory': {**corpus.metadata['source'], 'films': int(corpus.n)},
           'context_definition': {
               'slots_et': list(SLOTS), 'tolerance_minutes': a.tol,
               'declared_before_comparison': True,
               'rule': 'входная свеча (T0+11) покрывает слот',
               'calendar_coverage': 'НЕИЗВЕСТНО, какие даты несли реальную '
                                    'публикацию; слот есть каждый торговый день',
               'event_unit': 'дата + слот'},
           'support': {}, 'comparison': {}, 'execution_sensitivity': {}}

    for cname, cmask in ctx.items():
        res['support'][cname] = {
            'films': int((cmask & forms['полная конструкция']).sum()),
            'films_any': int(cmask.sum()),
            'days': int(len(np.unique(date_key(
                t0[cmask & forms['полная конструкция']])))) if
            (cmask & forms['полная конструкция']).any() else 0,
            'events': int(len(np.unique(
                date_key(t0[cmask & forms['полная конструкция']]) * 10000
                + np.maximum(slot_id[cmask & forms['полная конструкция']], 0))))
            if (cmask & forms['полная конструкция']).any() else 0}

    side = -1                       # сторона зафиксирована в S-13: продажа
    for cname, cmask in ctx.items():
        block = {}
        for fname, fmask in forms.items():
            per_hold = {}
            for h in a.holds:
                sel_tr = fmask & cmask & train
                sel_ev = fmask & cmask & ~train
                per_hold[f'hold_{h}'] = {
                    'выдвижение': summarise(stream(corpus, t0, sel_tr, side, h,
                                                   a.instrument, slot_id), cost),
                    'оценка': summarise(stream(corpus, t0, sel_ev, side, h,
                                               a.instrument, slot_id), cost)}
            block[fname] = per_hold
        res['comparison'][cname] = block

    # чувствительность исполнения: слоты исполняются хуже объявленной модели
    sel_all = forms['полная конструкция']
    for slip in a.slip_scenarios:
        tr = stream(corpus, t0, sel_all & train, side, 5, a.instrument,
                    slot_id, extra_slip=slip)
        tr2 = stream(corpus, t0, sel_all & ~train, side, 5, a.instrument,
                     slot_id, extra_slip=slip)
        res['execution_sensitivity'][f'+${slip:.0f} на входах в слоте'] = {
            'выдвижение': summarise(tr, cost), 'оценка': summarise(tr2, cost)}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1,
                                      default=str), encoding='utf-8')
    brief = {'support': res['support'],
             'comparison': {c: {f: {h: {k: (round(v['mean_gross'], 1),
                                           v['n'], v.get('events'))
                                       for k, v in d.items() if v['n']}
                                   for h, d in hh.items()}
                                for f, hh in b.items()}
                            for c, b in res['comparison'].items()}}
    print(json.dumps(brief, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
