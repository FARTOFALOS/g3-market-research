#!/usr/bin/env python3
"""Один предметный цикл: что добавляет история сверх текущего положения цены.

ВОПРОС, ОБЪЯВЛЕННЫЙ ДО СЧЁТА
============================
У зафиксированных конструкций пятой минуты отличается ли дальнейшее устройство
движения ПОСЛЕ учёта доступного порядкового положения — или различие
исчерпывается достижением ранее выбранного уровня?

Две альтернативы разводятся так:
  * история первых минут содержит различие в последующем развитии;
  * история лишь сообщает, где цена уже находится относительно выбранной цели.

Конструкции зафиксированы предыдущим проходом и здесь НЕ ищутся:
  юг    lastupdateH[1..5] < argminL_last[1..5]
  север argmaxH_last[1..5] > lastupdateL[1..5]
Новых сочетаний не перебирается, поэтому множественности и калибровки выбора
в этом файле нет.

ЧТО КОНТРОЛЬ ДЕЛАЕТ И ЧЕГО НЕ ДЕЛАЕТ
====================================
Сравнение идёт внутри сопоставимых ПОРЯДКОВЫХ состояний на минуте решения:
положение закрытия относительно доступных опор и уже совершённые прохождения
этих опор. Это уравнивает доступное порядковое положение.

Метрическую близость к уровню оно НЕ уравнивает и не обещает: одинаковый
порядок цен совместим с разными расстояниями между ними. Порядковый прибор
не способен полностью устранить объяснение близостью — это предел языка,
а не недоделка замера. Величины сюда не возвращаются ни под каким названием.

Страта, где встретилась только одна группа, — отсутствие подходящего сравнения,
а не отрицательный результат. Такие страты перечисляются отдельно.

БУДУЩЕЕ СНОВА ФИЛЬМ
===================
Одна метка достижения уровня сохраняется как один конкретный вопрос, но
универсальным определением продолжения не является. Рядом разворачивается
устройство продолжения существующими операторами: прохождения тенью, закрытия
за опорами, адреса экстремумов и последних обновлений. Для порядка двух
событий рядом всегда стоит полная картина популяции.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from relational_stencil import Corpus                                # noqa: E402
from anchor_control import sample_control                            # noqa: E402
from ordinal_events import (LEVELS, NONE, UNKNOWN, Contract, VERSION,  # noqa: E402
                            apply_contract, crossing_event, event_profile,
                            extremum_event, last_update_event, order_picture,
                            resolve_level, state_key, stratified_compare,
                            unfold_future)

CURSOR, HORIZON = 5, 30

FIXED = {
    'south': {'text': 'lastupdateH[1..5] < argminL_last[1..5]',
              'level': 'L0', 'field': 'C', 'above': False},
    'north': {'text': 'argmaxH_last[1..5] > lastupdateL[1..5]',
              'level': 'H0', 'field': 'C', 'above': True},
}


def construction(corpus, side):
    """Зафиксированная конструкция. Ничего не подбирается и не расширяется."""
    if side == 'south':
        a = last_update_event(corpus, 1, CURSOR, 'H')
        b = extremum_event(corpus, 1, CURSOR, 'L', True)
        known = (a.values >= 0) & (b.values >= 0)
        return known & (a.values < b.values), known
    a = extremum_event(corpus, 1, CURSOR, 'H', True)
    b = last_update_event(corpus, 1, CURSOR, 'L')
    known = (a.values >= 0) & (b.values >= 0)
    return known & (a.values > b.values), known


def declared_pairs(events, side):
    """Пары для картины порядка объявлены ДО чтения результата."""
    by = {e.name: e for e in events}
    goal = 'L0' if side == 'south' else 'H0'
    against = 'maxH_prefix' if side == 'south' else 'minL_prefix'
    goal_lvl = f'{goal[0]}[0]'
    against_lvl = f'{"max" if side == "south" else "min"}{against[3]}[1..{CURSOR}]'
    want = [
        (f'close_{"below" if side == "south" else "above"}({goal_lvl})'
         f'[{CURSOR + 1}..{HORIZON}]',
         f'close_{"above" if side == "south" else "below"}({against_lvl})'
         f'[{CURSOR + 1}..{HORIZON}]'),
        (f'wick_below(minL[1..{CURSOR}])[{CURSOR + 1}..{HORIZON}]',
         f'wick_above(maxH[1..{CURSOR}])[{CURSOR + 1}..{HORIZON}]'),
    ]
    out = []
    for a, b in want:
        if a in by and b in by:
            out.append((by[a], by[b]))
    return out


def analyse(corpus, side, label):
    spec = FIXED[side]
    contract = Contract(cursor=CURSOR, horizon=HORIZON, side=side,
                        field=spec['field'], level=spec['level'],
                        above=spec['above'], achievement='first_ever')
    applied = apply_contract(corpus, contract)
    group, group_known = construction(corpus, side)
    usable, labels = applied['usable'], applied['labels']

    raw_with = usable & group_known & group
    raw_without = usable & group_known & ~group
    def rate(mask):
        n = int(mask.sum())
        return {'n': n, 'hit': int((mask & labels).sum()),
                'frequency': int((mask & labels).sum()) / n if n else None}

    keys, key_known, key_names = state_key(corpus, CURSOR)
    strat = stratified_compare(keys, key_known, group, group_known, labels, usable)

    future = unfold_future(corpus, CURSOR, HORIZON)
    profile_split = event_profile(future, usable & group_known, group)
    pictures = [order_picture(a, b, usable & group_known)
                for a, b in declared_pairs(future, side)]
    pictures_split = []
    for a, b in declared_pairs(future, side):
        pictures_split.append({
            'with_construction': order_picture(a, b, raw_with),
            'without_construction': order_picture(a, b, raw_without)})

    return {
        'population_label': label, 'units': int(corpus.n),
        'contract': applied['contract'], 'level': applied['level_name'],
        'audit': applied['audit'],
        'construction': spec['text'],
        'unstratified': {'with': rate(raw_with), 'without': rate(raw_without),
                         'difference': (rate(raw_with)['frequency'] -
                                        rate(raw_without)['frequency']
                                        if rate(raw_with)['frequency'] is not None
                                        and rate(raw_without)['frequency'] is not None
                                        else None)},
        'state_description': key_names,
        'stratified': strat,
        'future_event_profile': profile_split,
        'order_pictures_whole_population': pictures,
        'order_pictures_split': pictures_split,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--riz-input', required=True)
    ap.add_argument('--side', choices=['north', 'south'], required=True)
    ap.add_argument('--instrument', default='NQ')
    ap.add_argument('--controls-per-anchor', type=int, default=8)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    riz = Corpus.load(a.riz_input)
    control = sample_control(riz, a.instrument, a.controls_per_anchor, a.seed)
    result = {
        'version': VERSION,
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'question': 'добавляет ли история первых минут сверх доступного '
                    'порядкового положения на минуте решения',
        'constructions_are_fixed': FIXED,
        'no_search_performed': True,
        'riz': analyse(riz, a.side, 'RIZ T0'),
        'control': analyse(control, a.side, 'обычные минуты, раскладка по времени суток'),
        'control_metadata': control.metadata,
        'limits': {
            'controls': 'доступное порядковое положение на минуте решения',
            'does_not_control': 'метрическую близость к уровню',
            'ordinal_language_limit': 'одинаковый порядок цен совместим с разными '
                                      'расстояниями; полностью устранить объяснение '
                                      'близостью порядковый прибор не может',
            'economics': 'экономическая проверка здесь не проводится и находку '
                         'задним числом не переписывает'}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str),
                   encoding='utf-8')
    brief = {}
    for key in ('riz', 'control'):
        b = result[key]
        brief[key] = {
            'units_usable': b['audit']['usable'],
            'unstratified_difference': b['unstratified']['difference'],
            'pooled_difference_within_state': b['stratified']['pooled_difference'],
            'coverage': b['stratified']['coverage'],
            'strata_compared': len(b['stratified']['strata_compared']),
            'strata_thin': len(b['stratified']['strata_thin']),
            'strata_no_comparison': len(b['stratified']['strata_no_comparison'])}
    print(json.dumps({'out': str(out.resolve()), 'side': a.side, **brief},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
