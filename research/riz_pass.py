#!/usr/bin/env python3
"""Ограниченный проход по фильмам RIZ языком порядковых событий.

ПОПУЛЯЦИЯ И ГРАНИЦЫ — НАЗВАНЫ ДО СЧЁТА
======================================
NQ, паспорта RIZ с каноническим T0, ТФ 5, сторона выхода объявляется явно и
не смешивается. Территория 2020-01-01 … 2025-11-01. Единица наблюдения —
ценовое окно `NQ:T0`: повторные риз-строки одного и того же T0 схлопываются
ядром, все исходные `riz_id` сохраняются. Окно наблюдения — ОБЪЯВЛЕННЫЙ бюджет
T0 … T0+30 минут, а не горизонт сделки; событийного обрезания нет.

ЧТО СПРАШИВАЕМ
==============
1. Повторяемость: какие порядковые конструкции устойчиво встречаются в фильме.
2. Разделение: взять объявленное ЗАРАНЕЕ продолжение, найти среди отношений,
   доступных к явно названному курсору, лучшее различие, и тут же повторить
   весь этот выбор при разрушенной связи.

Знаменатель вероятности продолжения — ВСЕ единицы, достигшие того же
наблюдаемого префикса, включая те, где продолжение не наступило. Неизвестные
исходы показываются отдельно и в ноль не превращаются.

Три утверждения не смешиваются: конструкция повторяется; конструкция специфична
для якоря; конструкция помогает предсказать продолжение. Калибруется третье.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from relational_stencil import Corpus                              # noqa: E402
from ordinal_events import (NONE, UNKNOWN, CompactJournal, Lens, VERSION,  # noqa: E402
                            calibrate, crossing_event, dissect, expand,
                            extremum_event, last_update_event, negative_space,
                            survey)

SEGMENTS = ((1, 5), (1, 10), (1, 30))
BUDGET_END = 30


def build_events(corpus):
    """Явно названные участки. Ни один адрес заранее не назначен."""
    ev = []
    for a, b in SEGMENTS:
        for field in ('H', 'L'):
            ev.append(extremum_event(corpus, a, b, field, False))
            ev.append(extremum_event(corpus, a, b, field, True))
            ev.append(last_update_event(corpus, a, b, field))
    for field, level_field, above in (('H', 'H', True), ('C', 'H', True),
                                      ('L', 'L', False), ('C', 'L', False)):
        ev.append(crossing_event(corpus, 1, BUDGET_END, field, (0, level_field), above))
    return ev


CONTINUATIONS = {
    'close_below_L0': ('C', 'L', False),
    'close_above_H0': ('C', 'H', True),
    'wick_below_L0': ('L', 'L', False),
    'wick_above_H0': ('H', 'H', True),
}


def subcorpus(corpus, keep):
    """Подкорпус с сохранением происхождения: все исходные riz_id остаются."""
    sub = Corpus(corpus.ohlc[keep], corpus.ordinals,
                 [corpus.unit_ids[i] for i in keep],
                 [corpus.unit_ids[i] for i in keep], dict(corpus.metadata))
    sub.members = [corpus.members[i] for i in keep]
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--cursor', type=int, default=5)
    ap.add_argument('--continuation', required=True,
                    help='имя события-продолжения, объявляется до поиска')
    ap.add_argument('--repeats', type=int, default=20)
    ap.add_argument('--top', type=int, default=15)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    corpus = Corpus.load(a.input)
    events = build_events(corpus)
    lens = Lens(corpus, events)

    value_order = [i for i, d in enumerate(lens.atoms)
                   if d['kind'] in ('event_value', 'event_order')]
    distances = [i for i, d in enumerate(lens.atoms) if d['kind'] == 'event_distance']

    # --- 1. повторяемость: одиночные факты полностью, пары по значениям и порядку
    single, marks = survey(lens, sizes=(1,), budget=10 ** 9, mode='exact', seed=11,
                           top=a.top, atom_pool=list(range(len(lens.atoms))))
    pair_journal = CompactJournal()
    pairs, _ = survey(lens, sizes=(2,), budget=10 ** 9, mode='exact', seed=11,
                      top=a.top, atom_pool=value_order, journal=pair_journal)
    pair_journal.save(out / 'pairs_journal.npz',
                      {'version': VERSION, 'seed': 11, 'sizes': [2],
                       'units': lens.n, 'by_size': pairs['by_size'],
                       'atom_pool': 'event_value + event_order',
                       'input': corpus.metadata})

    # --- 2. разделение ЕЩЁ НЕ ПРОИСШЕДШЕГО продолжения
    # Окно продолжения начинается ПОСЛЕ курсора, и из популяции убираются
    # единицы, где событие уже случилось к курсору: иначе «продолжение»
    # частью уже наблюдалось и разделение было бы круговым.
    field, level_field, above = CONTINUATIONS[a.continuation]
    already = crossing_event(corpus, 1, a.cursor, field, (0, level_field), above)
    after = crossing_event(corpus, a.cursor + 1, BUDGET_END, field, (0, level_field), above)
    not_yet = (already.values == NONE)
    usable = not_yet & (after.values != UNKNOWN)
    keep = np.flatnonzero(usable)
    if len(keep) == 0:
        raise SystemExit('после курсора не осталось единиц с известным продолжением')
    sub = subcorpus(corpus, keep)
    sub_events = build_events(sub)
    sub_lens = Lens(sub, sub_events)
    sub_after = crossing_event(sub, a.cursor + 1, BUDGET_END, field, (0, level_field), above)
    labels = sub_after.values != NONE
    label_known = sub_after.values != UNKNOWN
    # Из префикса исключены атомы, ссылающиеся на само событие продолжения.
    forbidden = {e.name for e in sub_events
                 if e.kind in (f'{"wick" if field in ("H", "L") else "close"}'
                               f'_{"above" if above else "below"}',)}
    prefix_pool = [i for i, d in enumerate(sub_lens.atoms)
                   if d['known_at'] <= a.cursor
                   and not (set(d.get('events', [])) & forbidden)]
    split, _ = survey(sub_lens, sizes=(1, 2), budget=10 ** 9, mode='exact', seed=11,
                      top=a.top, labels=labels, atom_pool=prefix_pool)
    cal = calibrate(sub_lens, labels=labels, repeats=a.repeats, sizes=(1, 2),
                    budget=10 ** 9, mode='exact', seed=11, top=a.top,
                    atom_pool=prefix_pool, exchangeability_justified=False)

    best = split['views']['more_in_group'][0] if split['views']['more_in_group'] else None
    detail = None
    if best is not None:
        candidate = tuple(best['atoms'])
        truth, known = sub_lens.intersection(candidate)
        cont_mask = int.from_bytes(np.packbits(labels & label_known,
                                               bitorder='little').tobytes(), 'little')
        known_cont = int.from_bytes(np.packbits(label_known, bitorder='little').tobytes(),
                                    'little')
        reached = truth & known_cont
        others = known & ~truth & known_cont & sub_lens.all
        def rate(mask):
            n = mask.bit_count()
            hit = (cont_mask & mask).bit_count()
            return {'known': n, 'count': hit, 'frequency': hit / n if n else None}
        # что ещё остаётся после курсора: адрес экстремума полного окна
        full_max = next(x for x in sub_lens.events
                        if x.name == f'argmaxH_first[1..{BUDGET_END}]')
        support = [i for i in range(sub_lens.n) if (truth >> i) & 1]
        addresses = full_max.values[support]
        ahead = addresses[addresses >= 0]
        detail = {
            'construction': [sub_lens.describe(x) for x in candidate],
            'recognisable_at_minute': max(sub_lens.describe(x)['known_at']
                                          for x in candidate),
            'observed_now': rate(reached),
            'same_cursor_construction_absent': rate(others),
            'prefix_true_but_continuation_unknown':
                (truth & ~known_cont & sub_lens.all).bit_count(),
            'still_ahead_extremum_address': {
                'event': full_max.name,
                'median': float(np.median(ahead)) if len(ahead) else None,
                'quartiles': [float(v) for v in np.percentile(ahead, [25, 75])]
                if len(ahead) else None,
                'note': 'сколько ещё времени фильма впереди, в свечах; '
                        'величина хода здесь не измеряется'},
            'dissection': dissect(sub_lens, candidate),
            'example_expansion': expand(sub_lens, candidate, support[0])
            if support else None}

    result = {
        'version': VERSION,
        'implementation_sha256': hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        'input': corpus.metadata,
        'units': lens.n, 'input_rows': corpus.input_rows,
        'budget_window': [0, BUDGET_END],
        'segments_declared': [list(s) for s in SEGMENTS],
        'events': [e.describe() for e in events],
        'atom_counts': {'event_value_or_order': len(value_order),
                        'event_distance': len(distances),
                        'relation_atoms_available_for_expansion': len(lens.space.masks)},
        'repeatability_singles': single,
        'repeatability_pairs': pairs,
        'pair_pool_note': 'пары перебраны по значениям и порядку; расстояния '
                          'перебраны полностью как одиночные факты',
        'declared_continuation': a.continuation,
        'continuation_window': [a.cursor + 1, BUDGET_END],
        'population_after_cursor': int(len(keep)),
        'dropped_already_happened_by_cursor': int((~not_yet).sum()),
        'continuation_known_units': int(label_known.sum()),
        'continuation_happened_units': int((labels & label_known).sum()),
        'cursor': a.cursor,
        'circularity_guards': ['окно продолжения начинается после курсора',
                               'единицы с уже случившимся событием исключены',
                               'атомы самого события продолжения удалены из префикса'],
        'prefix_atoms_available_at_cursor': len(prefix_pool),
        'separation': split,
        'calibration': cal,
        'best_detail': detail,
        'claims_kept_apart': {
            'repeats': 'витрина frequent',
            'anchor_specific': 'НЕ проверено этим проходом: нужен корпус '
                               'сопоставимых минут без якоря',
            'predicts_continuation': 'витрина more_in_group плюс calibration'},
        'denominator_note': 'вероятность продолжения считается среди всех единиц, '
                            'достигших того же префикса; неизвестные исходы отдельно'}
    (out / 'result.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(json.dumps({
        'out': str(out.resolve()), 'units': lens.n,
        'event_atoms': len(lens.atoms),
        'pairs_tested': pairs['tested'], 'pairs_complete': pairs['complete'],
        'separation_tested': split['tested'],
        'best_separation': best['display_value'] if best else None,
        'calibration_status': cal['status'],
        'calibration_best': cal['best_under_broken_link']['more_in_group']},
        ensure_ascii=False))


if __name__ == '__main__':
    main()
