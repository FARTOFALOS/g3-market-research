#!/usr/bin/env python3
"""Специфичен ли якорь: та же конструкция на обычных минутах ленты.

Конструкция и продолжение здесь УЖЕ ЗАФИКСИРОВАНЫ проходом по RIZ и никакому
подбору не подвергаются. Поэтому множественности нет и калибровка перебора не
нужна: измеряется одно заранее названное отношение на другой популяции.

Контрольные минуты берутся из той же ленты, той же территории и с ТОЙ ЖЕ
раскладкой по времени суток, что и T0 популяции RIZ: иначе разница объяснялась
бы внутридневным профилем, а не якорем. Стороны не смешиваются: для каждой
стороны берётся её собственная конструкция и её собственное продолжение.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from relational_stencil import Corpus                        # noqa: E402
from ordinal_events import (NONE, UNKNOWN, crossing_event,   # noqa: E402
                            extremum_event, last_update_event)

ROOT = Path(__file__).resolve().parents[1]
MIN = 60_000_000_000
BUDGET_END = 30


def tape(instrument):
    m = {k: np.load(ROOT / 'data/market' / instrument / (k + '.npy'))
         for k in ('close_ts_utc_ns', 'open', 'high', 'low', 'close')}
    return m


def window_corpus(m, positions, ids, meta):
    ts = m['close_ts_utc_ns']
    ords = np.arange(0, BUDGET_END + 1, dtype=np.int64)
    films = np.full((len(positions), len(ords), 4), np.nan)
    for i, p in enumerate(positions):
        want = ts[p] + ords * MIN
        at = np.searchsorted(ts, want)
        ok = at < len(ts)
        good = ok.copy()
        good[ok] &= ts[at[ok]] == want[ok]
        for f, name in enumerate(('open', 'high', 'low', 'close')):
            films[i, good, f] = m[name][at[good]]
    return Corpus(films, ords, ids, list(ids), meta)


def construction_mask(corpus, side):
    """Зафиксированные проходом конструкции. Ничего не подбирается."""
    upd_h = last_update_event(corpus, 1, 5, 'H')
    upd_l = last_update_event(corpus, 1, 5, 'L')
    max_last = extremum_event(corpus, 1, 5, 'H', True)
    min_last = extremum_event(corpus, 1, 5, 'L', True)
    if side == 'south':
        a, b = upd_h, min_last
        known = (a.values >= 0) & (b.values >= 0)
        return known & (a.values < b.values), known, 'lastupdateH[1..5] < argminL_last[1..5]'
    a, b = max_last, upd_l
    known = (a.values >= 0) & (b.values >= 0)
    return known & (a.values > b.values), known, 'argmaxH_last[1..5] > lastupdateL[1..5]'


def measure(corpus, side, cursor=5):
    field, level_field, above = (('C', 'L', False) if side == 'south'
                                 else ('C', 'H', True))
    already = crossing_event(corpus, 1, cursor, field, (0, level_field), above)
    after = crossing_event(corpus, cursor + 1, BUDGET_END, field, (0, level_field), above)
    usable = (already.values == NONE) & (after.values != UNKNOWN)
    happened = after.values != NONE
    truth, known, text = construction_mask(corpus, side)
    with_it = usable & known & truth
    without = usable & known & ~truth
    def rate(mask):
        n = int(mask.sum())
        hit = int((mask & happened).sum())
        return {'known': n, 'count': hit, 'frequency': hit / n if n else None}
    return {'construction': text,
            'continuation': f'close_{"below_L0" if side == "south" else "above_H0"}'
                            f' in [{cursor + 1}..{BUDGET_END}]',
            'population_after_cursor': int(usable.sum()),
            'dropped_already_happened': int((already.values != NONE).sum()),
            'with_construction': rate(with_it), 'without_construction': rate(without)}


def sample_control(riz, instrument, controls_per_anchor, seed):
    """Контрольные минуты той же ленты с ТОЙ ЖЕ раскладкой по времени суток.

    Выбор контроля и сравниваемые обстоятельства фиксируются ДО чтения его
    результата: раскладка по минуте суток, та же территория, сами якоря
    исключены. Адаптивного подбора здесь нет и быть не должно.
    """
    m = tape(instrument)
    ts = m['close_ts_utc_ns']
    anchors = np.array([int(u.split(':')[1]) for u in riz.unit_ids], dtype=np.int64)
    pos = np.searchsorted(ts, anchors)
    assert np.all(ts[pos] == anchors), 'якоря должны лежать на ленте'
    lo = int(np.searchsorted(ts, anchors.min()))
    hi = int(np.searchsorted(ts, anchors.max()))
    minute_of_day = ((ts // MIN) % 1440).astype(np.int64)
    want = minute_of_day[pos]
    rng = np.random.default_rng(seed)
    taken = set(int(p) for p in pos)
    picks = []
    for mod, need in zip(*np.unique(want, return_counts=True)):
        pool = np.flatnonzero(minute_of_day[lo:hi] == int(mod)) + lo
        pool = np.array([p for p in pool if int(p) not in taken], dtype=np.int64)
        if len(pool) == 0:
            continue
        picks.append(rng.choice(pool, min(len(pool), int(need) * controls_per_anchor),
                                replace=False))
    control_pos = np.unique(np.concatenate(picks))
    ids = [f'{instrument}:{int(ts[p])}' for p in control_pos]
    return window_corpus(m, control_pos, ids, {
        'instrument': instrument,
        'source': {'tape': f'data/market/{instrument}', 'seed': seed,
                   'matched_on': 'раскладка по минуте суток у якорей RIZ',
                   'excluded': 'сами якоря RIZ этой популяции',
                   'territory': [int(anchors.min()), int(anchors.max())],
                   'fixed_before_reading_result': True},
        'anchor': 'обычная минута ленты, не T0',
        'unit_definition': 'ценовое окно контрольной минуты'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instrument', default='NQ')
    ap.add_argument('--riz-input', required=True)
    ap.add_argument('--side', choices=['north', 'south'], required=True)
    ap.add_argument('--controls-per-anchor', type=int, default=8)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    riz = Corpus.load(a.riz_input)
    control = sample_control(riz, a.instrument, a.controls_per_anchor, a.seed)

    result = {'side': a.side,
              'riz': {'input': riz.metadata.get('input_path'), 'units': riz.n,
                      **measure(riz, a.side)},
              'control': {'units': control.n, **measure(control, a.side)},
              'claim_under_test': 'специфична ли конструкция для якоря T0',
              'no_search_here': 'конструкция и продолжение зафиксированы ранее; '
                                'множественности и калибровки перебора нет'}
    rz, ct = result['riz'], result['control']
    for block in (rz, ct):
        w, o = block['with_construction'], block['without_construction']
        block['lift'] = (w['frequency'] - o['frequency']
                         if w['frequency'] is not None and o['frequency'] is not None
                         else None)
    result['lift_difference_riz_minus_control'] = (
        rz['lift'] - ct['lift'] if rz['lift'] is not None and ct['lift'] is not None
        else None)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=1),
                           encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
