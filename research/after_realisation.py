#!/usr/bin/env python3
"""Что происходит ПОСЛЕ исполнения ближайшего ожидания и когда это различимо.

ПРЕДМЕТ
=======
Не «чем отличались длинные прибыльные ходы» — зная конец, отличия описываются
почти всегда. Предмет: **узнаваемое начало ещё не завершённой формы**. Нужно
установить, КОГДА различие продолжений становится наблюдаемым по закрытым
свечам и что после этого момента ещё остаётся впереди.

СМЫСЛОВАЯ ПОПРАВКА, ПРИНЯТАЯ ЗДЕСЬ
==================================
Будущая свеча не отменяет исторический порядок экстремумов. Она может
опровергнуть ожидание, выведенное из этого порядка. Поэтому определение
исторической конструкции и основание прекращать действие живут раздельно:
конструкция первых минут остаётся истинной навсегда, а «закрытие за
противоположным краем» — это отмена ОЖИДАНИЯ, не отмена конструкции.

ЯКОРЬ И ЧАСЫ
============
Якорь этого прохода — минута исполнения ближайшего ожидания (реализации):
первое закрытие за краем первых пяти минут в сторону конструкции. Локальные
часы: m = 0 — минута реализации, m > 0 — после неё. Исходное T0 и абсолютный
адрес реализации сохраняются в идентификаторе каждой единицы, поэтому связь с
исходной нумерацией не теряется.

ЗНАМЕНАТЕЛЬ
===========
Форма ищется в завершённых фильмах, но доля продолжений считается по ВСЕМ
случаям, где присутствовало её наблюдаемое начало, включая другое продолжение
и незавершённое наблюдение. Иначе ответ замкнётся: фильмы отобраны по
известному завершению, и завершение у них же обнаружено.

ПОКРЫТИЕ
========
Прибор просматривает совместные конфигурации из объявленного словаря событий
и отношений. Что именно просмотрено и что осталось вне вычислительной
территории, печатается в результат: полная матрица отношений на 21 свече даёт
3 486 пар точек и порядка 10 тысяч атомов, парный перебор по ним — 5·10⁷
сочетаний, поэтому парный поиск ведётся по событиям, а отношения остаются для
раскрытия и для одиночного просмотра. Это ограничение покрытия, а не знание.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from relational_stencil import Corpus, Space                       # noqa: E402
from anchor_control import sample_control, window_corpus, tape     # noqa: E402
from candidate_check import CURSOR, HORIZON, POINT, COST, sides    # noqa: E402
from ordinal_events import (LEVELS, NONE, UNKNOWN, Lens, VERSION,  # noqa: E402
                            calibrate, crossing_event, dissect, event_profile,
                            expand, extremum_event, last_update_event,
                            order_picture, point_level, prefix_extreme_level,
                            survey)

MIN = 60_000_000_000
BACK, AHEAD = 5, 15          # окно вокруг минуты реализации


def realisation_anchors(anchors, side, m):
    """Минута исполнения ближайшего ожидания. Возвращает абсолютные адреса."""
    ts = m['close_ts_utc_ns']
    pos = np.searchsorted(ts, anchors)
    out = []
    for i in range(len(anchors)):
        if side[i] == 0:
            continue
        p = int(pos[i])
        want = ts[p] + np.arange(0, HORIZON + 1) * MIN
        at = np.searchsorted(ts, want)
        if at[-1] >= len(ts) or not np.all(ts[at] == want):
            continue
        hi, lo, cl = m['high'][at], m['low'][at], m['close'][at]
        s = int(side[i])
        edge = float(lo[1:CURSOR + 1].min()) if s < 0 else float(hi[1:CURSOR + 1].max())
        fut = np.arange(CURSOR + 1, HORIZON + 1)
        hit = (cl[fut] < edge) if s < 0 else (cl[fut] > edge)
        if not hit.any():
            continue
        k = int(np.argmax(hit))
        r = int(at[fut[k]])
        out.append({'t0_index': p, 't0_ts': int(ts[p]), 'r_index': r,
                    'minutes_from_decision': k + 1, 'side': s, 'edge': edge})
    return out


def build_corpus(rows, m, instrument, label):
    """Корпус, заякоренный на минуте реализации. T0 сохраняется в адресе."""
    ts = m['close_ts_utc_ns']
    ords = np.arange(-BACK, AHEAD + 1, dtype=np.int64)
    films, ids, keep = [], [], []
    for row in rows:
        want = ts[row['r_index']] + ords * MIN
        at = np.searchsorted(ts, want)
        ok = at < len(ts)
        good = ok.copy()
        good[ok] &= ts[at[ok]] == want[ok]
        film = np.full((len(ords), 4), np.nan)
        for f, name in enumerate(('open', 'high', 'low', 'close')):
            film[good, f] = m[name][at[good]]
        films.append(film)
        ids.append(f'{instrument}:{row["t0_ts"]}:r+{row["minutes_from_decision"]}')
        keep.append(row)
    films = np.asarray(films)
    corpus = Corpus(films, ords, ids, list(ids), {
        'instrument': instrument,
        'source': {'population': label, 'window_around_realisation': [-BACK, AHEAD],
                   'realisation': 'первое закрытие за краем первых пяти минут '
                                  'в сторону конструкции',
                   'origin': 'T0 и локальный сдвиг сохранены в адресе единицы'},
        'anchor': 'минута исполнения ближайшего ожидания; m=0 — она сама',
        'unit_definition': 'один фильм, первая реализация'})
    return corpus, keep


def orient(corpus, keep):
    """Север и юг не смешиваются: юг отражается в общую систему «по ходу».

    Отражение — операция над ЦЕНАМИ, а не над порядком; порядковые отношения
    при нём сохраняются, а стороны становятся сравнимыми. Это объявлено, а не
    сделано молча.
    """
    x = corpus.ohlc.copy()
    side = np.array([r['side'] for r in keep])
    flip = side < 0
    if flip.any():
        o, h, l, c = (x[flip, :, k].copy() for k in range(4))
        x[flip, :, 0] = -o
        x[flip, :, 1] = -l
        x[flip, :, 2] = -h
        x[flip, :, 3] = -c
    meta = dict(corpus.metadata)
    meta['source'] = {**meta['source'],
                      'orientation': 'юг отражён по цене; «вверх» = по ходу конструкции'}
    out = Corpus(x, corpus.ordinals, list(corpus.ids), list(corpus.unit_ids), meta)
    out.members = list(corpus.members)
    return out


def build_events(corpus, cursor):
    """Словарь событий после реализации. Опоры доступны к m=cursor."""
    ev = []
    pre_hi = prefix_extreme_level(corpus, -BACK, 0, 'H')
    pre_lo = prefix_extreme_level(corpus, -BACK, 0, 'L')
    c0 = point_level(corpus, 0, 'C')
    levels = {'maxH_pre': pre_hi, 'minL_pre': pre_lo, 'C0': c0}
    if cursor > 0:
        levels['maxH_cur'] = prefix_extreme_level(corpus, 1, cursor, 'H')
        levels['minL_cur'] = prefix_extreme_level(corpus, 1, cursor, 'L')
    for name, lvl in levels.items():
        for field, above in (('H', True), ('L', False)):
            ev.append(crossing_event(corpus, cursor + 1, AHEAD, field, lvl, above))
        for above in (True, False):
            ev.append(crossing_event(corpus, cursor + 1, AHEAD, 'C', lvl, above))
    for field in ('H', 'L'):
        ev.append(extremum_event(corpus, cursor + 1, AHEAD, field, False))
        ev.append(last_update_event(corpus, cursor + 1, AHEAD, field))
    return ev, levels


def observable_events(corpus, cursor):
    """События, целиком закрытые к m=cursor. Ничего из будущего."""
    if cursor < 1:
        return []
    ev = []
    pre_hi = prefix_extreme_level(corpus, -BACK, 0, 'H')
    pre_lo = prefix_extreme_level(corpus, -BACK, 0, 'L')
    c0 = point_level(corpus, 0, 'C')
    for lvl in (pre_hi, pre_lo, c0):
        for field, above in (('H', True), ('L', False)):
            ev.append(crossing_event(corpus, 1, cursor, field, lvl, above))
        for above in (True, False):
            ev.append(crossing_event(corpus, 1, cursor, 'C', lvl, above))
    for field in ('H', 'L'):
        ev.append(extremum_event(corpus, 1, cursor, field, False))
        ev.append(extremum_event(corpus, 1, cursor, field, True))
        ev.append(last_update_event(corpus, 1, cursor, field))
    return ev


def continuation_classes(corpus, cursor):
    """Альтернативные продолжения. Имена даны ПОСЛЕ восстановления отношений.

    Все три считаются на окне строго после курсора и против опор, доступных
    к нему. Случаи без выраженного продолжения сохраняются отдельным классом.
    """
    hi_cur = prefix_extreme_level(corpus, -BACK, cursor, 'H')
    lo_cur = prefix_extreme_level(corpus, -BACK, cursor, 'L')
    extend = crossing_event(corpus, cursor + 1, AHEAD, 'C', hi_cur, True)
    give_back = crossing_event(corpus, cursor + 1, AHEAD, 'C', lo_cur, False)
    known = (extend.values != UNKNOWN) & (give_back.values != UNKNOWN)
    e, g = extend.values >= 0, give_back.values >= 0
    first_extend = known & e & (~g | (extend.values < give_back.values))
    first_give = known & g & (~e | (give_back.values < extend.values))
    neither = known & ~e & ~g
    tie = known & e & g & (extend.values == give_back.values)
    return {'extend_first': first_extend & ~tie, 'give_back_first': first_give & ~tie,
            'tie_same_minute': tie, 'neither': neither, 'known': known,
            'events': (extend, give_back)}


def describe_population(corpus, cursor, classes):
    known = classes['known']
    n = corpus.n
    return {'units': n, 'cursor': cursor,
            'known_outcome': int(known.sum()),
            'unfinished_observation': int((~known).sum()),
            'extend_first': int(classes['extend_first'].sum()),
            'give_back_first': int(classes['give_back_first'].sum()),
            'tie_same_minute': int(classes['tie_same_minute'].sum()),
            'no_pronounced_continuation': int(classes['neither'].sum()),
            'share_extend': float(classes['extend_first'].sum() / known.sum())
            if known.sum() else None,
            'timing': {
                'extend_median_minute': float(np.median(
                    classes['events'][0].values[classes['extend_first']]))
                if classes['extend_first'].any() else None,
                'give_back_median_minute': float(np.median(
                    classes['events'][1].values[classes['give_back_first']]))
                if classes['give_back_first'].any() else None}}


def subcorpus(corpus, keep_idx, keep_rows):
    sub = Corpus(corpus.ohlc[keep_idx], corpus.ordinals,
                 [corpus.unit_ids[i] for i in keep_idx],
                 [corpus.unit_ids[i] for i in keep_idx], dict(corpus.metadata))
    sub.members = [corpus.members[i] for i in keep_idx]
    return sub, [keep_rows[i] for i in keep_idx]


def stage_search(corpus, rows, cursor, repeats, top, seed=17):
    """Различимо ли продолжение по закрытым к m=cursor свечам."""
    classes = continuation_classes(corpus, cursor)
    known = classes['known']
    idx = np.flatnonzero(known)
    sub, sub_rows = subcorpus(corpus, idx, rows)
    sub_classes = continuation_classes(sub, cursor)
    labels = sub_classes['extend_first']
    obs = observable_events(sub, cursor)
    lens = Lens(sub, obs)
    pool = list(range(len(lens.atoms)))
    min_cell = max(30, lens.n // 20)
    report, marks = survey(lens, sizes=(1, 2), budget=10 ** 9, mode='exact',
                           seed=seed, top=top, labels=labels, atom_pool=pool,
                           min_cell=min_cell)
    cal = calibrate(lens, labels=labels, repeats=repeats, sizes=(1, 2),
                    budget=10 ** 9, mode='exact', seed=seed, top=top,
                    atom_pool=pool, min_cell=min_cell,
                    exchangeability_justified=False)
    best = report['views']['more_in_group'][0] if report['views']['more_in_group'] else None
    drawn = [v for v in cal['best_under_broken_link']['more_in_group'] if v is not None]
    out = {
        'cursor': cursor,
        'population': describe_population(corpus, cursor, classes),
        'searched_units': lens.n, 'observable_atoms': len(lens.atoms),
        'min_cell': min_cell,
        'combinations_tested': report['tested'], 'complete': report['complete'],
        'coverage_note': 'парный перебор по событиям, доступным к курсору; '
                         'полная матрица отношений на 21 свече в парный перебор '
                         'не входила — ограничение покрытия, не знание',
        'calibration_status': cal['status'],
        'calibration_max': max(drawn) if drawn else None,
        'calibration_share_not_worse': (sum(1 for v in drawn if v >= best['display_value'])
                                        / len(drawn)) if drawn and best else None,
        'best': None}
    if best is not None:
        atoms = tuple(best['atoms'])
        out['best'] = {
            'display_value': best['display_value'],
            'relations': [lens.describe(a) for a in atoms],
            'recognised_at_minute_after_realisation':
                max(lens.describe(a)['known_at'] for a in atoms),
            'atoms': list(atoms)}
        out['_lens'] = lens
        out['_atoms'] = atoms
        out['_sub'] = sub
        out['_rows'] = sub_rows
        out['_labels'] = labels
        out['_classes'] = sub_classes
    return out


def stage_denominator(found):
    """Знаменатель — ВСЕ случаи с наблюдаемым началом, включая иное продолжение."""
    lens, atoms, classes = found['_lens'], found['_atoms'], found['_classes']
    truth, known = lens.intersection(atoms)
    n = lens.n
    mask = np.array([(truth >> i) & 1 for i in range(n)], dtype=bool)
    known_mask = np.array([(known >> i) & 1 for i in range(n)], dtype=bool)
    def split(sel):
        total = int(sel.sum())
        if total == 0:
            return {'n': 0}
        return {'n': total,
                'extend_first': int((sel & classes['extend_first']).sum()),
                'give_back_first': int((sel & classes['give_back_first']).sum()),
                'tie_same_minute': int((sel & classes['tie_same_minute']).sum()),
                'no_pronounced_continuation': int((sel & classes['neither']).sum()),
                'share_extend': float((sel & classes['extend_first']).sum() / total)}
    with_prefix = mask & known_mask
    without = ~mask & known_mask
    return {'with_observable_start': split(with_prefix),
            'without_it': split(without),
            'difference_share_extend': (split(with_prefix)['share_extend']
                                        - split(without)['share_extend'])
            if with_prefix.any() and without.any() else None,
            'denominator_rule': 'все случаи с наблюдаемым началом, включая другое '
                                'продолжение и невыраженное',
            '_mask': with_prefix, '_known': known_mask}


def stage_money(found, denom, m, cursor, instrument='NQ'):
    """Что остаётся после фактически возможного решения и исполнения.

    Вход по open минуты, следующей за узнаванием. Всё до неё — история
    формирования сигнала. Сравнение — с сопоставимым удержанием в том же
    контексте, то есть с той же сделкой без распознавания.
    """
    rows, sub = found['_rows'], found['_sub']
    mask, known_mask = denom['_mask'], denom['_known']
    ts = m['close_ts_utc_ns']
    cost = COST[instrument]
    recognised_at = found['best']['recognised_at_minute_after_realisation']
    entry_offset = recognised_at + 1
    out = {'with_recognition': [], 'without_recognition': []}
    busy = {'with_recognition': -1, 'without_recognition': -1}
    order = np.argsort([r['r_index'] for r in rows])
    for i in order:
        if not known_mask[i]:
            continue
        r = rows[i]
        base = int(r['r_index'])
        e_idx, x_idx = base + entry_offset, base + AHEAD
        if x_idx >= len(ts):
            continue
        want = ts[base] + np.arange(entry_offset, AHEAD + 1) * MIN
        at = np.searchsorted(ts, want)
        if at[-1] >= len(ts) or not np.all(ts[at] == want):
            continue
        s = r['side']
        entry, exit_price = float(m['open'][at[0]]), float(m['close'][at[-1]])
        net = s * (exit_price - entry) * POINT[instrument] - cost
        for tag, take in (('with_recognition', bool(mask[i])),
                          ('without_recognition', True)):
            if not take:
                continue
            if int(ts[at[0]]) <= busy[tag]:
                continue
            out[tag].append({'ts': int(ts[at[0]]), 'net': net})
            busy[tag] = int(ts[at[-1]])
    def block(tr):
        if not tr:
            return {'n': 0}
        net = np.array([t['net'] for t in tr])
        ts_ = np.array([t['ts'] for t in tr])
        year = 1970 + (ts_ // MIN) // int(365.2425 * 1440)
        by = {int(y): float(net[year == y].sum()) for y in np.unique(year)}
        eq = np.cumsum(net)
        return {'n': len(tr), 'mean_net': float(net.mean()),
                'total_net': float(net.sum()),
                'positive_years': int(sum(v > 0 for v in by.values())),
                'years': len(by),
                'drawdown': float(np.max(np.maximum.accumulate(eq) - eq)),
                'by_year': by}
    return {'entry': f'open минуты m={entry_offset} после реализации',
            'exit': f'close минуты m={AHEAD}',
            'recognised_at': recognised_at,
            'with_recognition': block(out['with_recognition']),
            'comparable_holding_same_context': block(out['without_recognition']),
            'note': 'всё до минуты входа — история формирования сигнала и в '
                    'доступное движение не входит'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--riz-inputs', nargs='+', required=True)
    ap.add_argument('--instrument', default='NQ')
    ap.add_argument('--cursors', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--repeats', type=int, default=60)
    ap.add_argument('--top', type=int, default=10)
    ap.add_argument('--controls-per-anchor', type=int, default=8)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    m = tape(a.instrument)
    riz_anchors, riz_side = [], []
    riz_corpora = []
    for path in a.riz_inputs:
        c = Corpus.load(path)
        s, _, _ = sides(c)
        riz_corpora.append(c)
        riz_anchors.append(np.array([int(u.split(':')[1]) for u in c.unit_ids]))
        riz_side.append(s)
    riz_anchors = np.concatenate(riz_anchors)
    riz_side = np.concatenate(riz_side)
    keep = np.unique(riz_anchors, return_index=True)[1]
    riz_anchors, riz_side = riz_anchors[keep], riz_side[keep]

    control = sample_control(riz_corpora[0], a.instrument, a.controls_per_anchor, a.seed)
    ctrl_anchors = np.array([int(u.split(':')[1]) for u in control.unit_ids])
    ctrl_side, _, _ = sides(control)

    result = {'version': VERSION,
              'implementation_sha256': hashlib.sha256(
                  Path(__file__).read_bytes()).hexdigest(),
              'subject': 'узнаваемое начало ещё не завершённой формы после '
                         'исполнения ближайшего ожидания',
              'semantic_note': 'будущая свеча не отменяет исторический порядок '
                               'экстремумов; она может опровергнуть выведенное '
                               'из него ожидание',
              'populations': {}, 'by_cursor': {}, 'chain': None}

    for label, anchors, side in (('control', ctrl_anchors, ctrl_side),
                                 ('riz', riz_anchors, riz_side)):
        rows = realisation_anchors(anchors, side, m)
        corpus, rows = build_corpus(rows, m, a.instrument, label)
        corpus = orient(corpus, rows)
        result['populations'][label] = {
            'decisions_with_side': int((side != 0).sum()),
            'reached_nearest_expectation': len(rows),
            'median_minutes_to_realisation': float(np.median(
                [r['minutes_from_decision'] for r in rows])) if rows else None}
        if label == 'control':
            ctrl_corpus, ctrl_rows = corpus, rows
        else:
            riz_corpus, riz_rows = corpus, rows

    best_overall = None
    for cursor in a.cursors:
        found = stage_search(ctrl_corpus, ctrl_rows, cursor, a.repeats, a.top)
        public = {k: v for k, v in found.items() if not k.startswith('_')}
        if found.get('best'):
            denom = stage_denominator(found)
            public['denominator'] = {k: v for k, v in denom.items()
                                     if not k.startswith('_')}
            beats = (found['calibration_share_not_worse'] is not None
                     and found['calibration_share_not_worse'] <= 0.05)
            public['beats_calibration'] = beats
            if beats and best_overall is None:
                best_overall = (cursor, found, denom)
        result['by_cursor'][str(cursor)] = public

    if best_overall is not None:
        cursor, found, denom = best_overall
        riz_found = None
        try:
            riz_found = stage_search(riz_corpus, riz_rows, cursor, 1, a.top)
        except Exception as exc:                       # noqa: BLE001
            riz_found = {'error': str(exc)}
        money = stage_money(found, denom, m, cursor, a.instrument)
        result['chain'] = {
            'cursor': cursor,
            'form': found['best'],
            'denominator': {k: v for k, v in denom.items() if not k.startswith('_')},
            'money_from_recognition': money}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=1,
                                      default=str), encoding='utf-8')
    brief = {'populations': result['populations'],
             'by_cursor': {k: {'share_extend': v['population']['share_extend'],
                               'no_pronounced': v['population']
                               ['no_pronounced_continuation'],
                               'best': (v['best'] or {}).get('display_value'),
                               'calibration_max': v['calibration_max'],
                               'share_not_worse': v.get(
                                   'calibration_share_not_worse'),
                               'beats': v.get('beats_calibration')}
                           for k, v in result['by_cursor'].items()}}
    print(json.dumps(brief, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
