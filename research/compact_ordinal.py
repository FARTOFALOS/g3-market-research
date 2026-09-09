#!/usr/bin/env python3
"""Компактные порядковые конструкции: полное пространство + сжатая запись сцены.

ЧТО ЭТО ДОБАВЛЯЕТ К 066
=======================
В 066 полнота представления сочеталась с ограниченным поиском: одиночное
сравнение точек или конъюнкция двух сравнений. Многие простые для человека
конструкции требуют большего числа связей и потому в тот перебор не входили.
Положение экстремума среди десяти свечей — это сразу десяток сравнений, но
конструкция остаётся **полностью порядковой**.

Здесь такие конструкции введены как компактная запись поверх того же языка. У
каждой есть:

- **точное раскрытие** в отношения свечей (булева формула над парами точек),
  проверяемое на данных: `selftest` сверяет вектор события с вектором формулы;
- **время доступности** — ординал, к которому все нужные сведения закрыты;
- **конкретные фильмы**, которые её выполняют;
- **правило для равенств, отсутствующих событий и неизвестных наблюдений**:
  равенство экстремумов разрешается в пользу ПЕРВОГО вхождения и это входит в
  раскрытие; ненаступившее событие — отдельное значение `нет`, не ноль;
  неизвестное наблюдение исключается из знаменателя и считается отдельно.

Размеры и расстояния НЕ вводятся. Язык остаётся порядковым.

ТЕРРИТОРИЯ ЭТОГО ЦИКЛА (объявлена до расчёта)
=============================================
NQ, паспорта RIZ из готового поля, ТФ 3, 5, 10, 15, 30, 60, T0 в
2020-01-01 … 2025-10-31, дедупликация по минуте T0. Прежние циклы использовали
только ТФ 5 — 2 074 единицы из 7 198 доступных по этим ТФ, то есть 29% готового
поля в этой территории. Пересечение по T0 между ТФ мало (6–15%), поэтому
добавленные строки дают новые рыночные наблюдения, а не копии соседних ТФ.

Фильм — окно T0−10 … T0+30 минутной ленты; T0 сохранён в адресе единицы. Юг
отражён по цене, «вверх» = по ходу конструкции (операция над ценами, порядок
внутри фильма при этом сохраняется как система).

Курсоры 5 и 10 объявлены до счёта. Исходы исполнимы: вход `open` минуты после
курсора, выходы через 1, 5 и 15 минут. Календарь разделён: выдвижение
конструкции на T0 до 2024-01-01, оценка — на 2024-01-01 … 2025-10-31. Прежняя
экспозиция названа прямо: обе части этой истории уже просматривались в 062–066,
поэтому оценочный участок независимым подтверждением не является.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calendar_utils import year_of, quarter_of, date_key  # noqa: E402

from relational_stencil import Corpus, Space                        # noqa: E402
from candidate_check import POINT, COST                             # noqa: E402
from ordinal_events import (NONE, UNKNOWN, crossing_event,          # noqa: E402
                            extremum_event, last_update_event,
                            point_level, prefix_extreme_level)
from ordinal_search import unpack, matrix, scan, pick, ceiling, counts  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MIN = 60_000_000_000
BACK, AHEAD = 10, 30
TFS = (3, 5, 10, 15, 30, 60)


# ------------------------------------------------------------- популяция ----

def passports(instrument, start, stop, tfs):
    lo = np.datetime64(start, 'ns').astype('int64')
    hi = np.datetime64(stop, 'ns').astype('int64')
    rows, per_tf = [], {}
    for tf in tfs:
        f = ROOT / f'data/field/{instrument}/cells/tf_{tf:04d}/passports.parquet'
        if not f.exists():
            continue
        p = pd.read_parquet(f, columns=['riz_id', 't0_ts_ns', 'tf_minutes',
                                        'direction', 't0_exit_side', 'censored'])
        p = p[(p.t0_ts_ns >= lo) & (p.t0_ts_ns < hi)]
        per_tf[tf] = {'rows': int(len(p)), 'unique_t0': int(p.t0_ts_ns.nunique())}
        rows.append(p)
    allp = pd.concat(rows, ignore_index=True)
    allp = allp.sort_values(['t0_ts_ns', 'tf_minutes'])
    first = allp.drop_duplicates('t0_ts_ns', keep='first')
    return first, per_tf, int(len(allp))


def build(instrument, start, stop, tfs):
    m = {k: np.load(ROOT / 'data/market' / instrument / (k + '.npy'))
         for k in ('close_ts_utc_ns', 'open', 'high', 'low', 'close')}
    ts = m['close_ts_utc_ns']
    first, per_tf, total_rows = passports(instrument, start, stop, tfs)
    t0 = first.t0_ts_ns.to_numpy()
    pos = np.searchsorted(ts, t0)
    ok = (pos < len(ts)) & (ts[np.minimum(pos, len(ts) - 1)] == t0)
    pos, first = pos[ok], first[ok]
    ords = np.arange(-BACK, AHEAD + 1)
    idx = pos[:, None] + ords[None, :]
    good = (idx[:, 0] >= 0) & (idx[:, -1] < len(ts))
    pos, first, idx = pos[good], first[good], idx[good]
    cont = ((ts[idx[:, 1:]] - ts[idx[:, :-1]]) == MIN).all(axis=1)
    pos, first, idx = pos[cont], first[cont], idx[cont]
    films = np.stack([m['open'][idx], m['high'][idx], m['low'][idx],
                      m['close'][idx]], axis=2)
    side = np.where(first.t0_exit_side.to_numpy() == 'south', -1, 1)
    flip = side < 0
    if flip.any():
        o, h, l, c = (films[flip, :, k].copy() for k in range(4))
        films[flip, :, 0], films[flip, :, 1] = -o, -l
        films[flip, :, 2], films[flip, :, 3] = -h, -c
    ids = [f'{instrument}:{int(v)}' for v in first.t0_ts_ns.to_numpy()]
    corpus = Corpus(films, ords, list(ids), list(ids),
                    {'instrument': instrument,
                     'source': {'population': 'паспорта RIZ готового поля',
                                'tfs': list(tfs), 'start': start, 'stop': stop,
                                'dedup': 'по минуте T0, оставлен наименьший ТФ',
                                'orientation': 'юг отражён по цене'},
                     'anchor': 'T0 = ординал 0',
                     'unit_definition': 'один T0'})
    corpus.members = [[i] for i in range(len(ids))]
    coverage = {'per_tf': per_tf, 'rows_all_tfs': total_rows,
                'unique_t0': int(len(first)), 'films_built': int(corpus.n),
                'previous_cycles_used': 'только ТФ 5',
                'note': 'дедупликация по минуте T0; пересечение T0 между ТФ мало'}
    return corpus, np.array([int(v) for v in first.t0_ts_ns.to_numpy()]), side, coverage


# ------------------------------ компактные конструкции и их раскрытие -------

def compact_events(corpus, cursor):
    """Компактные порядковые конструкции. Все закрыты к `cursor`."""
    ev = {}
    segs = [(-BACK, -1), (1, cursor), (-BACK, cursor)]
    for a, b in segs:
        for field in ('H', 'L'):
            ev[f'arg{field}_first[{a}..{b}]'] = extremum_event(corpus, a, b, field, False)
            ev[f'arg{field}_last[{a}..{b}]'] = extremum_event(corpus, a, b, field, True)
            ev[f'upd{field}[{a}..{b}]'] = last_update_event(corpus, a, b, field)
    pre_hi = prefix_extreme_level(corpus, -BACK, 0, 'H')
    pre_lo = prefix_extreme_level(corpus, -BACK, 0, 'L')
    c0 = point_level(corpus, 0, 'C')
    for nm, lvl in (('maxH_pre', pre_hi), ('minL_pre', pre_lo), ('C0', c0)):
        ev[f'close_above({nm})[1..{cursor}]'] = crossing_event(corpus, 1, cursor, 'C', lvl, True)
        ev[f'close_below({nm})[1..{cursor}]'] = crossing_event(corpus, 1, cursor, 'C', lvl, False)
        ev[f'wick_above({nm})[1..{cursor}]'] = crossing_event(corpus, 1, cursor, 'H', lvl, True)
        ev[f'wick_below({nm})[1..{cursor}]'] = crossing_event(corpus, 1, cursor, 'L', lvl, False)
    return ev


def expansion_vector(corpus, name, ev, value):
    """Точное раскрытие конструкции в отношения свечей. Возвращает булев вектор.

    Правило равенств: экстремум разрешается в пользу ПЕРВОГО вхождения (для
    `_first`) или ПОСЛЕДНЕГО (для `_last`) — это входит в формулу, а не
    подразумевается.
    """
    col = {int(o): i for i, o in enumerate(corpus.ordinals)}
    H, L = corpus.ohlc[:, :, 1], corpus.ohlc[:, :, 2]
    C = corpus.ohlc[:, :, 3]
    e = ev[name]
    a, b = e.segment
    j = int(value)
    fin = np.isfinite(corpus.ohlc).all(axis=2)
    span = [int(o) for o in corpus.ordinals if a <= o <= b]
    known = np.ones(corpus.n, bool)
    for o in span:
        known &= fin[:, col[o]]
    if name.startswith(('argH', 'argL')):
        field = H if name[3] == 'H' else L
        better = (lambda x, y: x >= y) if name[3] == 'H' else (lambda x, y: x <= y)
        strict = (lambda x, y: x > y) if name[3] == 'H' else (lambda x, y: x < y)
        last = '_last[' in name
        out = known.copy()
        for o in span:
            if o == j:
                continue
            if (o < j) if not last else (o > j):
                out &= strict(field[:, col[j]], field[:, col[o]])
            else:
                out &= better(field[:, col[j]], field[:, col[o]])
        return out, known
    if name.startswith('upd'):
        field = H if name[3] == 'H' else L
        strict = (lambda x, y: x > y) if name[3] == 'H' else (lambda x, y: x < y)
        weak = (lambda x, y: x >= y) if name[3] == 'H' else (lambda x, y: x <= y)
        out = known.copy()
        for o in span:
            if o < j:
                out &= strict(field[:, col[j]], field[:, col[o]])
            elif o > j:
                out &= weak(field[:, col[j]], field[:, col[o]])
        return out, known
    # события пересечения: формула «пробито на m и не пробито раньше»
    kind, rest = name.split('(', 1)
    lvl_name = rest.split(')')[0]
    seg = rest.split('[')[1].rstrip(']')
    c1, c2 = (int(x) for x in seg.split('..'))
    src = C if kind.startswith('close') else (H if kind.startswith('wick_above') else L)
    above = 'above' in kind
    if lvl_name == 'C0':
        lv = C[:, col[0]][:, None]
        lv_span = [0]
        lvv = C[:, [col[0]]]
    else:
        f = H if lvl_name == 'maxH_pre' else L
        lv_span = [int(o) for o in corpus.ordinals if -BACK <= o <= 0]
        lvv = f[:, [col[o] for o in lv_span]]
    known2 = known.copy()
    for o in lv_span:
        known2 &= fin[:, col[o]]
    for o in range(c1, c2 + 1):
        known2 &= fin[:, col[o]]
    # агрегат уровня задаётся ИМЕНЕМ опоры, а не направлением пересечения:
    # maxH_pre — максимум префикса, minL_pre — минимум, C0 — одна точка.
    level = lvv.min(axis=1) if lvl_name == 'minL_pre' else lvv.max(axis=1)
    beyond = np.stack([(src[:, col[o]] > level) if above
                       else (src[:, col[o]] < level)
                       for o in range(c1, c2 + 1)], axis=1)
    if value == NONE:
        return known2 & ~beyond.any(axis=1), known2
    m = int(value)
    out = known2 & beyond[:, m - c1]
    if m > c1:
        out &= ~beyond[:, :m - c1].any(axis=1)
    return out, known2


def selftest_expansion(corpus, cursor, max_values=6):
    """Каждая компактная конструкция сверяется со своим раскрытием на данных."""
    ev = compact_events(corpus, cursor)
    checked, mismatched, cases = 0, 0, []
    for name, e in ev.items():
        vals = [v for v in np.unique(e.values) if v != UNKNOWN][:max_values]
        for v in vals:
            direct = (e.values == v) & (e.values != UNKNOWN)
            formula, known = expansion_vector(corpus, name, ev, v)
            formula = formula & (e.values != UNKNOWN)
            checked += 1
            if not np.array_equal(direct, formula):
                mismatched += 1
                cases.append({'construction': f'{name} = {int(v)}',
                              'differs_on': int((direct != formula).sum())})
    return {'constructions': len(ev), 'checks': checked,
            'mismatched': mismatched, 'examples': cases[:5],
            'rule_equal_extremes': 'равенство разрешается в пользу первого '
                                   'вхождения (_first) или последнего (_last); '
                                   'правило входит в формулу раскрытия',
            'rule_absent_event': 'ненаступившее событие — отдельное значение, '
                                 'а не ноль',
            'rule_unknown': 'неизвестное наблюдение исключается из знаменателя',
            'passed': mismatched == 0}


# ---------------------------------------------------------------- атомы -----

def compact_atoms(corpus, cursor, min_cell):
    """Атомы компактного слоя: значения конструкций и порядок между ними."""
    ev = compact_events(corpus, cursor)
    names, vecs, meta = [], [], []
    for name, e in ev.items():
        for v in np.unique(e.values):
            if v == UNKNOWN:
                continue
            sel = e.values == v
            if sel.sum() < min_cell or (corpus.n - sel.sum()) < min_cell:
                continue
            label = 'нет' if v == NONE else int(v)
            names.append(f'{name} = {label}')
            vecs.append(sel)
            meta.append({'kind': 'значение конструкции', 'construction': name,
                         'value': label, 'known_at': cursor})
    keys = list(ev)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = ev[keys[i]].values, ev[keys[j]].values
            ok = (a >= 0) & (b >= 0)
            sel = ok & (a < b)
            if sel.sum() < min_cell or (corpus.n - sel.sum()) < min_cell:
                continue
            names.append(f'{keys[i]} < {keys[j]}')
            vecs.append(sel)
            meta.append({'kind': 'порядок конструкций', 'known_at': cursor})
    return names, (np.stack(vecs, axis=1) if vecs else
                   np.zeros((corpus.n, 0), bool)), meta


def direct_atoms(space, cursor, min_cell, n):
    keep, names = [], []
    for atom in range(len(space.masks)):
        d = space.describe(atom)
        if d['known_at'] > cursor:
            continue
        known = unpack(space.known[atom // 3], n)
        truth = unpack(space.masks[atom], n) & known
        c = truth.sum()
        if c < min_cell or (known.sum() - c) < min_cell:
            continue
        keep.append(atom)
        names.append(d['text'])
    B = np.zeros((n, len(keep)), bool)
    for j, atom in enumerate(keep):
        B[:, j] = unpack(space.masks[atom], n)
    return names, B


# --------------------------------------------------------------- деньги -----

def streams(corpus, t0, sel, side, cursor, holds, instrument):
    col = {int(o): i for i, o in enumerate(corpus.ordinals)}
    O = corpus.ohlc[:, :, 0]
    p, cost = POINT[instrument], COST[instrument]
    out = {}
    order = np.argsort(t0)
    for h in holds:
        busy, tr = -1, []
        for i in order:
            if not sel[i]:
                continue
            a, b = col[cursor + 1], col[cursor + 1 + h]
            if not (np.isfinite(O[i, a]) and np.isfinite(O[i, b])):
                continue
            t_in = int(t0[i]) + (cursor + 1) * MIN
            if t_in <= busy:
                continue
            tr.append((t_in, float(side * (O[i, b] - O[i, a]) * p)))
            busy = int(t0[i]) + (cursor + 1 + h) * MIN
        out[f'hold_{h}'] = summarise(tr, cost)
    return out


def summarise(tr, cost):
    if not tr:
        return {'n': 0}
    g = np.array([t[1] for t in tr])
    ts = np.array([t[0] for t in tr])
    day = date_key(ts)
    year = year_of(ts)
    net = g - cost
    eq = np.cumsum(net)
    sem = float(g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else 0.0
    byg = {int(y): float(g[year == y].mean()) for y in np.unique(year)}
    days = np.unique(day)
    by_day = np.array([g[day == d].sum() for d in days])
    order = np.argsort(-by_day)
    return {'n': len(g), 'days': int(len(days)),
            'mean_gross': float(g.mean()), 'sem_gross': sem,
            't_gross': float(g.mean() / sem) if sem else None,
            'median_gross': float(np.median(g)),
            'share_positive': float((g > 0).mean()),
            'mean_net_at_15': float(net.mean()),
            'breakeven_cost_usd': float(g.mean()),
            'total_net_at_15': float(net.sum()),
            'drawdown_net_at_15': float(np.max(np.maximum.accumulate(eq) - eq)),
            'by_year_mean_gross': byg,
            'positive_years_gross': int(sum(v > 0 for v in byg.values())),
            'years': len(byg),
            'top5_days_share': float(by_day[order[:5]].sum() / g.sum())
            if g.sum() else None,
            'mean_gross_drop_best_5_days':
                float(g[~np.isin(day, days[order[:5]])].mean())}


# ----------------------------------------------------------------- main -----

def run(a):
    corpus, t0, side, coverage = build(a.instrument, a.start, a.stop, TFS)
    train = t0 < np.datetime64(a.split, 'ns').astype('int64')
    space = Space(corpus)
    cost = COST[a.instrument]
    res = {'implementation_sha256': hashlib.sha256(
               Path(__file__).read_bytes()).hexdigest(),
           'language': 'порядковый: прямые отношения точек + компактные '
                       'конструкции над ними; размеров и расстояний нет',
           'territory': {**corpus.metadata['source'], 'films': int(corpus.n),
                         'window': [-BACK, AHEAD]},
           'field_coverage': coverage,
           'calendar_split': {'search_until': a.split,
                              'search_films': int(train.sum()),
                              'evaluation_films': int((~train).sum()),
                              'prior_exposure': 'обе части уже просматривались в '
                                                '062–066; оценочный участок '
                                                'независимым подтверждением не '
                                                'является'},
           'selection_rule': 'объявлено до счёта: среди ячеек с опорой не меньше '
                             f'{a.min_cell}, числом различных дней не меньше '
                             f'{a.min_days} и переживших калибровку — берётся '
                             'наибольшее |среднее|; если не выжила ни одна, '
                             'кандидат не выдвигается',
           'cursors': {}}
    for cursor in a.cursors:
        col = {int(o): i for i, o in enumerate(corpus.ordinals)}
        O = corpus.ohlc[:, :, 0]
        p = POINT[a.instrument]
        y_all = {f'R{h}': (O[:, col[cursor + 1 + h]] - O[:, col[cursor + 1]]) * p
                 for h in a.horizons}
        usable = np.isfinite(O[:, col[cursor + 1]])
        for y in y_all.values():
            usable &= np.isfinite(y)
        pool = usable & train
        cn, cB, cmeta = compact_atoms(corpus, cursor, a.min_cell)
        dn, dB = direct_atoms(space, cursor, a.min_cell, corpus.n)
        names = cn + dn
        B = np.concatenate([cB, dB], axis=1)[pool]
        nc, nd = len(cn), len(dn)
        tested = len(names) + nc * (nc - 1) // 2 + nc * nd
        block = {'films_in_search': int(pool.sum()),
                 'compact_atoms': nc, 'direct_atoms': nd,
                 'atoms_available_direct': len(space.masks),
                 'combinations_tested': int(tested),
                 'not_tested': 'пары прямое×прямое — они разобраны в 066',
                 'min_cell': a.min_cell, 'min_days': a.min_days,
                 'outcomes': {}}
        # ограничиваем пары: компактное×всё
        mask_pair = np.zeros((len(names), len(names)), bool)
        mask_pair[:nc, :] = True
        mask_pair[:, :nc] = True
        np.fill_diagonal(mask_pair, False)
        cached = counts(B)
        day = ((t0[pool] // MIN) // 1440)
        for label, y in y_all.items():
            yy = y[pool]
            m1, c1, m2, c2 = scan(B, yy, a.min_cell, cached)
            m2 = np.where(mask_pair, m2, np.nan)
            drawn = ceiling_masked(B, yy, a.min_cell, a.repeats, a.seed, cached,
                                   mask_pair)
            best = pick(m1, c1, m2, c2)
            beats = sum(1 for v in drawn if v >= best['score'])
            sel_train = B[:, best['atoms'][0]].copy()
            for i in best['atoms'][1:]:
                sel_train &= B[:, i]
            n_days = int(len(np.unique(day[sel_train])))
            block['outcomes'][label] = {
                'baseline_mean_usd': float(yy.mean()),
                'best_construction': [names[i] for i in best['atoms']],
                'best_kind': best['kind'],
                'best_mean_gross_usd': best['mean'], 'best_n': best['n'],
                'best_distinct_days': n_days,
                'best_side': 'long' if best['mean'] > 0 else 'short',
                'calibration_max': max(drawn),
                'calibration_median': float(np.median(drawn)),
                'survives_calibration': bool(beats == 0),
                'passes_day_support': bool(n_days >= a.min_days),
                'promoted': bool(beats == 0 and n_days >= a.min_days)}
            if block['outcomes'][label]['promoted']:
                full = np.concatenate([cB, dB], axis=1)
                s = full[:, best['atoms'][0]].copy()
                for i in best['atoms'][1:]:
                    s &= full[:, i]
                sd = 1 if best['mean'] > 0 else -1
                block['outcomes'][label]['streams_search'] = streams(
                    corpus, t0, s & usable & train, sd, cursor, a.horizons,
                    a.instrument)
                block['outcomes'][label]['streams_evaluation'] = streams(
                    corpus, t0, s & usable & ~train, sd, cursor, a.horizons,
                    a.instrument)
                block['outcomes'][label]['examples'] = [
                    corpus.unit_ids[i] for i in np.flatnonzero(s & usable)[:5]]
                block['outcomes'][label]['dissection'] = {
                    names[i]: streams(corpus, t0,
                                      full[:, i] & usable & train, sd, cursor,
                                      [a.horizons[0]], a.instrument)
                    for i in best['atoms']}
        res['cursors'][str(cursor)] = block
    return res


def ceiling_masked(B, y, min_cell, repeats, seed, cached, mask_pair):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(repeats):
        z = rng.permutation(y)
        m1, c1, m2, c2 = scan(B, z, min_cell, cached)
        m2 = np.where(mask_pair, m2, np.nan)
        out.append(pick(m1, c1, m2, c2)['score'])
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    st = sub.add_parser('selftest')
    st.add_argument('--instrument', default='NQ')
    st.add_argument('--start', default='2020-01-01')
    st.add_argument('--stop', default='2025-11-01')
    st.add_argument('--cursor', type=int, default=5)
    st.add_argument('--out', required=True)
    rn = sub.add_parser('run')
    rn.add_argument('--instrument', default='NQ')
    rn.add_argument('--start', default='2020-01-01')
    rn.add_argument('--stop', default='2025-11-01')
    rn.add_argument('--split', default='2024-01-01')
    rn.add_argument('--cursors', nargs='+', type=int, default=[5, 10])
    rn.add_argument('--horizons', nargs='+', type=int, default=[1, 5, 15])
    rn.add_argument('--min-cell', type=int, default=400)
    rn.add_argument('--min-days', type=int, default=150)
    rn.add_argument('--repeats', type=int, default=40)
    rn.add_argument('--seed', type=int, default=13)
    rn.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.cmd == 'selftest':
        corpus, t0, side, coverage = build(a.instrument, a.start, a.stop, TFS)
        res = {'field_coverage': coverage,
               'expansion': selftest_expansion(corpus, a.cursor)}
        res['passed'] = res['expansion']['passed']
    else:
        res = run(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1,
                                      default=str), encoding='utf-8')
    if a.cmd == 'selftest':
        print(json.dumps(res, ensure_ascii=False, indent=1)[:3000])
    else:
        brief = {'films': res['territory']['films'],
                 'split': res['calendar_split'],
                 'cursors': {c: {'compact': b['compact_atoms'],
                                 'direct': b['direct_atoms'],
                                 'tested': b['combinations_tested'],
                                 'outcomes': {k: {'best': round(v['best_mean_gross_usd'], 2),
                                                  'n': v['best_n'],
                                                  'days': v['best_distinct_days'],
                                                  'ceil': round(v['calibration_max'], 2),
                                                  'promoted': v['promoted'],
                                                  'what': v['best_construction']}
                                              for k, v in b['outcomes'].items()}}
                             for c, b in res['cursors'].items()}}
        print(json.dumps(brief, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
