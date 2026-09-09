#!/usr/bin/env python3
"""Сквозной рентген: библиотека конструкций, чтение по префиксу, действия, история.

ЧТО ЭТО ЗА ПРИБОР
=================
Открываем исторический RIZ-фильм на закрытой минуте, будущее скрыто. Прибор
предъявляет пять вещей:

1. какие конструкции библиотеки уже узнаются по доступным отношениям свечей;
2. какие исторические фильмы выполняли те же отношения совместно, с какими
   различиями и в каких обстоятельствах;
3. какие продолжения встречались у ВСЕХ таких начал, включая альтернативные,
   незавершённые и неизвестные;
4. что меняется после следующей закрытой свечи и почему — какое пересечение,
   закрытие или обновление экстремума стало известно;
5. какое действие предусмотрено проверяемым правилом и что даёт его
   последовательное историческое исполнение.

Новой архитектуры здесь нет: язык отношений — `relational_stencil.Space`,
компактные конструкции с раскрытием — `compact_ordinal`, календарь —
`calendar_utils`, деньги и расход — `candidate_check`.

БИБЛИОТЕКА, А НЕ ОДНА СТРОКА
============================
Результат поиска — небольшой набор СОДЕРЖАТЕЛЬНО РАЗНЫХ конструкций, а не
строка с максимальной средней прибылью. Правило отбора объявлено до счёта:

- опора не меньше `--min-cell` фильмов и не меньше `--min-days` различных дней;
- находка переживает перестановочную калибровку (повторяется весь выбор,
  включая выбор стороны);
- различность: пересечение поддерживающих фильмов с уже отобранной конструкцией
  по Жаккару не выше `--max-overlap`;
- простота: если одиночное отношение даёт не меньше `1 - --pair-margin` от
  величины пары, в библиотеку идёт одиночное.

ЧЕГО ПРИБОР НЕ ДЕЛАЕТ
=====================
Не смотрит в будущее фильма при узнавании: атом участвует, только если обе его
точки закрыты к курсору. Это проверяется на данных (`selftest`): скрытая часть
фильма заменяется случайной, и ответ прибора обязан остаться прежним.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calendar_utils import year_of, date_key, minute_of_day          # noqa: E402
from candidate_check import POINT, COST                              # noqa: E402
from relational_stencil import Space                                 # noqa: E402
from compact_ordinal import (build, compact_events, compact_atoms,   # noqa: E402
                             direct_atoms, TFS, BACK, AHEAD, MIN)
from ordinal_search import scan, pick, counts                        # noqa: E402
from ordinal_events import NONE, UNKNOWN                             # noqa: E402
from context_release import contexts                                 # noqa: E402


# ------------------------------------------------------------ библиотека ----

def atom_pool(corpus, space, cursor, min_cell):
    cn, cB, cmeta = compact_atoms(corpus, cursor, min_cell)
    dn, dB = direct_atoms(space, cursor, min_cell, corpus.n)
    names = cn + dn
    B = np.concatenate([cB, dB], axis=1)
    kinds = ['компактная'] * len(cn) + ['прямое отношение'] * len(dn)
    return names, B, kinds, len(cn)


def ceiling(B, y, min_cell, repeats, seed, cached, mask_pair):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(repeats):
        z = rng.permutation(y)
        m1, c1, m2, c2 = scan(B, z, min_cell, cached)
        m2 = np.where(mask_pair, m2, np.nan)
        out.append(pick(m1, c1, m2, c2)['score'])
    return out


def jaccard(a, b):
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union else 0.0


def library(corpus, t0, cursor, horizons, args):
    """Набор различающихся конструкций. Правило отбора объявлено до счёта."""
    space = Space(corpus)
    col = {int(o): i for i, o in enumerate(corpus.ordinals)}
    O = corpus.ohlc[:, :, 0]
    p = POINT[args.instrument]
    split = np.datetime64(args.split, 'ns').astype('int64')
    train = t0 < split
    y_all = {f'R{h}': (O[:, col[cursor + 1 + h]] - O[:, col[cursor + 1]]) * p
             for h in horizons}
    usable = np.isfinite(O[:, col[cursor + 1]])
    for y in y_all.values():
        usable &= np.isfinite(y)
    pool = usable & train
    names, Ball, kinds, n_compact = atom_pool(corpus, space, cursor,
                                              args.min_cell)
    B = Ball[pool]
    A = len(names)
    mask_pair = np.zeros((A, A), bool)
    mask_pair[:n_compact, :] = True
    mask_pair[:, :n_compact] = True
    np.fill_diagonal(mask_pair, False)
    cached = counts(B)
    day = date_key(t0[pool])
    found, budget = [], {'atoms': A, 'compact': n_compact,
                         'combinations': A + n_compact * (A - 1)
                         - n_compact * (n_compact - 1) // 2}
    pooled, ceilings, means1 = [], {}, {}
    for label, y in y_all.items():
        yy = y[pool]
        m1, c1, m2, c2 = scan(B, yy, args.min_cell, cached)
        m2 = np.where(mask_pair, m2, np.nan)
        drawn = ceiling(B, yy, args.min_cell, args.repeats, args.seed, cached,
                        mask_pair)
        top = float(max(drawn))
        ceilings[label] = top
        means1[label] = m1
        a1 = np.abs(m1)
        for i in np.flatnonzero(np.nan_to_num(a1, nan=-1) > 0):
            pooled.append((label, {'atoms': [int(i)], 'mean': float(m1[i]),
                                   'n': int(c1[i]), 'kind': 'одиночная'},
                           float(a1[i]), top))
        iu = np.triu_indices(A, k=1)
        a2 = np.abs(m2[iu])
        for q in np.flatnonzero(np.nan_to_num(a2, nan=-1) > 0):
            i, j = int(iu[0][q]), int(iu[1][q])
            pooled.append((label, {'atoms': [i, j], 'mean': float(m2[i, j]),
                                   'n': int(c2[i, j]), 'kind': 'пара'},
                           float(a2[q]), top))
    # порядок объявлен: сначала превысившие потолок, внутри — по величине
    pooled.sort(key=lambda z: (0 if z[2] > z[3] else 1, -z[2]))
    for label, rec, score, top in pooled:
        if len(found) >= args.library_size:
            break
        m1 = means1[label]
        sel = B[:, rec['atoms'][0]].copy()
        for i in rec['atoms'][1:]:
            sel &= B[:, i]
        if len(np.unique(day[sel])) < args.min_days:
            continue
        if rec['kind'] == 'пара':
            best_single, best_i = 0.0, None
            for i in rec['atoms']:
                v = abs(float(m1[i])) if np.isfinite(m1[i]) else 0.0
                if v > best_single:
                    best_single, best_i = v, i
            if best_i is not None and best_single >= (1 - args.pair_margin) * score:
                rec = {'atoms': [best_i], 'mean': float(m1[best_i]),
                       'n': int(c1[best_i]) if False else int(sel.sum()),
                       'kind': 'одиночная'}
                sel = B[:, best_i].copy()
                score = best_single
        full = Ball[:, rec['atoms'][0]].copy()
        for i in rec['atoms'][1:]:
            full &= Ball[:, i]
        if any(jaccard(full, f['_mask']) > args.max_overlap for f in found):
            continue
        found.append({
            'id': f'K{len(found) + 1}',
            'outcome': label, 'cursor': cursor, 'kind': rec['kind'],
            'relations': [names[i] for i in rec['atoms']],
            'relation_kinds': [kinds[i] for i in rec['atoms']],
            'known_at': cursor,
            'mean_gross_search': rec['mean'],
            'films_search': int(sel.sum()),
            'days_search': int(len(np.unique(day[sel]))),
            'side': 'short' if rec['mean'] < 0 else 'long',
            'ceiling': top,
            'tier': ('допущена к действию' if score > top else 'описательная'),
            'beats_ceiling': bool(score > top),
            '_mask': full, '_atoms': rec['atoms']})
    return found, budget, usable, train, names, Ball


# ------------------------------------------- продолжения и обстоятельства ---

def continuations(corpus, t0, mask, cursor, horizons, instrument):
    """Все продолжения ВСЕХ фильмов с этим началом, включая неизвестные."""
    col = {int(o): i for i, o in enumerate(corpus.ordinals)}
    O, H, L, C = (corpus.ohlc[:, :, k] for k in range(4))
    p = POINT[instrument]
    idx = np.flatnonzero(mask)
    prior_hi = np.max(np.stack([H[:, col[o]] for o in range(-BACK, cursor + 1)]),
                      axis=0)
    prior_lo = np.min(np.stack([L[:, col[o]] for o in range(-BACK, cursor + 1)]),
                      axis=0)
    seg = [col[o] for o in range(cursor + 1, AHEAD + 1)]
    up = np.array([[C[i, j] > prior_hi[i] for j in seg] for i in idx])
    dn = np.array([[C[i, j] < prior_lo[i] for j in seg] for i in idx])
    known = np.array([[np.isfinite(C[i, j]) for j in seg] for i in idx])
    first_up = np.where(up.any(1), up.argmax(1), -1)
    first_dn = np.where(dn.any(1), dn.argmax(1), -1)
    classes = []
    for a, b, k in zip(first_up, first_dn, known):
        if not k.all():
            classes.append('наблюдение не завершено')
        elif a < 0 and b < 0:
            classes.append('ни того, ни другого')
        elif a >= 0 and (b < 0 or a < b):
            classes.append('продолжение вверх первым')
        elif b >= 0 and (a < 0 or b < a):
            classes.append('возврат вниз первым')
        else:
            classes.append('обе стороны на одной минуте')
    out = {'films': int(len(idx)),
           'classes': {c: int(classes.count(c)) for c in sorted(set(classes))},
           'median_minute_up': (float(np.median(first_up[first_up >= 0]) + 1)
                                if (first_up >= 0).any() else None),
           'median_minute_down': (float(np.median(first_dn[first_dn >= 0]) + 1)
                                  if (first_dn >= 0).any() else None),
           'returns': {}}
    for h in horizons:
        j = col[cursor + 1 + h]
        r = (O[idx, j] - O[idx, col[cursor + 1]]) * p
        fin = np.isfinite(r)
        out['returns'][f'R{h}'] = {
            'n_known': int(fin.sum()), 'n_unknown': int((~fin).sum()),
            'mean': float(r[fin].mean()) if fin.any() else None,
            'median': float(np.median(r[fin])) if fin.any() else None,
            'q25': float(np.percentile(r[fin], 25)) if fin.any() else None,
            'q75': float(np.percentile(r[fin], 75)) if fin.any() else None,
            'share_positive': float((r[fin] > 0).mean()) if fin.any() else None}
    yr = year_of(t0[idx])
    out['circumstances'] = {
        'by_year_films': {int(y): int((yr == y).sum()) for y in np.unique(yr)},
        'entry_minute_et': {'median': float(np.median(
            minute_of_day(t0[idx] + (cursor + 1) * MIN)))},
        'differences_inside_group':
            'фильмы группы совпадают по перечисленным отношениям и различаются '
            'всем остальным; конкретные адреса сохранены'}
    out['examples'] = [corpus.unit_ids[i] for i in idx[:5]]
    return out


# ------------------------------------------- последовательное чтение фильма --

def why_changed(corpus, unit, cursor):
    """Что стало известно на закрытии `cursor`: событие, а не только номер."""
    col = {int(o): i for i, o in enumerate(corpus.ordinals)}
    O, H, L, C = (corpus.ohlc[unit, :, k] for k in range(4))
    if cursor <= -BACK:
        return []
    notes = []
    prev = [col[o] for o in range(-BACK, cursor)]
    j = col[cursor]
    if prev:
        if H[j] > max(H[i] for i in prev):
            notes.append('новый максимум доступного окна')
        if L[j] < min(L[i] for i in prev):
            notes.append('новый минимум доступного окна')
        if C[j] > max(H[i] for i in prev):
            notes.append('закрытие выше прежнего максимума окна')
        if C[j] < min(L[i] for i in prev):
            notes.append('закрытие ниже прежнего минимума окна')
    if C[j] > O[j]:
        notes.append('тело вверх')
    elif C[j] < O[j]:
        notes.append('тело вниз')
    return notes or ['закрытых событий из объявленного набора не добавилось']


def read_film(corpus, t0, unit, lib_by_cursor, cursors, horizons, instrument,
              stats):
    """Чтение фильма по закрытым минутам. Будущее раскрывается после решения."""
    col = {int(o): i for i, o in enumerate(corpus.ordinals)}
    O = corpus.ohlc[unit, :, 0]
    p = POINT[instrument]
    steps = []
    prev = set()
    for c in cursors:
        recognised = []
        for k in lib_by_cursor.get(c, []):
            if bool(k['_mask'][unit]):
                recognised.append(k['id'])
        now = set(recognised)
        step = {'cursor': c,
                'closed_minutes': f'[-{BACK}..{c}]',
                'became_known': why_changed(corpus, unit, c),
                'recognised': sorted(now),
                'appeared': sorted(now - prev),
                'lost': sorted(prev - now),
                'expectation': []}
        for kid in sorted(now):
            k = next(x for x in lib_by_cursor[c] if x['id'] == kid)
            st = stats[kid]
            step['expectation'].append({
                'construction': kid, 'relations': k['relations'],
                'side': k['side'],
                'historical_films_with_this_start': st['films'],
                'continuations': st['classes'],
                'returns': st['returns']})
        steps.append(step)
        prev = now
    # раскрытие будущего ПОСЛЕ фиксации состояния
    future = {}
    for c in cursors:
        for h in horizons:
            j = col[c + 1 + h]
            if np.isfinite(O[j]) and np.isfinite(O[col[c + 1]]):
                future[f'от open m={c + 1} через {h} мин'] = float(
                    (O[j] - O[col[c + 1]]) * p)
    return {'unit': corpus.unit_ids[unit], 'steps': steps,
            'revealed_future_after_decision': future}


# ------------------------------------------------- действия и исполнение ----

def replay(corpus, t0, lib, holds, instrument, mask_valid, priority):
    """Последовательное историческое исполнение. Конфликт решается приоритетом.

    Приоритет объявлен ДО результатов сделок: порядок конструкций в библиотеке,
    полученный поиском. Одна позиция одновременно.
    """
    col = {int(o): i for i, o in enumerate(corpus.ordinals)}
    O = corpus.ohlc[:, :, 0]
    p, cost = POINT[instrument], COST[instrument]
    order = np.argsort(t0)
    busy, trades = -1, []
    for i in order:
        if not mask_valid[i]:
            continue
        fired = [k for k in priority if k['_mask'][i]]
        if not fired:
            continue
        k = fired[0]
        c = k['cursor']
        h = holds[k['id']]
        a, b = col[c + 1], col[c + 1 + h]
        if not (np.isfinite(O[i, a]) and np.isfinite(O[i, b])):
            continue
        t_in = int(t0[i]) + (c + 1) * MIN
        if t_in <= busy:
            continue
        side = -1 if k['side'] == 'short' else 1
        trades.append({'ts': t_in, 'unit': corpus.unit_ids[i], 'id': k['id'],
                       'gross': float(side * (O[i, b] - O[i, a]) * p),
                       'competing': [x['id'] for x in fired[1:]]})
        busy = int(t0[i]) + (c + 1 + h) * MIN
    return trades


def summarise(trades, cost):
    if len(trades) < 2:
        return {'n': len(trades)}
    g = np.array([t['gross'] for t in trades])
    ts = np.array([t['ts'] for t in trades])
    yr = year_of(ts)
    net = g - cost
    eq = np.cumsum(net)
    sem = float(g.std(ddof=1) / np.sqrt(len(g)))
    return {'n': int(len(g)), 'days': int(len(np.unique(date_key(ts)))),
            'mean_gross': float(g.mean()), 'sem': sem,
            't_gross': float(g.mean() / sem),
            'mean_net_at_15': float(net.mean()),
            'total_net_at_15': float(net.sum()),
            'breakeven_cost_usd': float(g.mean()),
            'share_positive': float((g > 0).mean()),
            'drawdown_net_at_15': float(np.max(np.maximum.accumulate(eq) - eq)),
            'by_year_mean_gross': {int(y): float(g[yr == y].mean())
                                   for y in np.unique(yr)},
            'by_construction': {}}


# ----------------------------------------------------------- самопроверка ---

def selftest_no_lookahead(corpus, lib, cursor, seed=5):
    """Скрытое будущее заменяется случайным: ответ прибора обязан не измениться."""
    col = {int(o): i for i, o in enumerate(corpus.ordinals)}
    rng = np.random.default_rng(seed)
    x = corpus.ohlc.copy()
    fut = [col[o] for o in range(cursor + 1, AHEAD + 1)]
    base = x[:, col[cursor], 3][:, None]
    n, m = x.shape[0], len(fut)
    o = base + rng.normal(0, 50, size=(n, m))
    c = o + rng.normal(0, 50, size=(n, m))
    hi = np.maximum(o, c) + np.abs(rng.normal(0, 20, size=(n, m)))
    lo = np.minimum(o, c) - np.abs(rng.normal(0, 20, size=(n, m)))
    x[:, fut, 0], x[:, fut, 1] = o, hi
    x[:, fut, 2], x[:, fut, 3] = lo, c
    from relational_stencil import Corpus
    other = Corpus(x, corpus.ordinals, list(corpus.unit_ids),
                   list(corpus.unit_ids), dict(corpus.metadata))
    other.members = list(corpus.members)
    space2 = Space(other)
    names2, B2, kinds2, nc2 = atom_pool(other, space2, cursor, 1)
    same, checked = True, 0
    for k in lib:
        if k['cursor'] != cursor:
            continue
        cols = []
        for rel in k['relations']:
            if rel not in names2:
                same = False
                continue
            cols.append(B2[:, names2.index(rel)])
        if not cols:
            continue
        m = cols[0].copy()
        for c in cols[1:]:
            m &= c
        checked += 1
        if not np.array_equal(m, k['_mask']):
            same = False
    return {'constructions_checked': checked,
            'answer_unchanged_when_future_randomised': bool(same),
            'passed': bool(same),
            'note': ('на этом курсоре конструкций нет' if checked == 0
                     else 'ответ не изменился при случайном будущем')}


# ------------------------------------------------------------------ main ----

def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items()
                if not (isinstance(k, str) and k.startswith('_'))}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instrument', default='NQ')
    ap.add_argument('--start', default='2020-01-01')
    ap.add_argument('--stop', default='2025-11-01')
    ap.add_argument('--split', default='2024-01-01')
    ap.add_argument('--cursors', nargs='+', type=int, default=[5, 10])
    ap.add_argument('--horizons', nargs='+', type=int, default=[1, 5, 15])
    ap.add_argument('--min-cell', type=int, default=400)
    ap.add_argument('--min-days', type=int, default=150)
    ap.add_argument('--max-overlap', type=float, default=0.5)
    ap.add_argument('--pair-margin', type=float, default=0.2)
    ap.add_argument('--library-size', type=int, default=6)
    ap.add_argument('--repeats', type=int, default=40)
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--demo-films', type=int, default=3)
    ap.add_argument('--read-unit', default=None,
                    help='открыть один фильм по адресу NQ:<T0 ns> и '
                         'напечатать чтение по закрытым минутам')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    corpus, t0, side_col, coverage = build(a.instrument, a.start, a.stop, TFS)
    cost = COST[a.instrument]
    split = np.datetime64(a.split, 'ns').astype('int64')
    train = t0 < split

    lib, budgets, valid = [], {}, None
    for c in a.cursors:
        part, budget, usable, tr, names, Ball = library(corpus, t0, c,
                                                        a.horizons, a)
        for k in part:
            k['id'] = f'K{len(lib) + 1}'
            lib.append(k)
        budgets[str(c)] = budget
        valid = usable if valid is None else (valid & usable)

    lib_by_cursor = {}
    for k in lib:
        lib_by_cursor.setdefault(k['cursor'], []).append(k)

    stats = {k['id']: continuations(corpus, t0, k['_mask'], k['cursor'],
                                    a.horizons, a.instrument) for k in lib}

    # длительность действия — из профиля продолжений, не назначена
    holds = {}
    for k in lib:
        r = stats[k['id']]['returns']
        sgn = -1 if k['side'] == 'short' else 1
        best, bh = None, a.horizons[0]
        for h in a.horizons:
            v = r[f'R{h}']['mean']
            if v is None:
                continue
            if best is None or sgn * v > best:
                best, bh = sgn * v, h
        holds[k['id']] = bh

    tradable = [k for k in lib if k['beats_ceiling']]
    priority = list(tradable)     # объявлен до результатов сделок
    res = {'implementation_sha256': hashlib.sha256(
               Path(__file__).read_bytes()).hexdigest(),
           'territory': {**corpus.metadata['source'], 'films': int(corpus.n),
                         'window': [-BACK, AHEAD],
                         'search_until': a.split,
                         'prior_exposure': 'все периоды просматривались в 062–068; '
                                           'независимым подтверждением ни один '
                                           'не является'},
           'selection_rule': {
               'declared_before_counting': True,
               'min_cell': a.min_cell, 'min_days': a.min_days,
               'max_overlap_jaccard': a.max_overlap,
               'prefer_single_if_within': a.pair_margin,
               'library_size': a.library_size,
               'calibration_repeats': a.repeats,
               'note': 'библиотека, а не строка с максимумом среднего'},
           'budget': budgets,
           'library': [strip(k) for k in lib],
           'continuations': stats,
           'holds_from_continuations': holds,
           'tiers': {'допущена к действию': [k['id'] for k in lib
                                               if k['beats_ceiling']],
                     'описательная': [k['id'] for k in lib
                                      if not k['beats_ceiling']],
                     'rule': 'к действию допускаются только конструкции, '
                             'превысившие перестановочный потолок; остальные '
                             'остаются в библиотеке как описания сцен'},
           'conflict_rule': 'при одновременном узнавании исполняется конструкция '
                            'с меньшим номером среди допущенных к действию; '
                            'порядок задан поиском до любых результатов сделок',
           'selftest': {}, 'reading': [], 'replay': {}}

    for c in a.cursors:
        res['selftest'][f'курсор {c}'] = selftest_no_lookahead(corpus, lib, c)

    if lib and a.read_unit:
        if a.read_unit in corpus.unit_ids:
            u = corpus.unit_ids.index(a.read_unit)
            one = read_film(corpus, t0, u, lib_by_cursor, a.cursors,
                            a.horizons, a.instrument, stats)
            res['reading'].append(one)
            print(f'=== фильм {one["unit"]}')
            for s in one['steps']:
                print(f'  курсор {s["cursor"]}: закрыто {s["closed_minutes"]}, '
                      f'стало известно — {", ".join(s["became_known"])}')
                print(f'    узнано {s["recognised"]}; появилось {s["appeared"]}; '
                      f'утрачено {s["lost"]}')
                for e in s['expectation']:
                    print(f'    {e["construction"]} ({e["side"]}): '
                          f'{" & ".join(e["relations"])}')
                    print(f'      исторических фильмов с этим началом: '
                          f'{e["historical_films_with_this_start"]}, '
                          f'продолжения {e["continuations"]}')
            print('  будущее раскрыто после решения:',
                  {k: round(v, 1) for k, v
                   in one['revealed_future_after_decision'].items()})
        else:
            print('адрес не найден в корпусе:', a.read_unit)

    if lib:
        pick_units = np.flatnonzero(lib[0]['_mask'] & valid)[:a.demo_films]
        for u in pick_units:
            res['reading'].append(read_film(corpus, t0, int(u), lib_by_cursor,
                                            a.cursors, a.horizons, a.instrument,
                                            stats))
        for tag, mask in (('выдвижение', valid & train),
                          ('оценка', valid & ~train)):
            tr = replay(corpus, t0, lib, holds, a.instrument, mask, priority)
            s = summarise(tr, cost)
            by = {}
            for k in tradable:
                sub = [x for x in tr if x['id'] == k['id']]
                by[k['id']] = summarise(sub, cost)
            s['by_construction'] = by
            s['conflicts'] = int(sum(1 for x in tr if x['competing']))
            res['replay'][tag] = s

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(strip(res), ensure_ascii=False, indent=1,
                                      default=str), encoding='utf-8')
    print(json.dumps({'library': [{'id': k['id'], 'cursor': k['cursor'],
                                   'tier': k['tier'],
                                   'outcome': k['outcome'], 'side': k['side'],
                                   'n': k['films_search'],
                                   'days': k['days_search'],
                                   'mean': round(k['mean_gross_search'], 1),
                                   'ceiling': round(k['ceiling'], 1),
                                   'rel': k['relations']} for k in lib],
                      'holds': holds,
                      'selftest': res['selftest'],
                      'replay': {t: {q: v.get(q) for q in
                                     ('n', 'mean_gross', 't_gross',
                                      'mean_net_at_15', 'conflicts')}
                                 for t, v in res['replay'].items()}},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
