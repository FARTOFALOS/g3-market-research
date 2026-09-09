#!/usr/bin/env python3
"""Рабочий поиск на ПОЛНОМ языке порядковых отношений O/H/L/C.

ЧТО ЗДЕСЬ ИСПРАВЛЕНО
====================
В `research/minute_edge.py` генератор признаков назывался порядковым, но три
атома сравнивали ДЛИНЫ: верхняя тень против тела, нижняя против тела, верхняя
против нижней. Это измерение расстояний между ценами, а не отношение точек
«выше / ниже / на уровне». Именно такой атом вошёл в лучшую находку прошлого
цикла, поэтому та находка сохраняется вместе с точным составом признаков, но
языком формы больше не считается. Кроме того окно там было ограничено пятью
свечами, а набор признаков — выбранным вручную.

Здесь рабочим источником поиска сделано существующее полное представление
отношений — `Space` из `relational_stencil.py`. Оно строит все пары точек фильма
и три отношения между ними (<, =, >). Длины и расстояния в язык не входят
вообще. Связь с ранней свечой остаётся доступной на любом более позднем курсоре:
окна в пять свечей нет, есть только «известно к курсору».

ВЫЧИСЛИТЕЛЬНАЯ ТЕРРИТОРИЯ РАЗЛИЧАЕТСЯ В ТРИ УРОВНЯ
==================================================
- **отношение доступно** — все пары точек фильма × три отношения;
- **отношение участвовало в поиске** — доступные к курсору, известные на
  достаточном числе фильмов и невырожденные по частоте;
- **сочетание проверено** — какие именно сочетания перебраны.
Все три числа печатаются в результат. Полнота языка не требует исчерпывающего
перебора; когда перебор всё же исчерпывающий, это сказано прямо.

ЧТО КАЛИБРУЕТСЯ
===============
Вместе с сочетанием подбирается сторона (знак среднего). При разрушенной связи
повторяется весь выбор целиком, включая выбор стороны. Статус `DIAGNOSTIC`:
это потолок шума перебора, а не проверка гипотезы.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from relational_stencil import Corpus, Space, FIELDS                # noqa: E402
from minute_edge import load as tape_minutes                        # noqa: E402
from candidate_check import POINT, COST                             # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MIN = 60_000_000_000


# ------------------------------------------------- язык как рабочий источник --

def unpack(mask, n):
    """Битовая маска Space → булев вектор длины n. Одно определение языка."""
    raw = int(mask).to_bytes((n + 7) // 8, 'little')
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8),
                         bitorder='little')[:n].astype(bool)


def language(space):
    """Все доступные атомы: пары точек фильма × три отношения."""
    return int(len(space.masks))


def searchable(space, cursor, min_known, freq_band):
    """Атомы, участвующие в поиске: известны к курсору и невырождены."""
    n = space.n
    keep, meta = [], []
    for atom in range(len(space.masks)):
        d = space.describe(atom)
        if d['known_at'] > cursor:
            continue
        known = unpack(space.known[atom // 3], n)
        if known.sum() < min_known:
            continue
        truth = unpack(space.masks[atom], n) & known
        f = truth.sum() / known.sum()
        if not (freq_band[0] <= f <= freq_band[1]):
            continue
        keep.append(atom)
        meta.append(d['text'])
    return keep, meta


def matrix(space, atoms, usable):
    n = space.n
    B = np.zeros((int(usable.sum()), len(atoms)), dtype=bool)
    for j, atom in enumerate(atoms):
        B[:, j] = unpack(space.masks[atom], n)[usable]
    return B


# ------------------------------------------------------------------ перебор ---

def counts(B):
    """Количества считаются один раз: от перестановки меток они не меняются."""
    Bf = B if B.dtype == np.float32 else B.astype(np.float32)
    return Bf, Bf.sum(axis=0), Bf.T @ Bf


def scan(B, y, min_cell, cached=None):
    """Полный перебор одиночных и всех парных сочетаний. Критерий — |среднее|."""
    Bf, c1, c2 = cached if cached is not None else counts(B)
    yf = y.astype(np.float32)
    s1 = (Bf * yf[:, None]).sum(axis=0)
    s2 = (Bf * yf[:, None]).T @ Bf
    total = len(y)
    with np.errstate(invalid='ignore', divide='ignore'):
        ok1 = (c1 >= min_cell) & ((total - c1) >= min_cell)
        ok2 = (c2 >= min_cell) & ((total - c2) >= min_cell)
        m1 = np.where(ok1, s1 / np.maximum(c1, 1), np.nan)
        m2 = np.where(ok2, s2 / np.maximum(c2, 1), np.nan)
    return m1, c1, m2, c2


def pick(m1, c1, m2, c2):
    a1 = np.abs(m1)
    iu = np.triu_indices(m2.shape[0], k=1)
    a2 = np.abs(m2[iu])
    b1 = np.nanmax(a1) if np.isfinite(a1).any() else -np.inf
    b2 = np.nanmax(a2) if np.isfinite(a2).any() else -np.inf
    if b2 >= b1:
        p = int(np.nanargmax(a2))
        i, j = int(iu[0][p]), int(iu[1][p])
        return {'kind': 'pair', 'atoms': [i, j], 'mean': float(m2[i, j]),
                'n': int(c2[i, j]), 'score': float(b2)}
    i = int(np.nanargmax(a1))
    return {'kind': 'single', 'atoms': [i], 'mean': float(m1[i]),
            'n': int(c1[i]), 'score': float(b1)}


def ceiling(B, y, min_cell, repeats, seed, cached=None):
    rng = np.random.default_rng(seed)
    cached = cached if cached is not None else counts(B)
    out = []
    for _ in range(repeats):
        z = rng.permutation(y)
        out.append(pick(*scan(B, z, min_cell, cached))['score'])
    return out


# ------------------------------------------------------------- самопроверка ---

def warp(ohlc, rng):
    """Строго возрастающая деформация уровней ВНУТРИ каждого фильма.

    Все отношения «выше / ниже / на уровне» сохраняются по построению;
    расстояния между уровнями меняются произвольно.
    """
    out = np.array(ohlc, dtype=float, copy=True)
    for i in range(out.shape[0]):
        v = out[i][np.isfinite(out[i])]
        levels = np.unique(v)
        if len(levels) < 2:
            continue
        gaps = rng.uniform(0.5, 8.0, size=len(levels) - 1)
        new = np.concatenate([[float(levels[0])], levels[0] + np.cumsum(gaps)])
        flat = out[i].reshape(-1)
        good = np.isfinite(flat)
        flat[good] = np.interp(flat[good], levels, new)
        out[i] = flat.reshape(out[i].shape)
    return out


def old_wick_atoms(ohlc, j):
    """Три признака прошлого цикла — сравнения ДЛИН, а не порядка точек."""
    o, h, l, c = (ohlc[:, j, k] for k in range(4))
    top, bot = np.maximum(o, c), np.minimum(o, c)
    up, dn, body = h - top, bot - l, np.abs(c - o)
    return np.stack([up > body, dn > body, up > dn], axis=1)


def synthetic(n_per_group, rng):
    """Фильмы с ЗАРАНЕЕ ИЗВЕСТНЫМ совместным фрагментом и дальней связью.

    A: C[12] > H[1];  B: L[9] < L[2].  Исход задан «исключающим или», поэтому
    каждое отношение по отдельности неинформативно ровно, а пара — максимальна.
    """
    M = 16
    films, labels, planted = [], [], []
    for a in (True, False):
        for b in (True, False):
            for _ in range(n_per_group):
                base = 100.0 + np.cumsum(rng.normal(0, 1.0, M))
                f = np.zeros((M, 4))
                for t in range(M):
                    o = base[t]
                    c = o + rng.normal(0, 0.8)
                    f[t] = [o, max(o, c) + abs(rng.normal(0, 0.6)),
                            min(o, c) - abs(rng.normal(0, 0.6)), c]
                d = abs(rng.normal(0, 1.0)) + 0.5
                c12 = f[1, 1] + d if a else f[1, 1] - d
                o12 = c12 - rng.normal(0, 0.3)
                f[12] = [o12, max(o12, c12) + 0.4, min(o12, c12) - 0.4, c12]
                d = abs(rng.normal(0, 1.0)) + 0.5
                l9 = f[2, 2] - d if b else f[2, 2] + d
                o9 = l9 + abs(rng.normal(0, 0.5)) + 0.2
                c9 = l9 + abs(rng.normal(0, 0.5)) + 0.2
                f[9] = [o9, max(o9, c9) + 0.3, l9, c9]
                films.append(f)
                labels.append(10.0 if a == b else -10.0)
                planted.append((0 if a else 2) + (0 if b else 1))
    ordinals = np.arange(M)
    ids = [f'X:{i}' for i in range(len(films))]
    corpus = Corpus(np.asarray(films), ordinals, ids, list(ids),
                    {'instrument': 'X', 'source': {'population': 'synthetic'},
                     'anchor': 'synthetic', 'unit_definition': 'film'})
    return corpus, np.array(labels), np.array(planted)


def selftest():
    rng = np.random.default_rng(3)
    report = {}

    # --- проверка 1: смысловая инвариантность рабочего маршрута
    src = Corpus.load(str(ROOT / 'work/riz-pass/tf5_south.npz'))
    take = np.arange(min(120, src.n))
    sub = Corpus(src.ohlc[take], src.ordinals,
                 [src.unit_ids[i] for i in take],
                 [src.unit_ids[i] for i in take], dict(src.metadata))
    sub.members = [src.members[i] for i in take]
    a = Space(sub)
    warped = Corpus(warp(sub.ohlc, rng), sub.ordinals, list(sub.unit_ids),
                    list(sub.unit_ids), dict(sub.metadata))
    warped.members = list(sub.members)
    b = Space(warped)
    same = (a.masks == b.masks) and (a.known == b.known)
    before = old_wick_atoms(sub.ohlc, 5)
    after = old_wick_atoms(warped.ohlc, 5)
    changed = int((before != after).sum())
    report['invariance'] = {
        'films': int(len(take)),
        'atoms_compared': len(a.masks),
        'ordinal_description_unchanged': bool(same),
        'old_length_atoms_changed_cells': changed,
        'verdict': 'порядковое описание инвариантно; признаки прошлого цикла, '
                   'сравнивающие длины, при той же деформации меняются',
        'passed': bool(same and changed > 0)}

    # --- проверка 2: обнаружимость совместного фрагмента с дальней связью
    corpus, y, quad = synthetic(100, rng)
    space = Space(corpus)
    atoms, texts = searchable(space, cursor=15, min_known=10, freq_band=(0.05, 0.95))
    index = {a: i for i, a in enumerate(atoms)}
    a_atom = space.atom(12, 'C', '>', 1, 'H')
    b_atom = space.atom(9, 'L', '<', 2, 'L')
    usable = np.ones(space.n, bool)
    B = matrix(space, atoms, usable)
    min_cell = 80
    m1, c1, m2, c2 = scan(B, y, min_cell=min_cell)
    best = pick(m1, c1, m2, c2)
    sel = B[:, best['atoms'][0]].copy()
    for i in best['atoms'][1:]:
        sel &= B[:, i]
    planted_sets = [quad == v for v in range(4)]
    pure = any(bool((sel & ~s).sum() == 0) and int(sel.sum()) > 0
               for s in planted_sets)
    coverage = max((int((sel & s).sum()) / max(1, int(s.sum())))
                   for s in planted_sets)
    singles = {}
    for nm, at in (('A: C[12] > H[1]', a_atom), ('B: L[9] < L[2]', b_atom)):
        j = index.get(at)
        singles[nm] = None if j is None else float(m1[j])
    ia, ib = index.get(a_atom), index.get(b_atom)
    exact = None
    if ia is not None and ib is not None:
        exact = {'mean': float(m2[ia, ib]), 'n': int(c2[ia, ib])}
    a_vec = unpack(space.masks[a_atom], space.n)
    b_vec = unpack(space.masks[b_atom], space.n)
    pair_mean = float(y[a_vec & b_vec].mean())
    top = float(np.nanmax(np.abs(m2)))
    report['joint_fragment'] = {
        'films': int(space.n), 'atoms_available': language(space),
        'atoms_searched': len(atoms), 'min_cell': min_cell,
        'planted_pair': ['C[12] > H[1]', 'L[9] < L[2]'],
        'planted_pair_mean_direct': pair_mean,
        'planted_pair_in_search': exact,
        'planted_pair_is_maximal': bool(
            exact is not None and abs(abs(exact['mean']) - top) < 1e-6),
        'planted_singles_mean': singles,
        'singles_uninformative': bool(
            all(v is not None and abs(v) < 1.0 for v in singles.values())),
        'best_kind': best['kind'],
        'best_relations': [texts[i] for i in best['atoms']],
        'best_mean': best['mean'], 'best_n': best['n'],
        'best_cell_pure_within_one_planted_group': pure,
        'best_cell_coverage_of_that_group': coverage,
        'verdict': 'составляющие по отдельности неинформативны ровно; засеянная '
                   'пара присутствует в поиске и достигает максимума; лучшая '
                   'найденная ячейка целиком лежит внутри одной засеянной группы',
        'passed': bool(pure and best['kind'] == 'pair'
                       and exact is not None
                       and abs(abs(exact['mean']) - top) < 1e-6
                       and all(v is not None and abs(v) < 1.0
                               for v in singles.values()))}
    report['passed'] = all(v['passed'] for v in report.values()
                           if isinstance(v, dict))
    return report




def tape_corpus(instrument, start, stop, back, ahead, stride):
    """Фильмы из обычной ленты основной сессии: та же единица, что у RIZ.

    Окно [-back..+ahead] вокруг решающей минуты; решающая минута — ординал 0,
    её метка времени становится адресом единицы. Прореживание задаётся шагом и
    объявляется: это вычислительная территория, а не свойство рынка.
    """
    m, k, _ = tape_minutes(instrument, start, stop)
    ts = m['close_ts_utc_ns']
    k = k[(k >= back) & (k + ahead < len(ts))]
    k = k[::stride]
    ords = np.arange(-back, ahead + 1)
    idx = k[:, None] + ords[None, :]
    step_ok = (ts[idx[:, 1:]] - ts[idx[:, :-1]]) == MIN
    keep = step_ok.all(axis=1)
    k, idx = k[keep], idx[keep]
    films = np.stack([m['open'][idx], m['high'][idx], m['low'][idx],
                      m['close'][idx]], axis=2)
    ids = [f'{instrument}:{int(ts[i])}' for i in k]
    corpus = Corpus(films, ords, list(ids), list(ids),
                    {'instrument': instrument,
                     'source': {'population': 'обычные минуты основной сессии',
                                'start': start, 'stop': stop, 'stride': stride},
                     'anchor': 'решающая минута = ординал 0',
                     'unit_definition': 'одна решающая минута'})
    corpus.members = [[i] for i in range(len(ids))]
    return corpus


# ---------------------------------------------------------------- проход -----

def outcomes(corpus, cursor, horizons, instrument):
    """Исполнимые исходы от `open` минуты после курсора."""
    col = {int(o): i for i, o in enumerate(corpus.ordinals)}
    O = corpus.ohlc[:, :, 0]
    p = POINT[instrument]
    e = O[:, col[cursor + 1]]
    out = {}
    for h in horizons:
        j = cursor + 1 + h
        if j not in col:
            continue
        out[f'R{h}'] = (O[:, col[j]] - e) * p
    return out, e


def money(corpus, sel, side, cursor, hold, instrument, unit_ids):
    """Фактический поток позиций: одна позиция, вход open c+1, выход open c+1+h."""
    col = {int(o): i for i, o in enumerate(corpus.ordinals)}
    O = corpus.ohlc[:, :, 0]
    p, cost = POINT[instrument], COST[instrument]
    t0 = np.array([int(u.split(':')[1]) for u in unit_ids])
    order = np.argsort(t0)
    busy, tr = -1, []
    for i in order:
        if not sel[i]:
            continue
        a, b = col[cursor + 1], col[cursor + 1 + hold]
        if not (np.isfinite(O[i, a]) and np.isfinite(O[i, b])):
            continue
        t_in = int(t0[i]) + (cursor + 1) * MIN
        if t_in <= busy:
            continue
        tr.append((t_in, float(side * (O[i, b] - O[i, a]) * p)))
        busy = int(t0[i]) + (cursor + 1 + hold) * MIN
    return tr


def summarise(tr, cost):
    if not tr:
        return {'n': 0}
    g = np.array([t[1] for t in tr])
    ts = np.array([t[0] for t in tr])
    year = (ts // MIN) // int(365.2425 * 1440) + 1970
    net = g - cost
    eq = np.cumsum(net)
    sem = float(g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else 0.0
    by = {int(y): float(net[year == y].sum()) for y in np.unique(year)}
    byg = {int(y): float(g[year == y].mean()) for y in np.unique(year)}
    keep = np.sort(g)[:-max(1, len(g) // 100)]
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
            'gross_without_top1pct': float(keep.mean()) if len(keep) else None}


def run(a):
    if getattr(a, 'tape', False):
        corpus = tape_corpus(a.instrument, a.start, a.stop, a.back, a.ahead,
                             a.stride)
        source = {'tape': [a.start, a.stop], 'back': a.back, 'ahead': a.ahead,
                  'stride': a.stride}
    else:
        corpora = [Corpus.load(path) for path in a.inputs]
        ohlc = np.concatenate([c.ohlc for c in corpora])
        unit_ids = [u for c in corpora for u in c.unit_ids]
        corpus = Corpus(ohlc, corpora[0].ordinals, list(unit_ids),
                        list(unit_ids), dict(corpora[0].metadata))
        corpus.members = [m for c in corpora for m in c.members]
        source = {'inputs': list(a.inputs)}
    unit_ids = list(corpus.unit_ids)
    ordinals = corpus.ordinals
    space = Space(corpus)
    cost = COST[a.instrument]

    res = {'implementation_sha256': hashlib.sha256(
               Path(__file__).read_bytes()).hexdigest(),
           'language': 'полное представление отношений O/H/L/C: все пары точек '
                       'фильма × три отношения (<, =, >); длин и расстояний нет',
           'territory': {**source, 'films': int(corpus.n),
                         'window': [int(ordinals[0]), int(ordinals[-1])],
                         'anchor': 'T0 сохранён в идентификаторе каждой единицы',
                         'unit': 'один T0'},
           'atoms_available': language(space),
           'cursors': {}}

    for cursor in a.cursors:
        out, entry = outcomes(corpus, cursor, a.horizons, a.instrument)
        usable = np.isfinite(entry)
        for y in out.values():
            usable &= np.isfinite(y)
        atoms, texts = searchable(space, cursor, a.min_known, tuple(a.freq_band))
        B = matrix(space, atoms, usable)
        A = len(atoms)
        cached = counts(B)
        block = {'films_usable': int(usable.sum()),
                 'films_dropped_unknown': int((~usable).sum()),
                 'atoms_searched': A,
                 'combinations_tested': A + A * (A - 1) // 2,
                 'exhaustive': True,
                 'min_cell': a.min_cell,
                 'entry': f'open m={cursor + 1}',
                 'outcomes': {}}
        for label, y in out.items():
            yy = y[usable]
            m1, c1, m2, c2 = scan(B, yy, a.min_cell, cached)
            best = pick(m1, c1, m2, c2)
            drawn = ceiling(B, yy, a.min_cell, a.repeats, a.seed, cached)
            beats = sum(1 for v in drawn if v >= best['score'])
            iu = np.triu_indices(A, k=1)
            order2 = np.argsort(-np.nan_to_num(np.abs(m2[iu]), nan=-1))[:5]
            block['outcomes'][label] = {
                'baseline_mean_usd': float(yy.mean()),
                'baseline_sd_usd': float(yy.std(ddof=1)),
                'best_relations': [texts[i] for i in best['atoms']],
                'best_kind': best['kind'],
                'best_mean_gross_usd': best['mean'], 'best_n': best['n'],
                'best_side': 'long' if best['mean'] > 0 else 'short',
                'calibration_max': max(drawn),
                'calibration_median': float(np.median(drawn)),
                'repeats_not_worse': beats,
                'survives_calibration': bool(beats == 0),
                'clears_cost': bool(abs(best['mean']) > cost),
                'top5_pairs': [
                    {'relations': [texts[int(iu[0][q])], texts[int(iu[1][q])]],
                     'mean_gross': float(m2[int(iu[0][q]), int(iu[1][q])]),
                     'n': int(c2[int(iu[0][q]), int(iu[1][q])])}
                    for q in order2]}
            sel_full = np.zeros(corpus.n, bool)
            s = B[:, best['atoms'][0]].copy()
            for i in best['atoms'][1:]:
                s &= B[:, i]
            sel_full[np.flatnonzero(usable)] = s
            side = 1 if best['mean'] > 0 else -1
            streams = {}
            for h in a.horizons:
                if cursor + 1 + h > int(ordinals[-1]):
                    continue
                streams[f'hold_{h}'] = summarise(
                    money(corpus, sel_full, side, cursor, h, a.instrument,
                          unit_ids), cost)
            block['outcomes'][label]['streams'] = streams
            block['outcomes'][label]['example_units'] = [
                unit_ids[i] for i in np.flatnonzero(sel_full)[:5]]
        res['cursors'][str(cursor)] = block
    return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    st = sub.add_parser('selftest')
    st.add_argument('--out', required=True)
    rn = sub.add_parser('run')
    rn.add_argument('--inputs', nargs='+', default=[])
    rn.add_argument('--tape', action='store_true')
    rn.add_argument('--start', default='2020-01-01')
    rn.add_argument('--stop', default='2025-11-01')
    rn.add_argument('--back', type=int, default=9)
    rn.add_argument('--ahead', type=int, default=16)
    rn.add_argument('--stride', type=int, default=5)
    rn.add_argument('--instrument', default='NQ')
    rn.add_argument('--cursors', nargs='+', type=int, default=[3, 5, 10])
    rn.add_argument('--horizons', nargs='+', type=int, default=[1, 5, 15])
    rn.add_argument('--min-known', type=int, default=200)
    rn.add_argument('--min-cell', type=int, default=150)
    rn.add_argument('--freq-band', nargs=2, type=float, default=[0.05, 0.95])
    rn.add_argument('--repeats', type=int, default=40)
    rn.add_argument('--seed', type=int, default=7)
    rn.add_argument('--out', required=True)
    a = ap.parse_args()

    if a.cmd == 'selftest':
        res = selftest()
    else:
        res = run(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1,
                                      default=str), encoding='utf-8')
    if a.cmd == 'selftest':
        print(json.dumps(res, ensure_ascii=False, indent=1))
    else:
        brief = {'films': res['territory']['films'],
                 'atoms_available': res['atoms_available'],
                 'cursors': {}}
        for c, b in res['cursors'].items():
            brief['cursors'][c] = {
                'atoms_searched': b['atoms_searched'],
                'tested': b['combinations_tested'],
                'usable': b['films_usable'],
                'outcomes': {k: {'best': round(v['best_mean_gross_usd'], 2),
                                 'n': v['best_n'],
                                 'ceil': round(v['calibration_max'], 2),
                                 'surv': v['survives_calibration'],
                                 'cost': v['clears_cost'],
                                 'rel': v['best_relations']}
                             for k, v in b['outcomes'].items()}}
        print(json.dumps(brief, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
