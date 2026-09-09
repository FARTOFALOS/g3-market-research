#!/usr/bin/env python3
"""Карта повторений: частота как ось, точное перечисление в объявленной области.

ТЕРРИТОРИЯ КАРТЫ
================
Фильм — T0 и минуты +1…+50, 51 свеча. Карта относится ко ВСЕМУ этому окну.
Язык — порядковые отношения {<,=,>} между точками O/H/L/C любых двух свечей
фильма. Ни размеры, ни объём, ни доходность, ни исход сделки в обнаружении,
ранжировании и отсечении не участвуют.

ЧТО ТОЧНО, А ГДЕ КОНЧИЛСЯ БЮДЖЕТ
================================
- Глубина 1 (одиночные отношения): перечисление ТОЧНОЕ и ПОЛНОЕ на всех
  51 свече, без всякого порога. Спектр поддержки сохраняется целиком.
- Глубина 2 (пары): ТОЧНЫЙ ПОДСЧЁТ ЧИСЛА пар на каждом уровне поддержки —
  фронтир по уровням, а не одна отсечка. Абсолютная поддержка антимонотонна,
  поэтому предварительное отсечение атомов ниже самого низкого уровня не
  теряет ни одной пары этого фронтира. Ниже нижнего уровня область не
  перечислялась и так и называется.
- Глубина 3: точное расширение пар, попавших в витрину, в порядке убывания
  поддержки. Excess в этом порядке не участвует. Непросмотренная часть
  называется числом нерасширенных пар.

Порог поддержки — вычислительная граница, не утверждение о важности.

ЗНАМЕНАТЕЛЬ И ПРОПУСКИ
======================
Пропущенная минута ленты — NaN на своём месте. Знаменатель отношения — фильмы,
где известны обе точки; знаменатель конъюнкции — где известны все её свечи.
Ось отсечения — абсолютная поддержка (число фильмов), она монотонна.

ЗАМЫКАНИЕ И ДЕДУПЛИКАЦИЯ
========================
Конструкции схлопываются только при СОВПАДАЮЩЕМ множестве поддерживающих
фильмов (extent), а не при равном их числе.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from film_corpus import build, projections, FIELDS  # noqa: E402

SIGNS = ('<', '=', '>')


# ------------------------------------------------------------- язык ---------

def anatomy_constant(a, b, s):
    """Отношение, вынужденное анатомией одной свечи: всегда ложно."""
    (ka, fa), (kb, fb) = a, b
    if ka != kb:
        return False
    op = SIGNS[s]
    if fa == 'H' and fb in ('O', 'C', 'L'):
        return op == '<'
    if fb == 'H' and fa in ('O', 'C', 'L'):
        return op == '>'
    if fa == 'L' and fb in ('O', 'C', 'H'):
        return op == '>'
    if fb == 'L' and fa in ('O', 'C', 'H'):
        return op == '<'
    return False


def language(lo, hi):
    pts = [(k, f) for k in range(lo, hi + 1) for f in FIELDS]
    li, ri = np.triu_indices(len(pts), 1)
    out = []
    for q in range(len(li)):
        a, b = pts[int(li[q])], pts[int(ri[q])]
        for s in range(3):
            if anatomy_constant(a, b, s):
                continue
            out.append((int(li[q]), int(ri[q]), s, a, b))
    return pts, out


def atom_name(rec):
    _, _, s, a, b = rec
    return f'{a[1]}[{a[0]}] {SIGNS[s]} {b[1]}[{b[0]}]'


def atom_kind(rec):
    _, _, _, a, b = rec
    if a[0] == b[0]:
        return 'внутри свечи'
    return 'соседние' if b[0] - a[0] == 1 else 'дальняя'


def atom_candles(rec):
    _, _, _, a, b = rec
    return (a[0], b[0])


# ------------------------------------------------- упакованное хранилище ----

class Packed:
    """Столбцы истинности, упакованные по фильмам. Разворот блоками по запросу."""

    def __init__(self, corpus, recs, lo, hi):
        x = corpus['ohlc'][:, lo:hi + 1, :]
        self.n = x.shape[0]
        flat = x.reshape(self.n, -1)
        self.A = len(recs)
        self.data = np.zeros(((self.n + 7) // 8, self.A), dtype=np.uint8)
        self.support = np.zeros(self.A, dtype=np.int64)
        step = 2048
        for s in range(0, self.A, step):
            e = min(s + step, self.A)
            blk = np.zeros((self.n, e - s), dtype=bool)
            for j, (l, r, sg, a, b) in enumerate(recs[s:e]):
                u, v = flat[:, l], flat[:, r]
                ok = np.isfinite(u) & np.isfinite(v)
                rel = (u < v) if sg == 0 else ((u == v) if sg == 1 else (u > v))
                blk[:, j] = rel & ok
            self.data[:, s:e] = np.packbits(blk, axis=0)
            self.support[s:e] = blk.sum(0)

    def take(self, idx):
        """Подмножество столбцов как новый Packed-подобный контейнер."""
        out = object.__new__(Packed)
        out.n, out.A = self.n, len(idx)
        out.data = np.ascontiguousarray(self.data[:, idx])
        out.support = self.support[idx]
        return out

    def block(self, s, e, dtype=np.float32):
        return np.unpackbits(self.data[:, s:e], axis=0)[:self.n].astype(dtype)

    def bool_block(self, s, e):
        return np.unpackbits(self.data[:, s:e], axis=0)[:self.n].astype(bool)


# ------------------------------------------------- точный фронтир по парам --

def pair_frontier(P, levels_abs, showcase_level, block=512, log=print):
    """Точный счёт пар на каждом уровне + витрина выше showcase_level."""
    A = P.A
    counts = {int(L): 0 for L in levels_abs}
    keep_i, keep_j, keep_t = [], [], []
    nb = (A + block - 1) // block
    t_start = time.time()
    # разворот один раз в uint8: побитовая распаковка на каждом блоке была
    # в шесть раз дороже самого умножения
    X = np.zeros((P.n, A), dtype=np.uint8)
    for s in range(0, A, block):
        e = min(s + block, A)
        X[:, s:e] = P.bool_block(s, e)
    for bi in range(nb):
        s1, e1 = bi * block, min((bi + 1) * block, A)
        L = X[:, s1:e1].astype(np.float32)
        for bj in range(bi, nb):
            s2, e2 = bj * block, min((bj + 1) * block, A)
            R = L if bj == bi else X[:, s2:e2].astype(np.float32)
            T = L.T @ R
            if bj == bi:
                T = np.triu(T, 1)
            for lv in levels_abs:
                counts[int(lv)] += int((T >= lv).sum())
            loc = np.argwhere(T >= showcase_level)
            if len(loc):
                keep_i.append(loc[:, 0] + s1)
                keep_j.append(loc[:, 1] + s2)
                keep_t.append(T[loc[:, 0], loc[:, 1]])
        done = (bi + 1) / nb
        log(f'  пары: блок {bi + 1}/{nb}, витрина {sum(len(v) for v in keep_i)}, '
            f'{time.time() - t_start:.0f}s, оценка всего '
            f'{(time.time() - t_start) / max(done, 1e-9) * (1 - 0) :.0f}s')
    if not keep_i:
        z = np.zeros(0, dtype=np.int64)
        return counts, z, z, z
    return (counts, np.concatenate(keep_i).astype(np.int64),
            np.concatenate(keep_j).astype(np.int64),
            np.concatenate(keep_t).astype(np.int64))


def extent_key(mask):
    return hashlib.blake2b(np.packbits(mask).tobytes(), digest_size=16).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instrument', default='NQ')
    ap.add_argument('--lo', type=int, default=0)
    ap.add_argument('--hi', type=int, default=50)
    ap.add_argument('--levels', nargs='+', type=float,
                    default=[0.90, 0.80, 0.70, 0.60, 0.50, 0.40])
    ap.add_argument('--showcase-level', type=float, default=0.80)
    ap.add_argument('--block', type=int, default=512)
    ap.add_argument('--triple-budget', type=int, default=2000)
    ap.add_argument('--informative', nargs=2, type=float, default=None,
                    metavar=('LO', 'HI'),
                    help='оставить в языке только отношения с частотой в полосе '
                         '[LO,HI]. Это объявленное ОГРАНИЧЕНИЕ ЯЗЫКА, а не '
                         'отбор по исходу: около-постоянные отношения (H>L, '
                         'перекрытие соседних диапазонов) почти ничего не '
                         'сообщают о том, какой это фильм, и забивают витрину.')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    c = build(a.instrument)
    n = c['ohlc'].shape[0]
    pts, recs = language(a.lo, a.hi)
    names = [atom_name(r) for r in recs]
    kinds = np.array([atom_kind(r) for r in recs])
    cand = np.array([atom_candles(r) for r in recs])
    print(f'{a.instrument}: фильмов {n}, точек {len(pts)}, атомов {len(recs)}',
          flush=True)

    P = Packed(c, recs, a.lo, a.hi)
    present = np.isfinite(c['ohlc'][:, a.lo:a.hi + 1, 0])
    kn1 = present[:, cand[:, 0]] & present[:, cand[:, 1]]
    kn1 = kn1.sum(0).astype(np.int64)
    sup1 = P.support
    f1 = np.divide(sup1, kn1, out=np.full(len(recs), np.nan), where=kn1 > 0)

    lo_level = min(a.levels)
    sel = np.flatnonzero(sup1 >= int(round(lo_level * n)))
    band = None
    if a.informative is not None:
        band = tuple(a.informative)
        near_const = int(np.nansum(f1 > band[1]))
        near_never = int(np.nansum(f1 < band[0]))
        sel = np.flatnonzero((sup1 >= int(round(lo_level * n)))
                             & (f1 >= band[0]) & (f1 <= band[1]))
        print(f'полоса информативности {band}: выброшено около-постоянных '
              f'{near_const}, около-никогда {near_never}', flush=True)
    print(f'атомов выше нижнего уровня {lo_level}: {len(sel)}', flush=True)
    Ps = P.take(sel)
    del P

    levels_abs = [int(round(L * n)) for L in a.levels]
    counts, i2, j2, t2 = pair_frontier(
        Ps, levels_abs, int(round(a.showcase_level * n)), block=a.block)
    print(f'витрина пар: {len(t2)}', flush=True)

    res = {
        'instrument': a.instrument, 'films': int(n),
        'window': [a.lo, a.hi],
        'territory': 'весь фильм T0…T0+50',
        'unit': c['meta']['unit'],
        'language': {'points': len(pts), 'atoms': len(recs),
                     'dropped': 'только всегда-ложные отношения анатомии свечи'},
        'depth1': {
            'enumeration': 'точная и полная, без порога',
            'spectrum_by_frequency': {
                str(L): int(np.nansum(f1 >= L))
                for L in [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3,
                          0.2, 0.1, 0.05]},
            'by_kind_above_lowest_level': {
                k: int(((kinds == k) & (sup1 >= int(round(lo_level * n)))).sum())
                for k in ('внутри свечи', 'соседние', 'дальняя')}},
        'depth2': {
            'enumeration': 'точный счёт числа пар на каждом уровне поддержки',
            'frontier_pairs_at_or_above': {
                str(L): counts[int(round(L * n))] for L in a.levels},
            'informative_band': band,
            'atoms_kept_for_pairs': int(len(sel)),
            'lowest_level_examined': lo_level,
            'unexamined': f'пары с поддержкой ниже {lo_level} не перечислялись',
            'showcase_level': a.showcase_level,
            'showcase_pairs': int(len(t2))},
    }

    # ---------------------------------------------------- витрина ----------
    def card(idxs):
        m = np.ones(n, bool)
        kn = np.ones(n, bool)
        cs = set()
        for i in idxs:
            col = Ps.bool_block(i, i + 1)[:, 0]
            m &= col
            cs.update(cand[sel[i]].tolist())
        for k in cs:
            kn &= present[:, k]
        pr = projections(c, m)
        return {'relations': [names[sel[i]] for i in idxs],
                'candles_used': sorted(cs),
                'known_at_minute': int(max(cs)),
                'support_films': int(m.sum()),
                'known_films': int(kn.sum()),
                'frequency_on_known': float(m.sum() / max(kn.sum(), 1)),
                'days': pr.get('days'), 'riz_ids': pr.get('riz_ids'),
                'years_covered': len(pr.get('years', {})),
                'side': pr.get('side'),
                '_extent': extent_key(m), '_mask': m}

    show, seen = [], set()
    for q in np.argsort(-t2):
        r = card([int(i2[q]), int(j2[q])])
        if r['_extent'] in seen:
            continue
        seen.add(r['_extent'])
        pi, pj = f1[sel[i2[q]]], f1[sel[j2[q]]]
        r['frequency_if_independent'] = float(pi * pj)
        r['excess'] = float(r['frequency_on_known'] - pi * pj)
        show.append(r)
        if len(show) >= 60:
            break
    res['depth2_showcase_by_support'] = [
        {k: v for k, v in r.items() if not k.startswith('_')} for r in show]

    # ---------------------------------------------------- глубина 3 --------
    order = np.argsort(-t2)[:a.triple_budget]
    trip_count = 0
    trips = []
    for q in order:
        m = (Ps.bool_block(int(i2[q]), int(i2[q]) + 1)[:, 0]
             & Ps.bool_block(int(j2[q]), int(j2[q]) + 1)[:, 0])
        cnt = np.zeros(Ps.A, dtype=np.int64)
        step = 8192
        for s in range(0, Ps.A, step):
            e = min(s + step, Ps.A)
            cnt[s:e] = Ps.bool_block(s, e)[m].sum(0)
        thr = int(round(a.showcase_level * n))
        for k in np.flatnonzero(cnt >= thr):
            if k in (i2[q], j2[q]):
                continue
            trip_count += 1
            if len(trips) < 4000:
                trips.append((int(i2[q]), int(j2[q]), int(k), int(cnt[k])))
    res['depth3'] = {
        'enumeration': 'точное расширение пар витрины, порядок — по поддержке',
        'pairs_extended': int(len(order)),
        'showcase_pairs_total': int(len(t2)),
        'unextended_pairs': int(max(0, len(t2) - len(order))),
        'triples_at_or_above_showcase_level': trip_count,
        'note': 'excess в порядке расширения не участвует'}

    out = Path(a.out or f'work/070/repmap_{a.instrument}_{a.lo}_{a.hi}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    res['implementation_sha256'] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                   encoding='utf-8')
    np.savez_compressed(str(out).replace('.json', '_atoms.npz'),
                        sel=sel, sup1=sup1, kn1=kn1, i2=i2, j2=j2, t2=t2,
                        names=np.array(names), kinds=kinds, cand=cand)
    print(json.dumps({k: res[k] for k in ('instrument', 'films', 'window',
                                          'language', 'depth1', 'depth2',
                                          'depth3')},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
