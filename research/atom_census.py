#!/usr/bin/env python3
"""Спектр поддержки одиночных отношений: ПОЛНОЕ перечисление, без порога.

204 точки (51 свеча x O/H/L/C), 20 706 пар, 62 118 отношений {<,=,>}.
Для каждого отношения считаются: число фильмов, где оно истинно, и число
фильмов, где обе точки известны (знаменатель этого отношения). Порога здесь
нет — это перепись, а не отбор.

Отдельно помечаются:
- анатомически вынужденные отношения внутри одной свечи (H>=O,H>=C,H>=L,L<=O,
  L<=C и т.п.) — они не сообщают о рынке ничего сверх определения свечи;
- сцепка соседних свечей C[k] ? O[k+1] — измеряется и показывается отдельно.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from film_corpus import build, FIELDS  # noqa: E402

SIGNS = ('<', '=', '>')


def point_names(n_candles):
    return [(k, f) for k in range(n_candles) for f in FIELDS]


def forced(a, b):
    """Отношение вынуждено анатомией свечи (одна и та же свеча)."""
    (ka, fa), (kb, fb) = a, b
    if ka != kb:
        return None
    hi, lo = 'H', 'L'
    if fa == hi and fb != hi:
        return '>='
    if fb == hi and fa != hi:
        return '<='
    if fa == lo and fb != lo:
        return '<='
    if fb == lo and fa != lo:
        return '>='
    return None


def census(c, chunk=256):
    x = c['ohlc']
    n, k, _ = x.shape
    flat = x.reshape(n, k * 4)
    pts = point_names(k)
    li, ri = np.triu_indices(k * 4, 1)
    P = len(li)
    cnt = np.zeros((P, 3), dtype=np.int64)
    known = np.zeros(P, dtype=np.int64)
    for s in range(0, P, chunk):
        e = min(s + chunk, P)
        a = flat[:, li[s:e]]
        b = flat[:, ri[s:e]]
        ok = np.isfinite(a) & np.isfinite(b)
        known[s:e] = ok.sum(0)
        cnt[s:e, 0] = ((a < b) & ok).sum(0)
        cnt[s:e, 1] = ((a == b) & ok).sum(0)
        cnt[s:e, 2] = ((a > b) & ok).sum(0)
    return pts, li, ri, cnt, known


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--instrument', default='NQ')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    c = build(a.instrument)
    pts, li, ri, cnt, known = census(c)
    n = c['ohlc'].shape[0]
    freq = np.divide(cnt, known[:, None], out=np.full(cnt.shape, np.nan),
                     where=known[:, None] > 0)
    kinds = np.zeros(len(li), dtype='<U12')
    adj = np.zeros(len(li), dtype=bool)
    same = np.zeros(len(li), dtype=bool)
    for q in range(len(li)):
        pa, pb = pts[li[q]], pts[ri[q]]
        if pa[0] == pb[0]:
            same[q] = True
            kinds[q] = 'внутри' if forced(pa, pb) else 'внутри-своб'
        else:
            kinds[q] = 'между'
            if pb[0] - pa[0] == 1:
                adj[q] = True
    # спектр: сколько отношений имеет частоту >= уровня
    levels = [0.999999, 0.999, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3,
              0.2, 0.1, 0.05, 0.01, 0.001]
    f = freq.reshape(-1)
    kin = np.repeat(kinds, 3)
    free = kin != 'внутри'
    spectrum = {str(L): {'все': int(np.nansum(f >= L)),
                         'без анатомии свечи': int(np.nansum((f >= L) & free))}
                for L in levels}
    chain = {}
    K = c['ohlc'].shape[1]
    for q in np.flatnonzero(adj):
        pa, pb = pts[li[q]], pts[ri[q]]
        if pa[1] == 'C' and pb[1] == 'O':
            chain[pa[0]] = {'C=O_next': float(freq[q, 1]),
                            'C<O_next': float(freq[q, 0]),
                            'C>O_next': float(freq[q, 2]),
                            'known': int(known[q])}
    eq = np.array([v['C=O_next'] for v in chain.values()])
    res = {'instrument': a.instrument, 'films': int(n),
           'points': len(pts), 'pairs': int(len(li)),
           'relations_enumerated': int(3 * len(li)),
           'enumeration': 'полное; порога нет',
           'candle_anatomy_forced_relations':
               int(3 * int(same.sum()) - int(np.nansum(f[np.repeat(same, 3)] > 0))),
           'support_spectrum_share': spectrum,
           'adjacent_chaining_C_eq_O_next': {
               'minutes': len(chain), 'mean': float(eq.mean()),
               'min': float(eq.min()), 'max': float(eq.max()),
               'note': 'сцепка НЕ является законом: равенство держится в '
                       'меньшинстве случаев, поэтому отношения между соседними '
                       'свечами несут рыночное сведение'},
           }
    out = Path(a.out or f'work/070/atom_census_{a.instrument}.npz')
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, cnt=cnt, known=known, li=li, ri=ri,
                        pts=np.array([f'{k}{f}' for k, f in pts]))
    Path(str(out).replace('.npz', '.json')).write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
