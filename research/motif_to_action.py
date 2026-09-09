#!/usr/bin/env python3
"""От карты повторений к раннему действию: семейства, продолжения, контроли.

ЦЕПЬ
====
1. Язык — порядковые отношения O/H/L/C на префиксе фильма 0..cursor. Всё, что
   участвует в узнавании, закрыто к минуте `cursor` по построению языка.
2. Область поиска — ТОЧНОЕ перечисление пар выше объявленного уровня поддержки.
3. Витрина семейств отбирается ТОЛЬКО по структуре: уровень поддержки, число
   дней, охват лет, различность по Жаккару. Ни деньги, ни доходность, ни MFE,
   ни объём в отборе не участвуют. Размер замыкания записывается как свойство
   находки и в отборе НЕ участвует.
   Порядок рассмотрения внутри уровня — псевдослучайный с объявленным зерном:
   иначе край уровня сам определяет состав витрины (при плотности в десятки
   пар на единицу поддержки максимум внутри уровня всегда лежит у его границы).
4. Продолжение описывается порядково И проверяется двумя контролями:
   - **контроль по расстоянию**: фильмы всей популяции с тем же отношением
     расстояний до двух барьеров префикса. Если различие исчезает — это
     геометрия текущего состояния, а не историческая добавка;
   - **симметричные барьеры**: барьер по обе стороны на одинаковом удалении
     b = min(вверх, вниз). Здесь асимметрия расстояний устранена построением.
5. Только после этого считаются деньги. Числа снимаются с продолжений.

КАЛЕНДАРЬ (объявлен до счёта)
=============================
Разведка: T0 < --split. Второй участок: T0 >= --split. Нетронутость второго
участка НЕ установлена (циклы 062–069 работали на 2020–2025), поэтому это
разведочное разделение, а не подтверждение.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from film_corpus import build, projections                      # noqa: E402
from repetition_map import (language, atom_name, atom_candles,  # noqa: E402
                            Packed, extent_key)
from calendar_utils import year_of, date_key                    # noqa: E402
from candidate_check import POINT, COST                         # noqa: E402


# ------------------------------------------------ барьеры и продолжения ----

def barriers(c, cursor):
    """Расстояния от закрытия курсора до двух барьеров префикса, в пунктах."""
    H = c['ohlc'][:, :cursor + 1, 1]
    L = c['ohlc'][:, :cursor + 1, 2]
    C = c['ohlc'][:, cursor, 3]
    hi = np.nanmax(H, axis=1)
    lo = np.nanmin(L, axis=1)
    return hi, lo, C


def first_touch(c, cursor, up_lvl, dn_lvl):
    """Первое касание уровней тенями после курсора. -1 — не наступило."""
    H = c['ohlc'][:, cursor + 1:, 1]
    L = c['ohlc'][:, cursor + 1:, 2]
    up = H > up_lvl[:, None]
    dn = L < dn_lvl[:, None]
    fu = np.where(up.any(1), up.argmax(1), -1)
    fd = np.where(dn.any(1), dn.argmax(1), -1)
    return fu, fd


def race(fu, fd, mask):
    u, d = fu[mask], fd[mask]
    first_up = (u >= 0) & ((d < 0) | (u < d))
    first_dn = (d >= 0) & ((u < 0) | (d < u))
    same = (u >= 0) & (d >= 0) & (u == d)
    dec = int((first_up & ~same).sum()) + int((first_dn & ~same).sum())
    return {'films': int(mask.sum()),
            'first_up': int((first_up & ~same).sum()),
            'first_down': int((first_dn & ~same).sum()),
            'same_minute': int(same.sum()),
            'neither': int(((u < 0) & (d < 0)).sum()),
            'share_up_among_decided': (int((first_up & ~same).sum()) / dec
                                       if dec else None),
            'median_minutes_up': (float(np.median(u[u >= 0]) + 1)
                                  if (u >= 0).any() else None),
            'median_minutes_down': (float(np.median(d[d >= 0]) + 1)
                                    if (d >= 0).any() else None)}


def distance_matched_control(ratio, pool, mask, fu, fd, bins=20):
    """Контроль: та же популяция, то же отношение расстояний до барьеров."""
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(ratio, edges) - 1, 0, bins - 1)
    tgt = np.bincount(b[mask], minlength=bins).astype(float)
    if tgt.sum() == 0:
        return None
    w = tgt / tgt.sum()
    up = np.zeros(bins)
    dn = np.zeros(bins)
    for k in range(bins):
        sel = pool & (b == k) & ~mask
        if sel.sum() < 30:
            up[k] = np.nan
            dn[k] = np.nan
            continue
        r = race(fu, fd, sel)
        up[k] = r['first_up']
        dn[k] = r['first_down']
    ok = np.isfinite(up) & ((up + dn) > 0)
    if not ok.any():
        return None
    share = up[ok] / (up[ok] + dn[ok])
    ww = w[ok] / w[ok].sum()
    return {'share_up_matched_control': float((share * ww).sum()),
            'bins_used': int(ok.sum()),
            'coverage_of_family_weight': float(w[ok].sum())}


# ------------------------------------------------------------- деньги ------

def money(c, mask, cursor, hold, side, instrument, cost=None):
    O = c['ohlc'][:, :, 0]
    j_in, j_out = cursor + 1, cursor + 1 + hold
    if j_out > O.shape[1] - 1:
        return None
    idx = np.flatnonzero(mask)
    ok = np.isfinite(O[idx, j_in]) & np.isfinite(O[idx, j_out])
    miss = int((~ok).sum())
    idx = idx[ok]
    if len(idx) < 30:
        return {'n': int(len(idx)), 'note': 'слишком мало исполнимых случаев'}
    g = side * (O[idx, j_out] - O[idx, j_in]) * POINT[instrument]
    cost = COST[instrument] if cost is None else cost
    ts = c['t0'][idx]
    yr = year_of(ts)
    sem = float(g.std(ddof=1) / np.sqrt(len(g)))
    eq = np.cumsum(g - cost)
    return {'n': int(len(g)), 'days': int(len(np.unique(date_key(ts)))),
            'mean_gross': float(g.mean()), 'sem': sem,
            't_gross': float(g.mean() / sem) if sem else None,
            'median_gross': float(np.median(g)),
            'share_positive': float((g > 0).mean()),
            'breakeven_cost_usd': float(g.mean()),
            'mean_net_at_cost': float(g.mean() - cost),
            'total_net_at_cost': float((g - cost).sum()),
            'drawdown_net_at_cost': float(np.max(np.maximum.accumulate(eq) - eq)),
            'by_year_mean_gross': {int(y): float(g[yr == y].mean())
                                   for y in np.unique(yr)},
            'unexecutable_missing_price': miss}


def jac(a, b):
    u = int((a | b).sum())
    return int((a & b).sum()) / u if u else 0.0


# ------------------------------------------------------------- перебор -----

def exact_pairs(P, min_abs, block=1024, log=print):
    """Точное перечисление пар с поддержкой >= min_abs. uint8 в памяти."""
    A = P.A
    X = np.zeros((P.n, A), dtype=np.uint8)
    for s in range(0, A, block):
        e = min(s + block, A)
        X[:, s:e] = P.bool_block(s, e)
    ii, jj, tt = [], [], []
    nb = (A + block - 1) // block
    for bi in range(nb):
        s1, e1 = bi * block, min((bi + 1) * block, A)
        L = X[:, s1:e1].astype(np.float32)
        for bj in range(bi, nb):
            s2, e2 = bj * block, min((bj + 1) * block, A)
            R = L if bj == bi else X[:, s2:e2].astype(np.float32)
            T = L.T @ R
            if bj == bi:
                T = np.triu(T, 1)
            loc = np.argwhere(T >= min_abs)
            if len(loc):
                ii.append(loc[:, 0] + s1)
                jj.append(loc[:, 1] + s2)
                tt.append(T[loc[:, 0], loc[:, 1]])
        log(f'  пары: {bi + 1}/{nb}, найдено {sum(len(v) for v in ii)}')
    del X
    if not ii:
        z = np.zeros(0, np.int64)
        return z, z, z
    return (np.concatenate(ii).astype(np.int64),
            np.concatenate(jj).astype(np.int64),
            np.concatenate(tt).astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instrument', default='NQ')
    ap.add_argument('--cursor', type=int, default=12)
    ap.add_argument('--split', default='2020-01-01')
    ap.add_argument('--levels', nargs='+', type=float,
                    default=[0.60, 0.40, 0.25, 0.15, 0.08, 0.04, 0.02])
    ap.add_argument('--informative', nargs=2, type=float, default=[0.05, 0.90])
    ap.add_argument('--per-bucket', type=int, default=3)
    ap.add_argument('--min-days', type=int, default=200)
    ap.add_argument('--min-years', type=int, default=8)
    ap.add_argument('--min-support-train', type=int, default=400)
    ap.add_argument('--max-overlap', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=70)
    ap.add_argument('--holds', nargs='+', type=int, default=[1, 3, 5, 10, 20])
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    c = build(a.instrument)
    n = c['ohlc'].shape[0]
    split = np.datetime64(a.split, 'ns').astype('int64')
    train = c['t0'] < split
    pts, recs = language(0, a.cursor)
    names = [atom_name(r) for r in recs]
    cand = np.array([atom_candles(r) for r in recs])
    P = Packed(c, recs, 0, a.cursor)
    present = np.isfinite(c['ohlc'][:, :a.cursor + 1, 0])
    kn1 = (present[:, cand[:, 0]] & present[:, cand[:, 1]]).sum(0)
    f1 = np.divide(P.support, kn1, out=np.full(len(recs), np.nan), where=kn1 > 0)
    lo_b, hi_b = a.informative
    sel = np.flatnonzero((f1 >= lo_b) & (f1 <= hi_b))
    Ps = P.take(sel)
    del P
    print(f'{a.instrument} курсор {a.cursor}: атомов {len(recs)}, '
          f'информативных {len(sel)}, фильмов {n}, разведка {int(train.sum())}',
          flush=True)

    min_abs = int(round(min(a.levels) * n))
    i2, j2, t2 = exact_pairs(Ps, min_abs, log=lambda *x: None)
    print(f'пар в точной области (>= {min_abs}): {len(t2)}', flush=True)

    # ---- барьеры и контроли считаются один раз на всей популяции ---------
    hi, lo, Cc = barriers(c, a.cursor)
    d_up = hi - Cc
    d_dn = Cc - lo
    ratio = np.divide(d_up, d_up + d_dn,
                      out=np.full(n, np.nan), where=(d_up + d_dn) > 0)
    fu, fd = first_touch(c, a.cursor, hi, lo)
    b_sym = np.minimum(d_up, d_dn)
    fus, fds = first_touch(c, a.cursor, Cc + b_sym, Cc - b_sym)
    pool = np.isfinite(ratio)

    base = {'population_prefix_barriers': race(fu, fd, pool & train),
            'population_symmetric_barriers': race(fus, fds, pool & train)}

    # ---- витрина семейств: порядок псевдослучайный внутри уровня ---------
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(t2))
    fam, seen = [], set()
    buckets = list(zip(a.levels[:-1], a.levels[1:]))
    share = t2 / n
    for hi_l, lo_l in buckets:
        taken = 0
        for q in order:
            if not (lo_l <= share[q] < hi_l):
                continue
            m = (Ps.bool_block(int(i2[q]), int(i2[q]) + 1)[:, 0]
                 & Ps.bool_block(int(j2[q]), int(j2[q]) + 1)[:, 0])
            mt = m & train
            if mt.sum() < a.min_support_train:
                continue
            key = extent_key(mt)
            if key in seen:
                continue
            pr = projections(c, mt)
            if pr['days'] < a.min_days or len(pr['years']) < a.min_years:
                continue
            if any(jac(mt, f['_mt']) > a.max_overlap for f in fam):
                continue
            clo = np.zeros(Ps.A, bool)
            for s0 in range(0, Ps.A, 8192):
                e0 = min(s0 + 8192, Ps.A)
                clo[s0:e0] = Ps.bool_block(s0, e0)[mt].all(0)
            seen.add(key)
            fam.append({'id': f'F{len(fam) + 1}',
                        'relations': [names[sel[int(i2[q])]],
                                      names[sel[int(j2[q])]]],
                        'support_bucket': [lo_l, hi_l],
                        'support_all': int(m.sum()),
                        'support_scout': int(mt.sum()),
                        'closure_size_property_not_selection': int(clo.sum()),
                        'closure_sample': [names[sel[k]]
                                           for k in np.flatnonzero(clo)[:10]],
                        'projections_scout': pr,
                        '_m': m, '_mt': mt})
            taken += 1
            if taken >= a.per_bucket:
                break
    print(f'семейств: {len(fam)}', flush=True)

    out_fam = []
    for f in fam:
        mt = f['_mt'] & pool
        rec = {k: v for k, v in f.items() if not k.startswith('_')}
        rec['continuation_prefix_barriers'] = race(fu, fd, mt)
        rec['continuation_symmetric_barriers'] = race(fus, fds, mt)
        rec['distance_matched_control'] = distance_matched_control(
            ratio, pool & train, mt, fu, fd)
        rec['barrier_distance'] = {
            'median_up_points': float(np.nanmedian(d_up[mt])),
            'median_down_points': float(np.nanmedian(d_dn[mt])),
            'median_ratio_up': float(np.nanmedian(ratio[mt]))}
        sym = rec['continuation_symmetric_barriers']['share_up_among_decided']
        side = 1 if (sym is not None and sym >= 0.5) else -1
        rec['side_from_symmetric_continuation'] = 'long' if side > 0 else 'short'
        rec['money_by_hold_scout'] = {
            str(h): money(c, f['_mt'], a.cursor, h, side, a.instrument)
            for h in a.holds}
        rec['money_by_hold_second_part'] = {
            str(h): money(c, f['_m'] & ~train, a.cursor, h, side, a.instrument)
            for h in a.holds}
        idx = np.flatnonzero(f['_mt'])
        yr = year_of(c['t0'][idx])
        rec['examples'] = [
            {'unit': f'{a.instrument}:{int(c["t0"][int(idx[yr == y][0])])}',
             'year': int(y),
             'riz_ids': list(c['members'][int(idx[yr == y][0])])[:3],
             'tfs': list(c['tf_sets'][int(idx[yr == y][0])])[:8]}
            for y in sorted(set(yr.tolist()))[:5]]
        out_fam.append(rec)

    res = {'instrument': a.instrument, 'cursor': a.cursor, 'films': int(n),
           'split': a.split,
           'split_status': 'разведочное разделение; нетронутость второго '
                           'участка не установлена, подтверждением он не является',
           'prior_exposure': 'циклы 062–069 работали на 2020–2025',
           'selection_rule': {
               'declared_before_counting': True, 'levels': a.levels,
               'informative_band': a.informative, 'per_bucket': a.per_bucket,
               'min_days': a.min_days, 'min_years': a.min_years,
               'min_support_scout': a.min_support_train,
               'max_overlap_jaccard': a.max_overlap,
               'order_within_level': f'псевдослучайный, зерно {a.seed}',
               'closure_size_is_property_not_criterion': True,
               'money_not_used_in_selection': True},
           'language': {'atoms': len(recs), 'informative_atoms': int(len(sel))},
           'exact_region': {'min_support_absolute': min_abs,
                            'pairs': int(len(t2))},
           'population_baseline_scout': base,
           'families': out_fam,
           'implementation_sha256': hashlib.sha256(
               Path(__file__).read_bytes()).hexdigest()}
    out = Path(a.out or f'work/070/action_{a.instrument}_c{a.cursor}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                   encoding='utf-8')
    print('популяция, барьеры префикса:', base['population_prefix_barriers'])
    print('популяция, симметричные:', base['population_symmetric_barriers'])
    for r in out_fam:
        dm = r['distance_matched_control']
        print(f"{r['id']} n={r['support_scout']} дн={r['projections_scout']['days']} "
              f"замык={r['closure_size_property_not_selection']} "
              f"| префикс up={r['continuation_prefix_barriers']['share_up_among_decided']:.3f} "
              f"контроль={dm and round(dm['share_up_matched_control'], 3)} "
              f"| симметр up={r['continuation_symmetric_barriers']['share_up_among_decided']:.3f} "
              f"| {r['relations']}")


if __name__ == '__main__':
    main()
