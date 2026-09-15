#!/usr/bin/env python3
"""Две семантические проверки ДО постройки популяции 082.

ПРОВЕРКА 1 — можно ли задать пригодность q вообще без ссылки на F
=================================================================
`certified_fresh_until_pos_strict` у сертифицированного фильма равен `c - 1`,
то есть построен из будущего контакта. Пользоваться им как ОПРЕДЕЛЕНИЕМ
популяции нельзя: определение обязано читаться на закрытии q.

Определение 082 — только префикс, три условия на барах `t0 … q`:

    (i)   q <= last
    (ii)  kind[j] == 0 для всех j из (t0, q]        непрерывность наблюдения
    (iii) НЕ (low[j] <= e <= high[j]) для всех j из (t0, q]   контакта не было

Ординал 0 исключён, как в `first_exit_contact_v1`: минута T0 сама контактом не
считается, поэтому при q = t0 интервал пуст и оба условия выполнены пусто.

Аналитически `eligible(q) ⇔ t0 <= q <= min(g-1, c-1 если контакт есть, last)`,
где `g = nxt_any[t0]`, и это ровно `certified_fresh_until_pos_strict` с его
подъёмом `max(t0, ·)`. Здесь равенство проверяется НЕЗАВИСИМЫМ прямым
просмотром баров, без `first_observed_contact_pos` и без `nxt_any`.

ПРОВЕРКА 2 — гарантирует ли семантика контакта положительный d_close
====================================================================
Предикат контакта — `low[j] <= e <= high[j]`. FREEZE_080A §2 прямо оставляет
гэп через уровень БЕЗ такого бара вне контакта. Значит бар, целиком лежащий по
другую сторону границы, Film-1 не заканчивает, а `d_close` на нём отрицателен.
Возможно это или нет — вопрос к данным, а не к интуиции. Считается доля таких q
и то, чем они кончаются.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit, prange

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'work/082'
P081 = ROOT / 'work/081a'
sys.path.insert(0, str(ROOT / 'base/080'))
from tape import presence_grid, gap_kinds                              # noqa: E402

AGES = (5, 15, 60)


@njit(parallel=True, cache=True)
def _prefix_eligible(t0s, es, north, ages, kind, high, low, close, last,
                     elig, dsign, dclose):
    """Прямой просмотр баров t0+1 … q. Ни F, ни первый контакт не читаются."""
    for i in prange(t0s.size):
        t0 = t0s[i]; e = es[i]; up = north[i]
        alive = True
        k = 0
        amax = ages[ages.size - 1]
        for step in range(0, amax + 1):
            q = t0 + step
            if q > last:
                alive = False
            elif step > 0 and alive:
                if kind[q] != 0:
                    alive = False
                else:
                    hi = high[q]; lo = low[q]
                    if lo <= e and e <= hi:
                        alive = False
            while k < ages.size and ages[k] == step:
                elig[i, k] = alive
                if alive:
                    d = (close[q] - e) if up else (e - close[q])
                    dclose[i, k] = d
                    dsign[i, k] = 1 if d > 0 else (0 if d == 0 else -1)
                k += 1


@njit(parallel=True, cache=True)
def _after_negative(t0s, es, north, kind, high, low, close, last, amax,
                    first_neg_step, neg_resolved, neg_bars_to_return):
    """Для каждого фильма: первый шаг с d_close < 0 при живом Film-1 и что дальше.

    `neg_resolved`: 1 — позже наблюдается бар с диапазоном, содержащим границу
    (Film-1 всё-таки кончается контактом); 0 — наблюдение обрывается раньше.
    `neg_bars_to_return` — сколько баров прошло до такого контакта.
    """
    for i in prange(t0s.size):
        t0 = t0s[i]; e = es[i]; up = north[i]
        first_neg_step[i] = -1
        neg_resolved[i] = -1
        neg_bars_to_return[i] = -1
        q = t0
        step = 0
        while step < amax and q < last:
            q += 1; step += 1
            if kind[q] != 0:
                break
            hi = high[q]; lo = low[q]
            if lo <= e and e <= hi:
                break
            d = (close[q] - e) if up else (e - close[q])
            if d < 0:
                first_neg_step[i] = step
                # чем это кончается: ищем контакт дальше по непрерывной ленте
                j = q
                n = 0
                res = 0
                while j < last:
                    j += 1; n += 1
                    if kind[j] != 0:
                        break
                    if low[j] <= e and e <= high[j]:
                        res = 1
                        break
                neg_resolved[i] = res
                neg_bars_to_return[i] = n if res == 1 else -1
                break


def main(inst='NQ', terr='discovery'):
    m = ROOT / 'data/market' / inst
    high = np.load(m / 'high.npy'); low = np.load(m / 'low.npy')
    close = np.load(m / 'close.npy')
    last = high.size - 1
    grid, lo_, hi_ = presence_grid()
    kind, _ = gap_kinds(inst, grid, lo_, hi_)

    f = pd.read_parquet(P081 / f'paths/films_{inst}_{terr}.parquet')
    t0 = f.t0_spine_pos.to_numpy().astype(np.int64)
    F = f.certified_fresh_until_pos_strict.to_numpy().astype(np.int64)
    e = f.exit_boundary.to_numpy().astype(np.float64)
    north = (f.side.to_numpy() == 'north')
    ages = np.array(AGES, dtype=np.int64)
    n = len(f)

    elig = np.zeros((n, len(AGES)), np.bool_)
    dsign = np.zeros((n, len(AGES)), np.int8)
    dclose = np.full((n, len(AGES)), np.nan, np.float64)
    _prefix_eligible(t0, e, north, ages, kind, high, low, close, last,
                     elig, dsign, dclose)

    res = {'instrument': inst, 'territory': terr, 'films': int(n)}

    # --- ПРОВЕРКА 1 -------------------------------------------------------
    chk1 = {}
    for k, a in enumerate(AGES):
        via_F = (t0 + a) <= F
        same = int((via_F == elig[:, k]).sum())
        chk1[f'age_{a}'] = {
            'prefix_only_eligible': int(elig[:, k].sum()),
            'via_F_eligible': int(via_F.sum()),
            'agree': same, 'disagree': int(n - same)}
    res['check1_eligibility'] = chk1

    # --- ПРОВЕРКА 2 -------------------------------------------------------
    chk2 = {}
    for k, a in enumerate(AGES):
        m_ = elig[:, k]
        s = dsign[m_, k]
        chk2[f'age_{a}'] = {
            'eligible': int(m_.sum()),
            'd_close_positive': int((s == 1).sum()),
            'd_close_zero': int((s == 0).sum()),
            'd_close_negative': int((s == -1).sum()),
            'negative_share': round(float((s == -1).mean()), 6),
            'zero_share': round(float((s == 0).mean()), 6)}
        if (s == -1).any():
            v = dclose[m_, k][s == -1]
            chk2[f'age_{a}']['negative_d_close_p50'] = round(float(np.median(v)), 4)
            chk2[f'age_{a}']['negative_d_close_min'] = round(float(v.min()), 4)
    res['check2_d_close_sign'] = chk2

    # что происходит с фильмом, ушедшим за границу без контакта
    amax = max(AGES)
    fn = np.empty(n, np.int64); nr = np.empty(n, np.int64); nb = np.empty(n, np.int64)
    _after_negative(t0, e, north, kind, high, low, close, last, amax, fn, nr, nb)
    has_neg = fn >= 0
    res['check2_crossing_without_contact'] = {
        'films_with_a_negative_d_close_within_60_bars': int(has_neg.sum()),
        'share_of_films': round(float(has_neg.mean()), 6),
        'first_negative_step_p50': int(np.median(fn[has_neg])) if has_neg.any() else None,
        'later_bar_range_contains_boundary': int((nr[has_neg] == 1).sum()),
        'observation_broke_first': int((nr[has_neg] == 0).sum()),
        'bars_from_crossing_to_that_contact_p50':
            float(np.median(nb[has_neg & (nr == 1)])) if (has_neg & (nr == 1)).any() else None,
        'bars_from_crossing_to_that_contact_p90':
            float(np.quantile(nb[has_neg & (nr == 1)], .9)) if (has_neg & (nr == 1)).any() else None}

    # 081A закрывал этот вопрос фильтром слоя ИСПОЛНЕНИЯ (`usable` = ref_ok & gross>0)
    q = pd.read_parquet(P081 / f'paths/q_{inst}_{terr}.parquet',
                        columns=['film', 'age', 'usable', 'd_close', 'gross'])
    qa = q[q.age == 5]
    res['note_081a_filter'] = {
        'age_5_rows': int(len(qa)),
        'usable_share': round(float(qa.usable.mean()), 6),
        'rows_with_d_close_le_0': int((qa.d_close <= 0).sum()),
        'of_those_usable': int(qa.usable[qa.d_close <= 0].sum())}

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f'semantics_{inst}_{terr}.json'
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print(p, 'written')
    return res


if __name__ == '__main__':
    main(*(sys.argv[1:] or ['NQ', 'discovery']))
