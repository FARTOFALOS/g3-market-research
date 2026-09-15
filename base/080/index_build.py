#!/usr/bin/env python3
"""Компактный per-RIZ индекс Film-1. Одна строка на riz_id, свечи не копируются.

ТРИ ОСИ, А НЕ ОДНО ПОЛЕ
=======================
`first_observed_contact_pos` — чистый факт наблюдения: позиция спины первого
бара после T0, чей диапазон содержит собственную exit boundary, либо -1, если
такого бара до края архива нет. Слой промежутков это поле НЕ меняет и не может
отменить.

`first_observed_contact_primacy` — доказано ли, что этот наблюдавшийся контакт
был первым произошедшим:

    certified        наблюдение от T0 до контакта непрерывно
    unknown          до контакта встретился промежуток, внутри которого цена
                     могла коснуться границы ненаблюдаемо
    not_applicable   контакт не наблюдался

`film1_status` — чем кончилось наблюдение самого Film-1:

    contact_certified              конец Film-1 установлен
    contact_seen_primacy_unknown   контакт есть, но более ранний не исключён
    no_contact_through_archive     непрерывное наблюдение T0 → край архива,
                                   контакта нет нигде
    freshness_lost_before_contact  контакта не наблюдалось, а достоверность
                                   наблюдения кончилась на промежутке
    t0_at_archive_edge             после T0 ленты уже нет

Это три разных вопроса, и ни один не кодируется значением другого.

ДВА ВАРИАНТА СЕРТИФИКАЦИИ
=========================
Отличаются ровно одним объявленным заранее допущением: переживается ли
промежуток рода `shared_cause_unknown` (минута отсутствует у ES, NQ и YM
одновременно). `strict` не переживает ни одного промежутка и остаётся основным:
календаря с провенансом в репозитории нет, поэтому KNOWN SCHEDULED CLOSED не
используется, и strict-вариант не перепрыгивает промежутки.

DECISION GRID НЕ МАТЕРИАЛИЗУЕТСЯ
================================
Здесь лежат только адреса. Сетка решений выводится лениво: q пробегает бары
от `t0_spine_pos` до `certified_fresh_until_pos_*` включительно, где
`certified_fresh_until_pos` — последняя позиция, на закрытии которой ещё
известно, что post-T0 контакта не происходило. При сертифицированном контакте
в `c` это ровно `c - 1`: контакт при `T0+1` оставляет сеткой одну минуту T0.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

import os
#: Где лежит компактный индекс. По умолчанию рядом с кодом; переопределяется
#: `G3_080A_INDEX`, чтобы несколько копий кода читали один и тот же индекс и не
#: плодили 120 МБ на каждую.
INDEX = Path(os.environ.get('G3_080A_INDEX', str(HERE / 'index')))
sys.path.insert(0, str(HERE))
from tape import presence_grid, gap_kinds, next_of_kind_fast          # noqa: E402
from contact import blocks, first_contacts, B                        # noqa: E402

MIN = 60_000_000_000
PCOLS = ['riz_id', 'instrument', 'tf_minutes', 'zone_top', 'zone_bottom',
         't0_spine_pos', 't0_ts_ns', 't0_exit_side', 't0_close',
         'c1_deletion_spine_pos', 'blue_eligibility_end_spine_pos',
         'native_blue_confirmation_spine_pos', 'last_observed_spine_pos',
         'censored']


def read_passports(inst):
    base = ROOT / f'data/field/{inst}/cells'
    parts = [pd.read_parquet(d / 'passports.parquet', columns=PCOLS)
             for d in sorted(base.iterdir()) if (d / 'passports.parquet').exists()]
    return pd.concat(parts, ignore_index=True)


def build(inst, grid=None, lo=None, hi=None):
    m = ROOT / 'data/market' / inst
    low = np.load(m / 'low.npy')
    high = np.load(m / 'high.npy')
    opn = np.load(m / 'open.npy')
    ts = np.load(m / 'close_ts_utc_ns.npy')
    sid = np.load(m / 'session_id.npy')
    last = low.size - 1

    kind, missing = gap_kinds(inst, grid, lo, hi)
    # `nxt[p]` — первый бар ПОСЛЕ p, перед которым лежит промежуток данного рода
    nxt_any = next_of_kind_fast(kind, (1, 2, 3))      # strict
    nxt_hard = next_of_kind_fast(kind, (2, 3))        # shared_survived
    cum_missing = np.concatenate(([0], np.cumsum(missing)))
    cum_shared = np.concatenate(([0], np.cumsum(np.where(kind == 1, missing, 0))))

    p = read_passports(inst)
    # Отсутствующее событие в паспорте хранится как null. В индексе оно
    # становится -1 явно: NaN в позиционном столбце молча проваливает любое
    # сравнение и читается как «события не было» по случайности, а не по
    # смыслу.
    for col in ('c1_deletion_spine_pos', 'blue_eligibility_end_spine_pos',
                'native_blue_confirmation_spine_pos'):
        p[col] = p[col].fillna(-1).astype('int64')
    north = p.t0_exit_side.to_numpy() == 'north'
    e = np.where(north, p.zone_top.to_numpy(), p.zone_bottom.to_numpy()).astype(np.float64)
    far = np.where(north, p.zone_bottom.to_numpy(), p.zone_top.to_numpy()).astype(np.float64)
    t0 = p.t0_spine_pos.to_numpy().astype(np.int64)

    blo, bhi = blocks(low, high)
    t = time.time()
    contact = first_contacts(low, high, blo, bhi, t0, e, B)
    secs = time.time() - t
    has = contact >= 0

    def certify(gap_after):
        """`(primacy, status, fresh_until, last_bar_before_gap)`.

        `fresh_until` — последний бар, на закрытии которого отсутствие
        post-T0 контакта ещё установлено. Сертифицированный контакт в `c`
        даёт `c - 1`; промежуток перед баром `g` даёт `g - 1`.
        """
        before_gap = np.minimum(gap_after - 1, last)
        certified = has & (contact <= before_gap)
        primacy = np.where(certified, 'certified',
                           np.where(has, 'unknown', 'not_applicable'))
        fresh = np.where(certified, contact - 1, before_gap)
        status = np.where(
            certified, 'contact_certified',
            np.where(has, 'contact_seen_primacy_unknown',
                     np.where(t0 >= last, 't0_at_archive_edge',
                              np.where(gap_after > last, 'no_contact_through_archive',
                                       'freshness_lost_before_contact'))))
        return primacy, status, np.maximum(fresh, t0), before_gap

    prim_s, stat_s, fresh_s, bg_s = certify(nxt_any[t0])
    prim_h, stat_h, fresh_h, bg_h = certify(nxt_hard[t0])

    endpos = np.where(has, contact, last)
    out = pd.DataFrame({
        'riz_id': p.riz_id.to_numpy(), 'instrument': inst,
        'tf_minutes': p.tf_minutes.to_numpy().astype(np.int16),
        'side': p.t0_exit_side.to_numpy(),
        'zone_top': p.zone_top.to_numpy(), 'zone_bottom': p.zone_bottom.to_numpy(),
        'exit_boundary': e, 'far_boundary': far,
        't0_spine_pos': t0, 't0_ts_ns': p.t0_ts_ns.to_numpy(),
        't0_close': p.t0_close.to_numpy(), 't0_session_id': sid[t0],

        # --- ось 1: факт наблюдения; слой промежутков его не трогает ---
        'first_observed_contact_pos': contact,
        'first_observed_contact_ts_ns': np.where(has, ts[np.maximum(contact, 0)], -1),

        # --- ось 2: первичность наблюдавшегося контакта ---
        'first_observed_contact_primacy': prim_s,
        'first_observed_contact_primacy_shared_survived': prim_h,

        # --- ось 3: чем кончилось наблюдение Film-1 ---
        'film1_status': stat_s,
        'film1_status_shared_survived': stat_h,

        # --- достоверность наблюдения ---
        'last_bar_before_first_gap_strict': np.where(bg_s >= last, -1, bg_s),
        'last_bar_before_first_gap_shared_survived': np.where(bg_h >= last, -1, bg_h),
        'certified_fresh_until_pos_strict': fresh_s,
        'certified_fresh_until_pos_shared_survived': fresh_h,

        # --- край наблюдения и жизненный цикл ---
        'last_observed_spine_pos': p.last_observed_spine_pos.to_numpy(),
        'censored': p.censored.to_numpy(),
        'c1_deletion_spine_pos': p.c1_deletion_spine_pos.to_numpy(),
        'blue_eligibility_end_spine_pos': p.blue_eligibility_end_spine_pos.to_numpy(),
        'native_blue_confirmation_spine_pos': p.native_blue_confirmation_spine_pos.to_numpy(),

        # --- три разные длительности, не подменяющие друг друга ---
        'observed_bars_t0_to_end': endpos - t0,
        'wall_clock_minutes_t0_to_end': (ts[endpos] - ts[t0]) // MIN,
        'missing_minutes_inside': cum_missing[endpos + 1] - cum_missing[t0 + 1],
        'missing_minutes_inside_shared': cum_shared[endpos + 1] - cum_shared[t0 + 1],
        'sessions_spanned': (sid[endpos] - sid[t0] + 1).astype(np.int32),
    })
    # ленивый счёт решений: q от T0 до последней достоверной минуты включительно
    out['q_observed'] = np.where(has, contact - t0, last - t0 + 1)
    out['q_certified_strict'] = fresh_s - t0 + 1
    out['q_certified_shared_survived'] = fresh_h - t0 + 1

    # next-open reference на самой минуте T0
    nxt = np.minimum(t0 + 1, last)
    at_edge = t0 >= last
    out['t0_next_bar_kind'] = np.where(at_edge, -1, kind[nxt]).astype(np.int8)
    out['t0_next_open'] = np.where(at_edge, np.nan, opn[nxt])
    out['t0_next_open_wait_minutes'] = np.where(at_edge, -1, (ts[nxt] - ts[t0]) // MIN)
    out.attrs['build_seconds'] = round(secs, 2)
    return out


if __name__ == '__main__':
    grid, lo, hi = presence_grid()
    outdir = INDEX
    outdir.mkdir(exist_ok=True)
    order = sys.argv[1:] or ['NQ', 'ES', 'YM']
    for inst in order:
        t = time.time()
        df = build(inst, grid, lo, hi)
        f = outdir / f'film1_{inst}.parquet'
        df.to_parquet(f, index=False)
        print(f'{inst}: {len(df)} RIZ, contact search {df.attrs["build_seconds"]}s, '
              f'total {time.time()-t:.1f}s, {f.stat().st_size/1e6:.1f} MB', flush=True)
