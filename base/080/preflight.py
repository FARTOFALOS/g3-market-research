#!/usr/bin/env python3
"""Инженерный preflight 080A: бюджет, не находка.

Считается по компактному индексу Film-1. Ни одна строка `(riz_id, q)` не
материализуется: число решений выводится из адресов.

Лаг до контакта называется в двух разных единицах, потому что они расходятся:
наблюдённые бары ленты и минуты настенных часов. Подменять одно другим здесь
нельзя — спина хранит только наблюдённые минуты.
"""
from __future__ import annotations
import json, sys
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
sys.path.insert(0, str(ROOT / 'research'))
from calendar_utils import year_of, date_key                      # noqa: E402

INSTRUMENTS = ('ES', 'NQ', 'YM')
KIND = {0: 'contiguous', 1: 'shared_cause_unknown',
        2: 'instrument_specific', 3: 'outside_common_window', -1: 't0_at_archive_edge'}


def q(a, qs=(0.5, 0.75, 0.9, 0.99, 1.0)):
    if not len(a):
        return {}
    return {str(x): int(np.quantile(a, x)) for x in qs}


def one(inst):
    d = pd.read_parquet(INDEX / f'film1_{inst}.parquet')
    c = d.first_observed_contact_pos.to_numpy()
    t0 = d.t0_spine_pos.to_numpy()
    has = c >= 0
    bars = d.observed_bars_t0_to_end.to_numpy()
    wall = d.wall_clock_minutes_t0_to_end.to_numpy()
    stat = d.film1_status.to_numpy()
    stat_h = d.film1_status_shared_survived.to_numpy()
    yr = year_of(d.t0_ts_ns.to_numpy())

    def cuts(v, m):
        n = int(m.sum())
        return {f'<= {k}': int(((v <= k) & m).sum()) for k in (1, 5, 15, 60, 240, 1024)} | {
            'total_with_contact': n}

    r = {
        '1_riz_rows': int(len(d)),
        '1_tf_cells_with_rows': int(d.tf_minutes.nunique()),
        '1_distinct_t0_minutes': int(d.t0_ts_ns.nunique()),
        '1_distinct_t0_minute_boundary_pairs': int(
            d.groupby('t0_ts_ns').exit_boundary.nunique().sum()),
        '1_side': {k: int(v) for k, v in d.side.value_counts().items()},
        '1_t0_years': f'{int(yr.min())}..{int(yr.max())}',

        '2_lag_observed_bars_to_contact': q(bars[has]),
        '2_lag_wall_clock_minutes_to_contact': q(wall[has]),
        '2_lag_observed_bars_no_contact_to_archive': q(bars[~has]),

        '3_contact_within_observed_bars': cuts(bars, has),
        '3_contact_within_wall_clock_minutes': cuts(wall, has),

        '4_engineering_tail_beyond_1024_observed_bars': int((has & (bars > 1024)).sum()),
        '4_engineering_tail_beyond_1024_note':
            'счёт только инженерный: поиск шёл до края архива, горизонта не вводилось',
        '4_no_observed_contact_to_archive_edge': int((~has).sum()),

        '5_films_crossing_session_boundary': int((d.sessions_spanned.to_numpy() > 1).sum()),
        '5_films_crossing_calendar_day': int(
            (date_key(d.t0_ts_ns.to_numpy())
             != date_key(np.where(has, d.first_observed_contact_ts_ns.to_numpy(),
                                  d.t0_ts_ns.to_numpy()))).sum()),
        '5_sessions_spanned': q(d.sessions_spanned.to_numpy()),

        '6_film1_status_strict': {k: int(v) for k, v in pd.Series(stat).value_counts().items()},
        '6_film1_status_shared_survived': {k: int(v) for k, v in pd.Series(stat_h).value_counts().items()},
        '6_primacy_strict': {k: int(v) for k, v in
                             d.first_observed_contact_primacy.value_counts().items()},
        '6_primacy_shared_survived': {k: int(v) for k, v in
                                      d.first_observed_contact_primacy_shared_survived.value_counts().items()},
        '6_missing_minutes_inside_film_quantiles': q(d.missing_minutes_inside.to_numpy()),
        # Статусы обязаны РАЗБИВАТЬ популяцию. Четвёртый случай — контакта нет,
        # но достоверность наблюдения дошла до края архива — редок и именно
        # поэтому его легко потерять в сводке. Здесь он не растворяется.
        '6_status_partition_ok': bool(
            sum(pd.Series(stat).value_counts()) == len(d)
            and sum(pd.Series(stat_h).value_counts()) == len(d)),

        # ВНИМАНИЕ: `censored` — это паспортный факт жизненного цикла RIZ
        # («объект был ещё жив, когда кончился архив»), а НЕ статус Film-1.
        # Он строгое надмножество: им помечены все фильмы без контакта, но
        # подавляющее большинство помеченных свой контакт давно получили.
        # Терминальным статусом Film-1 он становиться не имеет права.
        '7_passport_riz_alive_at_archive_end_flag': int(d.censored.sum()),
        '7_passport_flag_but_film1_contact_observed': int(
            (d.censored.to_numpy() & has).sum()),
        # Film-1 не закрыт наблюдавшимся контактом. Две РАЗНЫЕ причины внутри:
        # край архива и потеря достоверности на промежутке. Не смешивать.
        '7_film1_not_closed_by_observed_contact': int(
            (stat == 'no_contact_through_archive').sum()
            + (stat == 'freshness_lost_before_contact').sum()),
        '7_film1_censored_freshness_intact_strict': int(
            (stat == 'no_contact_through_archive').sum()),
        '7_film1_censored_freshness_intact_shared_survived': int(
            (stat_h == 'no_contact_through_archive').sum()),
        '7_deletion_recorded_before_contact': int(
            (has & (d.c1_deletion_spine_pos.to_numpy() >= 0)
             & (d.c1_deletion_spine_pos.to_numpy() < c)).sum()),
        '7_deletion_recorded_at_or_after_contact': int(
            (has & (d.c1_deletion_spine_pos.to_numpy() >= c)).sum()),

        '9_q_observed_total': int(d.q_observed.to_numpy().sum()),
        '9_q_certified_strict_total': int(d.q_certified_strict.to_numpy().sum()),
        '9_q_certified_shared_survived_total': int(d.q_certified_shared_survived.to_numpy().sum()),
        '9_q_certified_strict_quantiles': q(d.q_certified_strict.to_numpy()),

        'next_open_at_t0_bar_kind': {KIND[int(k)]: int(v) for k, v in
                                     d.t0_next_bar_kind.value_counts().items()},
        'next_open_at_t0_wait_minutes': q(
            d.t0_next_open_wait_minutes.to_numpy()[d.t0_next_open_wait_minutes.to_numpy() >= 0]),
    }
    # next-open на T0. Род следующего бара и направленная пригодность —
    # РАЗНЫЕ вопросы, поэтому они скрещены: в strict-версии reference через
    # промежуток не считается доступной, сколько бы удобной ни была цена.
    nxt = d.t0_next_open.to_numpy()
    north = d.side.to_numpy() == 'north'
    e = d.exit_boundary.to_numpy()
    bar_kind = d.t0_next_bar_kind.to_numpy()
    known = np.isfinite(nxt)
    direction = np.where(np.where(north, nxt > e, nxt < e), 'outside_boundary_trade_intact',
                         np.where(nxt == e, 'exactly_on_boundary',
                                  'beyond_boundary_trade_void'))
    direction = np.where(known, direction, 'reference_price_unknown')
    cross = {}
    for k in np.unique(bar_kind):
        m = bar_kind == k
        cross[KIND[int(k)]] = {str(v): int(((direction == v) & m).sum())
                               for v in np.unique(direction[m])}
    r['next_open_at_t0_by_bar_kind_x_direction'] = cross
    contiguous = bar_kind == 0
    r['next_open_at_t0_strict_available'] = int(contiguous.sum())
    r['next_open_at_t0_strict_available_and_trade_intact'] = int(
        (contiguous & (direction == 'outside_boundary_trade_intact')).sum())
    r['next_open_at_t0_through_gap_not_available_strict'] = int(
        (~contiguous & known).sum())
    r['index_bytes'] = int((INDEX / f'film1_{inst}.parquet').stat().st_size)
    return r


if __name__ == '__main__':
    out = {i: one(i) for i in INSTRUMENTS}
    out['TOTAL'] = {
        'riz_rows': sum(out[i]['1_riz_rows'] for i in INSTRUMENTS),
        'q_observed_total': sum(out[i]['9_q_observed_total'] for i in INSTRUMENTS),
        'q_certified_strict_total': sum(out[i]['9_q_certified_strict_total'] for i in INSTRUMENTS),
        'q_certified_shared_survived_total': sum(
            out[i]['9_q_certified_shared_survived_total'] for i in INSTRUMENTS),
        'index_megabytes': round(sum(out[i]['index_bytes'] for i in INSTRUMENTS) / 1e6, 1),
    }
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    (HERE / 'preflight.json').write_text(txt, encoding='utf-8', newline='\n')
    print(txt)
