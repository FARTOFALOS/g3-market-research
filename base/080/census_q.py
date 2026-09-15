#!/usr/bin/env python3
"""Перепись decision points и покрытия next-open reference на ВСЕХ q.

Обходится вся сертифицированная сетка, но ни одна строка `(riz_id, q)` не
хранится: numba идёт по диапазону адресов каждого фильма и возвращает счётчики.

Два разных вопроса на каждом q, и они скрещены, а не смешаны:
  доступна ли reference    — род следующего бара (strict: только contiguous);
  цела ли сделка по ней    — где open относительно собственной exit boundary.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

import os
#: Где лежит компактный индекс. По умолчанию рядом с кодом; переопределяется
#: `G3_080A_INDEX`, чтобы несколько копий кода читали один и тот же индекс и не
#: плодили 120 МБ на каждую.
INDEX = Path(os.environ.get('G3_080A_INDEX', str(HERE / 'index')))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'research'))
from tape import presence_grid, gap_kinds                            # noqa: E402
from calendar_utils import year_of, minute_of_day                    # noqa: E402

INSTRUMENTS = ('ES', 'NQ', 'YM')
TF_BANDS = ([0, 1, 2, 5, 15, 60, 240, 1440],
            ['1', '2', '3-5', '6-15', '16-60', '61-240', '241-1440'])


@njit(cache=True)
def scan(t0s, fresh, e, north, opn, kind, last):
    """Счётчики по всем q каждого фильма. Возвращает матрицу per-RIZ."""
    n = t0s.size
    out = np.zeros((n, 6), dtype=np.int64)
    dist = np.zeros(n, dtype=np.float64)     # сумма |open - e| по годным q
    for i in range(n):
        a, b = t0s[i], fresh[i]
        lvl, up = e[i], north[i]
        for q in range(a, b + 1):
            out[i, 0] += 1                    # всего q
            nxt = q + 1
            if nxt > last:
                out[i, 5] += 1                # reference за краем архива
                continue
            if kind[nxt] != 0:
                out[i, 4] += 1                # только через промежуток
                continue
            out[i, 1] += 1                    # reference доступна в strict
            o = opn[nxt]
            s = (o - lvl) if up else (lvl - o)
            if s > 0.0:
                out[i, 2] += 1                # снаружи, сделка цела
                dist[i] += s
            elif s == 0.0:
                out[i, 3] += 1                # ровно на границе
    return out, dist


def one(inst, kind):
    d = pd.read_parquet(INDEX / f'film1_{inst}.parquet')
    opn = np.asarray(np.load(ROOT / f'data/market/{inst}/open.npy'))
    last = opn.size - 1
    t0 = d.t0_spine_pos.to_numpy().astype(np.int64)
    fresh = d.certified_fresh_until_pos_strict.to_numpy().astype(np.int64)
    e = d.exit_boundary.to_numpy().astype(np.float64)
    north = (d.side.to_numpy() == 'north')
    m, dist = scan(t0, fresh, e, north, opn, kind, last)

    q_all, q_ref, q_out, q_on, q_gap, q_edge = (m[:, k] for k in range(6))
    width = (d.zone_top.to_numpy() - d.zone_bottom.to_numpy())
    yr = year_of(d.t0_ts_ns.to_numpy())
    mod = minute_of_day(d.t0_ts_ns.to_numpy())
    band = pd.cut(d.tf_minutes, TF_BANDS[0], labels=TF_BANDS[1])

    def by(group_key, series=None):
        g = pd.DataFrame({'k': group_key, 'films': 1, 'q': q_all,
                          'q_ref': q_ref, 'q_out': q_out}).groupby('k', observed=True).sum()
        return {str(i): {'films': int(r.films), 'q': int(r.q),
                         'q_reference_available': int(r.q_ref),
                         'q_trade_intact': int(r.q_out)} for i, r in g.iterrows()}

    beyond = q_ref - q_out - q_on
    r = {
        'films': int(len(d)),
        'q_certified_strict': int(q_all.sum()),
        'q_reference_available_strict': int(q_ref.sum()),
        'q_reference_only_through_gap': int(q_gap.sum()),
        'q_reference_past_archive_edge': int(q_edge.sum()),
        'q_trade_intact_open_outside': int(q_out.sum()),
        'q_open_exactly_on_boundary': int(q_on.sum()),
        'q_open_beyond_boundary_trade_void': int(beyond.sum()),
        'share_q_reference_available': round(float(q_ref.sum() / q_all.sum()), 4),
        'share_q_trade_intact': round(float(q_out.sum() / q_all.sum()), 4),
        'films_with_at_least_one_usable_q': int((q_out > 0).sum()),
        'films_with_no_usable_q': int((q_out == 0).sum()),
        'mean_points_open_to_boundary_on_usable_q': round(
            float(dist.sum() / max(q_out.sum(), 1)), 3),
        'mean_zone_widths_open_to_boundary': round(float(
            (dist[q_out > 0] / np.maximum(width[q_out > 0], 1e-9)).sum()
            / max(q_out.sum(), 1)), 4),
        'by_tf_band': by(band),
        'by_side': by(d.side.to_numpy()),
        'by_epoch': by((yr // 5) * 5),
        'rth_0930_1600_et_films': int(((mod >= 570) & (mod < 960)).sum()),
        'distinct_session_days_with_a_film': int(d.t0_session_id.nunique()),
    }
    # multiplicity: что считается за один случай
    r['multiplicity'] = {
        'riz': int(len(d)),
        'distinct_t0_minutes': int(d.t0_ts_ns.nunique()),
        'distinct_t0_minute_boundary_pairs': int(
            d.groupby('t0_ts_ns').exit_boundary.nunique().sum()),
        'distinct_exit_boundary_prices': int(d.exit_boundary.nunique()),
        'distinct_session_days': int(d.t0_session_id.nunique()),
        'distinct_tf_cells': int(d.tf_minutes.nunique()),
        'max_riz_on_one_t0_minute': int(d.groupby('t0_ts_ns').size().max()),
        'max_distinct_boundaries_on_one_t0_minute': int(
            d.groupby('t0_ts_ns').exit_boundary.nunique().max()),
        'films_sharing_their_t0_minute': int(
            (d.groupby('t0_ts_ns').riz_id.transform('size') > 1).sum()),
    }
    return r


if __name__ == '__main__':
    grid, lo, hi = presence_grid()
    out = {}
    for inst in INSTRUMENTS:
        kind, _ = gap_kinds(inst, grid, lo, hi)
        out[inst] = one(inst, kind)
        print(inst, 'done', flush=True)
    keys = ('films', 'q_certified_strict', 'q_reference_available_strict',
            'q_trade_intact_open_outside', 'q_open_exactly_on_boundary',
            'q_open_beyond_boundary_trade_void', 'q_reference_only_through_gap',
            'films_with_no_usable_q')
    out['TOTAL'] = {k: sum(out[i][k] for i in INSTRUMENTS) for k in keys}
    out['TOTAL']['share_q_trade_intact'] = round(
        out['TOTAL']['q_trade_intact_open_outside'] / out['TOTAL']['q_certified_strict'], 4)
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    (HERE / 'census_q.json').write_text(txt, encoding='utf-8', newline='\n')
    print(json.dumps(out['TOTAL'], ensure_ascii=False, indent=1))
