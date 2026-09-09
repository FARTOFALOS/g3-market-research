#!/usr/bin/env python3
"""Корпус фильмов RIZ: T0 и минуты +1…+50, один физический путь на единицу.

ЕДИНИЦА ПОВТОРЕНИЯ (объявлена до счёта)
=======================================
Один физический ценовой путь = (инструмент, минута T0). Все `riz_id` этой
минуты со всех ТФ сохраняются как члены единицы: окно считается один раз,
обратное отображение в паспорта сохранено. Перепись показала, что сторона ухода
на одной минуте T0 всегда одна (unique_t0 == unique_t0×side во всех трёх
инструментах), поэтому сторона — свойство единицы, а не часть её адреса.

ОКНО
====
Ординалы 0…50: T0 и пятьдесят минут после него. 51 свеча, а не длительность
сделки. Пропущенная минута ленты остаётся NaN на своём месте и выпадает из
знаменателя конкретного отношения, а не выбрасывает фильм целиком.

СТОРОНА
=======
Исходная сторона сохраняется. Зеркального отражения здесь нет; если оно
понадобится, это отдельно названное преобразование.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calendar_utils import year_of, date_key, minute_of_day  # noqa: E402

MIN = 60_000_000_000
AHEAD = 50
FIELDS = ('O', 'H', 'L', 'C')
PCOLS = ['riz_id', 't0_ts_ns', 'tf_minutes', 't0_exit_side', 'censored',
         't0_kind', 'direction']


def passports(instrument, tfs=None):
    base = ROOT / f'data/field/{instrument}/cells'
    parts = []
    for d in sorted(base.iterdir()):
        tf = int(d.name.split('_')[1])
        if tfs is not None and tf not in tfs:
            continue
        f = d / 'passports.parquet'
        if f.exists():
            parts.append(pd.read_parquet(f, columns=PCOLS))
    return pd.concat(parts, ignore_index=True)


def build(instrument, start=None, stop=None, tfs=None, ahead=AHEAD):
    p = passports(instrument, tfs)
    if start is not None:
        p = p[p.t0_ts_ns >= np.datetime64(start, 'ns').astype('int64')]
    if stop is not None:
        p = p[p.t0_ts_ns < np.datetime64(stop, 'ns').astype('int64')]
    m = {k: np.load(ROOT / 'data/market' / instrument / (k + '.npy'))
         for k in ('close_ts_utc_ns', 'open', 'high', 'low', 'close')}
    ts = m['close_ts_utc_ns']
    t0 = np.sort(p.t0_ts_ns.unique())
    want = t0[:, None] + np.arange(0, ahead + 1)[None, :] * MIN
    pos = np.searchsorted(ts, want)
    safe = np.minimum(pos, len(ts) - 1)
    hit = (pos < len(ts)) & (ts[safe] == want)
    films = np.full((len(t0), ahead + 1, 4), np.nan)
    for k, key in enumerate(('open', 'high', 'low', 'close')):
        v = m[key][safe]
        films[:, :, k] = np.where(hit, v, np.nan)
    # T0 сама обязана быть на ленте: паспорт её оттуда и взял
    assert hit[:, 0].all(), 'T0 отсутствует на минутной ленте'
    g = p.groupby('t0_ts_ns')
    side = g.t0_exit_side.agg(lambda s: s.iloc[0]).reindex(t0).to_numpy()
    nside = g.t0_exit_side.nunique().reindex(t0).to_numpy()
    assert (nside == 1).all(), 'на одной минуте T0 две стороны ухода'
    members = g.riz_id.apply(list).reindex(t0)
    tf_sets = g.tf_minutes.apply(lambda s: sorted(set(int(x) for x in s))).reindex(t0)
    meta = {
        'instrument': instrument, 'ordinals': [0, ahead],
        'units': int(len(t0)), 'passport_rows': int(len(p)),
        'tfs': 'все' if tfs is None else list(tfs),
        'start': start, 'stop': stop,
        'unit': '(инструмент, минута T0); все riz_id — члены',
        'orientation': 'исходная сторона сохранена',
        'missing_minutes': 'NaN на своём месте, из знаменателя отношения',
    }
    return {'ohlc': films, 't0': t0, 'side': side,
            'members': members.to_numpy(), 'tf_sets': tf_sets.to_numpy(),
            'known_candles': hit, 'meta': meta}


def projections(c, mask=None):
    """Проекции поддержки: единицы, riz_id, дни, ТФ, эпохи, сессии."""
    t0 = c['t0'] if mask is None else c['t0'][mask]
    members = c['members'] if mask is None else c['members'][mask]
    tfs = c['tf_sets'] if mask is None else c['tf_sets'][mask]
    side = c['side'] if mask is None else c['side'][mask]
    if len(t0) == 0:
        return {'units': 0}
    yr = year_of(t0)
    mod = minute_of_day(t0)
    allt = set()
    for s in tfs:
        allt.update(s)
    return {'units': int(len(t0)),
            'riz_ids': int(sum(len(x) for x in members)),
            'days': int(len(np.unique(date_key(t0)))),
            'years': {int(y): int((yr == y).sum()) for y in np.unique(yr)},
            'tfs_touched': len(allt),
            'side': {'north': int((side == 'north').sum()),
                     'south': int((side == 'south').sum())},
            'rth_0930_1600_et': int(((mod >= 570) & (mod < 960)).sum())}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--instrument', default='NQ')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    c = build(a.instrument)
    print(json.dumps({'meta': c['meta'], 'projections': projections(c),
                      'candle_known_share': float(c['known_candles'].mean())},
                     ensure_ascii=False, indent=1, default=str))
