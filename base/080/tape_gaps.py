#!/usr/bin/env python3
"""Структура пропусков ленты: что вообще можно установить о торговом времени.

Спина хранит только наблюдённые минуты (`synthetic_minutes: false`), поэтому
соседние позиции спины могут быть разнесены по часам. Для Film-1 важно
единственное: мог ли контакт с границей произойти внутри пропуска. Здесь
считается только фактура пропусков — без объявления их причины.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research'))
from calendar_utils import eastern  # noqa: E402

MIN = 60_000_000_000
INSTRUMENTS = ('ES', 'NQ', 'YM')


def ts(inst):
    return np.load(ROOT / f'data/market/{inst}/close_ts_utc_ns.npy', mmap_mode='r')


def gap_table(inst):
    t = np.asarray(ts(inst))
    d = (t[1:] - t[:-1]) // MIN
    idx = np.flatnonzero(d > 1)
    e = eastern(t[idx])              # закрытие бара ПЕРЕД пропуском
    return pd.DataFrame({
        'pos_before': idx,
        'missing': d[idx] - 1,
        'et_dow': e.dayofweek.to_numpy(),
        'et_hm': e.hour.to_numpy() * 60 + e.minute.to_numpy(),
        'ts_before': t[idx],
        'ts_after': t[idx + 1],
    })


def main():
    out = {}
    minute_sets = {}
    for inst in INSTRUMENTS:
        t = np.asarray(ts(inst))
        g = gap_table(inst)
        minute_sets[inst] = t // MIN
        e0, e1 = eastern(t[[0, -1]])
        out[inst] = {
            'rows': int(t.size),
            'first_close_et': str(e0), 'last_close_et': str(e1),
            'clock_minutes_spanned': int((t[-1] - t[0]) // MIN) + 1,
            'observed_share_of_clock': round(float(t.size / (((t[-1]-t[0])//MIN)+1)), 4),
            'gaps': int(len(g)),
            'missing_minutes_total': int(g.missing.sum()),
            'gap_size_quantiles': {q: int(np.quantile(g.missing, q))
                                   for q in (0.5, 0.9, 0.99, 1.0)},
            'gaps_by_size': {
                '1': int((g.missing == 1).sum()),
                '2-5': int(((g.missing >= 2) & (g.missing <= 5)).sum()),
                '6-60': int(((g.missing >= 6) & (g.missing <= 60)).sum()),
                '61-120': int(((g.missing >= 61) & (g.missing <= 120)).sum()),
                '121-1440': int(((g.missing >= 121) & (g.missing <= 1440)).sum()),
                '>1440': int((g.missing > 1440).sum()),
            },
        }
        # где по часам ET начинаются крупные пропуски
        big = g[g.missing >= 6]
        hh = (big.et_hm // 60).value_counts().sort_index()
        out[inst]['big_gap_start_hour_et'] = {int(k): int(v) for k, v in hh.items()}
        out[inst]['big_gap_start_dow'] = {int(k): int(v)
                                          for k, v in big.et_dow.value_counts().sort_index().items()}

    # пересечение: какие отсутствующие минуты общие для трёх инструментов
    lo = max(int(minute_sets[i][0]) for i in INSTRUMENTS)
    hi = min(int(minute_sets[i][-1]) for i in INSTRUMENTS)
    n = hi - lo + 1
    present = {}
    for inst in INSTRUMENTS:
        m = np.zeros(n, dtype=bool)
        v = minute_sets[inst]
        v = v[(v >= lo) & (v <= hi)] - lo
        m[v] = True
        present[inst] = m
    k = sum(present[i].astype(np.int8) for i in INSTRUMENTS)
    out['_overlap_window'] = {
        'clock_minutes': int(n),
        'present_in_all_three': int((k == 3).sum()),
        'present_in_two': int((k == 2).sum()),
        'present_in_one': int((k == 1).sum()),
        'absent_in_all_three': int((k == 0).sum()),
    }
    for inst in INSTRUMENTS:
        miss = ~present[inst]
        out[inst]['missing_in_overlap'] = int(miss.sum())
        out[inst]['missing_but_another_has_it'] = int((miss & (k > 0)).sum())
        out[inst]['missing_shared_by_all_three'] = int((miss & (k == 0)).sum())
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
