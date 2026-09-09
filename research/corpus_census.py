#!/usr/bin/env python3
"""Фактическая перепись покрытия готового поля: инструменты x годы x ТФ.

Считает не «наличие каталога», а строки паспортов и различные минуты T0.
Единица повторения объявлена здесь же: один физический ценовой путь —
(instrument, T0, сторона ухода). Все riz_id этой минуты сохраняются как члены.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calendar_utils import year_of  # noqa: E402

COLS = ['riz_id', 't0_ts_ns', 'tf_minutes', 't0_exit_side', 'censored',
        't0_kind', 'direction']


def read_instrument(instrument, tfs=None):
    base = ROOT / f'data/field/{instrument}/cells'
    parts = []
    for d in sorted(base.iterdir()):
        tf = int(d.name.split('_')[1])
        if tfs is not None and tf not in tfs:
            continue
        f = d / 'passports.parquet'
        if f.exists():
            parts.append(pd.read_parquet(f, columns=COLS))
    return pd.concat(parts, ignore_index=True)


def census(instrument):
    p = read_instrument(instrument)
    yr = year_of(p.t0_ts_ns.to_numpy())
    p = p.assign(year=yr)
    unit = p.t0_ts_ns.astype('int64').astype(str) + ':' + p.t0_exit_side.astype(str)
    p = p.assign(unit=unit)
    out = {
        'instrument': instrument,
        'passport_rows': int(len(p)),
        'tf_cells_with_rows': int(p.tf_minutes.nunique()),
        'unique_t0_minutes': int(p.t0_ts_ns.nunique()),
        'unique_units_t0_side': int(p.unit.nunique()),
        'rows_per_unit_mean': float(len(p) / p.unit.nunique()),
        'by_year': {int(y): {'rows': int((p.year == y).sum()),
                             'units': int(p.loc[p.year == y, 'unit'].nunique())}
                    for y in sorted(p.year.unique())},
        'by_tf_decade': {},
        'exit_side': {str(k): int(v) for k, v in p.t0_exit_side.value_counts().items()},
        't0_kind': {str(k): int(v) for k, v in p.t0_kind.value_counts().items()},
        'censored': {str(k): int(v) for k, v in p.censored.value_counts().items()},
    }
    band = pd.cut(p.tf_minutes, [0, 1, 2, 5, 15, 60, 240, 1440],
                  labels=['1', '2', '3-5', '6-15', '16-60', '61-240', '241-1440'])
    g = p.groupby(band, observed=True)
    out['by_tf_decade'] = {str(k): {'rows': int(len(v)),
                                    'units': int(v.unit.nunique())}
                           for k, v in g}
    return out, p


if __name__ == '__main__':
    res = {}
    for ins in ('ES', 'NQ', 'YM'):
        c, _ = census(ins)
        res[ins] = c
        print(ins, c['passport_rows'], c['unique_t0_minutes'], c['unique_units_t0_side'])
    Path('work/070').mkdir(parents=True, exist_ok=True)
    Path('work/070/census.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
