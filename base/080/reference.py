#!/usr/bin/env python3
"""Эталонная реализация первого post-T0 контакта. Медленная и очевидная.

Единственное определение здесь — каноническое: наблюдаемый диапазон бара
содержит собственную exit boundary этого RIZ, ординал >= 1. Никаких бюджетов,
никаких блоков, никакого раннего выхода: полный проход по всей оставшейся
ленте. Этот файл существует, чтобы оптимизированная версия могла быть с ним
сверена строка в строку.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

PCOLS = ['riz_id', 'instrument', 'tf_minutes', 'zone_top', 'zone_bottom', 't0_spine_pos',
         't0_ts_ns', 't0_exit_side', 'c1_deletion_spine_pos',
         'blue_eligibility_end_spine_pos', 'last_observed_spine_pos', 'censored']


def spine(inst):
    d = ROOT / 'data/market' / inst
    return {k: np.load(d / f'{k}.npy') for k in ('low', 'high', 'open', 'close',
                                                 'close_ts_utc_ns')}


def passports(inst, tf):
    f = ROOT / f'data/field/{inst}/cells/tf_{tf:04d}/passports.parquet'
    return pd.read_parquet(f, columns=PCOLS)


def exit_boundary(p):
    return np.where(p.t0_exit_side.to_numpy() == 'north',
                    p.zone_top.to_numpy(), p.zone_bottom.to_numpy())


def first_contact_reference(low, high, t0, e):
    """Первый индекс j > t0 с low[j] <= e <= high[j], или -1. Полный проход."""
    m = (low[t0 + 1:] <= e) & (high[t0 + 1:] >= e)
    j = int(np.argmax(m))
    return (t0 + 1 + j) if (m.size and m[j]) else -1


def run(inst, tf):
    s = spine(inst)
    p = passports(inst, tf)
    e = exit_boundary(p)
    t0 = p.t0_spine_pos.to_numpy().astype(np.int64)
    out = np.empty(len(p), dtype=np.int64)
    for i in range(len(p)):
        out[i] = first_contact_reference(s['low'], s['high'], int(t0[i]), float(e[i]))
    return p.assign(exit_boundary=e, contact_pos=out)


if __name__ == '__main__':
    import argparse, time
    a = argparse.ArgumentParser()
    a.add_argument('--instrument', default='NQ')
    a.add_argument('--tf', type=int, default=54)
    a.add_argument('--out', default=None)
    a = a.parse_args()
    t = time.time()
    r = run(a.instrument, a.tf)
    print(f'{a.instrument} TF{a.tf}: {len(r)} RIZ, {time.time()-t:.1f}s')
    print('no contact to tape end:', int((r.contact_pos < 0).sum()))
    d = r.contact_pos.to_numpy() - r.t0_spine_pos.to_numpy()
    ok = r.contact_pos.to_numpy() >= 0
    print('observed-bar lag quantiles:',
          {q: int(np.quantile(d[ok], q)) for q in (0.5, 0.9, 0.99, 1.0)})
    if a.out:
        r.to_parquet(a.out, index=False)
