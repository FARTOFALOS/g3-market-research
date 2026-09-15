#!/usr/bin/env python3
"""Suffix-проход против прямого пересчёта. Быстрый примитив не имеет права
отличаться от наивного чтения оставшейся траектории ни на одной строке.

Наивный эталон не знает про suffix: для каждого q он заново пробегает бары
`q+1 … end` и берёт экстремумы. Это то же определение, записанное дословно.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / 'base/080'))
from tape import presence_grid, gap_kinds                              # noqa: E402
import paths as P                                                      # noqa: E402

RNG = np.random.default_rng(20260915)


def naive(inst, f, kind, opn, high, low, close, last):
    """Дословное чтение определения, без suffix-структуры."""
    rows = []
    for _, r in f.iterrows():
        t0 = int(r.t0_spine_pos); F = int(r.certified_fresh_until_pos_strict)
        cert = r.film1_status == 'contact_certified'
        end = int(r.first_observed_contact_pos) if cert else F
        e = float(r.exit_boundary); up = r.side == 'north'
        for q in range(t0, F + 1):
            j = q + 1
            dc = (close[q] - e) if up else (e - close[q])
            if j > end:
                rows.append((q - t0, 1, False, dc, np.nan, np.nan, np.nan, 0))
                continue
            o = opn[j]
            g = (o - e) if up else (e - o)
            seg_all = slice(j, end + 1)
            seg_exc = slice(j, end)                     # строго до касательной
            if up:
                mb = float(high[seg_all].max()) - o
                mc = (float(high[seg_exc].max()) - o) if end > j else 0.0
            else:
                mb = o - float(low[seg_all].min())
                mc = (o - float(low[seg_exc].min())) if end > j else 0.0
            mb = max(mb, 0.0); mc = max(mc, 0.0)
            if not cert:
                mc = mb
            rows.append((q - t0, 0 if cert else 1,
                         bool(j <= last and kind[j] == 0), dc, g, mc, mb, end - q))
    return pd.DataFrame(rows, columns=['age', 'outcome', 'ref_ok', 'd_close',
                                       'gross', 'mae_certain', 'mae_bound', 'dur'])


def main(inst='NQ'):
    grid, lo, hi = presence_grid()
    m = ROOT / 'data/market' / inst
    opn = np.load(m / 'open.npy'); high = np.load(m / 'high.npy')
    low = np.load(m / 'low.npy'); close = np.load(m / 'close.npy')
    kind, _ = gap_kinds(inst, grid, lo, hi)
    last = opn.size - 1

    allf = P.load_films(inst, 'discovery')
    nq = allf.q_certified_strict.to_numpy()
    cert = allf.film1_status.to_numpy() == 'contact_certified'
    # намеренно трудные срезы, а не только случайные строки
    buckets = {
        'contact на T0+1 (c == q+1)': np.flatnonzero(cert & (nq == 1)),
        'самые длинные сертифицированные': np.flatnonzero(cert)[np.argsort(-nq[cert])[:400]],
        'несертифицированные, длинные': np.flatnonzero(~cert)[np.argsort(-nq[~cert])[:400]],
        'несертифицированные, короткие': np.flatnonzero(~cert & (nq <= 3)),
        'north': np.flatnonzero(allf.side.to_numpy() == 'north'),
        'south': np.flatnonzero(allf.side.to_numpy() == 'south'),
        'цена на линии в T0': np.flatnonzero(
            (low[allf.t0_spine_pos.to_numpy()] <= allf.exit_boundary.to_numpy()) &
            (high[allf.t0_spine_pos.to_numpy()] >= allf.exit_boundary.to_numpy())),
    }
    _, full = P.build(inst, 'discovery', grid, lo, hi)     # полный слой один раз
    off = np.concatenate(([0], np.cumsum(allf.q_certified_strict.to_numpy())))
    bad = 0
    for name, idx in buckets.items():
        if idx.size == 0:
            print(f'  {name}: пусто'); continue
        take = idx if idx.size <= 120 else RNG.choice(idx, 120, replace=False)
        sub = allf.iloc[np.sort(take)].reset_index(drop=True)
        ref = naive(inst, sub, kind, opn, high, low, close, last)
        pos = np.sort(take)
        sel = np.concatenate([np.arange(off[i], off[i + 1]) for i in pos])
        got = full.iloc[sel].reset_index(drop=True)
        d = 0
        for col in ('age', 'outcome', 'ref_ok', 'dur'):
            d += int((got[col].to_numpy() != ref[col].to_numpy()).sum())
        for col in ('d_close', 'gross', 'mae_certain', 'mae_bound'):
            a = got[col].to_numpy().astype(float); b = ref[col].to_numpy().astype(float)
            both_nan = np.isnan(a) & np.isnan(b)
            d += int((~both_nan & ~np.isclose(a, b, rtol=0, atol=2e-3)).sum())
        bad += d
        print(f'  {name}: {len(sub)} фильмов, {len(ref)} q, расхождений {d}')
    print('ИТОГО расхождений:', bad)
    return bad


if __name__ == '__main__':
    sys.exit(1 if main(*(sys.argv[1:] or ['NQ'])) else 0)
