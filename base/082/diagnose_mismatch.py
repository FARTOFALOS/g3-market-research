#!/usr/bin/env python3
"""Почему prefix-only критерий и `q <= F` разошлись. Причина, а не заплатка."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'work/082'
P081 = ROOT / 'work/081a'
sys.path.insert(0, str(ROOT / 'base/080'))
sys.path.insert(0, str(HERE))
from tape import presence_grid, gap_kinds, next_of_kind_fast            # noqa: E402
from semantics import _prefix_eligible, AGES                            # noqa: E402


def main(inst='NQ', terr='discovery'):
    m = ROOT / 'data/market' / inst
    high = np.load(m / 'high.npy'); low = np.load(m / 'low.npy')
    close = np.load(m / 'close.npy'); ts = np.load(m / 'close_ts_utc_ns.npy')
    last = high.size - 1
    grid, lo_, hi_ = presence_grid()
    kind, _ = gap_kinds(inst, grid, lo_, hi_)
    nxt_any = next_of_kind_fast(kind, (1, 2, 3))

    f = pd.read_parquet(P081 / f'paths/films_{inst}_{terr}.parquet')
    t0 = f.t0_spine_pos.to_numpy().astype(np.int64)
    F = f.certified_fresh_until_pos_strict.to_numpy().astype(np.int64)
    c = f.first_observed_contact_pos.to_numpy().astype(np.int64)
    e = f.exit_boundary.to_numpy().astype(np.float64)
    north = (f.side.to_numpy() == 'north')
    ages = np.array(AGES, dtype=np.int64)
    n = len(f)

    elig = np.zeros((n, len(AGES)), np.bool_)
    dsign = np.zeros((n, len(AGES)), np.int8)
    dclose = np.full((n, len(AGES)), np.nan, np.float64)
    _prefix_eligible(t0, e, north, ages, kind, high, low, close, last,
                     elig, dsign, dclose)

    res = {}
    for k, a in enumerate(AGES):
        via_F = (t0 + a) <= F
        only_prefix = elig[:, k] & ~via_F
        only_F = via_F & ~elig[:, k]
        # гипотеза: `nxt_any[t0]` включает промежуток ПЕРЕД самим баром T0
        gap_at_t0 = kind[t0] != 0
        res[f'age_{a}'] = {
            'prefix_only_eligible': int(elig[:, k].sum()),
            'via_F_eligible': int(via_F.sum()),
            'eligible_by_prefix_not_by_F': int(only_prefix.sum()),
            'eligible_by_F_not_by_prefix': int(only_F.sum()),
            'of_prefix_only__gap_immediately_before_T0_bar':
                int((only_prefix & gap_at_t0).sum()),
            'of_prefix_only__other': int((only_prefix & ~gap_at_t0).sum()),
            'status_of_prefix_only': {
                str(kk): int(vv) for kk, vv in
                f.film1_status[only_prefix].value_counts().items()},
        }
        if (only_prefix & ~gap_at_t0).any():
            idx = np.flatnonzero(only_prefix & ~gap_at_t0)[:5]
            res[f'age_{a}']['other_examples'] = [{
                'riz_id': str(f.riz_id.iloc[i]), 't0': int(t0[i]), 'F': int(F[i]),
                'contact': int(c[i]), 'nxt_any_t0': int(nxt_any[t0[i]]),
                'kind_t0': int(kind[t0[i]]),
                'status': str(f.film1_status.iloc[i])} for i in idx]

    # сводно: доля фильмов с промежутком прямо перед баром T0
    res['films_with_gap_immediately_before_T0_bar'] = int((kind[t0] != 0).sum())
    res['share'] = round(float((kind[t0] != 0).mean()), 6)
    res['их F'] = {'F_equals_t0': int((F[kind[t0] != 0] == t0[kind[t0] != 0]).sum()),
                   'n': int((kind[t0] != 0).sum())}

    p = OUT / f'mismatch_{inst}_{terr}.json'
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return res


if __name__ == '__main__':
    main(*(sys.argv[1:] or ['NQ', 'discovery']))
