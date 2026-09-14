"""075: the composite early sign, with its own noise ceiling.

074 reported no second early separator after checking four elements of one minute
separately. Separate checks do not close a composite sign, so here every single,
pair and triple of the named candle conditions is counted exactly on the whole
node population, and the identical search is repeated against a rotated outcome.
Rotation keeps the outcome's own serial structure -- neighbouring minutes of the
same film stay neighbours -- so the ceiling is not the ceiling of an i.i.d.
shuffle, which would be far too generous.
"""
from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from predicates import build

FLOOR = 2000
NPERM = 20
SEED = 75


def literals(f):
    P = build(f)
    lit = {}
    for k, v in P.items():
        lit[k] = v
        if 0.15 < v.mean() < 0.85:
            lit['not ' + k] = ~v
    return lit


def main():
    f = pd.read_parquet(common.CACHE / 'feat_all.parquet').reset_index(drop=True)
    f = f.sort_values(['t0', 'k']).reset_index(drop=True)
    lit = literals(f)
    names = list(lit)
    A = [np.packbits(lit[n]) for n in names]
    N = len(f)
    y = f.R2.to_numpy().astype(bool)
    rng = np.random.default_rng(SEED)
    shifts = rng.integers(N // 20, N - N // 20, NPERM)
    Y = np.vstack([np.packbits(y)] + [np.packbits(np.roll(y, int(s))) for s in shifts])
    base = np.array([np.bitwise_count(Y[i]).sum() for i in range(len(Y))], np.float64) / N
    L = len(names)
    print(f'literals {L}   events {N}   floor {FLOOR}   permutations {NPERM}')
    best = np.full(len(Y), -9.0)
    best_name = [None] * len(Y)
    keep = []
    combos = ([(i,) for i in range(L)] + list(itertools.combinations(range(L), 2))
              + list(itertools.combinations(range(L), 3)))
    cache2 = {}
    for cb in combos:
        if len(cb) == 1:
            m = A[cb[0]]
        elif len(cb) == 2:
            m = A[cb[0]] & A[cb[1]]
            cache2[cb] = m
        else:
            m = cache2.get(cb[:2])
            if m is None:
                m = A[cb[0]] & A[cb[1]]
                cache2[cb[:2]] = m
            m = m & A[cb[2]]
        n = int(np.bitwise_count(m).sum())
        if n < FLOOR:
            continue
        hits = np.bitwise_count(Y & m[None, :]).sum(1).astype(np.float64)
        rate = hits / n
        lift = rate - base
        imp = lift > best
        if imp.any():
            nm = ' & '.join(names[t] for t in cb)
            for q in np.flatnonzero(imp):
                best_name[q] = nm
            best = np.maximum(best, lift)
        keep.append((cb, n, rate[0], lift[0]))
    obs, null = best[0], best[1:]
    print(f'\ncombinations above the floor: {len(keep)} of {len(combos)}')
    print(f'best observed lift         : {obs:+.4f}   {best_name[0]}')
    print(f'ceiling of the same search : median {np.median(null):+.4f}   '
          f'p95 {np.percentile(null, 95):+.4f}   max {null.max():+.4f}')
    print(f'observed above the ceiling : {bool(obs > np.percentile(null, 95))}    '
          f'p = {float((null >= obs).mean()):.3f}')
    t = pd.DataFrame([dict(sign=' & '.join(names[i] for i in cb), n=n,
                           rate=round(float(r), 4), lift=round(float(l), 4))
                      for cb, n, r, l in keep]).sort_values('lift', ascending=False)
    t.to_parquet(common.CACHE / 'compose.parquet')
    print('\n--- eight strongest composites')
    print(t.head(8).to_string(index=False))
    json.dump(dict(observed=float(obs), sign=best_name[0], floor=FLOOR, base=float(base[0]),
                   ceiling_p95=float(np.percentile(null, 95)), ceiling_max=float(null.max()),
                   nperm=NPERM, n_combos=len(keep), n_events=int(N)),
              open(common.CACHE / 'compose.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
