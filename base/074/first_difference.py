"""074: shared loaders for the cycle.

`scenes` cuts the frozen field to one scene per RIZ zone and hands over both the
8-minute and the 51-minute price windows; `membership` rebuilds the core plus the
admissible variants of a frozen 073 predicate.

The scan of prefix relations against a later label that once lived here was
dropped: the question of 074 is the process, not a feature that correlates with
a chosen horizon. The trajectory work is in trajectories.py and branches.py.
"""

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'work/074'
OUT = Path(__file__).resolve().parent
F = 'OHLC'
PRE = 6           # minutes 0..5 are available before the trajectories part
LATE = 20         # where the later development is read
PERM = 300


def parse(r):
    m = re.match(r'(\w)\[(\d)\] ([<>]) (\w)\[(\d)\]', r)
    return 4 * int(m.group(2)) + F.index(m.group(1)), m.group(3), 4 * int(m.group(5)) + F.index(m.group(4))


def scenes():
    z8 = np.load(CACHE / 'NQ_prefix.npz')
    t0, x8 = z8['t0'], z8['x']
    X = x8.reshape(len(t0), 32)
    xl = np.load(CACHE / 'NQ_prefix51.npz')['x']
    d = pd.read_parquet(CACHE / 'NQ_zones.parquet')
    d = d.drop_duplicates(['t0_ts_ns', 'zone_top', 'zone_bottom', 't0_exit_side'])
    row = pd.Series(np.arange(len(t0)), index=t0.astype('int64'))
    d = d[d.t0_ts_ns.isin(row.index)].copy()
    d['row'] = row.loc[d.t0_ts_ns].to_numpy()
    return t0, X, xl, d.reset_index(drop=True)


def membership(X, p):
    ok = np.ones(len(X), bool)
    miss = np.zeros(len(X), int)
    for rl in p['basis_relations']:
        i, op, k = parse(rl)
        v = X[:, i] - X[:, k]
        ok &= np.isfinite(v)
        miss += ~((v > 0) if op == '>' else (v < 0))
    lad = np.ones(len(X), bool)
    for rl in p['ladder_relations']:
        i, op, k = parse(rl)
        v = X[:, i] - X[:, k]
        lad &= (v > 0) if op == '>' else (v < 0)
    return ok & ((miss == 0) | lad)      # core plus admissible variants
