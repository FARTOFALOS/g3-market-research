"""075: plain signed drift after a node, no stop, no target.

If the tape prices the node honestly the mean is zero at every horizon. This is
the denominator every construction has to beat, and it is measured before any
bracket is chosen.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

H = [1, 2, 3, 5, 10, 20, 30, 60]


def paths(kind):
    o, h, l, c, ts, sid = common.tape()
    pos = common.positions(ts)
    f = pd.read_parquet(common.CACHE / f'feat_{kind}.parquet')
    r = f.row.to_numpy(); k = f.k.to_numpy(); n = len(f)
    p = pos[r]
    ar = np.arange(n)
    ent = p[ar, np.minimum(k + 1, common.HOR)]
    entry = o[np.maximum(ent, 0)].astype(np.float64)
    sgn = np.where(f.long.to_numpy(), 1.0, -1.0)
    out = {}
    for hh in H:
        j = np.minimum(k + hh, common.HOR)
        pp = p[ar, j]
        ok = (pp >= 0) & (ent >= 0) & (k + hh <= common.HOR)
        v = np.where(ok, sgn * (c[np.maximum(pp, 0)] - entry) * 20.0, np.nan)
        out[f'm{hh}'] = v
    return f, pd.DataFrame(out, index=f.index)


def report(f, P, name):
    print(f'\n=== {name}  n = {len(f)}')
    rows = []
    for col in P.columns:
        v = P[col].to_numpy()
        v = v[np.isfinite(v)]
        se = v.std(ddof=1) / np.sqrt(len(v))
        rows.append(dict(hor=col, n=len(v), mean=round(v.mean(), 2), se=round(se, 2),
                         t=round(v.mean() / se, 2), median=round(float(np.median(v)), 2)))
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    F, PP = [], []
    for kind in ('ZE', 'ZF'):
        f, P = paths(kind)
        f = f.copy(); f['kind'] = kind
        F.append(f); PP.append(P)
    f = pd.concat(F, ignore_index=True); P = pd.concat(PP, ignore_index=True)
    f.to_parquet(common.CACHE / 'feat_all.parquet')
    P.to_parquet(common.CACHE / 'drift_all.parquet')
    report(f, P, 'all nodes, both kinds')
    for kind in ('ZE', 'ZF'):
        m = (f.kind == kind).to_numpy()
        report(f[m], P[m], kind)
    m = (f.rth == 1).to_numpy()
    report(f[m], P[m], 'regular session only')
    report(f[~m], P[~m], 'outside the regular session')
    # by hour of the New York day
    print('\n--- mean 10-minute move by the hour of the node, $ per trade')
    g = pd.DataFrame(dict(hour=f.minute_of_day // 60, v=P.m10.to_numpy(), w=P.m30.to_numpy()))
    t = g.groupby('hour').agg(n=('v', 'size'), m10=('v', 'mean'), m30=('w', 'mean')).round(2)
    t['t10'] = (g.groupby('hour').v.mean() / (g.groupby('hour').v.std() / np.sqrt(g.groupby('hour').v.count()))).round(2)
    print(t.to_string())


if __name__ == '__main__':
    main()
