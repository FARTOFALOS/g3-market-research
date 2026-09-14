"""075: the plain question under everything -- after T0, does price drift in the
direction in which it left the zone?

Unit of count: one distinct price path, i.e. one T0 minute. The field cuts one
row per riz_id, so a single minute can carry dozens of zones; counting those as
separate cases would multiply the same tape path. Here every T0 minute enters
once, with the exit side that its zones agree on.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

H = [1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120]


def main():
    o, h, l, c, ts, sid = common.tape()
    pos = common.positions(ts)
    d = common.scene_table()
    g = d.groupby('row').t0_exit_side.agg(['nunique', 'first', 'size']).rename(columns={'nunique': 'nu'})
    print('distinct T0 minutes:', len(g))
    print('minutes whose zones disagree on the exit side:', int((g.nu > 1).sum()),
          f'({(g.nu > 1).mean():.4f})')
    print('median zones per minute:', int(g['size'].median()))
    u = g[g.nu == 1].copy()
    rows = u.index.to_numpy()
    sgn = np.where(u['first'].to_numpy() == 'north', 1.0, -1.0)
    p = pos[rows]
    ar = np.arange(len(rows))
    ent = p[ar, 1]
    entry = np.where(ent >= 0, o[np.maximum(ent, 0)], np.nan).astype(np.float64)
    t0ts = pd.to_datetime(common.t0_list()[rows]).tz_localize('UTC').tz_convert('America/New_York')
    year = t0ts.year.to_numpy()
    mod = (t0ts.hour * 60 + t0ts.minute).to_numpy()
    out = {}
    for hh in H:
        pp = p[ar, hh]
        v = np.where(pp >= 0, sgn * (c[np.maximum(pp, 0)] - entry) * 20.0, np.nan)
        out[hh] = v
    P = pd.DataFrame(out)
    print('\n=== drift in the T0 exit direction, $ per one contract, entry = open of T0+1')
    rep = []
    for hh in H:
        v = P[hh].to_numpy(); v = v[np.isfinite(v)]
        se = v.std(ddof=1) / np.sqrt(len(v))
        rep.append(dict(min=hh, n=len(v), mean=round(v.mean(), 2), se=round(se, 2),
                        t=round(v.mean() / se, 2), median=round(float(np.median(v)), 2),
                        share_pos=round(float((v > 0).mean()), 3)))
    print(pd.DataFrame(rep).to_string(index=False))
    print('\n=== the same by epoch (60 minutes)')
    ep = pd.cut(year, [2005, 2010, 2015, 2020, 2026])
    t = pd.DataFrame(dict(ep=ep, v=P[60].to_numpy(), v10=P[10].to_numpy()))
    print(t.groupby('ep', observed=True).agg(n=('v', 'size'), m60=('v', 'mean'), m10=('v10', 'mean')).round(2).to_string())
    print('\n=== by session phase of T0 (New York)')
    ph = pd.cut(mod, [-1, 240, 570, 700, 840, 960, 1020, 1440],
                labels=['00:00-04:00', '04:00-09:30', '09:30-11:40', '11:40-14:00',
                        '14:00-16:00', '16:00-17:00', '17:00-24:00'])
    t = pd.DataFrame(dict(ph=ph, v10=P[10].to_numpy(), v60=P[60].to_numpy()))
    q = t.groupby('ph', observed=True).agg(n=('v10', 'size'), m10=('v10', 'mean'), m60=('v60', 'mean'))
    q['t10'] = t.groupby('ph', observed=True).v10.mean() / (t.groupby('ph', observed=True).v10.std() / np.sqrt(t.groupby('ph', observed=True).v10.count()))
    print(q.round(2).to_string())
    np.save(common.CACHE / 't0_rows.npy', rows)
    np.save(common.CACHE / 't0_sgn.npy', sgn)
    P.to_parquet(common.CACHE / 't0_drift.parquet')


if __name__ == '__main__':
    main()
