#!/usr/bin/env python3
"""099 step 2: the field not as one number and not as a predictor of the mean.

Declared before the count. 099 compressed the field into a scalar density and asked it
to move the MEAN of S-07. Its established role (050/055) is symmetric: it cuts tails.
So the honest reading is the SHAPE of the S-07 path, and the field must keep its
direction relative to the trade. No new metric: the natural RIZ unit is the distance to
the nearest LIVE boundary ahead of the trade and behind it, in candles (050's liveness).
  ahead = along the S-07 side from the entry, behind = against it; "empty" = no live
  boundary on that side at all (043: after T0 this is the usual case).
Read (120 bars, candles): favourable excursion MFE, adverse excursion MAE, the result,
  share of trades that reach the tail (MFE >= 20 candles), share hurt deep (MAE >= 16,
  the S-07 catastrophic limit).
"Coordinate system of risk" would mean, with ONE sign on NQ, ES and YM:
  MAE smaller when a boundary sits close behind (something holds the trade), or
  MFE smaller when a boundary sits close ahead (something caps it), empty ahead -> larger MFE.
Rule: low-vs-high tercile difference of the median with one sign on all three indexes
  and |t| >= 3 (on the mean) on at least two of them. Otherwise geometry of live
  boundaries does not shape the S-07 path at this resolution.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
MIN = 60_000_000_000


def run(inst):
    mk = ROOT / f'data/market/{inst}'
    O, H, L, C, TS = (np.load(mk / f'{n}.npy') for n in ('open', 'high', 'low', 'close', 'close_ts_utc_ns'))
    et = pd.to_datetime(TS, utc=True).tz_convert('America/New_York')
    mod = (et.hour * 60 + et.minute).to_numpy(); pos = np.arange(len(TS))
    k = pos[mod == 9 * 60 + 33]; k = k[(k > 61) & (k + 121 < len(TS))]
    k = k[(TS[k] - TS[k - 60] == 60 * MIN) & (TS[k + 121] - TS[k] == 121 * MIN)]
    z = pq.read_table(ROOT / f'work/080a/index/film1_{inst}.parquet', columns=[
        'zone_top', 'zone_bottom', 't0_spine_pos', 'c1_deletion_spine_pos', 'last_observed_spine_pos']).to_pandas()
    end = np.where(z.c1_deletion_spine_pos > 0, z.c1_deletion_spine_pos, z.last_observed_spine_pos).astype(np.int64)
    bp = np.concatenate([z.zone_top, z.zone_bottom]); bs = np.concatenate([z.t0_spine_pos, z.t0_spine_pos]); be = np.concatenate([end, end])
    o = np.argsort(bs); bp, bs, be = bp[o], bs[o], be[o]
    rows = []
    for m in k:
        j = np.searchsorted(bs, m, 'right'); live = bp[:j][be[:j] >= m]
        hi, lo = H[m - 29:m + 1].max(), L[m - 29:m + 1].min()
        s = 1.0 if C[m] > (hi + lo) / 2 else -1.0
        pc = C[m - 30:m]; cand = np.median(np.maximum(H[m - 29:m + 1], pc) - np.minimum(L[m - 29:m + 1], pc))
        if cand <= 0:
            continue
        e = O[m + 1]; d = s * (live - e)
        da = d[d > 0].min() / cand if (d > 0).any() else np.inf
        db = (-d[d < 0]).min() / cand if (d < 0).any() else np.inf
        seg = slice(m + 1, m + 122)
        mfe = (H[seg].max() - e if s > 0 else e - L[seg].min()) / cand
        mae = (e - L[seg].min() if s > 0 else H[seg].max() - e) / cand
        rp = (H[m - 60:m + 1].max() - L[m - 60:m + 1].min()) / cand
        rows.append(dict(da=da, db=db, mfe=mfe, mae=mae, res=s * (C[m + 121] - e) / cand, rp=rp))
    d = pd.DataFrame(rows)
    print(f'\n##### {inst}: sessions {len(d)} | empty ahead {np.isinf(d.da).mean():.3f} | empty behind {np.isinf(d.db).mean():.3f}'
          f' | nearest ahead p50 {d.da[np.isfinite(d.da)].median():.1f} candles, behind p50 {d.db[np.isfinite(d.db)].median():.1f}')

    def block(name, col, read):
        f = d[np.isfinite(d[col])].copy(); f['tc'] = pd.qcut(f[col].rank(method='first'), 3, labels=False)
        g = [f[f.tc == q] for q in range(3)] + [d[np.isinf(d[col])]]
        a, b = g[0][read], g[2][read]
        t = (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))
        print(' %-34s near %5.2f | mid %5.2f | far %5.2f | empty %5.2f   near-far mean %+.2f (t %.2f)'
              % (name, *[x[read].median() for x in g], a.mean() - b.mean(), t))
    block('MFE by nearest boundary AHEAD', 'da', 'mfe')
    block('MAE by nearest boundary BEHIND', 'db', 'mae')
    block('result by nearest boundary AHEAD', 'da', 'res')
    # control of the shared denominator and of direction (051: damping is not directional):
    # the same excursions against the boundary on the OTHER side, and distances in R_prev units
    block('MFE by nearest boundary BEHIND (cross)', 'db', 'mfe')
    block('MAE by nearest boundary AHEAD (cross)', 'da', 'mae')
    d['da_r'] = d.da / d.rp; d['db_r'] = d.db / d.rp
    block('MFE by AHEAD in R_prev units', 'da_r', 'mfe')
    block('MFE by BEHIND in R_prev units', 'db_r', 'mfe')
    block('MFE by R_prev itself (scale only)', 'rp', 'mfe')
    # final control: volatility state. A large prior hour both clears boundaries ahead (da grows)
    # and predicts a large MFE. Near-vs-far AHEAD inside terciles of rp, pooled with equal weights.
    g = d[np.isfinite(d.da)].copy(); g['rt'] = pd.qcut(g.rp.rank(method='first'), 3, labels=False)
    for read, col in [('mfe', 'da'), ('mfe', 'db'), ('mae', 'db')]:
        h = d[np.isfinite(d[col])].copy(); h['rt'] = pd.qcut(h.rp.rank(method='first'), 3, labels=False)
        diffs, vars_ = [], []
        for q in range(3):
            x = h[h.rt == q].copy(); x['tc'] = pd.qcut(x[col].rank(method='first'), 3, labels=False)
            a, b = x[x.tc == 0][read], x[x.tc == 2][read]
            diffs.append(a.mean() - b.mean()); vars_.append(a.var() / len(a) + b.var() / len(b))
        print(' inside rp terciles: %s by %s near-far %+.2f (t %.2f) | by tercile %s'
              % (read.upper(), 'AHEAD' if col == 'da' else 'BEHIND', np.mean(diffs), np.mean(diffs) / (np.sqrt(np.sum(vars_)) / 3),
                 ' / '.join('%+.2f' % v for v in diffs)))
    f = d.copy(); f['ea'] = np.isinf(f.da)
    for nm, x in [('boundary ahead', f[~f.ea]), ('EMPTY ahead', f[f.ea])]:
        print('   %-15s n %5d | result mean %+.2f sd %.1f | reach tail MFE>=20: %.3f | hurt MAE>=16: %.3f'
              % (nm, len(x), x.res.mean(), x.res.std(), (x.mfe >= 20).mean(), (x.mae >= 16).mean()))


if __name__ == '__main__':
    import sys
    for inst in (sys.argv[1:] or ['NQ', 'ES', 'YM']):
        run(inst)
