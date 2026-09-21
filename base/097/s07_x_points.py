#!/usr/bin/env python3
"""099 step 4: the same relation with NO shared denominator. Declared before the count.

Steps 2-3 measured distances and excursions in candles; a candle that underestimates the
scale inflates both and can fake "far level -> large excursion". Here everything is in
points and the scale enters as explicit controls:
  log(range_pts) ~ log(candle) + log(R_prev) + year + log(dist RIZ ahead) + log(dist RIZ behind)
                   + log(dist session extreme ahead) + log(dist yesterday cash extreme ahead)
range = MFE + MAE of the S-07 side over 120 bars; cursor 09:33 ET; sessions where all four
distances exist. Same for log(MFE) and for the directional log(MFE/MAE).
Reading: a distance keeps a positive coefficient with |t| >= 3 on at least two indexes and
one sign on three -> "known level within reach = compressed movement" is not a candle
artifact; which generators survive jointly shows whether RIZ adds to non-RIZ levels.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
MIN = 60_000_000_000


def rows(inst):
    mk = ROOT / f'data/market/{inst}'
    O, H, L, C, TS, SID = (np.load(mk / f'{n}.npy') for n in ('open', 'high', 'low', 'close', 'close_ts_utc_ns', 'session_id'))
    et = pd.to_datetime(TS, utc=True).tz_convert('America/New_York')
    mod = (et.hour * 60 + et.minute).to_numpy(); yr = et.year.to_numpy(); pos = np.arange(len(TS))
    first = pd.Series(pos).groupby(SID).min()
    cash = (mod > 9 * 60 + 30) & (mod <= 16 * 60)
    ch = pd.Series(np.where(cash, H, -np.inf)).groupby(SID).max(); cl = pd.Series(np.where(cash, L, np.inf)).groupby(SID).min()
    sids = np.array(sorted(first.index)); prev_sid = dict(zip(sids[1:], sids[:-1]))
    z = pq.read_table(ROOT / f'work/080a/index/film1_{inst}.parquet', columns=[
        'zone_top', 'zone_bottom', 't0_spine_pos', 'c1_deletion_spine_pos', 'last_observed_spine_pos']).to_pandas()
    end = np.where(z.c1_deletion_spine_pos > 0, z.c1_deletion_spine_pos, z.last_observed_spine_pos).astype(np.int64)
    bp = np.concatenate([z.zone_top, z.zone_bottom]); bs = np.concatenate([z.t0_spine_pos, z.t0_spine_pos]); be = np.concatenate([end, end])
    o = np.argsort(bs); bp, bs, be = bp[o], bs[o], be[o]
    k = pos[mod == 9 * 60 + 33]; k = k[(k > 61) & (k + 121 < len(TS))]
    k = k[(TS[k] - TS[k - 60] == 60 * MIN) & (TS[k + 121] - TS[k] == 121 * MIN)]
    out = []
    for m in k:
        j = np.searchsorted(bs, m, 'right'); live = bp[:j][be[:j] >= m]
        hi, lo = H[m - 29:m + 1].max(), L[m - 29:m + 1].min(); s = 1.0 if C[m] > (hi + lo) / 2 else -1.0
        pc = C[m - 30:m]; cand = np.median(np.maximum(H[m - 29:m + 1], pc) - np.minimum(L[m - 29:m + 1], pc))
        e = O[m + 1]; d = s * (live - e)
        if cand <= 0 or not (d > 0).any() or not (d < 0).any():
            continue
        sid = SID[m]; ext = H[first[sid]:m + 1].max() if s > 0 else L[first[sid]:m + 1].min()
        ps = prev_sid.get(sid); lvl = (ch.get(ps, np.nan) if s > 0 else cl.get(ps, np.nan)) if ps is not None else np.nan
        dse = s * (ext - e); dpd = s * (lvl - e) if np.isfinite(lvl) else np.nan
        if not (dse > 0 and np.isfinite(dpd) and dpd > 0):
            continue
        seg = slice(m + 1, m + 122)
        mfe = (H[seg].max() - e) if s > 0 else (e - L[seg].min()); mae = (e - L[seg].min()) if s > 0 else (H[seg].max() - e)
        if mfe <= 0 or mae <= 0:
            continue
        rpv = H[m - 60:m + 1].max() - L[m - 60:m + 1].min()
        dens = int((np.abs(live - C[m]) <= rpv).sum())          # 050's density verbatim: live boundaries within +-R_prev
        nxt = H[m + 1:m + 61].max() - L[m + 1:m + 61].min()      # 050's outcome: range of the next hour
        out.append(dict(year=int(yr[m]), cand=cand, dens=dens, nxt=nxt, rprev=rpv,
                        da=d[d > 0].min(), db=(-d[d < 0]).min(), dse=dse, dpd=dpd, mfe=mfe, mae=mae))
    return pd.DataFrame(out)


def ols(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b
    Xi = np.linalg.inv(X.T @ X); V = Xi @ (X.T * r ** 2) @ X @ Xi          # HC0 robust
    return b, b / np.sqrt(np.diag(V))


if __name__ == '__main__':
    names = ['RIZ ahead', 'RIZ behind', 'session ext ahead', 'yesterday ext ahead']
    for inst in ['NQ', 'ES', 'YM']:
        d = rows(inst)
        fe = pd.get_dummies(d.year, drop_first=True).to_numpy(float)
        X = np.column_stack([np.ones(len(d)), np.log(d.cand), np.log(d.rprev), fe,
                             np.log(d.da), np.log(d.db), np.log(d.dse), np.log(d.dpd)])
        print(f'\n##### {inst}: sessions with all four levels {len(d)}')
        for nm, y in [('log range', np.log(d.mfe + d.mae)), ('log MFE', np.log(d.mfe)), ('log MFE/MAE', np.log(d.mfe / d.mae))]:
            b, t = ols(y.to_numpy(), X)
            print(' %-12s scale: candle %+.2f (t %.1f) R_prev %+.2f (t %.1f) | ' % (nm, b[1], t[1], b[2], t[2])
                  + ' | '.join('%s %+.3f (t %.2f)' % (n, bb, tt) for n, bb, tt in zip(names, b[-4:], t[-4:])))
        # 050/055 under the same scrutiny: log(next-hour range) on log(1+density) with both scales and year controlled
        X2 = np.column_stack([np.ones(len(d)), np.log(d.cand), np.log(d.rprev), fe, np.log1p(d.dens)])
        for nm, y in [('log R_next (050 outcome)', np.log(d.nxt)), ('log range 120', np.log(d.mfe + d.mae))]:
            b, t = ols(y.to_numpy(), X2)
            print(' 050 check    %-26s density %+.3f (t %.2f) | candle %+.2f R_prev %+.2f' % (nm, b[-1], t[-1], b[1], b[2]))
