#!/usr/bin/env python3
"""101: the opening drive read against the overnight inventory of UNRETURNED RIZ.
Declared before the count, 2026-09-22.

New object, visible only when today's facts are put together:
  - the return to own b is the most frequent RIZ regularity (0.86-0.93 within 30 min, 0.88 in 092);
  - S-07: the side set at the close of 09:33 ET persists ~2 h, front-loaded, transfers;
  - 099-4 (denominator-free): the S-07 side goes better when the overnight extreme in its
    direction is far, i.e. when the drive moves back INTO the overnight range;
  - 100: whether a return trade stands with or against the drive is only its current mark.
So neither a level near price nor one ignition matters, but a RELATION: is the drive itself
the pending return of the overnight field, or does it run away from it?
Inventory at the 09:33 close: ignitions (T0 minute, side) born in the current session
  (since 18:00 ET) whose own b has not been touched yet and whose RIZ is alive.
  ahead  = own b lies in the direction of the drive (the drive performs their return)
  behind = own b lies against the drive (the drive runs further away from it)
No distance, no threshold: counts of physical ignitions, share ahead = n_a / (n_a + n_b).
Prediction (direction fixed): the larger the share ahead, the better the S-07 side:
  log(MFE/MAE) over 120 bars (the candle cancels) and result in points with explicit scale
  (log candle, log R_prev, year). Rule: one sign on NQ, ES, YM and |t| >= 3 on two.
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
    z = pq.read_table(ROOT / f'work/080a/index/film1_{inst}.parquet', columns=[
        'side', 'exit_boundary', 't0_spine_pos', 'first_observed_contact_pos', 'c1_deletion_spine_pos', 'last_observed_spine_pos']).to_pandas()
    z['end'] = np.where(z.c1_deletion_spine_pos > 0, z.c1_deletion_spine_pos, z.last_observed_spine_pos)
    z['cp'] = np.where(z.first_observed_contact_pos > 0, z.first_observed_contact_pos, np.iinfo(np.int64).max)
    z = z.sort_values('t0_spine_pos'); t0 = z.t0_spine_pos.to_numpy()
    b = z.exit_boundary.to_numpy(); cp = z.cp.to_numpy(); en = z.end.to_numpy(); north = (z.side == 'north').to_numpy()
    k = pos[mod == 9 * 60 + 33]; k = k[(k > 61) & (k + 121 < len(TS))]
    k = k[(TS[k] - TS[k - 60] == 60 * MIN) & (TS[k + 121] - TS[k] == 121 * MIN)]
    out = []
    for m in k:
        a0 = first[SID[m]]
        i0, i1 = np.searchsorted(t0, a0, 'left'), np.searchsorted(t0, m, 'right')
        sel = (cp[i0:i1] > m) & (en[i0:i1] >= m)
        hi, lo = H[m - 29:m + 1].max(), L[m - 29:m + 1].min(); s = 1.0 if C[m] > (hi + lo) / 2 else -1.0
        pc = C[m - 30:m]; cand = np.median(np.maximum(H[m - 29:m + 1], pc) - np.minimum(L[m - 29:m + 1], pc))
        if cand <= 0:
            continue
        e = O[m + 1]
        mom = pd.DataFrame(dict(t=t0[i0:i1][sel], n=north[i0:i1][sel], b=b[i0:i1][sel])).drop_duplicates(['t', 'n'])
        ahead = int((s * (mom.b - e) > 0).sum()); behind = int((s * (mom.b - e) < 0).sum())
        seg = slice(m + 1, m + 122)
        mfe = (H[seg].max() - e) if s > 0 else (e - L[seg].min()); mae = (e - L[seg].min()) if s > 0 else (H[seg].max() - e)
        if mfe <= 0 or mae <= 0:
            continue
        out.append(dict(year=int(yr[m]), cand=cand, rprev=H[m - 60:m + 1].max() - L[m - 60:m + 1].min(),
                        ahead=ahead, behind=behind, lr=np.log(mfe / mae), res=s * (C[m + 121] - e), mfe=mfe, mae=mae))
    return pd.DataFrame(out)


def ols(y, X):
    bb, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ bb
    Xi = np.linalg.inv(X.T @ X); V = Xi @ (X.T * r ** 2) @ X @ Xi
    return bb, bb / np.sqrt(np.diag(V))


if __name__ == '__main__':
    for inst in ['NQ', 'ES', 'YM']:
        d = rows(inst); d['n'] = d.ahead + d.behind
        print(f'\n##### {inst}: sessions {len(d)} | with pending overnight returns {int((d.n > 0).sum())} ({(d.n > 0).mean():.3f}) | pending per session p50 {d.n.median():.0f} | share ahead mean {(d.ahead / d.n)[d.n > 0].mean():.3f}')
        for nm, g in [('none pending', d[d.n == 0]), ('all BEHIND (drive runs away)', d[(d.n > 0) & (d.ahead == 0)]),
                      ('mixed', d[(d.ahead > 0) & (d.behind > 0)]), ('all AHEAD (drive is the return)', d[(d.n > 0) & (d.behind == 0)])]:
            if len(g) < 30:
                continue
            print('  %-32s n %5d | log MFE/MAE %+.3f (t %.2f) | result pts %+7.2f (t %.2f) | MFE/candle %.1f MAE/candle %.1f'
                  % (nm, len(g), g.lr.mean(), g.lr.mean() / (g.lr.std() / np.sqrt(len(g))), g.res.mean(), g.res.mean() / (g.res.std() / np.sqrt(len(g))),
                     (g.mfe / g.cand).median(), (g.mae / g.cand).median()))
        x = d[d.n > 0]; fe = pd.get_dummies(x.year, drop_first=True).to_numpy(float)
        X = np.column_stack([np.ones(len(x)), np.log(x.cand), np.log(x.rprev), fe, (x.ahead / x.n).to_numpy()])
        for nm, y in [('log MFE/MAE', x.lr.to_numpy()), ('result, candles', (x.res / x.cand).to_numpy()), ('log MFE', np.log(x.mfe).to_numpy()), ('log MAE', np.log(x.mae).to_numpy())]:
            bb, t = ols(y, X); print('  regression on share ahead: %-16s %+.3f (t %.2f)' % (nm, bb[-1], t[-1]))
