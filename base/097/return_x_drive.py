#!/usr/bin/env python3
"""100: two established facts read as one object. Declared before the count, 2026-09-22.

  092  early no-stop participation in the return to own b wins 0.88 of the time and still has
       no equal-day mean: the money is lost on a few one-directional days (negative skew).
  S-07 the side set at the close of 09:33 ET persists for about two hours, its money lives in
       the tail days (positive skew); sign transfers over three indexes and three epochs.
These are the same market fact seen from two sides: trend days exist and are partly
recognisable at 09:33. 092 decisions are made before 08:00 ET (ttc >= 480), i.e. BEFORE the
drive is known, so the relation is a management event inside an open position:
"waited for the return -> saw the drive -> changed the action".
Prediction (direction fixed now): among 092 trades still open at the close of 09:33,
  those standing AGAINST the drive carry the loss and the left tail; those standing WITH it
  do not. No tunable number: 092 frozen as is, S-07 side rule as is.
Variants  V0 = 092 v1 unchanged;  V1 = every open trade leaves at the 09:34 open;
          V2 = only trades against the drive leave at the 09:34 open.
Read     remaining P&L from the 09:34 open by alignment; per-trade and equal-day economics
         of V0/V1/V2 after 1.00 pt, day bootstrap. Discovery NQ 2006-2018 only - 092's
         evaluation 2019-2026 was never opened and stays closed here.
Adaptation history: 092's tail structure was known from its card before this idea; the cut
  at 09:33 comes from S-07, not from scanning clock times.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
MK = ROOT / 'data/market/NQ'
O, H, L, C, TS, SID = (np.load(MK / f'{n}.npy') for n in ('open', 'high', 'low', 'close', 'close_ts_utc_ns', 'session_id'))
et = pd.to_datetime(TS, utc=True).tz_convert('America/New_York')
MOD = (et.hour * 60 + et.minute).to_numpy()
MIN = 60_000_000_000
COST = 1.0

if __name__ == '__main__':
    p = pd.read_parquet(ROOT / 'work/092/policy_v1_NQ_discovery.parquet')
    p = p[p.kind.isin([0, 1])].copy()                       # reached / time exit; unknown stays out as in 092
    ix = pq.read_table(ROOT / 'work/080a/index/film1_NQ.parquet', columns=['riz_id', 't0_spine_pos', 'exit_boundary']).to_pandas()
    p = p.merge(ix, on='riz_id', how='left')
    rows = []; miss = 0
    for r in p.itertuples():
        t0 = int(r.t0_spine_pos); e = r.exit_boundary; up = bool(r.north); q = -1
        for a in range(0, 3000):
            k = t0 + a
            dcl = (C[k] - e) if up else (e - C[k])
            if dcl >= 8.0:
                g = (O[k + 1] - e) if up else (e - O[k + 1])
                if abs(g - r.gross) < 1e-3 and abs(dcl - r.d_close) < 1e-3:
                    q = k; break
        if q < 0:
            miss += 1; continue
        j = q + 1; x = q + int(r.dur); o = O[j]; tdir = -1.0 if up else 1.0
        seg = np.arange(j, x + 1); hit = seg[MOD[seg] == 9 * 60 + 33]
        alive = hit.size > 0 and hit[0] < x
        row = dict(day=int(r.day), pay=float(r.pay), alive=bool(alive), tdir=tdir)
        if alive:
            m = int(hit[0])
            if TS[m] - TS[m - 29] != 29 * MIN or TS[m + 1] - TS[m] != MIN:
                row['alive'] = False
            else:
                mid = (H[m - 29:m + 1].max() + L[m - 29:m + 1].min()) / 2
                drive = 1.0 if C[m] > mid else -1.0
                row.update(with_drive=bool(drive == tdir), mtm=tdir * (O[m + 1] - o))
        rows.append(row)
    d = pd.DataFrame(rows); d['with_drive'] = d['with_drive'].astype('boolean')
    print('092 trades reconstructed %d of %d (no match %d) | open at the 09:33 close: %d' % (len(d), len(p), miss, d.alive.sum()))
    a = d[d.alive].copy(); a['rem'] = a.pay - a.mtm
    print('\nstate of the open trades at the 09:34 open and what came after, points:')
    for nm, g in [('WITH the drive', a[a.with_drive == True]), ('AGAINST the drive', a[a.with_drive == False])]:
        print('  %-18s n %4d days %3d | marked %+7.2f | remaining mean %+7.2f  p5 %+8.2f  p50 %+6.2f  p95 %+7.2f | reached b after %.3f | final %+7.2f'
              % (nm, len(g), g.day.nunique(), g.mtm.mean(), g.rem.mean(), g.rem.quantile(.05), g.rem.median(), g.rem.quantile(.95), (g.rem > 0).mean(), g.pay.mean()))
    dm = a.groupby('day').rem.mean(); yrA = pd.to_datetime(a.day, unit='D').dt.year
    print('  remaining P&L after the 09:34 open, DAY as the unit: days %d | mean of day means %+.2f (t %.2f) | median day %+.2f | days>0 %.3f'
          % (len(dm), dm.mean(), dm.mean() / (dm.std(ddof=1) / np.sqrt(len(dm))), dm.median(), (dm > 0).mean()))
    d['v0'] = d.pay; d['v1'] = np.where(d.alive, d.mtm, d.pay)
    d['v2'] = np.where(d.alive & (d.with_drive == False), d.mtm, d.pay)
    rng = np.random.default_rng(20260922)
    print('\neconomics after %.2f pt, all 092 trades:' % COST)
    for v in ['v0', 'v1', 'v2']:
        net = d[v] - COST; day = net.groupby(d.day).agg(['sum', 'mean'])
        bs = np.array([day['mean'].sample(len(day), replace=True, random_state=int(rng.integers(1 << 31))).mean() for _ in range(2000)])
        bs2 = np.array([day['sum'].sample(len(day), replace=True, random_state=int(rng.integers(1 << 31))).mean() for _ in range(2000)])
        print('  %s per trade %+6.2f | equal-day mean %+6.2f [%+.2f; %+.2f] | day-sum mean %+7.2f [%+.1f; %+.1f] | worst day %+8.1f | days+ %.3f | worst trade %+7.1f'
              % (v.upper(), net.mean(), day['mean'].mean(), *np.percentile(bs, [2.5, 97.5]), day['sum'].mean(), *np.percentile(bs2, [2.5, 97.5]),
                 day['sum'].min(), (day['sum'] > 0).mean(), net.min()))
    yr = pd.to_datetime(d.day, unit='D').dt.year
    print('  V1 minus V0 day-sum by year: ' + ' '.join('%d:%+.0f' % (y, ((d.v1 - d.v0)[yr == y]).sum()) for y in sorted(yr.unique())))
    print('  V2 minus V0 day-sum by year: ' + ' '.join('%d:%+.0f' % (y, ((d.v2 - d.v0)[yr == y]).sum()) for y in sorted(yr.unique())))
