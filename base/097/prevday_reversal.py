#!/usr/bin/env python3
"""098 step 2. Born from a defect of step 1: a placebo that inherited the RIZ side on
the neighbouring earlier session continued only 0.433 of the time, not 0.5. The side of
a TF>=60 ignition is therefore NOT independent of the previous session: the zone was
cut by an earlier move, and the ignition comes back through it the other way.

Dangerous reading of 095 to test: 0.566 is the DAILY REVERSAL seen through the RIZ
side (the ignition points against yesterday, and the index tends to take back part of
yesterday by the close), not a continuation carried by the ignition.

prevdir  sign of the previous session's cash move: close(16:00) - open of the 09:31 bar.
Scene    095 slice (TF>=60, T0 in 09:34..11:30 ET), one row per (minute, side).
Ordinary every minute 09:34..11:30 of every session, direction-free outcome
         M = sign(close(16:00) - open[k+1]).
Read     share of ignitions pointing against yesterday;
         R_all = P(M == -prevdir) on ordinary minutes (the reversal itself);
         RIZ: P(cont | against yesterday) vs R_all, P(cont | with yesterday) vs 1-R_all.
Rule     RIZ adds something only if P(cont) beats the ordinary-minute number inside a
         stratum with |t| >= 3 (session-clustered) in both eras.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from drive_alignment import O, C, TS, SID, MOD, MIN, i16, et, ROOT

pos = np.arange(len(TS))
i931 = pd.Series(np.where(MOD == 9 * 60 + 31, pos, -1)).groupby(SID).max()
sids = sorted(set(i16.index) & set(i931.index))
cash = {s: np.sign(C[i16[s]] - O[i931[s]]) for s in sids if i16[s] > 0 and i931[s] > 0}
prev = {}
for a, b in zip(sids[:-1], sids[1:]):
    if a in cash and cash[a] != 0:
        prev[b] = cash[a]


def est(v, sid):
    x = pd.DataFrame(dict(v=v, sid=sid)); day = x.groupby('sid').v.agg(['sum', 'size']); m = x.v.mean()
    return m, np.sqrt(((day['sum'] - m * day['size']) ** 2).sum()) / day['size'].sum()


if __name__ == '__main__':
    win = (MOD > 9 * 60 + 33) & (MOD <= 11 * 60 + 30)
    k_all = pos[win][:-1]
    ok = np.array([(int(SID[k]) in prev) and i16.get(int(SID[k]), -1) > k + 1 and TS[k + 1] - TS[k] == MIN for k in k_all])
    k_all = k_all[ok]
    sid_all = SID[k_all]; yr_all = et.year.to_numpy()[k_all]
    M = np.sign(C[np.array([i16[int(s)] for s in sid_all])] - O[k_all + 1])
    pd_all = np.array([prev[int(s)] for s in sid_all])
    t = pq.read_table(ROOT / 'work/080a/index/film1_NQ.parquet', columns=['tf_minutes', 'side', 't0_spine_pos']).to_pandas()
    t = t[t.tf_minutes >= 60].drop_duplicates(['t0_spine_pos', 'side'])
    idx = pd.Series(np.arange(len(k_all)), index=k_all)
    t = t[t.t0_spine_pos.isin(idx.index)]
    j = idx[t.t0_spine_pos].to_numpy(); s = np.where(t.side == 'north', 1.0, -1.0)
    for era, lo, hi in [('2006-2018', 2006, 2018), ('2019-2026', 2019, 2026)]:
        a = (yr_all >= lo) & (yr_all <= hi)
        R, Rs = est((M[a] == -pd_all[a]).astype(float), sid_all[a])
        b = a[j]
        jj, ss = j[b], s[b]
        against = ss == -pd_all[jj]; cont = (M[jj] == ss).astype(float)
        sh, shs = est(against.astype(float), sid_all[jj])
        pa, pas = est(cont[against], sid_all[jj][against]); pw, pws = est(cont[~against], sid_all[jj][~against])
        pu, pus = est(cont, sid_all[jj])
        print('\n=== %s' % era)
        print('ordinary minutes %d: reversal of yesterday by the close R = %.3f ±%.3f' % (a.sum(), R, Rs))
        print('RIZ moments %d: all P(cont) %.3f ±%.3f | share pointing against yesterday %.3f ±%.3f' % (len(jj), pu, pus, sh, shs))
        print('   against yesterday: P(cont) %.3f ±%.3f  vs ordinary %.3f  -> diff %+.3f (t %.2f)'
              % (pa, pas, R, pa - R, (pa - R) / np.hypot(pas, Rs)))
        print('   with yesterday   : P(cont) %.3f ±%.3f  vs ordinary %.3f  -> diff %+.3f (t %.2f)'
              % (pw, pws, 1 - R, pw - (1 - R), (pw - (1 - R)) / np.hypot(pws, Rs)))
        print('   what selection alone predicts for all: %.3f' % (sh * R + (1 - sh) * (1 - R)))
