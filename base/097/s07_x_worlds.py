#!/usr/bin/env python3
"""099 step 3: what is the object behind "a live boundary close ahead -> shorter MFE"?
Declared before the count, 2026-09-22. Three worlds, each must fit ALL numbers incl. the
failed "empty ahead = free space":
  W1 close boundary    the RIZ boundary itself matters: effect survives inside strata of
                       non-RIZ levels ahead; travel tends to end near the boundary (stall).
  W2 field topology    distance is not a continuous measure of room but a label of states:
                       profile over distance is step-like / non-monotone, "empty" is its own
                       state with its own path shape, not the limit of "far".
  W3 travelled ground  the boundary only encodes where price stands in recently travelled
                       territory: a non-RIZ level ahead (extreme of the session so far,
                       extreme of yesterday's cash session) gives the same effect and the RIZ
                       effect loses more than half inside its terciles.
Also  what is redistributed inside the path when MFE shrinks and the mean does not:
      time of MFE, order MFE-before-MAE, give-back (MFE - result), quantiles of the result.
And   what S-07 adds: the same reading at other fixed clock cursors with the same side rule
      (03:33, 07:33, 11:33, 13:33 ET). Same relation everywhere = a property of the field in
      general; only at 09:33 = a relation between the event and the field.
Frozen from step 2: side rule, candle, liveness, 120 bars, terciles, near-minus-far on the
mean inside terciles of rp, one sign on three indexes and |t| >= 3 on two.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'work' / '099'
MIN = 60_000_000_000
CURS = [3 * 60 + 33, 7 * 60 + 33, 9 * 60 + 33, 11 * 60 + 33, 13 * 60 + 33]


def build(inst):
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
    rows = []
    for cur in CURS:
        k = pos[mod == cur]; k = k[(k > 61) & (k + 121 < len(TS))]
        k = k[(TS[k] - TS[k - 60] == 60 * MIN) & (TS[k + 121] - TS[k] == 121 * MIN)]
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
            sid = SID[m]; a0 = first[sid]
            ext = H[a0:m + 1].max() if s > 0 else L[a0:m + 1].min()
            dse = s * (ext - e) / cand
            ps = prev_sid.get(sid)
            lvl = (ch.get(ps, np.nan) if s > 0 else cl.get(ps, np.nan)) if ps is not None else np.nan
            dpd = s * (lvl - e) / cand if np.isfinite(lvl) else np.nan
            seg = slice(m + 1, m + 122)
            fav = (H[seg] - e) if s > 0 else (e - L[seg]); adv = (e - L[seg]) if s > 0 else (H[seg] - e)
            rows.append(dict(cur=cur, year=int(yr[m]), da=da, db=db,
                             dse=dse if dse > 0 else np.inf, dpd=(dpd if dpd > 0 else np.inf) if np.isfinite(dpd) else np.nan,
                             rp=(H[m - 60:m + 1].max() - L[m - 60:m + 1].min()) / cand,
                             mfe=fav.max() / cand, mae=adv.max() / cand, tmfe=int(fav.argmax()), tmae=int(adv.argmax()),
                             res=s * (C[m + 121] - e) / cand))
    d = pd.DataFrame(rows); d.to_parquet(OUT / f'worlds_{inst}.parquet'); return d


def nf(x, col, read, strat=None):
    """near-minus-far tercile difference of the mean of `read` by `col`, optionally inside terciles of `strat`."""
    h = x[np.isfinite(x[col])].dropna(subset=[read]).copy()
    if strat is None:
        h['st'] = 0
    else:
        h = h[np.isfinite(h[strat])].copy(); h['st'] = pd.qcut(h[strat].rank(method='first'), 3, labels=False)
    df, vr = [], []
    for q, g in h.groupby('st'):
        g = g.copy(); g['tc'] = pd.qcut(g[col].rank(method='first'), 3, labels=False)
        a, b = g[g.tc == 0][read], g[g.tc == 2][read]
        df.append(a.mean() - b.mean()); vr.append(a.var() / len(a) + b.var() / len(b))
    n = len(df); return np.mean(df), np.mean(df) / (np.sqrt(np.sum(vr)) / n)


def report(inst, d):
    x = d[d.cur == 9 * 60 + 33]
    print(f'\n##### {inst}  cursor 09:33, sessions {len(x)}')
    f = x[np.isfinite(x.da)].copy(); f['q'] = pd.qcut(f.da.rank(method='first'), 5, labels=False)
    print(' 1 MFE median by quintile of distance AHEAD -> empty: ' + ' / '.join('%.1f' % f[f.q == i].mfe.median() for i in range(5))
          + ' -> %.1f' % x[np.isinf(x.da)].mfe.median()
          + ' | distance p50 by quintile: ' + ' / '.join('%.1f' % f[f.q == i].da.median() for i in range(5)))
    e = x[np.isinf(x.da)]
    print('   EMPTY ahead (n %d): prior-hour range rp p50 %.1f (all %.1f) | at session extreme %.2f (all %.2f) | beyond yesterday cash extreme %.2f (all %.2f) | MAE p50 %.1f (all %.1f) | result mean %+.2f'
          % (len(e), e.rp.median(), x.rp.median(), np.isinf(e.dse).mean(), np.isinf(x.dse).mean(),
             np.isinf(e.dpd[e.dpd.notna()]).mean(), np.isinf(x.dpd[x.dpd.notna()]).mean(), e.mae.median(), x.mae.median(), e.res.mean()))
    f['tc'] = pd.qcut(f.da.rank(method='first'), 3, labels=False)
    print(' 2 inside the path        tMFE p50 | MFE first | give-back | result p25 / p50 / p75 / p95 | MFE>=20')
    for nm, g in [('near', f[f.tc == 0]), ('far', f[f.tc == 2]), ('empty', e)]:
        print('   %-6s                %7.0f | %9.3f | %9.2f | %+6.1f / %+5.1f / %+5.1f / %+5.1f | %.3f' % (
            nm, g.tmfe.median(), (g.tmfe < g.tmae).mean(), (g.mfe - g.res).mean(),
            g.res.quantile(.25), g.res.median(), g.res.quantile(.75), g.res.quantile(.95), (g.mfe >= 20).mean()))
    print(' 3 worlds (near-minus-far MFE, mean, t):')
    for nm, col, st in [('RIZ ahead, inside rp', 'da', 'rp'), ('session extreme ahead, inside rp', 'dse', 'rp'),
                        ('yesterday cash extreme ahead, inside rp', 'dpd', 'rp'),
                        ('RIZ ahead INSIDE session-extreme terciles', 'da', 'dse'), ('RIZ ahead INSIDE yesterday-extreme terciles', 'da', 'dpd'),
                        ('session extreme INSIDE RIZ terciles', 'dse', 'da'), ('yesterday extreme INSIDE RIZ terciles', 'dpd', 'da')]:
        m, t = nf(x, col, 'mfe', st); print('   %-46s %+.2f (t %.2f)' % (nm, m, t))
    rng = np.random.default_rng(20260922)
    for nm, col in [('RIZ boundary', 'da'), ('session extreme', 'dse'), ('yesterday extreme', 'dpd')]:
        g = x[np.isfinite(x[col])].dropna(subset=[col]); r = g.mfe / g[col]; obs = ((r >= .85) & (r <= 1.15)).mean()
        sim = []
        for _ in range(200):
            rr = g.mfe.to_numpy() / rng.permutation(g[col].to_numpy())
            sim.append(((rr >= .85) & (rr <= 1.15)).mean())
        print('   stall near %-18s share MFE in [0.85,1.15] of distance %.3f vs shuffled %.3f (z %.2f)' % (nm, obs, np.mean(sim), (obs - np.mean(sim)) / np.std(sim)))
    print(' 4 other cursors, RIZ ahead near-minus-far MFE inside rp: ' + ' | '.join(
        '%02d:%02d %+.2f (t %.2f)' % ((c // 60, c % 60) + nf(d[d.cur == c], 'da', 'mfe', 'rp')) for c in CURS))


if __name__ == '__main__':
    for inst in ['NQ', 'ES', 'YM']:
        p = OUT / f'worlds_{inst}.parquet'
        d = build(inst) if ('run' in sys.argv or not p.exists()) else pd.read_parquet(p)
        report(inst, d)
