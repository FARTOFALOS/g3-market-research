#!/usr/bin/env python3
"""097 census: first post-T0 contact of own exit boundary b, NQ, no outcomes.

Reads the frozen Film-1 index of 080 (strict certification) and the NQ tape.
Builds one row per RIZ at the contact bar k using ONLY bars <= k, then groups
rows into physical episodes. Nothing after k is read here.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[2] / 'work' / '097'
MK = ROOT / 'data/market/NQ'

O = np.load(MK / 'open.npy', mmap_mode='r')
H = np.load(MK / 'high.npy', mmap_mode='r')
L = np.load(MK / 'low.npy', mmap_mode='r')
C = np.load(MK / 'close.npy', mmap_mode='r')
TS = np.load(MK / 'close_ts_utc_ns.npy', mmap_mode='r')
SID = np.load(MK / 'session_id.npy', mmap_mode='r')


def build():
    t = pq.read_table(ROOT / 'work/080a/index/film1_NQ.parquet').to_pandas()
    t = t[t.film1_status == 'contact_certified'].copy()
    k = t.first_observed_contact_pos.to_numpy()
    p = t.t0_spine_pos.to_numpy()
    kts = pd.to_datetime(np.asarray(TS)[k], utc=True).tz_convert('America/New_York')
    t['k'] = k
    t['k_et'] = kts
    t['year'] = kts.year
    t = t[(t.year >= 2021) & (t.year <= 2026)].copy()
    k = t.k.to_numpy(); p = t.t0_spine_pos.to_numpy()
    s = np.where(t.side.to_numpy() == 'north', 1.0, -1.0)
    B = t.exit_boundary.to_numpy()
    w = (t.zone_top - t.zone_bottom).to_numpy()
    n = len(t)
    U = np.empty(n); full_out = np.zeros(n, bool)
    for i in range(n):
        a, b_ = int(p[i]), int(k[i])
        if s[i] > 0:
            U[i] = float(np.max(H[a:b_])) - B[i]
            full_out[i] = bool(np.any(np.asarray(L[a + 1:b_]) > B[i])) if b_ > a + 1 else False
        else:
            U[i] = B[i] - float(np.min(L[a:b_]))
            full_out[i] = bool(np.any(np.asarray(H[a + 1:b_]) < B[i])) if b_ > a + 1 else False
    Hk, Lk, Ck, Ok = (np.asarray(x)[k] for x in (H, L, C, O))
    t['wait'] = k - p
    t['w'] = w
    t['U'] = U                                   # prior outward extreme vs b, points
    t['has_clean_bar'] = full_out                # >=1 bar fully outside b before contact
    t['k_close_out'] = s * (Ck - B)              # >0 closed outside
    t['k_open_out'] = s * (Ok - B)
    t['k_depth_in'] = np.where(s > 0, B - Lk, Hk - B)   # wick depth inside, >=0
    t['k_new_ext'] = np.where(s > 0, Hk - B, B - Lk) - U  # >0: contact bar made new outward extreme
    t['k_range'] = Hk - Lk
    t['sid'] = np.asarray(SID)[k]
    t['t0_same_session'] = np.asarray(SID)[p] == t.sid.to_numpy()
    return t


def main():
    t = build()
    t.to_parquet(OUT / 'contact_rows_NQ.parquet')
    disc = t[t.year <= 2025]
    print('RIZ rows, certified first contact, contact in 2021-2025:', len(disc),
          '| 2026 partial:', int((t.year == 2026).sum()))
    ep = disc.groupby(['k', 'side', 'exit_boundary']).agg(
        n_riz=('riz_id', 'size'), sid=('sid', 'first'), year=('year', 'first'),
        hour=('k_et', lambda x: x.iloc[0].hour)).reset_index()
    mom = disc.groupby(['k', 'side']).agg(n_levels=('exit_boundary', 'nunique'),
                                          sid=('sid', 'first')).reset_index()
    print('episodes (minute, side, b):', len(ep), '| moments (minute, side):', len(mom),
          '| distinct minutes:', disc.k.nunique())
    print('RIZ per episode: median %.0f p90 %.0f max %d' % (
        ep.n_riz.median(), ep.n_riz.quantile(.9), ep.n_riz.max()))
    print('levels per moment: median %.0f p90 %.0f max %d' % (
        mom.n_levels.median(), mom.n_levels.quantile(.9), mom.n_levels.max()))

    # calendar of eligible days: every session present on the tape in 2021-2025
    ts_all = pd.to_datetime(np.asarray(TS), utc=True).tz_convert('America/New_York')
    yr = ts_all.year
    m = (yr >= 2021) & (yr <= 2025)
    sid_all = np.asarray(SID)[m]
    bars_per_sid = pd.Series(sid_all).value_counts()
    days = bars_per_sid.index.to_numpy()
    full_days = bars_per_sid[bars_per_sid >= 600].index.to_numpy()
    print('sessions on tape 2021-2025:', len(days), '| with >=600 bars:', len(full_days))
    per_day = mom.groupby('sid').size()
    per_day = per_day.reindex(full_days, fill_value=0)
    print('moments per full session: mean %.1f median %.0f p10 %.0f p90 %.0f | share of sessions with >=1: %.3f'
          % (per_day.mean(), per_day.median(), per_day.quantile(.1), per_day.quantile(.9),
             float((per_day > 0).mean())))

    print('\nper-RIZ prefix facts at contact bar (2021-2025):')
    for c in ['wait', 'w', 'U', 'k_depth_in', 'k_close_out', 'k_range']:
        q = disc[c].quantile([.1, .25, .5, .75, .9]).to_numpy()
        print('  %-12s p10 %.2f  p25 %.2f  p50 %.2f  p75 %.2f  p90 %.2f' % ((c,) + tuple(q)))
    print('  wait==1 (contact on T0+1): %.3f' % float((disc.wait == 1).mean()))
    print('  has >=1 bar fully outside before contact: %.3f' % float(disc.has_clean_bar.mean()))
    print('  contact bar closed outside b: %.3f | on/inside: %.3f' % (
        float((disc.k_close_out > 0).mean()), float((disc.k_close_out <= 0).mean())))
    print('  contact bar made new outward extreme (order unknown): %.3f' % float((disc.k_new_ext > 0).mean()))
    print('  contact in same session as T0: %.3f' % float(disc.t0_same_session.mean()))

    # same facts on the moment level with a prefix-only representative: earliest T0, smaller TF, riz_id
    rep = disc.sort_values(['k', 'side', 't0_spine_pos', 'tf_minutes', 'riz_id']).groupby(['k', 'side']).head(1)
    rep.to_parquet(OUT / 'moments_NQ.parquet')
    print('\nmoment representatives (earliest T0, then smaller TF):', len(rep))
    for c in ['wait', 'w', 'U', 'k_depth_in', 'k_close_out']:
        q = rep[c].quantile([.1, .25, .5, .75, .9]).to_numpy()
        print('  %-12s p10 %.2f  p25 %.2f  p50 %.2f  p75 %.2f  p90 %.2f' % ((c,) + tuple(q)))
    hours = rep.k_et.dt.hour.value_counts().sort_index()
    print('  moments by ET hour:', dict(hours))
    print('  by year:', dict(rep.year.value_counts().sort_index()))


if __name__ == '__main__':
    main()
