#!/usr/bin/env python3
"""097 H3: the same first test of b, read in candles of the zone's OWN timeframe.

Declared before the count (2026-09-21):
  why      on minute bars the median zone (2.5 pt) is thinner than one bar (12 pt)
           and wide zones are tested by bars 10x smaller than the zone; "the wick
           probes, the body decides" was never read at the zone's own scale.
  who      NQ RIZ with tf 5..60 (a test candle and a few more must fit intraday),
           T0 in 2021-2025 (2026 printed apart). Candles on the RIZ's own grid,
           anchored at the session open. n0 = candle holding the T0 minute.
  event    first candle j>n0, same session, closing <= 16:00 ET, whose range
           reaches b. HELD: close outside b. ACCEPTED: close on/inside b.
  entry    open of the minute after the candle close.
  readout  G_close  outward gross to the 16:00 ET close
           G_struct outward gross until a later own-TF candle closes on/inside b
                    (exit next minute open), else the 16:00 close
           for ACCEPTED the mirror: inward, until a candle closes outside again
  unit     one bet per (minute, side); representative = earliest T0, smaller TF
  rule     supported only if HELD G_struct > 0 with day-block t >= 3 pooled and
           >0 in >= 4 of 5 years, and HELD-vs-ACCEPTED G_close contrast t >= 3.
No stop, target, threshold or horizon is tuned: there is none.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from race_money import cutoffs, O, H, L, C, TS, SID, MIN

OUT = Path(__file__).resolve().parents[2] / 'work' / '097'
ROOT = OUT.parents[1]


def build():
    z = np.load(ROOT / 'data/market/NQ/sessions.npz')
    s_open = dict(zip(z['session_id'].tolist(), z['session_open_utc_ns'].tolist()))
    cut, _ = cutoffs()
    t = pq.read_table(ROOT / 'work/080a/index/film1_NQ.parquet', columns=[
        'riz_id', 'tf_minutes', 'side', 'zone_top', 'zone_bottom', 'exit_boundary',
        't0_spine_pos', 't0_ts_ns']).to_pandas()
    yr = pd.to_datetime(t.t0_ts_ns, utc=True).dt.tz_convert('America/New_York').dt.year
    t = t[(t.tf_minutes >= 5) & (t.tf_minutes <= 60) & (yr >= 2021)].copy()
    t['year'] = yr[t.index]
    rows = []
    for r in t.itertuples():
        p = int(r.t0_spine_pos); tf = int(r.tf_minutes); sid = int(SID[p])
        an = s_open[sid]; ce = int(cut[p])
        if ce <= p:
            continue
        s = 1.0 if r.side == 'north' else -1.0
        B = r.exit_boundary
        n0 = ((int(TS[p]) - an) // MIN - 1) // tf
        j = n0 + 1; ev = None
        while True:
            end_ts = an + (j + 1) * tf * MIN
            if end_ts > TS[ce]:
                break
            a = int(np.searchsorted(TS, an + j * tf * MIN, 'right'))
            b = int(np.searchsorted(TS, end_ts, 'right'))
            if b <= a or TS[b - 1] != end_ts:
                break                                   # candle close not observed on grid
            lo = (L[a:b].min() - B) if s > 0 else (B - H[a:b].max())
            if lo <= 0:
                ev = (j, a, b); break
            j += 1
        if ev is None:
            continue
        j, a, b = ev
        q = b - 1
        if q + 1 > ce or TS[q + 1] - TS[q] != MIN:
            continue
        held = s * (C[q] - B) > 0
        entry = O[q + 1]
        g_close = s * (C[ce] - entry)
        # structural exit: later own-TF candle closing on the other side of b
        jj = j + 1; exit_ = C[ce]
        while True:
            end_ts = an + (jj + 1) * tf * MIN
            if end_ts > TS[ce]:
                break
            e = int(np.searchsorted(TS, end_ts, 'right')) - 1
            if e < 0 or TS[e] != end_ts:
                break
            out_side = s * (C[e] - B) > 0
            if out_side != held:
                exit_ = O[e + 1] if (e + 1 <= ce and TS[e + 1] - TS[e] == MIN) else C[e]
                break
            jj += 1
        g_struct = s * (exit_ - entry)
        rows.append(dict(riz_id=r.riz_id, tf=tf, side=r.side, year=int(r.year), sid=sid, q=q,
                         t0=p, w=r.zone_top - r.zone_bottom, candles_after_n0=j - n0,
                         held=bool(held), close_out=s * (C[q] - B),
                         g_close=g_close, g_struct=g_struct if held else -g_struct))
    d = pd.DataFrame(rows)
    d.to_parquet(OUT / 'h3_rows.parquet')
    return d


def stat(x, col):
    if len(x) < 30:
        return len(x), np.nan, np.nan
    day = x.groupby('sid')[col].agg(['sum', 'size'])
    m = x[col].mean()
    se = np.sqrt(((day['sum'] - m * day['size']) ** 2).sum()) / day['size'].sum()
    return len(x), m, m / se


if __name__ == '__main__':
    d = build() if 'run' in sys.argv else pd.read_parquet(OUT / 'h3_rows.parquet')
    rep = d.sort_values(['q', 'side', 't0', 'tf', 'riz_id']).groupby(['q', 'side']).head(1)
    disc = rep[rep.year <= 2025]
    print('RIZ rows with an own-TF test candle:', len(d), '| moments:', len(rep),
          '| discovery moments:', len(disc), '| sessions:', disc.sid.nunique())
    print('held share %.3f | median w %.2f | median candles after n0 %.0f'
          % (disc.held.mean(), disc.w.median(), disc.candles_after_n0.median()))
    print('\nG_struct is signed WITH the reading: outward for HELD, inward for ACCEPTED')
    print('%-26s %6s | %8s %6s | %8s %6s' % ('', 'n', 'G_close', 't', 'G_struct', 't'))
    for name, x in [('HELD, all', disc[disc.held]), ('ACCEPTED, all', disc[~disc.held])]:
        n, m1, t1 = stat(x, 'g_close'); _, m2, t2 = stat(x, 'g_struct')
        print('%-26s %6d | %+8.3f %6.2f | %+8.3f %6.2f' % (name, n, m1, t1, m2, t2))
    for lo, hi in [(5, 10), (11, 20), (21, 40), (41, 60)]:
        for name, x in [('HELD', disc[disc.held]), ('ACCEPTED', disc[~disc.held])]:
            x = x[(x.tf >= lo) & (x.tf <= hi)]
            n, m1, t1 = stat(x, 'g_close'); _, m2, t2 = stat(x, 'g_struct')
            print('%-26s %6d | %+8.3f %6.2f | %+8.3f %6.2f' % ('%s tf %d-%d' % (name, lo, hi), n, m1, t1, m2, t2))
    print('\nby year, HELD G_struct / ACCEPTED G_struct')
    for y in range(2021, 2027):
        a = rep[(rep.year == y) & rep.held]; b = rep[(rep.year == y) & ~rep.held]
        na, ma, ta = stat(a, 'g_struct'); nb, mb, tb = stat(b, 'g_struct')
        print(y, 'HELD n=%d %+.3f t %.2f | ACCEPTED n=%d %+.3f t %.2f' % (na, ma, ta, nb, mb, tb))
    # contrast HELD - ACCEPTED in outward G_close, day-block
    x = disc.copy(); x['sgn'] = np.where(x.held, 1.0, -1.0)
    dm = x.groupby(['sid', 'held']).g_close.mean().unstack()
    dd = (dm[True] - dm[False]).dropna()
    print('\nHELD minus ACCEPTED, outward G_close, per-day difference: mean %+.3f t %.2f n_days %d'
          % (dd.mean(), dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))), len(dd)))
