#!/usr/bin/env python3
"""097 diagnostic: is the B7 lean of 076 money or a counting artifact?

Bet without a free parameter, declared in RFC v3 before any count:
  scene   first certified post-T0 contact of own b (Film-1 index of 080), NQ
  entry   open[k+1], side = outward (the T0 exit side), k = contact bar
  cancel  first bar j>k whose CLOSE is on the line or inside      (A)
  target  first bar j>k whose CLOSE is >= one zone width outside  (B)
  exit    open[j+1]; if neither by the 16:00 ET bar of the session -> close there (C)
Under a fair game the mean gross of ANY stopping rule is zero, so the mean gross
is the test. Inference by session-day blocks. Bands of w are declared by
arithmetic (cost 1.00 pt round trip against the target w), not by result.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[2] / 'work' / '097'
MK = ROOT / 'data/market/NQ'
O = np.load(MK / 'open.npy'); H = np.load(MK / 'high.npy'); L = np.load(MK / 'low.npy')
C = np.load(MK / 'close.npy'); TS = np.load(MK / 'close_ts_utc_ns.npy')
SID = np.load(MK / 'session_id.npy')
MIN = 60_000_000_000
COST = 1.00
BANDS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)]


def cutoffs():
    et = pd.to_datetime(TS, utc=True).tz_convert('America/New_York')
    mod = et.hour * 60 + et.minute
    ok = (mod <= 16 * 60) | (mod > 18 * 60)
    pos = np.arange(len(TS))
    s = pd.Series(np.where(ok, pos, -1)).groupby(SID).max()
    return s.reindex(SID).to_numpy(), mod


def run():
    t = pd.read_parquet(OUT / 'contact_rows_NQ.parquet')
    cut, mod = cutoffs()
    k = t.k.to_numpy()
    sgn = np.where(t.side.to_numpy() == 'north', 1.0, -1.0)
    B = t.exit_boundary.to_numpy(); w = t.w.to_numpy()
    n = len(t)
    res = np.full(n, np.nan); kind = np.empty(n, object); dur = np.full(n, -1)
    mae_w = np.full(n, np.nan); mae_c = np.full(n, np.nan); mfe = np.full(n, np.nan)
    for i in range(n):
        ki = int(k[i]); ce = int(cut[ki])
        if ce <= ki or TS[ki + 1] - TS[ki] != MIN or SID[ki + 1] != SID[ki]:
            kind[i] = 'no_entry'; continue
        seg_ts = TS[ki + 1:ce + 1]
        if np.any(np.diff(seg_ts) != MIN):
            g = int(np.argmax(np.diff(seg_ts) != MIN))      # last contiguous bar index in seg
            ce = ki + 1 + g
        oc = sgn[i] * (C[ki + 1:ce + 1] - B[i])
        a = np.flatnonzero(oc <= 0); b = np.flatnonzero(oc >= w[i])
        ja = a[0] if a.size else 10 ** 9; jb = b[0] if b.size else 10 ** 9
        j = min(ja, jb)
        entry = O[ki + 1]
        if j == 10 ** 9:
            kind[i] = 'C'; jj = ce; exit_ = C[jj]
        else:
            kind[i] = 'A' if ja < jb else 'B'
            jj = ki + 1 + j
            exit_ = O[jj + 1] if (jj + 1 <= int(cut[ki]) and TS[jj + 1] - TS[jj] == MIN) else C[jj]
        res[i] = sgn[i] * (exit_ - entry); dur[i] = jj - ki
        lo = L[ki + 1:jj + 1] if sgn[i] > 0 else -H[ki + 1:jj + 1]
        hi = H[ki + 1:jj + 1] if sgn[i] > 0 else -L[ki + 1:jj + 1]
        cl = sgn[i] * C[ki + 1:jj + 1]
        e = sgn[i] * entry
        mae_w[i] = e - lo.min(); mae_c[i] = e - cl.min(); mfe[i] = hi.max() - e
    t['kind'] = kind; t['gross'] = res; t['dur'] = dur
    t['mae_wick'] = mae_w; t['mae_close'] = mae_c; t['mfe'] = mfe
    t['entry_out'] = sgn * (O[np.minimum(k + 1, len(O) - 1)] - B)
    t.to_parquet(OUT / 'race_rows_NQ.parquet')
    return t


def table(t, title):
    print('\n' + title)
    print('%-9s %7s %7s | %5s %5s %5s | %8s %8s %6s | %7s %7s | %8s' % (
        'w band', 'rows', 'moments', 'A', 'B', 'C', 'gross', 'net', 't_day', 'win', 'loss', 'dur p50'))
    for lo, hi in BANDS:
        d = t[(t.w >= lo) & (t.w < hi) & (t.kind != 'no_entry')]
        if len(d) < 50:
            continue
        rep = d.sort_values(['k', 'side', 't0_spine_pos', 'tf_minutes', 'riz_id']).groupby(['k', 'side']).head(1)
        day = rep.groupby('sid').gross.agg(['sum', 'size'])
        m = rep.gross.mean()
        # day-block t for the mean per bet: ratio estimator, delta method
        r = day['sum'] - m * day['size']
        se = np.sqrt((r ** 2).sum()) / day['size'].sum()
        print('%-9s %7d %7d | %.3f %.3f %.3f | %+8.3f %+8.3f %6.2f | %+7.2f %+7.2f | %8.0f' % (
            '%g-%g' % (lo, min(hi, 999)), len(d), len(rep),
            (rep.kind == 'A').mean(), (rep.kind == 'B').mean(), (rep.kind == 'C').mean(),
            m, m - COST, m / se if se > 0 else np.nan,
            rep.gross[rep.gross > 0].mean(), rep.gross[rep.gross <= 0].mean(), rep.dur.median()))


if __name__ == '__main__':
    import sys
    t = run() if 'run' in sys.argv else pd.read_parquet(OUT / 'race_rows_NQ.parquet')
    print('rows', len(t), '| no_entry', int((t.kind == 'no_entry').sum()))
    d = t[t.year <= 2025]
    table(d[d.wait >= 2], 'DISCOVERY 2021-2025, wait>=2 (departure, then first test)')
    table(d[d.wait == 1], 'DISCOVERY 2021-2025, wait==1 (contact on the bar right after T0)')
    for y in range(2021, 2026):
        table(d[(d.wait >= 2) & (d.year == y)], 'wait>=2, year %d' % y)
    table(t[(t.year == 2026) & (t.wait >= 2)], '2026 partial (exposed by other lines), wait>=2')
