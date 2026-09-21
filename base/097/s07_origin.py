#!/usr/bin/env python3
"""Line 097: where does S-07 come from - the opening auction itself, or the overnight repricing carried on by the cash session?
Declared before the count, 2026-09-22. Chosen because the answer changes BOTH the origin and the use (earlier entry, skipping days).

Known at 09:30:00 ET:   G = sign( last print before the open - yesterday's 16:00 close )   the overnight direction (gap)
                        g = |gap| / yesterday's cash range (09:30-16:00)                     its size, own units of the day before
Known at 09:33:         S = S-07 side;  B = sign( close 09:33 - close 09:30 )                the first three minutes of the cash session
P1 (auction): the edge of S-07 does not care whether S agrees with G; the side G entered at 09:31 earns nothing.
P2 (overnight repricing): the edge sits where S = G, grows with g, and the side G entered at the 09:31 open is itself positive
    (and takes the first, most loaded minutes that S-07 misses).
Rule for P2: result of S-07 where S = G minus where S != G has one sign on NQ, ES, YM with |t| >= 3 on two, AND side G from
09:31 is positive on the three. Rule for P1: no such difference and side G near zero. Anything else: neither picture.
All results: close 120 bars after the 09:33 decision (same exit for every variant), in candles (median true range of the 30
bars before 09:33), gross. 2006-2026; epochs printed for the decisive contrast.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MIN = 60_000_000_000


def run(inst):
    mk = ROOT / f'data/market/{inst}'
    O, H, L, C, TS = (np.load(mk / f'{n}.npy') for n in ('open', 'high', 'low', 'close', 'close_ts_utc_ns'))
    et = pd.to_datetime(TS, utc=True).tz_convert('America/New_York')
    mod = (et.hour * 60 + et.minute).to_numpy(); yr = et.year.to_numpy(); pos = np.arange(len(TS)); day = et.normalize().asi8
    k = pos[mod == 9 * 60 + 33]; k = k[(k > 61) & (k + 123 < len(TS))]
    k = k[(TS[k] - TS[k - 30] == 30 * MIN) & (TS[k + 121] - TS[k] == 121 * MIN)]
    p16 = pos[mod == 16 * 60]
    rows = []
    for g in k:
        j = np.searchsorted(p16, g) - 1
        if j < 0 or TS[g] - TS[p16[j]] > 4 * 24 * 60 * MIN:
            continue
        c16 = p16[j]
        a = np.searchsorted(pos[mod == 9 * 60 + 31], c16) - 1
        o931 = pos[mod == 9 * 60 + 31]
        if a < 0 or day[o931[a]] != day[c16]:
            continue
        pr = H[o931[a]:c16 + 1].max() - L[o931[a]:c16 + 1].min()          # yesterday's cash range
        hi, lo = H[g - 29:g + 1].max(), L[g - 29:g + 1].min(); S = 1.0 if C[g] > (hi + lo) / 2 else -1.0
        pc = C[g - 30:g]; cand = np.median(np.maximum(H[g - 29:g + 1], pc) - np.minimum(L[g - 29:g + 1], pc))
        if cand <= 0 or pr <= 0:
            continue
        gap = C[g - 3] - C[c16]; G = np.sign(gap); B = np.sign(C[g] - C[g - 3]); X = C[g + 121]
        rows.append(dict(year=int(yr[g]), S=S, G=G, B=B, gsz=abs(gap) / pr,
                         M=S * (X - O[g + 1]) / cand,                  # S-07 as is
                         Gearly=G * (X - O[g - 2]) / cand,             # side G, entered at the 09:31 open
                         Glate=G * (X - O[g + 1]) / cand,              # side G, entered where S-07 enters
                         first3=G * (C[g] - O[g - 2]) / cand))         # what side G takes in the first three minutes
    return pd.DataFrame(rows)


def ms(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return (v.mean(), v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 5 else (np.nan, np.nan)


if __name__ == '__main__':
    for inst in ['NQ', 'ES', 'YM']:
        d = run(inst); d = d[d.G != 0]; d.to_parquet((Path(__file__).resolve().parents[2] / 'work' / 'line-097') / f's07_origin_{inst}.parquet')
        al = d.S == d.G; sb = d.S == d.B
        a, b = d.M[al], d.M[~al]; dif = a.mean() - b.mean(); td = dif / np.sqrt(a.var() / len(a) + b.var() / len(b))
        print(f'\n##### {inst}: sessions {len(d)} | S-07 agrees with the overnight direction in {al.mean():.3f}, with the first 3 minutes in {sb.mean():.3f}')
        print('  S-07 all %+.3f (t %.1f) | where S = G %+.3f (t %.1f) | where S != G %+.3f (t %.1f) | difference %+.3f (t %.1f)' % (ms(d.M) + ms(a) + ms(b) + (dif, td)))
        print('  side G alone: from the 09:31 open %+.3f (t %.1f), of which the first three minutes %+.3f (t %.1f); from the S-07 entry %+.3f (t %.1f)'
              % (ms(d.Gearly) + ms(d.first3) + ms(d.Glate)))
        q = pd.qcut(d.gsz.rank(method='first'), 3, labels=False)
        print('  by size of the gap (terciles, in yesterday\'s cash ranges; cuts %.2f / %.2f):' % (d.gsz.quantile(1 / 3), d.gsz.quantile(2 / 3)))
        for i, nm in enumerate(['small', 'middle', 'large']):
            e = d[q == i]; ea = e.S == e.G
            print('     %-6s  S-07 %+.3f (t %.1f) | S = G (%.2f) %+.3f (t %.1f) | S != G %+.3f (t %.1f) | side G from 09:31 %+.3f (t %.1f)'
                  % ((nm,) + ms(e.M) + (ea.mean(),) + ms(e.M[ea]) + ms(e.M[~ea]) + ms(e.Gearly)))
        print('  decisive contrast by epoch (S = G minus S != G): ' + ' | '.join(
            '%d-%d %+.3f' % (y0, y1, d.M[al & d.year.between(y0, y1)].mean() - d.M[~al & d.year.between(y0, y1)].mean()) for y0, y1 in ((2006, 2012), (2013, 2019), (2020, 2026))))
        print('  S-07 where it agrees with the first three minutes %+.3f (t %.1f) | where it does not %+.3f (t %.1f)' % (ms(d.M[sb]) + ms(d.M[~sb])))
