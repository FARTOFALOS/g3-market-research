#!/usr/bin/env python3
"""Line 097: the gap lens placed where the tape is NOT a fair game. Declared before the count, 2026-09-22.

Side is given by S-07 (close of 09:33 ET against the middle of the last 30 bars). The question is the PRICE OF
PARTICIPATION, not the side: what does a fresh gap born along the drive add to the execution of a position whose
edge comes from elsewhere?
All variants share the side and the exit (close 120 bars after the decision), so on a filled session
   result(variant) - result(market) = price improvement exactly; delay cannot leak into the difference.
  M     market at the 09:34 open (S-07 as is, no management)
  G     resting order at the near edge of the FIRST one-minute gap born along the drive after the decision
        (machine's birth rule verbatim); placed at the birth close, valid to the exit; no fill -> no trade
  K_d   the same without any gap: resting order d candles behind the 09:34 open, d in 0.25 ... 3, placed at 09:34
Honest decomposition of G: fill rate; improvement on filled sessions; what M earned on the sessions G MISSED (S-07's
money lives in run-away days, so this is the named danger: the better price may arrive exactly where the edge is weak).
Role of the gap as structure, not as a ruler: G must lie ABOVE the curve traced by K_d in the plane
(fill rate, mean per session). If G sits on that curve the gap is only a convenient level: kills the branch.
Enough to call it a new role: G above the K curve on NQ, ES and YM and in at least two of the three S-07 epochs,
with the risk side (adverse excursion from entry) not worse. Units: candles (median true range of the 30 bars before
the decision) - one constant per session, so within-session differences are clean. Gross; both pay the same round trip.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MIN = 60_000_000_000
DS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)


def run(inst):
    mk = ROOT / f'data/market/{inst}'
    O, H, L, C, TS = (np.load(mk / f'{n}.npy') for n in ('open', 'high', 'low', 'close', 'close_ts_utc_ns'))
    et = pd.to_datetime(TS, utc=True).tz_convert('America/New_York')
    mod = (et.hour * 60 + et.minute).to_numpy(); yr = et.year.to_numpy(); pos = np.arange(len(TS))
    k = pos[mod == 9 * 60 + 33]; k = k[(k > 61) & (k + 122 < len(TS))]
    k = k[(TS[k] - TS[k - 30] == 30 * MIN) & (TS[k + 121] - TS[k] == 121 * MIN)]
    rows = []
    for m in k:
        hi, lo = H[m - 29:m + 1].max(), L[m - 29:m + 1].min(); s = 1.0 if C[m] > (hi + lo) / 2 else -1.0
        pc = C[m - 30:m]; cand = np.median(np.maximum(H[m - 29:m + 1], pc) - np.minimum(L[m - 29:m + 1], pc))
        if cand <= 0:
            continue
        a0 = m - 31; b0 = m + 123                     # local window only: orienting the whole tape per session was the slow part
        Os, Hs, Ls, Cs = O[a0:b0], H[a0:b0], L[a0:b0], C[a0:b0]
        o, h, l, c = (Os, Hs, Ls, Cs) if s > 0 else (-Os, -Ls, -Hs, -Cs)   # oriented: the drive is up
        g = m; m = 31; end = m + 121
        P0o, Xo = o[m + 1], c[end]
        row = dict(year=int(yr[g]), M=(Xo - P0o) / cand, cand=cand, Mmae=(P0o - l[m + 1:end + 1].min()) / cand)
        fill = None; born = None
        for i in range(m + 1, m + 111):
            if l[i] > h[i - 2] and max(o[i - 1], c[i - 1]) > h[i - 2] and min(o[i - 1], c[i - 1]) < l[i]:
                bt1 = max(o[i - 2], c[i - 2]); bt2 = max(o[i - 1], c[i - 1]); c2bb = min(o[i - 1], c[i - 1]); c3bb = min(o[i], c[i])
                nb_ = bt1 if c2bb > bt1 else h[i - 2]; nt_ = c3bb if c3bb > bt2 else l[i]
                if nt_ > nb_:
                    born = i; edge = nt_
                    for j in range(i + 1, end + 1):
                        if l[j] <= edge:
                            fill = (j, min(edge, o[j])); break
                    break
        row['born'] = (born - m) if born else np.nan
        if fill:
            j, pf = fill
            row.update(Gfill=1.0, Gimp=(P0o - pf) / cand, G=(Xo - pf) / cand, Gt=j - m, Gmae=(pf - l[j:end + 1].min()) / cand)
        else:
            row.update(Gfill=0.0, Gimp=np.nan, G=0.0, Gt=np.nan, Gmae=np.nan)
        for d in DS:
            lim = P0o - d * cand; f = np.flatnonzero(l[m + 1:end + 1] <= lim)
            if f.size:
                j = m + 1 + f[0]; pf = min(lim, o[j]); row[f'K{d}'] = (Xo - pf) / cand; row[f'F{d}'] = 1.0
            else:
                row[f'K{d}'] = 0.0; row[f'F{d}'] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def t(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return v.mean(), v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 2 else np.nan


if __name__ == '__main__':
    for inst in ['NQ', 'ES', 'YM']:
        d = run(inst); d.to_parquet((Path(__file__).resolve().parents[2] / 'work' / 'line-097') / f's07_gap_{inst}.parquet')
        print(f'\n##### {inst}: sessions {len(d)} | a drive-side gap was born in {d.born.notna().mean():.3f}, minutes after decision p25/50/75 '
              + ' / '.join('%.0f' % v for v in d.born.quantile([.25, .5, .75])))
        for nm, e in [('all', d)] + [(f'{a}-{b}', d[(d.year >= a) & (d.year <= b)]) for a, b in ((2006, 2012), (2013, 2019), (2020, 2026))]:
            f = e[e.Gfill == 1]; u = e[e.Gfill == 0]
            print(' %-9s M %+.3f (t %.1f) | G fill %.3f at +%.0f min | improvement %+.3f (t %.1f) | M on filled %+.3f, M on MISSED %+.3f (t %.1f) | G per session %+.3f (t %.1f) | G-M %+.3f (t %.1f)'
                  % ((nm,) + t(e.M) + (e.Gfill.mean(), f.Gt.median()) + t(f.Gimp) + (f.M.mean(),) + t(u.M) + t(e.G) + t(e.G - e.M)))
            print('           K curve (fill rate: mean per session): ' + '  '.join('d%.2g %.2f:%+.3f' % (dd, e[f'F{dd}'].mean(), e[f'K{dd}'].mean()) for dd in DS)
                  + ' | risk on G-filled sessions: MAE G %.2f vs M %.2f' % (f.Gmae.median(), f.Mmae.median()))
