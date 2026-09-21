#!/usr/bin/env python3
"""Line 097: the branch of the parent tree the field never sees - "own gap NOT revisited" - read inside the opening drive.
Declared before the count, 2026-09-22.

Found just before: all of S-07's expectancy sits in the ~10 % of sessions where price never came back even to the near
edge of the first gap born along the drive; that statement is conditioned on the future. Honest rewind:
  state at cursor c = decision + N minutes (N = 15, 30, 45):  the first drive-side one-minute gap born after the decision
  exists by c and has NOT been touched by c  (UNTOUCHED)  against  touched by c.
  outcome: the REMAINING move of the S-07 side from the open of bar c+1 to the same exit (close 120 bars after the decision).
Lesson of step 100 built in: "untouched" is largely "the position is already far in profit". So the state is read INSIDE
terciles of the current mark (open S-07 profit at c, candles). It carries knowledge only if, at an equal mark, the remaining
move differs. If it does not, the object is the mark, not the gap.
Three indexes, 2006-2026, candles (one constant per session), gross.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MIN = 60_000_000_000
NS = (15, 30, 45)


def run(inst):
    mk = ROOT / f'data/market/{inst}'
    O, H, L, C, TS = (np.load(mk / f'{n}.npy') for n in ('open', 'high', 'low', 'close', 'close_ts_utc_ns'))
    et = pd.to_datetime(TS, utc=True).tz_convert('America/New_York')
    mod = (et.hour * 60 + et.minute).to_numpy(); yr = et.year.to_numpy(); pos = np.arange(len(TS))
    k = pos[mod == 9 * 60 + 33]; k = k[(k > 61) & (k + 122 < len(TS))]
    k = k[(TS[k] - TS[k - 30] == 30 * MIN) & (TS[k + 121] - TS[k] == 121 * MIN)]
    rows = []
    for g in k:
        hi, lo = H[g - 29:g + 1].max(), L[g - 29:g + 1].min(); s = 1.0 if C[g] > (hi + lo) / 2 else -1.0
        pc = C[g - 30:g]; cand = np.median(np.maximum(H[g - 29:g + 1], pc) - np.minimum(L[g - 29:g + 1], pc))
        if cand <= 0:
            continue
        a0 = g - 31; Os, Hs, Ls, Cs = O[a0:g + 123], H[a0:g + 123], L[a0:g + 123], C[a0:g + 123]
        o, h, l, c = (Os, Hs, Ls, Cs) if s > 0 else (-Os, -Ls, -Hs, -Cs)
        m = 31; end = m + 121; born = touch = 10 ** 6
        for i in range(m + 1, m + 111):
            if l[i] > h[i - 2] and max(o[i - 1], c[i - 1]) > h[i - 2] and min(o[i - 1], c[i - 1]) < l[i]:
                bt2 = max(o[i - 1], c[i - 1]); c3bb = min(o[i], c[i]); bt1 = max(o[i - 2], c[i - 2]); c2bb = min(o[i - 1], c[i - 1])
                nb_ = bt1 if c2bb > bt1 else h[i - 2]; nt_ = c3bb if c3bb > bt2 else l[i]
                if nt_ > nb_:
                    born = i
                    f = np.flatnonzero(l[i + 1:end + 1] <= nt_)
                    touch = i + 1 + f[0] if f.size else 10 ** 6
                    break
        row = dict(year=int(yr[g]))
        for n in NS:
            cu = m + n
            row[f'st{n}'] = (0 if born > cu else (1 if touch > cu else 2))      # 0 no gap yet, 1 UNTOUCHED, 2 touched
            row[f'mk{n}'] = (c[cu] - o[m + 1]) / cand
            row[f'rm{n}'] = (c[end] - o[cu + 1]) / cand
        rows.append(row)
    return pd.DataFrame(rows)


def ms(v):
    v = np.asarray(v, float)
    return (v.mean(), v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 5 else (np.nan, np.nan)


if __name__ == '__main__':
    for inst in ['NQ', 'ES', 'YM']:
        d = run(inst); d.to_parquet((Path(__file__).resolve().parents[2] / 'work' / 'line-097') / f's07_untouched_{inst}.parquet')
        print(f'\n##### {inst}: sessions {len(d)}')
        for n in NS:
            u, t_ = d[d[f'st{n}'] == 1], d[d[f'st{n}'] == 2]
            print(' +%2d min: untouched %.3f of sessions | mark %+.2f vs %+.2f | REMAINING untouched %+.3f (t %.1f) vs touched %+.3f (t %.1f)'
                  % ((n, len(u) / len(d), u[f'mk{n}'].mean(), t_[f'mk{n}'].mean()) + ms(u[f'rm{n}']) + ms(t_[f'rm{n}'])))
            x = d[d[f'st{n}'] > 0].copy(); x['q'] = pd.qcut(x[f'mk{n}'].rank(method='first'), 3, labels=False)
            out = []
            for q in range(3):
                a, b = x[(x.q == q) & (x[f'st{n}'] == 1)][f'rm{n}'], x[(x.q == q) & (x[f'st{n}'] == 2)][f'rm{n}']
                dif = a.mean() - b.mean(); tt = dif / np.sqrt(a.var() / max(len(a), 1) + b.var() / max(len(b), 1)) if len(a) > 5 else np.nan
                out.append('mark T%d: n %d/%d  %+.2f vs %+.2f  diff %+.2f (t %.1f)' % (q + 1, len(a), len(b), a.mean(), b.mean(), dif, tt))
            print('        inside mark terciles, remaining untouched vs touched: ' + ' | '.join(out))
            allm = x.copy(); allm['q'] = pd.qcut(allm[f'mk{n}'].rank(method='first'), 3, labels=False)
            print('        remaining by the mark alone (all sessions): ' + ' | '.join('T%d %+.3f (t %.1f)' % ((q + 1,) + ms(allm[allm.q == q][f'rm{n}'])) for q in range(3)))
