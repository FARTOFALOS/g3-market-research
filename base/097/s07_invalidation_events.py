#!/usr/bin/env python3
"""Line 097: another CLASS of question (lens of 2026-09-22): not "how to predict the 10 % run-away sessions" but
"what is the earliest observable event that takes away the opening move's right to a big tail - and how much of the tail
is still ahead until then?"  The object is an INVALIDATION of continuation, not a prediction of it; the trade is the
cost of the right to stay in S-07 until the market itself proves that today is not a tail day.

A stopping event cannot add mean unless the path after it has negative mean; its honest value is elsewhere: the same mean
with less dispersion, less time at risk and the tail kept. So every event is read by four numbers:
  remaining  mean move AFTER the event to the normal exit (must be <= about 0, else the event throws money away)
  kept tail  share of the money of tail sessions (unmanaged result >= +15 candles) that survives the rule
  mean / sd  per session after the rule, against unmanaged S-07  (the scale-free yardstick of this line)
  time       share of the 120 minutes actually spent in the position
Events, declared before looking, one-minute bars, everything oriented so that the drive is up; exit = next open:
  A  first touch of the near edge of the first gap born along the drive        (known: 90 % of sessions; reference)
  B  a fresh gap born along the drive is sliced by ONE body against the drive  (the machine's span, against)
  C  first close beyond the FAR edge of the first gap born along the drive
  D  birth of the first gap AGAINST the drive (machine's birth rule verbatim)
  G  first close below the entry price                                           (price only, no gap)
  H  first close back beyond the middle of the 30-bar range that gave the side   (S-07's own premise undone; price only)
  R  S-07's existing rule: out at minute 15 if the position is under water       (reference from the S-07 card)
Three indexes, 2006-2026, candles, gross. No threshold is tuned; the tail cut-off (+15 candles) is used only to REPORT.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MIN = 60_000_000_000
EV = ['A', 'B', 'C', 'D', 'G', 'H', 'R']
SD = (1, 2, 3, 4, 6, 8, 12)
TD = (2, 3, 4, 6, 8, 12)        # gap-free TRAILING close-based stops (below the highest close so far): event C trails too, in a quarter of sessions its far edge is above the entry      # gap-free close-based stops, candles below entry: is event C more than a distance?


def run(inst):
    mk = ROOT / f'data/market/{inst}'
    O, H, L, C, TS = (np.load(mk / f'{n}.npy') for n in ('open', 'high', 'low', 'close', 'close_ts_utc_ns'))
    et = pd.to_datetime(TS, utc=True).tz_convert('America/New_York')
    mod = (et.hour * 60 + et.minute).to_numpy(); yr = et.year.to_numpy(); pos = np.arange(len(TS))
    k = pos[mod == 9 * 60 + 33]; k = k[(k > 61) & (k + 123 < len(TS))]
    k = k[(TS[k] - TS[k - 30] == 30 * MIN) & (TS[k + 122] - TS[k] == 122 * MIN)]
    rows = []
    for g in k:
        hi, lo = H[g - 29:g + 1].max(), L[g - 29:g + 1].min(); s = 1.0 if C[g] > (hi + lo) / 2 else -1.0
        pc = C[g - 30:g]; cand = np.median(np.maximum(H[g - 29:g + 1], pc) - np.minimum(L[g - 29:g + 1], pc))
        if cand <= 0:
            continue
        a0 = g - 31; sl = slice(a0, g + 124)
        o, h, l, c = (O[sl], H[sl], L[sl], C[sl]) if s > 0 else (-O[sl], -L[sl], -H[sl], -C[sl])
        mid = s * (hi + lo) / 2
        m = 31; end = m + 121; e0 = o[m + 1]; M = (c[end] - e0) / cand
        first = None; zones = []; t = {x: None for x in EV}; ts = {d_: None for d_ in SD}; tt = {d_: None for d_ in TD}; hc = -1e18
        for j in range(m + 1, end):
            # events read on the close of bar j against what was known before it
            if first is not None:
                if t['A'] is None and l[j] <= first[0]:
                    t['A'] = j
                if t['C'] is None and c[j] < first[1]:
                    t['C'] = j
            if t['B'] is None:
                for zt, zb in zones:
                    if o[j] > zt and c[j] < zb:
                        t['B'] = j; break
            for d_ in TD:
                if tt[d_] is None and hc > -1e17 and c[j] < hc - d_ * cand:
                    tt[d_] = j
            hc = max(hc, c[j])
            for d_ in SD:
                if ts[d_] is None and c[j] < e0 - d_ * cand:
                    ts[d_] = j
            if t['G'] is None and c[j] < e0:
                t['G'] = j
            if t['H'] is None and c[j] < mid:
                t['H'] = j
            if t['R'] is None and j == m + 15 and c[j] < e0:
                t['R'] = j
            # births known at the close of bar j
            if l[j] > h[j - 2] and max(o[j - 1], c[j - 1]) > h[j - 2] and min(o[j - 1], c[j - 1]) < l[j]:
                bt1 = max(o[j - 2], c[j - 2]); bt2 = max(o[j - 1], c[j - 1]); c2bb = min(o[j - 1], c[j - 1]); c3bb = min(o[j], c[j])
                nb_ = bt1 if c2bb > bt1 else h[j - 2]; nt_ = c3bb if c3bb > bt2 else l[j]
                if nt_ > nb_:
                    zones.append((nt_, nb_))
                    if first is None:
                        first = (nt_, nb_)
            if t['D'] is None and h[j] < l[j - 2] and min(o[j - 1], c[j - 1]) < l[j - 2] and max(o[j - 1], c[j - 1]) > h[j]:
                t['D'] = j
        row = dict(year=int(yr[g]), M=M)
        for x in EV:
            if t[x] is None:
                row[x] = M; row['t' + x] = np.nan
            else:
                row[x] = (o[t[x] + 1] - e0) / cand; row['t' + x] = t[x] - m
        for d_ in SD:
            row[f'S{d_}'] = M if ts[d_] is None else (o[ts[d_] + 1] - e0) / cand; row[f'tS{d_}'] = np.nan if ts[d_] is None else ts[d_] - m
        for d_ in TD:
            row[f'T{d_}'] = M if tt[d_] is None else (o[tt[d_] + 1] - e0) / cand; row[f'tT{d_}'] = np.nan if tt[d_] is None else tt[d_] - m
        row['cdist'] = np.nan if first is None else (e0 - first[1]) / cand
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    for inst in ['NQ', 'ES', 'YM']:
        d = run(inst); d.to_parquet((Path(__file__).resolve().parents[2] / 'work' / 'line-097') / f's07_invalidation_{inst}.parquet')
        tail = d.M >= 15
        print(f'\n##### {inst}: sessions {len(d)} | unmanaged S-07: mean {d.M.mean():+.3f}  sd {d.M.std():.1f}  mean/sd {d.M.mean() / d.M.std():+.4f}'
              f' | p5 {d.M.quantile(.05):+.1f} | tail sessions (>= +15) {tail.mean():.3f} carry {d.M[tail].sum() / d.M.sum():.2f} of the total')
        print(' ev | occurs | minute p25/50/75 | remaining after it (t) | kept tail | managed mean (t)   sd   mean/sd |   p5  | time in position')
        print('   far edge of the first drive gap sits %.1f / %.1f / %.1f candles below the entry (p25/50/75)' % tuple(d.cdist.quantile([.25, .5, .75])))
        for x in EV + [f'S{d_}' for d_ in SD] + [f'T{d_}' for d_ in TD]:
            occ = d['t' + x].notna(); rem = (d.M - d[x])[occ]
            tm = np.where(occ, d['t' + x], 120) / 120
            v = d[x]
            print(' %3s | %.3f  | %3.0f / %3.0f / %3.0f   | %+6.3f (%5.1f)        | %.3f     | %+6.3f (%4.1f) %5.1f  %+.4f | %+5.1f | %.2f' % (
                x, occ.mean(), *d['t' + x].quantile([.25, .5, .75]), rem.mean(), rem.mean() / (rem.std() / np.sqrt(len(rem))),
                v[tail].sum() / d.M[tail].sum(), v.mean(), v.mean() / (v.std() / np.sqrt(len(v))), v.std(), v.mean() / v.std(), v.quantile(.05), tm.mean()))
        print('   event C by epoch, mean/sd managed vs unmanaged vs the S-07 rule R: ' + ' | '.join(
            '%d-%d: C %+.4f  M %+.4f  R %+.4f' % (a, b, e.C.mean() / e.C.std(), e.M.mean() / e.M.std(), e.R.mean() / e.R.std())
            for a, b, e in [(a, b, d[(d.year >= a) & (d.year <= b)]) for a, b in ((2006, 2012), (2013, 2019), (2020, 2026))]))
