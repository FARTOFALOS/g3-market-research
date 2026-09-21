#!/usr/bin/env python3
"""Line 097: S-07 turned out to be an OPENING-AUCTION phenomenon (s07_origin.py: its side ignores the overnight direction and is the
direction of the first three minutes of the cash session). Most important unknown that follows: is this a CLASS of events or one
unique place?  Declared before the count, 2026-09-22.

Earlier looks could not see a class: the 389-minute scan of S-09 and my cursors used a 120-minute hold, while the residual of the
opening response lives about 30 minutes; and they scanned clock minutes, not scheduled liquidity events.
Identical rule everywhere, nothing tuned: decision on the close of event+3 minutes; side = close against the middle of the high-low
range of the last 30 bars; entry next open; read at +10 and +30 bars; candles = median true range of those 30 bars.
  scheduled events (ET): 03:00 European cash open | 08:30 US data slot | 09:30 US cash open (reference) | 10:00 US data slot |
                         15:50 closing-imbalance publication
  controls, same rule, no scheduled event: 05:00, 07:00, 11:30, 12:30, 13:30
An event belongs to the class if the +30 result is positive on NQ, ES and YM with t >= 3 on two. Controls must stay at zero,
otherwise the rule itself, not the event, produces the number. 2006-2026, gross, candles.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MIN = 60_000_000_000
EVENTS = [('03:00 Europe open', 180), ('08:30 US data slot', 510), ('09:30 US cash open', 570), ('10:00 US data slot', 600), ('15:50 closing imbalance', 950)]
CONTROLS = [('05:00', 300), ('07:00', 420), ('11:30', 690), ('12:30', 750), ('13:30', 810)]


def read(O, H, L, C, TS, mod, ev):
    pos = np.arange(len(TS)); k = pos[mod == ev + 3]; k = k[(k > 31) & (k + 32 < len(TS))]
    k = k[(TS[k] - TS[k - 30] == 30 * MIN) & (TS[k + 31] - TS[k] == 31 * MIN)]
    out = []
    for g in k:
        hi, lo = H[g - 29:g + 1].max(), L[g - 29:g + 1].min(); s = 1.0 if C[g] > (hi + lo) / 2 else -1.0
        pc = C[g - 30:g]; cand = np.median(np.maximum(H[g - 29:g + 1], pc) - np.minimum(L[g - 29:g + 1], pc))
        if cand > 0:
            out.append((s * (C[g + 10] - O[g + 1]) / cand, s * (C[g + 30] - O[g + 1]) / cand))
    return np.array(out)


def ms(v):
    return v.mean(), v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))


if __name__ == '__main__':
    for inst in ['NQ', 'ES', 'YM']:
        mk = ROOT / f'data/market/{inst}'
        O, H, L, C, TS = (np.load(mk / f'{n}.npy') for n in ('open', 'high', 'low', 'close', 'close_ts_utc_ns'))
        et = pd.to_datetime(TS, utc=True).tz_convert('America/New_York'); mod = (et.hour * 60 + et.minute).to_numpy()
        print(f'\n##### {inst}')
        for nm, ev in EVENTS + CONTROLS:
            r = read(O, H, L, C, TS, mod, ev)
            print('  %-24s n %5d | +10 bars %+.3f (t %4.1f) | +30 bars %+.3f (t %4.1f) | mean/sd at +30 %+.4f'
                  % ((nm, len(r)) + ms(r[:, 0]) + ms(r[:, 1]) + (r[:, 1].mean() / r[:, 1].std(),)))
