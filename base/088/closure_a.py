#!/usr/bin/env python3
"""(a) Narrow corpus-wide closure of the local close-fall-back relation. NOT a search.

Frozen literal event OCC(t): newM (Wo>running M from H[T0]) AND C <= Bo[t-1]
  (outward exploration by the extreme, close not held beyond previous bar's body).
Held front event HELD(t): newMB (Bo>running MB) AND C > Bo[t-1].
For each live OCC, the NEXT front-update event (newM or newMB) before the physical
Film-1 endpoint is classified: HELD / FALLBACK(=another OCC) / OTHER / NONE(terminal,
no further front event) / LOST(censored before any).
No thresholds, horizons, lookaheads, labels, bins, WOECB'. Episode=(t0,side), longest
life. Repeated occurrences kept as sequence; first-occurrence = independent unit.
"""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('C:/Users/Admin/Claude/g3-market-research')
sys.path.insert(0, str(ROOT / 'base/084'))
import race084
market, kind = race084.load_market('NQ'); high, low, close = market
opn = np.load(ROOT / 'data/market/NQ/open.npy'); last = close.size - 1
films = pd.read_parquet(ROOT / 'work/081a/paths/films_NQ_discovery.parquet',
                        columns=['side', 't0_spine_pos', 'exit_boundary', 'end_pos'])
# episode = physical act (t0,side); longest-life film gives fullest excursion
films['life'] = films.end_pos - films.t0_spine_pos
epi = (films.sort_values('life').groupby(['t0_spine_pos', 'side'], as_index=False)
       .last())


def classify_film(t0, up, e, end_pos):
    """Return list of front events (k, cls) and terminal reason. cls in HELD/FALLBACK/OTHER."""
    sgn = 1 if up else -1
    M = high[t0] if up else -low[t0]
    MB = max(sgn * opn[t0], sgn * close[t0])
    prevBo = MB  # previous bar body-front outward (T0 bar)
    events = []
    reason = 'contact_b'
    for k in range(1, end_pos - t0 + 1):
        j = t0 + k
        if j > last: reason = 'archive'; break
        if kind[j] != 0: reason = 'gap'; break
        hi = high[j]; lo = low[j]
        if lo <= e <= hi: reason = 'contact_b'; break
        Wo = hi if up else -lo
        Oo = sgn * opn[j]; Co = sgn * close[j]; Bo = max(Oo, Co)
        newM = Wo > M; newMB = Bo > MB
        if newM or newMB:
            if newMB and Co > prevBo:
                cls = 'HELD'
            elif newM and Co <= prevBo:
                cls = 'FALLBACK'          # == OCC
            else:
                cls = 'OTHER'
            events.append((k, cls))
        if newM: M = Wo
        if newMB: MB = Bo
        prevBo = Bo
    return events, reason


occ_sequel = Counter()      # occurrence-level (all OCC, sequence history)
first_sequel = Counter()    # first-OCC-per-episode (independent unit)
n_epi = 0; n_epi_with_occ = 0; occ_per_epi = []
t0a = epi.t0_spine_pos.to_numpy(); up_a = (epi.side.values == 'north')
ea = epi.exit_boundary.to_numpy(); ep = epi.end_pos.to_numpy()
for i in range(len(epi)):
    events, reason = classify_film(int(t0a[i]), bool(up_a[i]), float(ea[i]), int(ep[i]))
    n_epi += 1
    occ_idx = [n for n, (k, c) in enumerate(events) if c == 'FALLBACK']
    if occ_idx:
        n_epi_with_occ += 1
        occ_per_epi.append(len(occ_idx))
    for rank, n in enumerate(occ_idx):
        # next front event after this OCC
        if n + 1 < len(events):
            sq = events[n + 1][1]
        else:
            sq = 'NONE' if reason == 'contact_b' else 'LOST'
        occ_sequel[sq] += 1
        if rank == 0:
            first_sequel[sq] += 1

print(f'episodes (physical acts): {n_epi}')
print(f'episodes with >=1 occurrence: {n_epi_with_occ} ({n_epi_with_occ/n_epi:.3f})')
if occ_per_epi:
    print(f'occurrences per episode (when >=1) p50/p90/max: '
          f'{np.percentile(occ_per_epi,[50,90])}/{max(occ_per_epi)}')
tot = sum(occ_sequel.values())
print(f'\nOCCURRENCE-LEVEL next-front-event sequel (n={tot}, sequence history):')
for k in ('HELD', 'FALLBACK', 'OTHER', 'NONE', 'LOST'):
    print(f'   {k:9s} {occ_sequel[k]/tot:.3f}  ({occ_sequel[k]})')
ft = sum(first_sequel.values())
print(f'\nFIRST-OCCURRENCE-per-episode sequel (n={ft}, independent unit):')
for k in ('HELD', 'FALLBACK', 'OTHER', 'NONE', 'LOST'):
    print(f'   {k:9s} {first_sequel[k]/ft:.3f}  ({first_sequel[k]})')
print(f'\nresume = HELD+FALLBACK+OTHER (some further front event): '
      f'occ {(occ_sequel["HELD"]+occ_sequel["FALLBACK"]+occ_sequel["OTHER"])/tot:.3f}, '
      f'first {(first_sequel["HELD"]+first_sequel["FALLBACK"]+first_sequel["OTHER"])/ft:.3f}')
