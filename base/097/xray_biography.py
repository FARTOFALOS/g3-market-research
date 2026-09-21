#!/usr/bin/env python3
"""Line 097, X-ray without outcomes after T0: what did the market already do for a zone to reach T0?

Inside the settled domain (2026-08-28): "a qualifying zone is the same geometric object before and after
T0, its own precursor history is retained and tape before T0 stays inspectable". Never-qualifying zones
are NOT touched here.
Picture to check with eyes before any question about the future: a RIZ is a completed ROUND TRIP -
  impulse leaves a gap (birth) -> a full body goes back through the gap (span 1) -> price lives beyond it
  -> a full body takes the gap again in the forming direction (span 2 = T0).
Read, in the zone's own units (w = zone width, r0 = range of the native bar C3):
  dir1   share of first spans that go AGAINST the forming direction
  B      how far the impulse ran past the zone before the violation, (extreme - near edge) / w
  C      how deep price went beyond the zone between span 1 and T0, / w   (what a reclaim-bettor endured)
  bars   native bars span1 -> T0
NQ, T0 in 2021-2025.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
MK = ROOT / 'data/market/NQ'
O = np.load(MK / 'open.npy'); H = np.load(MK / 'high.npy'); L = np.load(MK / 'low.npy'); C_ = np.load(MK / 'close.npy')

print('%4s %6s | %-9s | %-22s | %-22s | %-22s | %s' % ('TF', 'n', 'span1 vs', 'B: run past zone, /w', 'C: depth beyond, /w', 'C in r0 (C3 ranges)', 'bars span1->T0'))
for tf in (1, 2, 3, 5, 10, 15, 30, 60):
    cell = ROOT / f'data/field/NQ/cells/tf_{tf:04d}'
    p = pq.read_table(cell / 'passports.parquet', columns=['riz_id', 'zone_top', 'zone_bottom', 'bullish', 'precursor_formed_spine_pos', 't0_spine_pos', 't0_ts_ns']).to_pandas()
    p = p[pd.to_datetime(p.t0_ts_ns).dt.year.between(2021, 2025) & (p.precursor_formed_spine_pos > 0)].set_index('riz_id')
    e = pq.read_table(cell / 'events.parquet', columns=['riz_id', 'event_kind', 'market_spine_pos', 'native_bar_index', 'event_seq']).to_pandas()
    s = e[(e.event_kind == 'accepted_span') & e.riz_id.isin(p.index)].sort_values(['riz_id', 'event_seq']).groupby('riz_id').head(1).set_index('riz_id')
    t0nb = e[e.event_kind == 't0'].groupby('riz_id').native_bar_index.min()
    d = p.join(s[['market_spine_pos', 'native_bar_index']].rename(columns={'market_spine_pos': 's1', 'native_bar_index': 'nb1'}), how='inner').join(t0nb.rename('nbT'))
    d = d[(d.s1 > d.precursor_formed_spine_pos) & (d.t0_spine_pos > d.s1)]
    out = []
    for r in d.itertuples():
        b0, s1, t0 = int(r.precursor_formed_spine_pos), int(r.s1), int(r.t0_spine_pos); w = r.zone_top - r.zone_bottom
        if w <= 0:
            continue
        body = C_[s1] - O[s1 - tf + 1]; up = bool(r.bullish)
        against = (body < 0) if up else (body > 0)
        r0 = H[b0 - tf + 1:b0 + 1].max() - L[b0 - tf + 1:b0 + 1].min()
        B = (H[b0 - tf + 1:s1 + 1].max() - r.zone_top) if up else (r.zone_bottom - L[b0 - tf + 1:s1 + 1].min())
        Cd = (r.zone_bottom - L[s1 - tf + 1:t0 + 1].min()) if up else (H[s1 - tf + 1:t0 + 1].max() - r.zone_top)
        out.append((against, B / w, Cd / w, Cd / r0 if r0 > 0 else np.nan, r.nbT - r.nb1))
    x = pd.DataFrame(out, columns=['ag', 'B', 'C', 'Cr', 'bars'])
    q = lambda c: ' / '.join('%.1f' % v for v in x[c].quantile([.25, .5, .75]))
    print('%4d %6d | %.3f     | %-22s | %-22s | %-22s | %s' % (tf, len(x), x.ag.mean(), q('B'), q('C'), q('Cr'), q('bars')))
