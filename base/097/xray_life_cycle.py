#!/usr/bin/env python3
"""Line 097, X-ray without outcomes: is the ~17-native-bar latency market time or definition time?

Blue-2X is born and matures in native bars, and I then measured its age in the same bars. Under a
scale-free (self-similar) tape every quantity of a definition written in native bars is TF-invariant
by construction, provided the zone is also scale-free (width ~ native bar range). So the census asks:
  geometry   zone width / range of the native bar that closed the precursor (C3), by TF
  life       native bars: birth -> first accepted span -> T0 -> deletion; spans per life; x3 share
  definition share of lives ended by the machine's own spacing rule (too_early_respan) - a lower
             bound on latency that is written in native bars, i.e. mechanical
If everything, including the definitional part, is flat across TF, the clock belongs to
"definition x self-similar tape" and says nothing about RIZ as a market object.
NQ, T0 in 2021-2025. No price outcome is read.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
MK = ROOT / 'data/market/NQ'
H = np.load(MK / 'high.npy'); L = np.load(MK / 'low.npy')
TFS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 45, 60, 120, 240]

print('%4s %6s | %-19s | %-28s | %-22s | %-17s | %s' % (
    'TF', 'n', 'width / C3 bar range', 'birth->1st span / ->T0 (bars)', 'T0->deletion (bars)', 'spans per life', 'ended by too_early_respan | reached x3'))
for tf in TFS:
    cell = ROOT / f'data/field/NQ/cells/tf_{tf:04d}'
    p = pq.read_table(cell / 'passports.parquet', columns=['riz_id', 'zone_width', 'precursor_formed_spine_pos', 't0_ts_ns', 'final_span_count']).to_pandas()
    p = p[pd.to_datetime(p.t0_ts_ns).dt.year.between(2021, 2025)]
    e = pq.read_table(cell / 'events.parquet', columns=['riz_id', 'event_kind', 'native_bar_index', 'deletion_cause', 'event_seq']).to_pandas()
    e = e[e.riz_id.isin(p.riz_id)]
    nb = lambda kind, how: getattr(e[e.event_kind == kind].groupby('riz_id').native_bar_index, how)()
    b0, s1, t0, dl = nb('precursor_formed', 'min'), nb('accepted_span', 'min'), nb('t0', 'min'), nb('deleted', 'min')
    d = pd.DataFrame(dict(b0=b0, s1=s1, t0=t0, dl=dl)).join(p.set_index('riz_id')[['zone_width', 'precursor_formed_spine_pos', 'final_span_count']])
    pos = d.precursor_formed_spine_pos.to_numpy()
    rng = np.array([H[max(0, q - tf + 1):q + 1].max() - L[max(0, q - tf + 1):q + 1].min() for q in pos])
    geo = (d.zone_width / np.where(rng > 0, rng, np.nan))
    cause = e[e.event_kind == 'deleted'].set_index('riz_id').deletion_cause
    x3 = e[e.event_kind == 'x3_t0'].riz_id.nunique() / len(d)
    q3 = lambda s: ' / '.join('%.1f' % v for v in s.quantile([.25, .5, .75]))
    print('%4d %6d | %-19s | %-13s ; %-13s | %-22s | %-17s | %.3f | %.3f' % (
        tf, len(d), q3(geo), q3(d.s1 - d.b0), q3(d.t0 - d.b0), q3(d.dl - d.t0), q3(d.final_span_count),
        (cause == 'too_early_respan').mean(), x3))
