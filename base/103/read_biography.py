"""Complete the 12 declared examples with native C1/C2/C3 and indexed extrema.
Read-only market/field; no Volume. This is discovery, not a trading estimate.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
from g3riz.entry_check import native_grid

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'work/103/scenes'
MK = ROOT / 'data/market/NQ'
a = {k: np.load(MK / f'{k}.npy', mmap_mode='r') for k in
     ('open', 'high', 'low', 'close', 'close_ts_utc_ns')}
with np.load(MK / 'sessions.npz') as z:
    m = SimpleNamespace(close_ts_utc_ns=a['close_ts_utc_ns'], sessions=dict(z))
rows = []
for tf in (1, 5, 54, 240):
    starts, stops = native_grid(m, tf)
    for path in sorted(OUT.glob(f'*_tf{tf:04d}_*_address.json')):
        r = json.loads(path.read_text())
        stem = path.name.removesuffix('_address.json')
        ev = json.loads((OUT / f'{stem}_prefix_events.json').read_text())
        birth = next(e for e in ev if e['event_kind'] == 'precursor_formed')
        first = next(e for e in ev if e['event_kind'] == 'accepted_span')
        j = birth['native_bar_index']; q = r['t0_spine_pos']
        assert stops[j]-1 == birth['market_spine_pos']
        assert stops[first['native_bar_index']]-1 == first['market_spine_pos']
        assert starts[r['t0_native_bar_index']] <= q < stops[r['t0_native_bar_index']]
        c1 = int(starts[j-2]); c3 = int(starts[j])
        pos = np.arange(c1, q+1)
        d = pd.DataFrame({'spine_pos': pos, 'relative_to_t0': pos-q,
                          'close_utc': pd.to_datetime(a['close_ts_utc_ns'][pos], utc=True)})
        for name in ('open', 'high', 'low', 'close'):
            d[name] = a[name][pos]
        d.to_csv(OUT / f'{stem}_ancestry.csv', index=False)
        direction = 1 if r['t0_exit_side'] == 'north' else -1
        # The original departure: C3 opening through first accepted span close.
        stop = first['market_spine_pos'] + 1
        name = 'high' if direction > 0 else 'low'
        values = a[name][c3:stop]
        xp = c3 + int(np.argmax(values) if direction > 0 else np.argmin(values))
        target = float(a[name][xp])
        f = pd.read_csv(OUT / f'{stem}_continuation.csv')
        crossed = f[direction*(f['close']-target) >= 0]
        far = r['zone_bottom'] if direction > 0 else r['zone_top']
        failed = f[direction*(f['close']-far) < 0]
        row = dict(riz_id=r['riz_id'], tf=tf, t0=str(d.close_utc.iloc[-1]),
                   C1_start=c1, C3_start=c3, span1=first['market_spine_pos'],
                   original_extreme_pos=xp, original_extreme=target,
                   extreme_time=str(pd.Timestamp(int(a['close_ts_utc_ns'][xp]), tz='UTC')),
                   t0_close=float(a['close'][q]), remaining=direction*(target-a['close'][q]),
                   origin_aligned=(direction == (1 if r['bullish'] else -1)),
                   first_close_at_extreme=None if crossed.empty else crossed.iloc[0].close_utc,
                   first_close_through_far=None if failed.empty else failed.iloc[0].close_utc)
        rows.append(row)
(OUT / 'biography_reading.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
for row in rows:
    print(json.dumps(row))
