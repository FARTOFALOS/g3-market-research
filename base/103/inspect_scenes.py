"""Small discovery sample, not a population estimate or trading simulation.

Selection uses only TF and T0: first admission after October 1, 13:30 UTC,
in each of 2021/2023/2025, TF 1/5/54/240. Preserve full indexed OHLC from
formation through T0 separately from continuation to the end of its UTC day.
That endpoint is an observation budget, NOT a proposed trading horizon.
No Volume, final-lifecycle selection, 2026, precursor population, or field writes.
Run: python -B base/103/inspect_scenes.py
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'work/103/scenes'
OUT.mkdir(parents=True, exist_ok=True)
market = ROOT / 'data/market/NQ'
manifest = json.loads((market / 'manifest.json').read_text())
source = json.loads((ROOT / 'SOURCE_DATA.json').read_text())
assert manifest['corpus_id'] == source['instruments']['NQ']['corpus_id']
arrays = {k: np.load(market / (k + '.npy'), mmap_mode='r')
          for k in ('open', 'high', 'low', 'close', 'close_ts_utc_ns')}
ts = arrays['close_ts_utc_ns']
cols = ['riz_id', 'corpus_id', 'tf_minutes', 'zone_top', 'zone_bottom',
        'bullish', 'precursor_formed_spine_pos', 't0_spine_pos', 't0_ts_ns',
        't0_exit_side', 't0_native_bar_index']
summary = []
for tf in (1, 5, 54, 240):
    cell = ROOT / f'data/field/NQ/cells/tf_{tf:04d}'
    p = pq.read_table(cell / 'passports.parquet', columns=cols).to_pandas()
    assert set(p.corpus_id) == {manifest['corpus_id']}
    for year in (2021, 2023, 2025):
        start = pd.Timestamp(f'{year}-10-01 13:30', tz='UTC').value
        stop = pd.Timestamp(f'{year+1}-01-01', tz='UTC').value
        candidates = p[(p.t0_ts_ns >= start) & (p.t0_ts_ns < stop)]
        if candidates.empty:
            summary.append({'tf': tf, 'year': year, 'status': 'no_admission'})
            continue
        r = candidates.sort_values(['t0_ts_ns', 'riz_id']).iloc[0]
        q, birth = int(r.t0_spine_pos), int(r.precursor_formed_spine_pos)
        assert birth >= 0 and birth <= q
        end_ns = (pd.Timestamp(int(r.t0_ts_ns), tz='UTC').normalize()
                  + pd.Timedelta(days=1)).value
        end = int(np.searchsorted(ts, end_ns)) - 1
        stem = f'{year}_tf{tf:04d}_{r.riz_id}'
        def frame(lo, hi):
            pos = np.arange(lo, hi+1)
            d = pd.DataFrame({'spine_pos': pos, 'relative_to_t0': pos-q,
                              'close_utc': pd.to_datetime(ts[pos], utc=True)})
            for name in ('open', 'high', 'low', 'close'):
                d[name] = arrays[name][pos]
            return d
        pre, future = frame(birth, q), frame(q+1, end)
        pre.to_csv(OUT / (stem + '_prefix.csv'), index=False)
        future.to_csv(OUT / (stem + '_continuation.csv'), index=False)
        events = pq.read_table(cell / 'events.parquet',
                               filters=[('riz_id', '=', r.riz_id)]).to_pandas()
        events[events.market_spine_pos <= q].to_json(
            OUT / (stem + '_prefix_events.json'), orient='records', indent=2)
        address = r.to_dict()
        address.update(prefix_rows=len(pre), continuation_rows=len(future),
                       observation_end='UTC day budget; not a lifecycle terminal',
                       prefix_sha256=hashlib.sha256(pre.to_csv(index=False).encode()).hexdigest())
        (OUT / (stem + '_address.json')).write_text(
            json.dumps(address, indent=2, default=int), encoding='utf-8')
        summary.append(dict(address, first_prefix=pre.iloc[0].to_dict(),
                            last_prefix=pre.iloc[-1].to_dict()))
(OUT / 'index.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
print(json.dumps([{'riz_id': x.get('riz_id'), 'tf': x.get('tf_minutes', x.get('tf')),
                   'prefix_rows': x.get('prefix_rows'), 'continuation_rows': x.get('continuation_rows'),
                   't0': x.get('last_prefix', {}).get('close_utc')} for x in summary], default=str))
