"""S-04: fresh per-RIZ candle census and a finite family of timed actions.

Run from the repository root. Never writes to market/field. No volume input.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import platform
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numba import njit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'data/research/S-04'
M = 60_000_000_000
INSTRUMENTS = ['ES', 'NQ', 'YM']
POINT = {'ES': 50., 'NQ': 20., 'YM': 5.}
TICK = {'ES': .25, 'NQ': .25, 'YM': 1.}
COST = {'ES': 30., 'NQ': 15., 'YM': 15.}
STATES = {0: 'hold', 1: 'reclaim', 2: 'inside', 3: 'reverse'}
COLS = ['riz_id', 'tf_minutes', 't0_spine_pos', 't0_ts_ns', 't0_exit_side',
        'zone_top', 'zone_bottom', 'precursor_formed_spine_pos']


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def market(ins):
    return {k: np.load(ROOT / 'data/market' / ins / (k+'.npy'), mmap_mode='r')
            for k in ['close_ts_utc_ns', 'open', 'high', 'low', 'close']}


def calendar():
    import exchange_calendars as xc
    cal = xc.get_calendar('XNYS', start='2006-01-01', end='2026-05-04')
    s = cal.schedule[['open', 'close']].copy()
    s['date'] = s.index.strftime('%Y-%m-%d')
    s['open_ns'] = pd.DatetimeIndex(s['open']).as_unit('ns').asi8
    s['close_ns'] = pd.DatetimeIndex(s['close']).as_unit('ns').asi8
    s = s.reset_index(drop=True)
    for d in ['2007-01-02', '2012-10-29', '2012-10-30', '2018-12-05', '2025-01-09']:
        assert d not in set(s.date)
    assert '2021-06-18' in set(s.date) and '2022-06-20' not in set(s.date)
    early = s.loc[s.date == '2025-11-28'].iloc[0]
    assert pd.Timestamp(early.close_ns, tz='UTC').tz_convert('America/New_York').hour == 13
    dump(HERE/'calendar_source.json', {'package': xc.__version__, 'calendar': 'XNYS',
         'convention': 'NYSE cash close, including historical early closes',
         'reference': 'https://www.nyse.com/trade/hours-calendars',
         'exceptional_closure_checks': 5, 'source_timezone_uncertainty':
         'Canonical timezone follows SOURCE_DATA.json; original feed metadata unknown.'})
    s.to_parquet(HERE/'calendar.parquet', index=False)
    return s


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    cal = calendar()
    inputs = {'source_data': sha(ROOT/'SOURCE_DATA.json'), 'market': {}, 'field': {}}
    for ins in INSTRUMENTS:
        print('read frozen field', ins, flush=True)
        directory = ROOT/'data/field'/ins/'cells'
        files = []
        for tf in range(1, 1441):
            folder = directory/f'tf_{tf:04d}'
            manifest = json.loads((folder/'manifest.json').read_text())
            assert manifest['status'] == 'complete' and manifest['tf_minutes'] == tf
            p = folder/'passports.parquet'
            digest = sha(p)
            assert digest == manifest['output_sha256']['passports.parquet']
            inputs['field'][f'{ins}/{tf}'] = {'manifest': sha(folder/'manifest.json'), 'passports': digest}
            files.append(p)
        passports = pq.read_table(files, columns=COLS).to_pandas()
        assert passports.riz_id.is_unique
        passports.to_parquet(OUT/f'{ins}_objects.parquet', index=False)
        m = market(ins)
        inputs['market'][ins] = {k: sha(ROOT/'data/market'/ins/(k+'.npy')) for k in m}
        inputs['market'][ins]['manifest'] = sha(ROOT/'data/market'/ins/'manifest.json')
        ts = m['close_ts_utc_ns']
        rows = []
        for r in cal.itertuples():
            a = int(np.searchsorted(ts, r.open_ns + M))
            b = int(np.searchsorted(ts, r.close_ns))
            expected = (r.close_ns-r.open_ns)//M
            good = (a < len(ts) and b < len(ts) and ts[a] == r.open_ns+M
                    and ts[b] == r.close_ns and b-a+1 == expected
                    and np.all(np.diff(ts[a:b+1]) == M))
            rows.append({'date': r.date, 'known': bool(good), 'first': a, 'last': b,
                         'expected_minutes': expected})
        pd.DataFrame(rows).to_parquet(OUT/f'{ins}_coverage.parquet', index=False)
        print(ins, 'objects', len(passports), 'complete cash days', sum(x['known'] for x in rows), flush=True)
    dump(OUT/'input_hashes.json', inputs)
    print('prepare done', flush=True)


@njit(cache=False)
def execute(ts, op, hi, lo, cl, q, close_ns, direction, stop, horizon, use_stop, delay=0):
    n = len(q)
    answer = np.full((n, 10), np.nan)
    for i in range(n):
        b = q[i]+1+delay
        if b >= len(ts) or ts[b] != ts[q[i]]+(1+delay)*M:
            answer[i, 0] = -2; continue
        entry = op[b]
        d = direction[i]
        if use_stop and d*(entry-stop[i]) <= 0:
            answer[i, 0] = -1; continue
        end_ns = min(ts[q[i]]+(horizon+delay)*M, close_ns[i])
        adverse = 0.; favorable = 0.
        for j in range(b, len(ts)):
            expected = ts[q[i]]+(j-q[i])*M
            if ts[j] != expected or ts[j] > end_ns:
                answer[i, 0] = -3; break
            # An opening gap through the stop fills at the opening price.
            adverse = max(adverse, (entry-op[j])*d)
            if use_stop and (op[j]-stop[i])*d <= 0:
                answer[i] = np.array([1., b, j, entry, op[j], (op[j]-entry)*d,
                                      (ts[j]-M-ts[q[i]]-delay*M)/M, adverse, favorable, 1.])
                break
            adverse = max(adverse, (entry-(lo[j] if d == 1 else hi[j]))*d)
            favorable = max(favorable, ((hi[j] if d == 1 else lo[j])-entry)*d)
            if use_stop and ((lo[j] <= stop[i]) if d == 1 else (hi[j] >= stop[i])):
                # Full-bar heat can overstate pre-stop heat; explicitly labelled bound.
                answer[i] = np.array([1., b, j, entry, stop[i], (stop[i]-entry)*d,
                                      (ts[j]-ts[q[i]]-delay*M)/M, adverse, favorable, 0.])
                break
            if ts[j] == end_ns:
                answer[i] = np.array([2., b, j, entry, cl[j], (cl[j]-entry)*d,
                                      (ts[j]-ts[q[i]]-delay*M)/M, adverse, favorable, 0.])
                break
        if np.isnan(answer[i, 0]):
            answer[i, 0] = -3
    return answer


def stats(daily, trades):
    x = np.asarray(daily, dtype=float); x = x[np.isfinite(x)]
    eq = np.r_[0., np.cumsum(x)]
    known = trades.loc[trades.status>0]
    return dict(days=len(x), trades=len(known), total=float(x.sum()),
        mean_day=float(x.mean()) if len(x) else None,
        mean_trade=float(known.net_dollars.mean()) if len(known) else None,
        win_rate=float((known.net_dollars>0).mean()) if len(known) else None,
        max_drawdown=float(np.max(np.maximum.accumulate(eq)-eq)),
        median_minutes=float(known.minutes.median()) if len(known) else None)


if __name__ == '__main__':
    prepare()
