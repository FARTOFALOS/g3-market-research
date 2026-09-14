"""077: materialise the rich prefix record behind the four H aggregates of 076.

Extraction logic is copied verbatim from base/076/support_probe.py (describe /
extract), parametrised only by instrument and native timeframe. Nothing about
the interval, the orientation or the readings is changed. Fidelity is asserted
against the frozen numbers of base/076/support_probe_result.md.

Read-only on the field. No Y. No Volume.
"""
from pathlib import Path
from collections import Counter
import json
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / '_states'
MINUTE = 60_000_000_000


def describe(bars, width, side, kind):
    o, h, l, c = bars.T
    e = (l <= 0) & (h >= 0)
    f = (l <= -width) & (h >= -width)
    location = np.select([c < -width, c == -width, c < 0, c == 0],
                         [-2, -1, 0, 1], default=2).astype(int)
    readings = [(int(a), int(b), int(d)) for a, b, d in zip(e, f, location)]
    runs = [(i, v) for i, v in enumerate(readings) if i == 0 or v != readings[i - 1]]
    g0 = (side, kind, float(width), float(o[0]), float(l.min()), float(h.max()),
          *[float(v) for v in bars[-1]], len(bars))
    counts = (int(e.sum()), int(f.sum()), *[int((location == k).sum()) for k in [-2, -1, 0, 1, 2]])
    return dict(g0=g0, g1=g0 + counts, h_order=tuple(v for _, v in runs),
                h_timed=tuple(runs), readings=readings, counts=counts)


def load(inst):
    m = ROOT / 'data/market' / inst
    names = ['open', 'high', 'low', 'close', 'close_ts_utc_ns', 'session_id']
    arrays = {n: np.load(m / (n + '.npy'), mmap_mode='r') for n in names}
    with np.load(m / 'sessions.npz') as z:
        sessions = {n: z[n].copy() for n in z.files}
    lookup = {int(v): i for i, v in enumerate(sessions['session_id'])}
    return arrays, sessions, lookup


def stamp(ns):
    return str(np.datetime64(int(ns), 'ns'))[:16]


def extract(row, A, sessions, lookup, tf):
    t0 = int(row['t0_spine_pos']); p = t0 + 2
    if p >= len(A['close']):
        return None, 'cursor_not_observed'
    si = lookup[int(A['session_id'][t0])]
    anchor = int(sessions['session_open_utc_ns'][si])
    t = int(A['close_ts_utc_ns'][t0])
    key = ((t - anchor) // MINUTE - 1) // tf
    begin_ns = anchor + key * tf * MINUTE
    start = int(np.searchsorted(A['close_ts_utc_ns'][:p + 1], begin_ns, side='right'))
    ts = np.asarray(A['close_ts_utc_ns'][start:p + 1])
    if int(ts[0]) != begin_ns + MINUTE or np.any(np.diff(ts) != MINUTE):
        return None, 'native_prefix_gap'
    raw = np.column_stack([A[n][start:p + 1] for n in ['open', 'high', 'low', 'close']])
    if (not np.isfinite(raw).all() or np.any(raw[:, 1] < raw[:, 2])
            or np.any(raw[:, 0] > raw[:, 1]) or np.any(raw[:, 0] < raw[:, 2])
            or np.any(raw[:, 3] > raw[:, 1]) or np.any(raw[:, 3] < raw[:, 2])):
        return None, 'invalid_ohlc'
    side = row['t0_exit_side']
    exit_price = row['zone_top'] if side == 'north' else row['zone_bottom']
    width = row['zone_top'] - row['zone_bottom']
    if width <= 0:
        return None, 'invalid_zone'
    bars = raw - exit_price if side == 'north' else (exit_price - raw)[:, [0, 2, 1, 3]]
    st = describe(bars, width, side, row['t0_kind'])
    st.update(riz_id=row['riz_id'], scene='%s:%d:%d' % (row['_inst'], t0, p), t0=t0, p=p,
              start=start, p_utc=stamp(ts[-1]), t0_utc=stamp(A['close_ts_utc_ns'][t0]),
              session_id=int(A['session_id'][p]), zone=[row['zone_bottom'], row['zone_top']],
              raw_ohlc=raw.tolist(), utc_minutes=[stamp(x) for x in ts])
    return st, None


def prior_u(A, start, k=60):
    a = start - k
    if a < 0:
        return None
    ts = np.asarray(A['close_ts_utc_ns'][a:start])
    if len(ts) < k or np.any(np.diff(ts) != MINUTE):
        return None
    h = np.asarray(A['high'][a:start], float)
    l = np.asarray(A['low'][a:start], float)
    if not np.isfinite(h).all() or not np.isfinite(l).all():
        return None
    v = float(np.median(h - l))
    return v if v > 0 else None


def build(inst, tf):
    A, sessions, lookup = load(inst)
    cell = ROOT / ('data/field/%s/cells/tf_%04d' % (inst, tf))
    cols = ['riz_id', 'corpus_id', 'zone_bottom', 'zone_top', 't0_spine_pos', 't0_ts_ns',
            't0_exit_side', 't0_kind', 't0_span_count']
    rows = pq.read_table(cell / 'passports.parquet', columns=cols).to_pylist()
    rows.sort(key=lambda r: (r['t0_spine_pos'], r['riz_id']))
    for r in rows:
        r['_inst'] = inst
    states, missing = [], []
    for row in rows:
        st, reason = extract(row, A, sessions, lookup, tf)
        if reason:
            missing.append(dict(riz_id=row['riz_id'], reason=reason)); continue
        st['u60'] = prior_u(A, st['start'], 60)
        st['u30'] = prior_u(A, st['start'], 30)
        st['u120'] = prior_u(A, st['start'], 120)
        st['inst'] = inst; st['tf'] = tf
        states.append(st)
    return rows, states, missing


if __name__ == '__main__':
    import sys
    OUT.mkdir(exist_ok=True)
    todo = [('NQ', 54), ('NQ', 30), ('ES', 54), ('YM', 54)]
    if len(sys.argv) > 1:
        todo = [(sys.argv[1], int(sys.argv[2]))]
    for inst, tf in todo:
        rows, states, missing = build(inst, tf)
        print('%s tf%-3d  passports %4d  usable %4d  distinct h_order %4d  distinct h_timed %4d  scenes %4d  days %4d  %s'
              % (inst, tf, len(rows), len(states),
                 len({s['h_order'] for s in states}), len({s['h_timed'] for s in states}),
                 len({s['scene'] for s in states}), len({s['session_id'] for s in states}),
                 dict(Counter(x['reason'] for x in missing))))
        for s in states:
            s['h_order'] = [list(v) for v in s['h_order']]
            s['h_timed'] = [[i, list(v)] for i, v in s['h_timed']]
            s['g0'] = list(s['g0']); s['g1'] = list(s['g1']); s['counts'] = list(s['counts'])
            s['readings'] = [list(v) for v in s['readings']]
        (OUT / ('states_%s_tf%d.json' % (inst, tf))).write_text(
            json.dumps(states, ensure_ascii=False), encoding='utf-8')
