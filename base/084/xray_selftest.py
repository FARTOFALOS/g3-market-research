#!/usr/bin/env python3
"""Synthetic scenes for the event operator of xray.py. Exact expected bars, both sides."""
from __future__ import annotations
import numpy as np
from xray import _scan, EVENTS, KH

# north film A, boundary e = 100; rows: (high, low, close), first row is T0
A = [(102.00, 99.50, 101.50),   # T0 (ordinal 0, contact not evaluated)
     (103.00, 101.00, 102.50),  # +1 update, P=+1, L[P]=101
     (102.75, 101.50, 102.00),  # +2 no update: pause1, leg_pause
     (102.50, 100.75, 100.90),  # +3 no update (run 2): pause2; L 100.75<101 wick_break; C 100.90<101 close_break
     (103.50, 102.00, 103.25),  # +4 update after pause: resume
     (103.25, 102.50, 103.00),  # +5 run 1
     (103.00, 102.25, 102.50),  # +6 run 2
     (103.40, 102.30, 102.40),  # +7 run 3: pause3
     (101.00, 99.75, 100.50)]   # +8 contact
A_EXP = dict(pause1=2, pause2=3, pause3=7, leg_pause=2, resume=4, wick_break=3, close_break=3)
# film C: tie on a flat bar, then an uncontacted crossing
C = [(101.00, 99.00, 100.50), (101.00, 101.00, 101.00), (99.50, 99.25, 99.25)]
C_EXP = dict(pause1=1)


def run_case(bars, e, north, gap_at=None, w=50):
    h = np.array([b[0] for b in bars]); lo = np.array([b[1] for b in bars]); c = np.array([b[2] for b in bars])
    kind = np.zeros(len(bars), np.int8)
    if gap_at is not None:
        kind[gap_at] = 2
    n, ne = 1, len(EVENTS)
    end_code = np.full(n, -1, np.int8); end_k = np.zeros(n, np.int32)
    r = np.full((n, ne), -1, np.int32); fl = np.zeros((n, ne), np.bool_); ti = np.zeros((n, ne), np.bool_)
    dc = np.full((n, ne), np.nan, np.float32)
    risk = np.zeros((1, ne, KH), np.int64); hits = np.zeros((1, ne, KH), np.int64)
    alive = np.zeros((1, KH), np.int64); ended = np.zeros((1, 2, KH), np.int64)
    _scan(np.array([0], np.int64), np.array([e]), np.array([north]), np.zeros(1, np.int8),
          h, lo, c, kind, len(bars) - 1, KH, w, end_code, end_k, r, fl, ti, dc, risk, hits, alive, ended)
    return dict(end=int(end_code[0]), end_k=int(end_k[0]),
                r={EVENTS[x]: int(r[0, x]) for x in range(ne)},
                flat={EVENTS[x]: bool(fl[0, x]) for x in range(ne)},
                tie={EVENTS[x]: bool(ti[0, x]) for x in range(ne)},
                risk={EVENTS[x]: [int(k) for k in np.flatnonzero(risk[0, x])] for x in range(ne)})


def mirror(bars, axis=200.0):
    return [(axis - l, axis - h, axis - c) for h, l, c in bars]


def check(res, exp, end, end_k, label):
    for name in EVENTS:
        assert res['r'][name] == exp.get(name, -1), (label, name, res['r'][name], exp.get(name, -1))
    assert (res['end'], res['end_k']) == (end, end_k), (label, res['end'], res['end_k'])


a = run_case(A, 100.0, True)
check(a, A_EXP, 0, 8, 'A north')
assert a['risk']['pause2'] == [3] and a['risk']['pause3'] == [4, 7]
assert a['risk']['resume'] == [3, 4] and a['risk']['leg_pause'] == [2]
assert a['risk']['pause1'] == [1, 2] and a['risk']['wick_break'] == [1, 2, 3]
b = run_case(mirror(A), 100.0, False)
check(b, A_EXP, 0, 8, 'A mirrored south')
assert b['risk'] == a['risk']
c = run_case(C, 100.0, True)
check(c, C_EXP, 1, 2, 'C tie+flat+crossing')
assert c['flat']['pause1'] and c['tie']['pause1']
d = run_case(A, 100.0, True, gap_at=3)
check(d, dict(pause1=2, leg_pause=2), 2, 3, 'A with gap before +3')
assert d['risk']['pause2'] == []            # bar +3 is behind a gap: not at risk
w = run_case(A, 100.0, True, w=3)          # window: nothing after bar +3 is read
check(w, dict(pause1=2, leg_pause=2, pause2=3, wick_break=3, close_break=3), 4, 3, 'A window 3')
assert w['risk']['pause3'] == [] and w['risk']['resume'] == [3]
print('selftest: 5 synthetic films, all event bars, ends, window and risk sets as expected')
