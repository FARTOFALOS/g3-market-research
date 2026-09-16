#!/usr/bin/env python3
"""084 — race after q_event: own boundary b against the mirror m = 2c - b (PRE_FREEZE_084 §3-§4).

One walker, two modes.
    full   own_first / mirror_first / same_bar / gap_own / gap_mirror / lost_observability / archive_edge
    blind  contact_one_level / same_bar / gap_transition / lost_observability / archive_edge
In blind mode own and mirror are pooled at computation time and the side of a gap transition
is not written: the blind census of PRE_FREEZE_084 §9 cannot see direction.

Per bar j = q+1, ... without horizon:
    j > last -> archive_edge;  gap before j -> lost_observability;
    bar contains b and m -> same_bar;  only b -> own_first;  only m -> mirror_first;
    bar entirely beyond b or beyond m -> gap transition;  else next bar.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit, prange

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(os.environ.get('G3_084_OUT', str(ROOT / 'work/084')))
P081 = ROOT / 'work/081a/paths'
sys.path.insert(0, str(ROOT / 'base/080'))
sys.path.insert(0, str(HERE))

FULL = {0: 'own_first', 1: 'mirror_first', 2: 'same_bar', 3: 'gap_own', 4: 'gap_mirror',
        5: 'lost_observability', 6: 'archive_edge'}
BLIND = {10: 'contact_one_level', 2: 'same_bar', 13: 'gap_transition',
         5: 'lost_observability', 6: 'archive_edge'}
EPOCH_LAB = {'discovery': ('2006-2009', '2010-2013', '2014-2018'),
             'evaluation': ('2019-2021', '2022-2024', '2025-2026')}


@njit(parallel=True, cache=True)
def _walk(qs, bs, north, high, low, close, kind, last, blind, code, kbar):
    for i in prange(qs.size):
        q = qs[i]; b = bs[i]; up = north[i]
        c = close[q]
        m = 2.0 * c - b
        j = q
        while True:
            j += 1
            if j > last:
                code[i] = 6
                break
            if kind[j] != 0:
                code[i] = 5
                break
            hb = low[j] <= b and b <= high[j]
            hm = low[j] <= m and m <= high[j]
            if hb and hm:
                code[i] = 2
                break
            if hb or hm:
                if blind:
                    code[i] = 10
                else:
                    code[i] = 0 if hb else 1
                break
            if up:
                gb = high[j] < b
                gm = low[j] > m
            else:
                gb = low[j] > b
                gm = high[j] < m
            if gb or gm:
                if blind:
                    code[i] = 13
                else:
                    code[i] = 3 if gb else 4
                break
        kbar[i] = j - q


def walk(qs, bs, north, market, kind, blind):
    high, low, close = market
    n = qs.size
    code = np.full(n, -1, np.int8); kbar = np.zeros(n, np.int64)
    d = np.where(north, close[qs] - bs, bs - close[qs])
    assert (d > 0).all(), 'population with d <= 0 at q'
    _walk(qs.astype(np.int64), bs.astype(np.float64), north, high, low, close, kind, close.size - 1,
          blind, code, kbar)
    assert (code >= 0).all()
    return code, kbar


def populations(inst, terr):
    """q_event (close_break) and q = 5 populations from the frozen X-ray scan."""
    x = pd.read_parquet(OUT / f'xray_{inst}_{terr}.parquet',
                        columns=['riz_id', 'side', 'tf_minutes', 't0_pos', 't0_ts_ns', 't0_day', 'epoch',
                                 'end_k', 'r_close_break'])
    f = pd.read_parquet(P081 / f'films_{inst}_{terr}.parquet', columns=['riz_id', 'exit_boundary'])
    x = x.merge(f, on='riz_id', how='left', validate='one_to_one')
    ev = x[x.r_close_break >= 0].copy(); ev['q'] = ev.t0_pos + ev.r_close_break
    q5 = x[x.end_k > 5].copy(); q5['q'] = q5.t0_pos + 5
    return {'close_break': ev.reset_index(drop=True), 'q5': q5.reset_index(drop=True)}


def load_market(inst):
    from tape import presence_grid, gap_kinds                              # noqa: E402
    m = ROOT / 'data/market' / inst
    market = tuple(np.load(m / f) for f in ('high.npy', 'low.npy', 'close.npy'))
    grid, lo_, hi_ = presence_grid()
    kind, _ = gap_kinds(inst, grid, lo_, hi_)
    return market, kind


def census_blind(inst='NQ', terr='discovery'):
    """PRE_FREEZE_084 §9: terminal type shares after q, direction-blind."""
    market, kind = load_market(inst)
    res = {'instrument': inst, 'territory': terr, 'mode': 'blind', 'anchors': {}}
    for name, p in populations(inst, terr).items():
        code, _ = walk(p.q.to_numpy(), p.exit_boundary.to_numpy(), (p.side == 'north').to_numpy(),
                       market, kind, blind=True)
        assert set(np.unique(code)) <= set(BLIND), 'blind walker produced a directional code'
        A = {'N': int(len(p)), 'all': {BLIND[c]: round(float((code == c).mean()), 5) for c in BLIND}}
        A['counts'] = {BLIND[c]: int((code == c).sum()) for c in BLIND}
        for g, lab in enumerate(EPOCH_LAB[terr]):
            s = (p.epoch == g).to_numpy()
            A[lab] = {'N': int(s.sum()), **{BLIND[c]: round(float((code[s] == c).mean()), 5) for c in BLIND}}
        for side in ('north', 'south'):
            s = (p.side == side).to_numpy()
            A[side] = {'N': int(s.sum()), **{BLIND[c]: round(float((code[s] == c).mean()), 5) for c in BLIND}}
        res['anchors'][name] = A
    p = OUT / f'census_blind_{inst}_{terr}.json'
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    for name, A in res['anchors'].items():
        print(f"{name}: N {A['N']}  counts {A['counts']}")
        for key in ('all',) + EPOCH_LAB[terr] + ('north', 'south'):
            v = A[key]
            print(f"   {key:>10}: contact one level {v['contact_one_level']:.4f}  same_bar {v['same_bar']:.4f}  "
                  f"gap_transition {v['gap_transition']:.5f}  lost {v['lost_observability']:.4f}  "
                  f"edge {v['archive_edge']:.5f}")
    print(p)


def selftest():
    """Synthetic bars: every terminal, both sides (south = mirror image), both modes."""
    def case(rows, kind_at=None, north=True):
        h = np.array([r[0] for r in rows]); lo = np.array([r[1] for r in rows]); c = np.array([r[2] for r in rows])
        if not north:
            h, lo, c = 200.0 - lo, 200.0 - h, 200.0 - c
        kind = np.zeros(len(rows), np.int8)
        if kind_at is not None:
            kind[kind_at] = 2
        b = 100.0
        full = walk(np.array([0]), np.array([b]), np.array([north]), (h, lo, c), kind, blind=False)
        bl = walk(np.array([0]), np.array([b]), np.array([north]), (h, lo, c), kind, blind=True)
        return int(full[0][0]), int(full[1][0]), int(bl[0][0])
    q = (101.25, 100.75, 101.00)                  # bar q: close 101 -> d 1, m 102 (north)
    neutral = (101.50, 100.50, 101.00)
    cases = {
        'own_first': ([q, neutral, (100.50, 99.75, 100.00)], None, 0, 2, 10),
        'mirror_first': ([q, (102.25, 101.50, 102.00)], None, 1, 1, 10),
        'same_bar': ([q, (102.25, 99.75, 101.00)], None, 2, 1, 2),
        'gap_own': ([q, (99.75, 99.25, 99.50)], None, 3, 1, 13),
        'gap_mirror': ([q, (102.75, 102.25, 102.50)], None, 4, 1, 13),
        'lost_observability': ([q, neutral, (100.50, 99.75, 100.00)], 2, 5, 2, 5),
        'archive_edge': ([q, neutral], None, 6, 2, 6),
    }
    for label, (rows, kat, want_full, want_k, want_blind) in cases.items():
        for north in (True, False):
            got = case(rows, kat, north)
            assert got == (want_full, want_k, want_blind), (label, north, got)
    print('race selftest: 7 terminals x 2 sides x 2 modes as expected')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'selftest'
    if cmd == 'selftest':
        selftest()
    elif cmd == 'census':
        census_blind(*(sys.argv[2:4] or ['NQ', 'discovery']))
    else:
        raise SystemExit('usage: race084.py selftest | census INST TERR')
