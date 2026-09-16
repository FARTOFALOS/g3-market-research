#!/usr/bin/env python3
"""084 — Event X-Ray: when does a live Film-1 first show a recognizable change of state.

OUTCOME-FREE
============
Nothing after the recognition bar of an event is read here. Film-1 end
(own-boundary contact), an uncontacted crossing and a tape gap are used only as
the end of the risk set BEFORE an event: an event is recorded only while the
scene is still alive. Time from an event to the later end of the film is never
aggregated. The own-vs-mirror race is not computed.

COORDINATES
===========
Outward = away from the own exit boundary (north: up, south: down).

    H[j] = high (north) | -low (south)      outward extreme of bar j
    L[j] = low  (north) | -high (south)     inward extreme of bar j
    C[j] = close (north) | -close (south)
    E    = e (north) | -e (south)

Ordinal language of `research/ordinal_events.py`: a strict update of the
running extreme, first passage of a level by wick and by close SEPARATELY,
levels addressed by a bar of the film, never by a number.

OBSERVATION WINDOW OF THIS X-RAY: T0 ... T0+50
==============================================
Declared by the architect (2026-09-16) for this map, as in 070-073. Events are
searched on bars +1 ... +50 only. A film still alive after bar +50 without an
event is `window_censored_50` (not recognized within 50): not "no event", not
the end of Film-1, not a market horizon. The window does NOT carry over to the
later own-vs-mirror race. `--unbounded-debug` reruns without the window into
work/084/debug_unbounded/ as a technical artifact; it takes no part in selection.

PER BAR k = 1, 2, ..., 50 AFTER T0, in this order
=================================================
    k > 50                          -> end window_censored_50
    no bar                          -> end archive_edge   (not at risk at k)
    tape gap before the bar         -> end gap            (not at risk at k)
    bar range contains e            -> end contact        (at risk, no event)
    bar entirely beyond e: H < E    -> end crossing       (at risk, no event)
    otherwise the bar is evaluated.

Running outward extreme M starts at H[t0]; P is the bar holding it (first bar
reaching the current value). A bar UPDATES iff H[j] > M (strict, as in
`last_update_event`).

EVENTS (first occurrence, recognized at the close of bar k)
===========================================================
    pause1       first bar without update
    pause2       first second consecutive bar without update
    pause3       first third consecutive bar without update
    leg_pause    first bar without update after at least one update in (t0, k)
    resume       first update after at least one bar without update
    wick_break   first bar without update with L[j] < L[P]
    close_break  first bar without update with C[j] < L[P]

A bar that updates is never a break: its internal order is not observed.

RISK SET (state in which the event can occur for the first time at bar k)
=========================================================================
bar k observed, event not yet recorded, and
    pause2: bar k-1 was without update;  pause3: bars k-2, k-1 without update;
    leg_pause: an update in (t0, k);      resume: a bar without update in (t0, k).
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(os.environ.get('G3_084_OUT', str(ROOT / 'work/084')))
P081 = Path(os.environ.get('G3_081A_OUT', str(ROOT / 'work/081a'))) / 'paths'
sys.path.insert(0, str(ROOT / 'base/080'))
from tape import presence_grid, gap_kinds                                # noqa: E402

EVENTS = ('pause1', 'pause2', 'pause3', 'leg_pause', 'resume', 'wick_break', 'close_break')
K_MIN = {'pause1': 1, 'pause2': 2, 'pause3': 3, 'leg_pause': 2, 'resume': 2,
         'wick_break': 1, 'close_break': 1}
ENDS = ('contact', 'crossing', 'gap', 'archive_edge', 'window_censored_50')
W = 50                          # observation window of this X-ray, bars after T0
KH = W + 1                      # histogram index = bar k (0 unused)
KH_DEBUG = 400                  # unbounded debug run only; last bin collects k >= 399
EPOCHS = {                      # declared before the first number of this map
    'discovery': (('2006-2009', '2010-01-01'), ('2010-2013', '2014-01-01'), ('2014-2018', None)),
    'evaluation': (('2019-2021', '2022-01-01'), ('2022-2024', '2025-01-01'), ('2025-2026', None)),
}


@njit(cache=True)
def _scan(t0s, es, north, grp, high, low, close, kind, last, kh, w,
          end_code, end_k, r_ev, flat_ev, tie_ev, dc_ev,
          risk, hits, alive, ended):
    for i in range(t0s.size):
        t0 = t0s[i]; e = es[i]; up = north[i]; g = grp[i]
        E = e if up else -e
        M = high[t0] if up else -low[t0]
        P = t0
        run = 0
        upd_after = False
        paused = False
        k = 0
        while True:
            k += 1
            if w > 0 and k > w:
                end_code[i] = 4; end_k[i] = w
                break
            kk = k if k < kh else kh - 1
            j = t0 + k
            if j > last:
                end_code[i] = 3; end_k[i] = k
                break
            if kind[j] != 0:
                end_code[i] = 2; end_k[i] = k
                break
            alive[g, kk] += 1
            if r_ev[i, 0] < 0:
                risk[g, 0, kk] += 1
            if r_ev[i, 1] < 0 and run >= 1:
                risk[g, 1, kk] += 1
            if r_ev[i, 2] < 0 and run >= 2:
                risk[g, 2, kk] += 1
            if r_ev[i, 3] < 0 and upd_after:
                risk[g, 3, kk] += 1
            if r_ev[i, 4] < 0 and paused:
                risk[g, 4, kk] += 1
            if r_ev[i, 5] < 0:
                risk[g, 5, kk] += 1
            if r_ev[i, 6] < 0:
                risk[g, 6, kk] += 1
            hi = high[j]; lo = low[j]
            if lo <= e and e <= hi:
                end_code[i] = 0; end_k[i] = k; ended[g, 0, kk] += 1
                break
            Hj = hi if up else -lo
            if Hj < E:
                end_code[i] = 1; end_k[i] = k; ended[g, 1, kk] += 1
                break
            Lj = lo if up else -hi
            Cj = close[j] if up else -close[j]
            dc = (close[j] - e) if up else (e - close[j])
            flat = hi == lo
            if Hj > M:
                if paused and r_ev[i, 4] < 0:
                    r_ev[i, 4] = k; flat_ev[i, 4] = flat; dc_ev[i, 4] = dc
                    hits[g, 4, kk] += 1
                M = Hj; P = j; run = 0; upd_after = True
            else:
                tie = Hj == M
                run += 1
                if r_ev[i, 0] < 0:
                    r_ev[i, 0] = k; flat_ev[i, 0] = flat; tie_ev[i, 0] = tie; dc_ev[i, 0] = dc
                    hits[g, 0, kk] += 1
                if run >= 2 and r_ev[i, 1] < 0:
                    r_ev[i, 1] = k; flat_ev[i, 1] = flat; tie_ev[i, 1] = tie; dc_ev[i, 1] = dc
                    hits[g, 1, kk] += 1
                if run >= 3 and r_ev[i, 2] < 0:
                    r_ev[i, 2] = k; flat_ev[i, 2] = flat; tie_ev[i, 2] = tie; dc_ev[i, 2] = dc
                    hits[g, 2, kk] += 1
                if upd_after and r_ev[i, 3] < 0:
                    r_ev[i, 3] = k; flat_ev[i, 3] = flat; tie_ev[i, 3] = tie; dc_ev[i, 3] = dc
                    hits[g, 3, kk] += 1
                LP = low[P] if up else -high[P]
                if Lj < LP and r_ev[i, 5] < 0:
                    r_ev[i, 5] = k; flat_ev[i, 5] = flat; dc_ev[i, 5] = dc
                    hits[g, 5, kk] += 1
                if Cj < LP and r_ev[i, 6] < 0:
                    r_ev[i, 6] = k; flat_ev[i, 6] = flat; dc_ev[i, 6] = dc
                    hits[g, 6, kk] += 1
                paused = True


def epoch_index(ts_ns, terr):
    edges = [pd.Timestamp(b, tz='UTC').value for _, b in EPOCHS[terr] if b is not None]
    return np.searchsorted(np.array(edges, dtype=np.int64), ts_ns, side='right').astype(np.int8)


def build(inst='NQ', terr='discovery', grid=None, lo_=None, hi_=None, window=W):
    m = ROOT / 'data/market' / inst
    high = np.load(m / 'high.npy'); low = np.load(m / 'low.npy'); close = np.load(m / 'close.npy')
    last = close.size - 1
    kind, _ = gap_kinds(inst, grid, lo_, hi_)
    f = pd.read_parquet(P081 / f'films_{inst}_{terr}.parquet',
                        columns=['riz_id', 'side', 'tf_minutes', 'exit_boundary',
                                 't0_spine_pos', 't0_ts_ns', 't0_day'])
    n = len(f)
    t0 = f.t0_spine_pos.to_numpy().astype(np.int64)
    e = f.exit_boundary.to_numpy().astype(np.float64)
    north = f.side.to_numpy() == 'north'
    grp = epoch_index(f.t0_ts_ns.to_numpy(), terr)
    ng, ne = len(EPOCHS[terr]), len(EVENTS)

    end_code = np.full(n, -1, np.int8); end_k = np.zeros(n, np.int32)
    r_ev = np.full((n, ne), -1, np.int32)
    flat_ev = np.zeros((n, ne), np.bool_); tie_ev = np.zeros((n, ne), np.bool_)
    dc_ev = np.full((n, ne), np.nan, np.float32)
    kh = KH if window > 0 else KH_DEBUG
    risk = np.zeros((ng, ne, kh), np.int64); hits = np.zeros((ng, ne, kh), np.int64)
    alive = np.zeros((ng, kh), np.int64); ended = np.zeros((ng, 2, kh), np.int64)
    t = time.time()
    _scan(t0, e, north, grp, high, low, close, kind, last, kh, window,
          end_code, end_k, r_ev, flat_ev, tie_ev, dc_ev, risk, hits, alive, ended)
    secs = round(time.time() - t, 1)
    assert (end_code >= 0).all()
    # hits never exceed the risk set of the same bar
    assert (hits <= risk).all()
    if window > 0:
        assert (r_ev <= window).all() and (end_k <= window).all()

    d = pd.DataFrame({'riz_id': f.riz_id, 'side': f.side, 'tf_minutes': f.tf_minutes,
                      't0_pos': t0, 't0_ts_ns': f.t0_ts_ns, 't0_day': f.t0_day,
                      'epoch': grp, 'end_code': end_code, 'end_k': end_k})
    for x, name in enumerate(EVENTS):
        d[f'r_{name}'] = r_ev[:, x]
        d[f'flat_{name}'] = flat_ev[:, x]
        d[f'tie_{name}'] = tie_ev[:, x]
        d[f'dclose_{name}'] = dc_ev[:, x]
    hist = {'instrument': inst, 'territory': terr, 'KH': kh, 'window': window, 'events': EVENTS,
            'epochs': [a for a, _ in EPOCHS[terr]], 'seconds': secs,
            'risk': risk.tolist(), 'hits': hits.tolist(), 'alive': alive.tolist(),
            'ended_contact': ended[:, 0].tolist(), 'ended_crossing': ended[:, 1].tolist()}
    return d, hist


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    debug = '--unbounded-debug' in sys.argv
    out = OUT / 'debug_unbounded' if debug else OUT
    tag = '_unbounded_debug' if debug else ''
    out.mkdir(parents=True, exist_ok=True)
    grid, lo_, hi_ = presence_grid()
    for inst, terr in [a.split(':') for a in (args or ['NQ:discovery'])]:
        d, hist = build(inst, terr, grid, lo_, hi_, window=0 if debug else W)
        d.to_parquet(out / f'xray{tag}_{inst}_{terr}.parquet', index=False, compression='zstd')
        (out / f'xray_hist{tag}_{inst}_{terr}.json').write_text(json.dumps(hist), encoding='utf-8')
        print(f'{inst} {terr}: {len(d)} films, window {hist["window"] or "none (debug)"}, '
              f'scan {hist["seconds"]}s', flush=True)
        print('  end codes:', {ENDS[c]: int((d.end_code == c).sum()) for c in range(len(ENDS))})
        for name in EVENTS:
            print(f'  {name:<12} recognized {int((d[f"r_{name}"] >= 0).sum()):>7}')
