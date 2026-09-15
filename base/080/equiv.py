#!/usr/bin/env python3
"""Эквивалентность: быстрый примитив против эталона, ноль расхождений или стоп."""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference import spine, passports, exit_boundary, first_contact_reference
from contact import blocks, first_contacts, B


def check(inst, tf, sample=None, seed=3, adversarial=True):
    s = spine(inst)
    p = passports(inst, tf)
    e = exit_boundary(p)
    t0 = p.t0_spine_pos.to_numpy().astype(np.int64)
    blo, bhi = blocks(s['low'], s['high'])
    t = time.time()
    fast = first_contacts(s['low'], s['high'], blo, bhi, t0, e.astype(np.float64), B)
    fast_s = time.time() - t

    idx = np.arange(len(p))
    if sample is not None and sample < len(p):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(p), size=sample, replace=False)
        if adversarial:
            lag = np.where(fast < 0, 10**12, fast - t0)
            extra = np.concatenate([np.argsort(-np.where(fast < 0, -1, lag))[:60],
                                    np.flatnonzero(lag == 1)[:60],
                                    np.flatnonzero(fast < 0)[:60],
                                    np.flatnonzero((lag > 1) & (lag < 6))[:60]])
            idx = np.unique(np.concatenate([idx, extra]))
    t = time.time()
    ref = np.array([first_contact_reference(s['low'], s['high'], int(t0[i]), float(e[i]))
                    for i in idx], dtype=np.int64)
    ref_s = time.time() - t
    bad = np.flatnonzero(ref != fast[idx])
    return {'instrument': inst, 'tf': tf, 'riz': int(len(p)),
            'checked': int(idx.size), 'mismatches': int(bad.size),
            'fast_seconds': round(fast_s, 2), 'reference_seconds': round(ref_s, 1),
            'bad_riz': p.riz_id.to_numpy()[idx[bad]].tolist()[:10]}


if __name__ == '__main__':
    cases = [('NQ', 54, None), ('NQ', 1, 300), ('NQ', 5, 400), ('NQ', 240, None),
             ('NQ', 1440, None), ('ES', 54, 400), ('ES', 3, 300), ('YM', 120, 400),
             ('YM', 7, 300), ('ES', 1440, None)]
    total = 0
    for inst, tf, n in cases:
        r = check(inst, tf, n)
        total += r['mismatches']
        print(r, flush=True)
    print('TOTAL MISMATCHES:', total)
    raise SystemExit(1 if total else 0)
