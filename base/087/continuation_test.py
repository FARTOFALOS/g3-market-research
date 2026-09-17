#!/usr/bin/env python3
"""Continuation from a COMMON parent: live object-scene whose last outward-update was joint.
Branch = the NEXT outward update after that joint (or NONE if the scene terminates first).
  WOECB'  : next outward update is M-only + inward body (Co<Oo)
  Mout    : next outward update is M-only + outward/doji body
  joint   : next outward update is joint
  MBonly  : next outward update is MB-only
  NONE    : scene ends (b_contact/cross/obs/cap) before any further outward update
Then read the WHOLE further film per branch: terminal mix, bars-to-terminal, further outward
updates, reached-b, censoring. Object-aware detection (own terminal); evidence deduped to
physical episodes (parent bar, side). NOT collapsed to one probability. NQ discovery.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('C:/Users/Admin/Claude/g3-market-research')
sys.path.insert(0, str(ROOT / 'base/084'))
import race084
CAP = 400
market, kind = race084.load_market('NQ'); high, low, close = market
opn = np.load(ROOT / 'data/market/NQ/open.npy'); last = close.size - 1
films = pd.read_parquet(ROOT / 'work/081a/paths/films_NQ_discovery.parquet',
                        columns=['riz_id', 'side', 'tf_minutes', 't0_spine_pos', 'exit_boundary'])


def analyse(t0, up, e):
    sgn = 1 if up else -1; Eo = e if up else -e
    M = high[t0] if up else -low[t0]; MB = max(sgn*opn[t0], sgn*close[t0])
    last_up = None; parent = -1; branch = None; parent_bar = -1
    n_upd_after = 0; n_joint_after = 0
    term = None; term_k = -1
    for k in range(1, CAP + 1):
        j = t0 + k
        if j > last or kind[j] != 0:
            term = 'obs'; term_k = k - 1; break
        hi = high[j]; lo = low[j]
        # terminal checks (object-aware, own b)
        if lo <= e <= hi:
            term = 'b'; term_k = k; break
        Ho = hi if up else -lo
        if Ho < Eo:
            term = 'cross'; term_k = k; break
        Oo = sgn*opn[j]; Co = sgn*close[j]; Bo = max(Oo, Co)
        Mupd = Ho > M; MBupd = Bo > MB
        cur = 'joint' if (Mupd and MBupd) else ('M' if Mupd else ('MB' if MBupd else None))
        if parent >= 0 and branch is None and cur is not None:
            # first outward update after the parent joint -> the branch
            if cur == 'M':
                branch = 'WOECB' if Co < Oo else 'Mout'
            elif cur == 'joint':
                branch = 'joint'
            else:
                branch = 'MBonly'
            parent_bar = parent
        if parent >= 0 and cur is not None:
            n_upd_after += 1
            if cur == 'joint': n_joint_after += 1
        if parent < 0 and cur == 'joint':
            parent = k                      # first joint predecessor state
        if Ho > M: M = Ho
        if Bo > MB: MB = Bo
    else:
        term = 'cap'; term_k = CAP
    if parent < 0:
        return None                          # no joint predecessor ever
    if branch is None:
        branch = 'NONE'                      # scene ended before any further outward update
    return dict(parent_abs=t0 + parent, branch=branch, term=term,
                bars_from_parent=term_k - parent, n_upd_after=n_upd_after,
                n_joint_after=n_joint_after, reached_b=(term == 'b'))


rows = []
t0a = films.t0_spine_pos.to_numpy(); up_a = (films.side.values == 'north'); ea = films.exit_boundary.to_numpy()
for i in range(len(films)):
    r = analyse(int(t0a[i]), bool(up_a[i]), float(ea[i]))
    if r:
        r['up'] = bool(up_a[i]); rows.append(r)
R = pd.DataFrame(rows)
print(f'object-scenes with a joint predecessor: {len(R)}')
D = R.drop_duplicates(subset=['parent_abs', 'up'])
print(f'physical episodes (parent bar,side): {len(D)}  (copies removed {len(R)-len(D)})\n')

print('branch distribution from the common joint-parent (physical-episode deduped):')
print(D.branch.value_counts(normalize=True).round(3).to_string())
print('\nWHOLE further film per branch (deduped), NOT one probability:')
print(' branch  |   n   | terminal mix (b/cross/obs/cap) | bars_from_parent p50 | further outward-upd p50 | reached_b | censored(obs/cap)')
for br in ['WOECB', 'Mout', 'joint', 'MBonly', 'NONE']:
    g = D[D.branch == br]
    if len(g) == 0: continue
    tm = g.term.value_counts(normalize=True)
    mix = '/'.join(f'{tm.get(x,0):.2f}' for x in ['b', 'cross', 'obs', 'cap'])
    print(f"  {br:7s} | {len(g):5d} | {mix} | {int(g.bars_from_parent.median()):4d} | "
          f"{int(g.n_upd_after.median()):3d} | {g.reached_b.mean():.3f} | {g.term.isin(['obs','cap']).mean():.3f}")
print('\nread: do the branches show a stably different further film, or reconverge (same terminal mix,'
      ' same path length, same reached-b)?  earliest-parent, survivorship-safe (NONE kept).')
