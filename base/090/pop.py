#!/usr/bin/env python3
"""090 prefix-honest population + CONT* + q-time (NQ discovery only).

Object-scene = triple (t0_spine_pos, side, exit_boundary). Prefix-identical TF/zone
copies collapse to the triple (end_pos is unique per triple -> terminal is
prefix-determined, so 089's life-selected representative is dropped). No future
selection. All cursors retained, including censored. No Volume.

Outcome CONT* (FREEZE_090 sec.2):
  1        next qualifying front event observed before physical terminal
  0        physical terminal (contact_b / retreat_in) before next event
  -1       observation lost (gap / archive) before either  -> censored/unknown

State (FREEZE_089 sec.3): dpos = sgn*(close[j]-e); psig = mean_{t0..j}(high-low).
Cursor q = live front-event bar; all read from closed prefix <= j. q-time from
close_ts_utc_ns[j].
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('C:/Users/Admin/Claude/g3-market-research')
sys.path.insert(0, str(ROOT / 'base/084'))
import race084
market, kind = race084.load_market('NQ'); high, low, close = market
opn = np.load(ROOT / 'data/market/NQ/open.npy')
ts = np.load(ROOT / 'data/market/NQ/close_ts_utc_ns.npy')
last = close.size - 1

cols = ['side', 't0_spine_pos', 'exit_boundary', 'end_pos', 'film1_status']
films = pd.read_parquet(ROOT / 'work/081a/paths/films_NQ_discovery.parquet', columns=cols)
# prefix-identical canonicalization: one row per (t0,side,exit_boundary)
tri = films.groupby(['t0_spine_pos', 'side', 'exit_boundary'], as_index=False).first()
print(f"parquet rows={len(films):,}   distinct object-scenes (triples)={len(tri):,}")

t0a = tri.t0_spine_pos.to_numpy().astype(np.int64)
up_a = (tri.side.values == 'north')
ea = tri.exit_boundary.to_numpy().astype(float)
ep = tri.end_pos.to_numpy().astype(np.int64)
status = tri.film1_status.to_numpy().astype(object)


def scene_cursors(t0, up, e, endp):
    """Rebuild outward live scene exactly as fullfilm.get_bars/annotate; return
    (list of (dpos,psig,ord,j), terminal_reason)."""
    sgn = 1.0 if up else -1.0
    j0 = t0
    if j0 > last or kind[j0] != 0:
        return [], 'archive'
    hi = high[j0]; lo = low[j0]
    Wo = hi if up else -lo
    Wi = lo if up else -hi
    Oo = sgn * opn[j0]; Co = sgn * close[j0]
    Bo = max(Oo, Co)
    M = Wo; MB = Bo
    rng_sum = (Wo - Wi); nb = 1
    cur = []
    reason = 'end'
    kmax = endp - t0
    for k in range(1, kmax + 1):
        j = t0 + k
        if j > last:
            reason = 'archive'; break
        if kind[j] != 0:
            reason = 'gap'; break
        hi = high[j]; lo = low[j]
        if lo <= e <= hi:
            reason = 'contact_b'; break
        Wo = hi if up else -lo
        Wi = lo if up else -hi
        Oo = sgn * opn[j]; Co = sgn * close[j]
        Bo = max(Oo, Co)
        newM = Wo > M; newMB = Bo > MB
        rng_sum += (Wo - Wi); nb += 1
        if newM or newMB:
            dpos = sgn * (close[j] - e)
            psig = rng_sum / nb
            cur.append((dpos, psig, len(cur), j))
        if newM: M = Wo
        if newMB: MB = Bo
    else:
        # loop completed to end_pos without contact/gap/archive: end_pos terminal
        reason = 'endpos'
    return cur, reason


rows = []
term_counter = {}
endpos_status = {}
for i in range(len(tri)):
    cur, reason = scene_cursors(int(t0a[i]), bool(up_a[i]), float(ea[i]), int(ep[i]))
    term_counter[reason] = term_counter.get(reason, 0) + 1
    if reason == 'endpos':
        s = str(status[i]); endpos_status[s] = endpos_status.get(s, 0) + 1
    ne = len(cur)
    if ne == 0:
        continue
    # map terminal -> physical vs lost-observability
    # gap/archive = lost; endpos is physical contact EXCEPT freshness_lost_before_contact
    lost = reason in ('gap', 'archive') or (reason == 'endpos' and 'freshness_lost' in str(status[i]))
    for idx, (dpos, psig, ordn, j) in enumerate(cur):
        if idx < ne - 1:
            cont = 1                      # next event observed before terminal
        else:
            cont = -1 if lost else 0      # last event: censored if obs lost, else physical 0
        rows.append((dpos, psig, cont, i, ne, ordn, j, int(ts[j]), reason))

R = pd.DataFrame(rows, columns=['dpos', 'psig', 'cont', 'scene', 'nev', 'ord', 'j', 'q_ts_ns', 'term'])
R = R[R.psig > 0].copy()
R['nz'] = R.dpos / R.psig
R['q_date'] = pd.to_datetime(R.q_ts_ns, utc=True)

print("\n--- terminal reason per scene (all triples) ---")
for k, v in sorted(term_counter.items(), key=lambda x: -x[1]):
    print(f"  {k:12s} {v:>8,}")
if endpos_status:
    print("  endpos film1_status breakdown:")
    for k, v in sorted(endpos_status.items(), key=lambda x: -x[1]):
        print(f"    {k:24s} {v:>8,}")

print("\n--- cursor population (>=1 live front event, psig>0) ---")
print(f"cursors={len(R):,}   object-scenes with >=1 event={R.scene.nunique():,}")
res = R[R.cont >= 0]
cens = R[R.cont < 0]
print(f"resolved cursors={len(res):,}  ({len(res)/len(R):.4f})   censored cursors={len(cens):,}  ({len(cens)/len(R):.4f})")
print(f"P(CONT*=1 | resolved) = {res.cont.mean():.4f}")
print(f"censored share among LAST events only: "
      f"{(R.groupby('scene').tail(1).cont < 0).mean():.4f}")
print(f"q-date span: {R.q_date.min()} .. {R.q_date.max()}")
print(f"dpos p5/25/50/75/95 = {np.percentile(R.dpos,[5,25,50,75,95]).round(2)}")
print(f"psig p5/25/50/75/95 = {np.percentile(R.psig,[5,25,50,75,95]).round(2)}")

outdir = ROOT / 'work/090'; outdir.mkdir(parents=True, exist_ok=True)
# scene-level table for split/embargo: t0, end_pos time, side, e
sc = tri.copy()
sc['scene'] = np.arange(len(tri))
sc['t0_ts_ns'] = ts[np.clip(t0a, 0, last)]
sc['end_ts_ns'] = ts[np.clip(ep, 0, last)]
sc.to_parquet(outdir / 'scenes.parquet')
R.to_parquet(outdir / 'cursors.parquet')
print(f"\nsaved {outdir/'cursors.parquet'} and {outdir/'scenes.parquet'}")
