#!/usr/bin/env python3
"""089 minimal descriptive bridge (NQ discovery only).

Question (frozen before any result): inside the 088 joint/outward Film-1 event
process, does the IMMEDIATE survival of the observable grammar depend on the
already-established current state?

CONT(q)=1 iff at least one further qualifying front event (newM OR newMB, exactly
as fullfilm.annotate) occurs before the scene's physical terminal; else 0. No fixed
horizon; not remaining count / minutes / b-contact / short-long / P&L / WOECB'.

State coordinates, frozen by reference to the established line:
  dpos(q) = sgn*(close[j]-e)                      -- d_close (081), outward distance to own boundary b
  psig(q) = mean_{t=t0..j}(high[t]-low[t])        -- prefix_sigma operator _sig (base/081/sigma_control.py)
  nz(q)   = dpos/psig                             -- deterministic combination of the two frozen quantities

Event detection is bit-identical to fullfilm.annotate (M,MB start at bar0; newM<=>Wo>M, newMB<=>Bo>MB).
Physical act = (t0_spine_pos, side) episode dedup, exactly as 088. No Volume.

Bins are QUANTILE edges of the state marginals (state distribution only, not CONT):
declared here, not searched. Two weightings reported, neither chosen for prettiness:
  event-level    -- one row per live front-event cursor
  scene-balanced -- each cursor weighted 1/n_events(film)
Uncertainty clustered at the physical act via block bootstrap over films.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('C:/Users/Admin/Claude/g3-market-research')
sys.path.insert(0, str(ROOT / 'base/084'))
import race084
market, kind = race084.load_market('NQ'); high, low, close = market
last = close.size - 1

films = pd.read_parquet(ROOT / 'work/081a/paths/films_NQ_discovery.parquet',
                        columns=['side', 't0_spine_pos', 'exit_boundary', 'end_pos'])
films['life'] = films.end_pos - films.t0_spine_pos
epi = films.sort_values('life').groupby(['t0_spine_pos', 'side'], as_index=False).last()

t0a = epi.t0_spine_pos.to_numpy().astype(np.int64)
up_a = (epi.side.values == 'north')
ea = epi.exit_boundary.to_numpy().astype(float)
ep = epi.end_pos.to_numpy().astype(np.int64)


def film_cursors(t0, up, e, endp):
    """Return list of (dpos, psig, event_ordinal) per front-event cursor, plus n_events.
    Rebuilds the outward live scene identically to fullfilm.get_bars/annotate but only
    keeps what the state + CONT need (O(life), no relational bookkeeping)."""
    sgn = 1.0 if up else -1.0
    # bar0
    j0 = t0
    if j0 > last or kind[j0] != 0:
        return [], 0
    hi = high[j0]; lo = low[j0]
    Wo = hi if up else -lo
    Wi = lo if up else -hi
    Co = sgn * close[j0]
    Oo_close = Co  # open not needed for Bo0? fullfilm Bo0 = bl[0]['Bo']=max(Oo,Co)
    import numpy as _np
    opn = _OPEN
    Oo = sgn * opn[j0]
    Bo = max(Oo, Co)
    M = Wo; MB = Bo
    rng_sum = (Wo - Wi); nb = 1
    cursors = []
    for k in range(1, endp - t0 + 1):
        j = t0 + k
        if j > last:
            break
        if kind[j] != 0:
            break
        hi = high[j]; lo = low[j]
        if lo <= e <= hi:            # contact_b terminal (k>0 always here)
            break
        Wo = hi if up else -lo
        Wi = lo if up else -hi
        Oo = sgn * opn[j]; Co = sgn * close[j]
        Bo = max(Oo, Co)
        newM = Wo > M; newMB = Bo > MB
        rng_sum += (Wo - Wi); nb += 1
        if newM or newMB:
            dpos = sgn * (close[j] - e)
            psig = rng_sum / nb
            cursors.append((dpos, psig, len(cursors)))
        if newM: M = Wo
        if newMB: MB = Bo
    n_events = len(cursors)
    return cursors, n_events


_OPEN = np.load(ROOT / 'data/market/NQ/open.npy')

rows = []   # dpos, psig, cont, film_id, n_events, ordinal
for i in range(len(epi)):
    cur, ne = film_cursors(int(t0a[i]), bool(up_a[i]), float(ea[i]), int(ep[i]))
    if ne == 0:
        continue
    for idx, (dpos, psig, ordn) in enumerate(cur):
        cont = 1 if idx < ne - 1 else 0
        rows.append((dpos, psig, cont, i, ne, ordn))

R = pd.DataFrame(rows, columns=['dpos', 'psig', 'cont', 'film', 'nev', 'ord'])
R = R[R.psig > 0].copy()
R['nz'] = R.dpos / R.psig
R['w_scene'] = 1.0 / R.nev
print(f"cursors: {len(R):,}   distinct physical acts with >=1 event: {R.film.nunique():,}")
print(f"overall P(CONT=1) event-level = {R.cont.mean():.4f}   "
      f"scene-balanced = {np.average(R.cont, weights=R.w_scene):.4f}")
print(f"dpos  (pts) p5/25/50/75/95 = {np.percentile(R.dpos,[5,25,50,75,95]).round(2)}")
print(f"psig  (pts) p5/25/50/75/95 = {np.percentile(R.psig,[5,25,50,75,95]).round(2)}")
print(f"nz=d/σ      p5/25/50/75/95 = {np.percentile(R.nz,[5,25,50,75,95]).round(3)}")

R.to_parquet(ROOT / 'work/089/cursors.parquet')


def curve(col, nbins=10):
    """P(CONT|state) along quantile bins of `col` (edges from state marginal only)."""
    edges = np.unique(np.quantile(R[col], np.linspace(0, 1, nbins + 1)))
    b = pd.cut(R[col], edges, include_lowest=True, duplicates='drop')
    g = R.groupby(b, observed=True)
    out = g.apply(lambda d: pd.Series({
        'n_cursor': len(d), 'n_act': d.film.nunique(),
        'mid': d[col].median(),
        'p_event': d.cont.mean(),
        'p_scene': np.average(d.cont, weights=d.w_scene),
    }), include_groups=False)
    return out


for col in ['dpos', 'psig', 'nz']:
    print(f"\n=== P(CONT=1) vs {col}  (deciles, state-only edges) ===")
    c = curve(col, 10)
    with pd.option_context('display.width', 140, 'display.max_columns', 20):
        print(c.round(4).to_string())


# 2-D surface: 6x6 quantile grid on (dpos, psig). Descriptive, not optimized.
print("\n=== 2-D surface P(CONT=1 | dpos x psig), 6x6 quantile grid (event-level) ===")
qd = pd.qcut(R.dpos, 6, duplicates='drop')
qs = pd.qcut(R.psig, 6, duplicates='drop')
piv = R.groupby([qd, qs], observed=True).cont.mean().unstack().round(3)
cnt = R.groupby([qd, qs], observed=True).film.nunique().unstack()
with pd.option_context('display.width', 200, 'display.max_columns', 20):
    print("P(CONT=1):"); print(piv.to_string())
    print("\ndistinct acts per cell:"); print(cnt.to_string())


# Maturity as a DIAGNOSTIC composition axis only (early/mid/late event ordinal),
# to see whether the dpos curve is confounded by where in the sequence the cursor sits.
print("\n=== diagnostic: P(CONT=1) vs dpos decile, split by event ordinal band ===")
R['ordband'] = pd.cut(R['ord'], [-1, 2, 9, 10**9], labels=['ord0-2', 'ord3-9', 'ord10+'])
edges = np.unique(np.quantile(R.dpos, np.linspace(0, 1, 11)))
R['dq'] = pd.cut(R.dpos, edges, include_lowest=True, duplicates='drop')
tab = R.groupby(['dq', 'ordband'], observed=True).cont.mean().unstack().round(3)
print(tab.to_string())


# Block bootstrap over films: CI on the nz decile curve (event-level), clustered at act.
print("\n=== block bootstrap (resample physical acts), 95% CI on P(CONT=1) vs nz decile ===")
edges = np.unique(np.quantile(R.nz, np.linspace(0, 1, 11)))
R['nzq'] = pd.cut(R.nz, edges, include_lowest=True, duplicates='drop')
film_ids = R.film.to_numpy()
uniq = np.unique(film_ids)
# group cursors by film for fast resample
order = np.argsort(film_ids, kind='stable')
Rs = R.iloc[order].reset_index(drop=True)
fs = Rs.film.to_numpy()
starts = np.searchsorted(fs, uniq)
ends = np.r_[starts[1:], len(fs)]
cont = Rs.cont.to_numpy(float)
nzq_codes = Rs.nzq.cat.codes.to_numpy()
K = nzq_codes.max() + 1
rng = np.random.default_rng(89)
B = 300
boot = np.full((B, K), np.nan)
for bi in range(B):
    pick = rng.integers(0, len(uniq), len(uniq))
    idx = np.concatenate([np.arange(starts[p], ends[p]) for p in pick])
    cc = cont[idx]; kk = nzq_codes[idx]
    num = np.bincount(kk, weights=cc, minlength=K)
    den = np.bincount(kk, minlength=K)
    boot[bi] = np.where(den > 0, num / den, np.nan)
lo = np.nanpercentile(boot, 2.5, axis=0)
hi = np.nanpercentile(boot, 97.5, axis=0)
point = Rs.groupby('nzq', observed=True).cont.mean()
mids = Rs.groupby('nzq', observed=True).nz.median()
print(" nz_decile_median   P(CONT)   95%CI          n_act")
nact = Rs.groupby('nzq', observed=True).film.nunique()
for k, lab in enumerate(point.index):
    print(f"  {mids.iloc[k]:+8.3f}        {point.iloc[k]:.3f}   [{lo[k]:.3f},{hi[k]:.3f}]   {nact.iloc[k]:>6}")
