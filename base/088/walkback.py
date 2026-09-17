#!/usr/bin/env python3
"""Paired walk-back. Find films with meaningfully similar EARLY histories that later
diverge in continuation, then inspect the last common event for an accumulated-history
distinction absent from the local view. Future used ONLY as retrospective contrast.
No classifier trained; no anchor on close-fall-back. NQ discovery.

Similarity key = sequence of first K front-event classes (HELD/FALLBACK/OTHER) AND the
integer reach bucket at each of those events (reach = M advance / mean-bar-range, floor).
Reach bucket keeps 'meaningfully similar' honest: same event grammar AND same coarse
outward progress. Then within a key contrast short-lived vs long-lived members and read
their accumulated annotations at the shared events.
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('C:/Users/Admin/Claude/g3-market-research')
sys.path.insert(0, str(ROOT / 'base/084'))
import race084
market, kind = race084.load_market('NQ'); high, low, close = market
opn = np.load(ROOT / 'data/market/NQ/open.npy'); last = close.size - 1
films = pd.read_parquet(ROOT / 'work/081a/paths/films_NQ_discovery.parquet',
                        columns=['side', 't0_spine_pos', 'exit_boundary', 'end_pos'])
films['life'] = films.end_pos - films.t0_spine_pos
epi = films.sort_values('life').groupby(['t0_spine_pos', 'side'], as_index=False).last()
K = 4  # length of shared early event prefix


def events(t0, up, e, end_pos):
    sgn = 1 if up else -1
    M = high[t0] if up else -low[t0]; MB = max(sgn*opn[t0], sgn*close[t0]); prevBo = MB
    W0 = M; rng = M - (low[t0] if up else -high[t0]); nb = 1
    ev = []
    for k in range(1, end_pos - t0 + 1):
        j = t0 + k
        if j > last or kind[j] != 0: break
        hi = high[j]; lo = low[j]
        if lo <= e <= hi: break
        Wo = hi if up else -lo
        Oo = sgn*opn[j]; Co = sgn*close[j]; Bo = max(Oo, Co)
        rng += Wo - (lo if up else -hi); nb += 1
        newM = Wo > M; newMB = Bo > MB
        if newM or newMB:
            cls = 'H' if (newMB and Co > prevBo) else ('F' if (newM and Co <= prevBo) else 'O')
            reach = int((M - W0) / (rng/nb)) if rng > 0 else 0
            ev.append(dict(k=k, cls=cls, reach=max(reach, 0)))
        if newM: M = Wo
        if newMB: MB = Bo
        prevBo = Bo
    return ev


# gather
recs = []
t0a = epi.t0_spine_pos.to_numpy(); up_a = (epi.side.values == 'north')
ea = epi.exit_boundary.to_numpy(); ep = epi.end_pos.to_numpy()
for i in range(len(epi)):
    ev = events(int(t0a[i]), bool(up_a[i]), float(ea[i]), int(ep[i]))
    if len(ev) < K: continue
    key = tuple((e['cls'], min(e['reach'], 3)) for e in ev[:K])
    recs.append((key, len(ev), int(t0a[i]), bool(up_a[i]), float(ea[i]), int(ep[i])))

groups = defaultdict(list)
for r in recs:
    groups[r[0]].append(r[1:])

# keys where continuation strongly diverges: has members with <=K+1 events AND >=K+15
div = []
for key, mem in groups.items():
    ne = [m[0] for m in mem]
    if len(mem) >= 20 and min(ne) <= K + 1 and max(ne) >= K + 15:
        div.append((key, len(mem), min(ne), max(ne)))
div.sort(key=lambda x: -x[1])
print(f"keys with >=K events: {len(groups)}; divergent keys (short & long members): {len(div)}")
print("top divergent shared-early-history keys (cls,reachbucket)x4 | nmembers | minEv | maxEv:")
for key, n, mn, mx in div[:12]:
    print(f"   {key} | {n} | {mn} | {mx}")

# summarize: within each divergent key, do short vs long differ in event-time depth only,
# or is there a per-event reach separation already at events 1..K?
print("\nPer-divergent-key: mean reach at each of the K shared events, short vs long members")
for key, n, mn, mx in div[:8]:
    mem = groups[key]
    short = [m for m in mem if m[0] <= K + 2]
    lng = [m for m in mem if m[0] >= K + 12]
    # recompute per-event reach for members
    def reaches(subset):
        R = []
        for (_ne, t0, up, e, endp) in subset:
            ev = events(t0, up, e, endp)
            R.append([ev[t]['reach'] for t in range(K)])
        return np.array(R)
    rs, rl = reaches(short), reaches(lng)
    ms = rs.mean(0) if len(rs) else np.zeros(K)
    ml = rl.mean(0) if len(rl) else np.zeros(K)
    print(f"   key {key}")
    print(f"      short(n={len(short)}) reach@1..K: {np.round(ms,2)}")
    print(f"      long (n={len(lng)}) reach@1..K: {np.round(ml,2)}")

# --- dump matched members of one divergent key for full-film reading ---
import importlib.util
spec = importlib.util.spec_from_file_location("ff", str(Path(__file__).with_name("fullfilm.py")))
ff = importlib.util.module_from_spec(spec)
KEY = (('H', 0), ('H', 0), ('H', 1), ('H', 1))
mem = groups[KEY]
short = sorted([m for m in mem if m[0] <= K + 2], key=lambda m: m[1])[:2]
lng = sorted([m for m in mem if m[0] >= K + 15], key=lambda m: -m[0])[:2]
print(f"\n\n#### MATCHED KEY {KEY}: reading 2 short + 2 long members ####")
spec.loader.exec_module(ff)
for tag, subset in [('SHORT', short), ('LONG', lng)]:
    for (_ne, t0, up, e, endp) in subset:
        print(f"\n----- {tag} member -----")
        ff.show(t0, up, e, endp)
