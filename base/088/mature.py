#!/usr/bin/env python3
"""Mature-cursor matched comparison. Birth of distinguishability DURING the process.

Cursor = a fixed event ordinal EC (event-time maturity control) in the live film.
Match live scenes on: same event maturity (EC by construction), same outward-progress
bucket (reach at cursor), same current local geometry (last-3 event classes at cursor),
both still live, both with >= EC events of accumulated history.
Do NOT tune matching to maximize future separation; it only controls opportunity.
Within a matched group, contrast members whose REMAINING continuation is short vs long,
and read the ACCUMULATED structure already present AT the cursor (literal, prefix-only):
  max_recon (deepest advances-back to an overlapped old body up to cursor),
  n_deep    (events with recon>=3), n_persist (persistent Bo edges >=2x within tick),
  max_pause (largest event gap), n_territory (distinct old bodies reconnected).
Full future visible for discovery. No close-fall-back/lifetime/terminal as admission.
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('C:/Users/Admin/Claude/g3-market-research')
sys.path.insert(0, str(ROOT / 'base/084'))
import race084
import importlib.util
spec = importlib.util.spec_from_file_location("ff", str(Path(__file__).with_name("fullfilm.py")))
ff = importlib.util.module_from_spec(spec); spec.loader.exec_module(ff)
market, kind = race084.load_market('NQ'); high, low, close = market
opn = np.load(ROOT / 'data/market/NQ/open.npy'); last = close.size - 1
films = pd.read_parquet(ROOT / 'work/081a/paths/films_NQ_discovery.parquet',
                        columns=['side', 't0_spine_pos', 'exit_boundary', 'end_pos'])
films['life'] = films.end_pos - films.t0_spine_pos
epi = films.sort_values('life').groupby(['t0_spine_pos', 'side'], as_index=False).last()
EC = 20                      # cursor event ordinal (mature)
TICK = 0.25


def reach_at(bl, upto_k):
    """coarse outward progress up to bar index (mean-bar-range units)."""
    W0 = bl[0]['Wo']; M = W0; rng = bl[0]['Wo'] - bl[0]['Wi']; nb = 1
    for b in bl[1:]:
        if b['k'] > upto_k: break
        rng += b['Wo'] - b['Wi']; nb += 1
        if b['Wo'] > M: M = b['Wo']
    return int((M - W0) / (rng / nb)) if rng > 0 else 0


def cursor_state(t0, up, e, endp):
    bl, reason = ff.get_bars(t0, up, e, endp)
    if len(bl) < EC + 3:
        return None
    rows = ff.annotate(bl)
    if len(rows) < EC + 3:
        return None
    cur = rows[EC - 1]                      # cursor event (0-indexed)
    pre = rows[:EC]
    last3 = tuple(r['cls'] for r in rows[EC - 3:EC])
    reach = reach_at(bl, cur['k'])
    # accumulated descriptors at cursor
    max_recon = max((r['recon'] or 0) for r in pre)
    n_deep = sum(1 for r in pre if (r['recon'] or 0) >= 3)
    n_persist = sum(1 for r in pre if len(r['edge_ks']) >= 2)
    gaps = [pre[i]['gap'] for i in range(1, len(pre))]
    max_pause = max(gaps) if gaps else 0
    n_territory = len(set(r['oldest'] for r in pre if r['oldest'] is not None))
    remaining = len(rows) - EC              # events after cursor
    return dict(t0=t0, up=up, e=e, endp=endp, last3=last3, reach=min(reach, 4),
                max_recon=max_recon, n_deep=n_deep, n_persist=n_persist,
                max_pause=max_pause, n_territory=n_territory, remaining=remaining,
                cursor_k=cur['k'])


states = []
t0a = epi.t0_spine_pos.to_numpy(); up_a = (epi.side.values == 'north')
ea = epi.exit_boundary.to_numpy(); ep = epi.end_pos.to_numpy()
for i in range(len(epi)):
    s = cursor_state(int(t0a[i]), bool(up_a[i]), float(ea[i]), int(ep[i]))
    if s: states.append(s)
S = pd.DataFrame(states)
print(f"live scenes reaching event {EC} with >=3 more: {len(S)}")

# match on (reach bucket, last-3 classes)
S['key'] = list(zip(S.reach, S.last3))
grp = S.groupby('key')
print("\nMatched mature-cursor groups (>=30 members) with divergent remaining continuation:")
print("key | n | remaining p10/p50/p90 | short(rem<=5) vs long(rem>=20): accumulated descriptors mean")
rows_out = []
for key, g in grp:
    if len(g) < 30: continue
    sh = g[g.remaining <= 5]; ln = g[g.remaining >= 20]
    if len(sh) < 8 or len(ln) < 8: continue
    p = np.percentile(g.remaining, [10, 50, 90])
    print(f"\n {key} | n={len(g)} | rem {p.astype(int)}")
    for tag, gg in [('short', sh), ('long', ln)]:
        print(f"    {tag}(n={len(gg):4d}): max_recon {gg.max_recon.mean():.2f}  n_deep {gg.n_deep.mean():.2f}  "
              f"n_persist {gg.n_persist.mean():.2f}  max_pause {gg.max_pause.mean():.1f}  "
              f"n_territory {gg.n_territory.mean():.2f}  cursor_min {gg.cursor_k.mean():.0f}")
    rows_out.append((key, len(g), sh, ln))

# read: pick the largest divergent group, render 2 short + 2 long around the cursor
if rows_out:
    key, n, sh, ln = max(rows_out, key=lambda x: x[1])
    print(f"\n\n#### READING matched group {key}: 2 short-remaining + 2 long-remaining ####")
    for tag, gg in [('SHORT-REMAINING', sh.head(2)), ('LONG-REMAINING', ln.head(2))]:
        for _, r in gg.iterrows():
            print(f"\n----- {tag} (remaining={r.remaining}) -----")
            ff.show(int(r.t0), bool(r.up), float(r.e), int(r.endp))
