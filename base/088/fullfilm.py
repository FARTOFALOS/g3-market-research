#!/usr/bin/env python3
"""Full-film retrospective reader. DISCOVERY stage: whole film visible.
Left  = accumulated post-T0 relational history up to each event q (prefix-only facts).
Right = continuation after q (the rest, visible during discovery).
Per event bar we annotate literal accumulated relations with EXACT bar refs:
  overlapK   = earlier bars whose BODY the current body overlaps (list ks)
  reenterK   = earlier bars whose BODY current CLOSE re-enters (list ks)
  reconDepth = oldest earlier bar current body overlaps, and #front-advances since it
  edgeTest   = current Bo level: how many earlier bars set Bo within 1 tick, span of ks
  region     = current close region VIRGIN (never in any prior bar range) or TRAVERSED
  front      = current M setter k, MB setter k ; gap = minutes since previous front event
No compression to scalar features; refs kept. No future in the left column.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('C:/Users/Admin/Claude/g3-market-research')
sys.path.insert(0, str(ROOT / 'base/084'))
import race084
market, kind = race084.load_market('NQ'); high, low, close = market
opn = np.load(ROOT / 'data/market/NQ/open.npy'); last = close.size - 1
TICK = 0.25
films = pd.read_parquet(ROOT / 'work/081a/paths/films_NQ_discovery.parquet',
                        columns=['side', 't0_spine_pos', 'exit_boundary', 'end_pos', 'tf_minutes'])
films['life'] = films.end_pos - films.t0_spine_pos
epi = films.sort_values('life').groupby(['t0_spine_pos', 'side'], as_index=False).last()


def get_bars(t0, up, e, end_pos):
    sgn = 1 if up else -1
    out = []; reason = 'contact_b'
    for k in range(0, end_pos - t0 + 1):
        j = t0 + k
        if j > last: reason = 'archive'; break
        if kind[j] != 0: reason = 'gap'; break
        hi = high[j]; lo = low[j]
        if k > 0 and lo <= e <= hi: reason = 'contact_b'; break
        Wo = hi if up else -lo
        Oo = sgn * opn[j]; Co = sgn * close[j]
        out.append(dict(k=k, Wo=Wo, Bo=max(Oo, Co), Bi=min(Oo, Co), Wi=(lo if up else -hi), C=Co))
    return out, reason


def annotate(bl):
    """Return per-event annotations. Event = newM or newMB bar."""
    M = bl[0]['Wo']; MB = bl[0]['Bo']; prevBo = bl[0]['Bo']
    m_set = 0; mb_set = 0; last_ev_k = 0
    rows = []
    fronts_adv = [0]                 # ks that advanced a front (for recon-depth counting)
    for t in range(1, len(bl)):
        b = bl[t]; C = b['C']
        newM = b['Wo'] > M; newMB = b['Bo'] > MB
        if newM or newMB:
            cls = 'HELD' if (newMB and C > prevBo) else ('FALLBACK' if (newM and C <= prevBo) else 'OTHER')
            overlapK = [bl[j]['k'] for j in range(t) if not (b['Bo'] < bl[j]['Bi'] or b['Bi'] > bl[j]['Bo'])]
            reenterK = [bl[j]['k'] for j in range(t) if bl[j]['Bi'] <= C <= bl[j]['Bo']]
            oldest = overlapK[0] if overlapK else None
            recon = None
            if oldest is not None and oldest < b['k']:
                recon = sum(1 for a in fronts_adv if oldest < a <= b['k'])
            edge_ks = [bl[j]['k'] for j in range(t) if abs(bl[j]['Bo'] - b['Bo']) < TICK / 2]
            virgin = not any(bl[j]['Wi'] <= C <= bl[j]['Wo'] for j in range(t))
            rows.append(dict(k=b['k'], cls=cls, typ=('j' if newM and newMB else ('m' if newM else 'b')),
                             C=C, Bo=b['Bo'], Wo=b['Wo'],
                             overlapK=overlapK, reenterK=reenterK, oldest=oldest, recon=recon,
                             edge_ks=edge_ks, virgin=virgin, m_set=m_set, mb_set=mb_set,
                             gap=b['k'] - last_ev_k))
            last_ev_k = b['k']
        if newM: M = b['Wo']; m_set = b['k']
        if newMB: MB = b['Bo']; mb_set = b['k']
        if newM or newMB: fronts_adv.append(b['k'])
        prevBo = b['Bo']
    return rows


def show(t0, up, e, endp, reason=None):
    bl, rsn = get_bars(t0, up, e, endp)
    rows = annotate(bl)
    print(f"\n=== t0={t0} {'north' if up else 'south'} b={e:.2f} life={bl[-1]['k']} "
          f"events={len(rows)} end={rsn} ===")
    print("   k  cls  typ gap  C      Bo     | overlap bodies (k) | close re-enters (k) | reconDepth | edgeTest | region")
    for r in rows:
        rec = f"old k{r['oldest']} ~{r['recon']}adv" if r['recon'] is not None else "-"
        et = f"Bo@{len(r['edge_ks'])}x k{r['edge_ks'][:4]}" if len(r['edge_ks']) > 1 else "-"
        reg = 'VIRGIN' if r['virgin'] else 'trav'
        print(f"  {r['k']:>3} {r['cls']:>8} {r['typ']} +{r['gap']:<2} {r['C']:.2f} {r['Bo']:.2f} | "
              f"{str(r['overlapK'][-8:]):22s} | {str(r['reenterK'][-6:]):16s} | {rec:16s} | {et:16s} | {reg}")


if __name__ == '__main__':
    # 1) the preserved counterexample pair
    show(495935, True, 1838.5, 495935 + 9)
    show(510345, False, 1980.25, 510345 + 187)

    # 2) sample films that EXERCISE accumulated depth: long life, many reconnections
    cand = epi[(epi.life >= 60) & (epi.life <= 400)].copy()
    cand = cand.sort_values('t0_spine_pos')
    shown = 0
    for _, f in cand.iterrows():
        t0 = int(f.t0_spine_pos); up = (f.side == 'north'); e = float(f.exit_boundary); endp = int(f.end_pos)
        bl, rsn = get_bars(t0, up, e, endp)
        rows = annotate(bl)
        deep = [r for r in rows if r['recon'] is not None and r['recon'] >= 3]
        if len(rows) >= 8 and deep:
            show(t0, up, e, endp)
            shown += 1
        if shown >= 4:
            break
