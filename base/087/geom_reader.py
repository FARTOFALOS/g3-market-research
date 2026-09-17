#!/usr/bin/env python3
"""Full candle-to-candle relational GEOMETRY (outward coords), WOECB as ONE reading of it.

Per bar, four outward points:
  W_out outer shadow/extreme | B_out outer body edge | B_in inner body edge | W_in inner shadow
References carried: previous bar's 4 points; current M (wick-extreme) & its bar;
current MB (body-front) & its bar; b (exit boundary, outward); RIZ zone.

Neutral relations printed each bar: wick<->wick, body<->body, body<->prev wick, wick<->prev
body, body vs M/MB, close vs b/zone. No pre-filtering: WOECB and its single-facet
counterexamples are flagged readings, not gates.

Frozen WOECB(t): W_out[t] > M_prev  AND  B_out[t] <= MB_prev   (body-front not advanced;
close-back entailed since Co<=B_out<=MB_prev). Counterexample CE-body: W_out[t]>M_prev AND
B_out[t] > MB_prev (body advanced the front). CE-pullback: no new M, Co<MB_prev.
Object-aware: WOECB judged inside the object's own live scene (<= its terminal). Evidence
deduplicated to physical episodes (t0_pos, side); TF copies are relations, counted once.
NO continuation. NQ discovery.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('C:/Users/Admin/Claude/g3-market-research')
sys.path.insert(0, str(ROOT / 'base/084'))
import race084
CAP = 300
market, kind = race084.load_market('NQ'); high, low, close = market
opn = np.load(ROOT / 'data/market/NQ/open.npy'); last = close.size - 1
films = pd.read_parquet(ROOT / 'work/081a/paths/films_NQ_discovery.parquet',
                        columns=['riz_id', 'side', 'tf_minutes', 't0_spine_pos', 'exit_boundary',
                                 'zone_top', 'zone_bottom', 'end_pos'])
films['end_k'] = films.end_pos - films.t0_spine_pos
epi = films.groupby(['t0_spine_pos', 'side']).agg(
    nriz=('riz_id', 'size'), tfs=('tf_minutes', lambda x: sorted(set(x))[:4]),
    end_ks=('end_k', list), max_end_k=('end_k', 'max'),
    zt=('zone_top', 'first'), zb=('zone_bottom', 'first'),
    e=('exit_boundary', 'first')).reset_index()


def pts(j, up):
    sgn = 1 if up else -1
    O, H, L, C = opn[j], high[j], low[j], close[j]
    Wo = H if up else -L; Wi = L if up else -H; Bo = max(sgn*O, sgn*C); Bi = min(sgn*O, sgn*C)
    return Wo, Bo, Bi, Wi, float(O), float(H), float(L), float(C)


def classify(t0, up, horizon):
    """earliest WOECB within horizon. REFINED: a real prior body-leg must exist
    (body-front MB advanced strictly beyond its T0 level before the bar). Also returns the
    RAW (unrefined) earliest for contrast, and CE-body / CE-pullback."""
    M = high[t0] if up else -low[t0]; MB0 = pts(t0, up)[1]; MB = MB0
    wo = wo_raw = ce_b = ce_p = -1; occ = []
    for k in range(1, horizon + 1):
        j = t0 + k
        if j > last or kind[j] != 0:
            break
        Wo, Bo, Bi, Wi, O, H, L, C = pts(j, up)
        Co = C if up else -C
        newM = Wo > M
        leg = MB > MB0                                  # a real outward body-leg has occurred
        if newM and Bo <= MB:
            if wo_raw < 0: wo_raw = k
            if leg:
                if wo < 0: wo = k
                occ.append(k)
        elif newM and Bo > MB:
            if ce_b < 0: ce_b = k
        elif (not newM) and Co < MB:
            if ce_p < 0: ce_p = k
        if Wo > M: M = Wo
        if Bo > MB: MB = Bo
    return wo, wo_raw, ce_b, ce_p, occ


def geom_window(t0, up, k0, e, zt, zb, klo=5, khi=2):
    M = high[t0] if up else -low[t0]; MB0 = pts(t0, up)[1]; MB = MB0
    Eo = e if up else -e
    out = []
    pWo = pBo = pBi = pWi = None
    for k in range(0, k0 + khi + 1):
        j = t0 + k
        Wo, Bo, Bi, Wi, O, H, L, C = pts(j, up)
        Co = C if up else -C
        newM = Wo > M; leg = MB > MB0
        if k >= k0 - klo:
            rel = []
            if pWo is not None:
                rel.append('Wo>pWo' if Wo > pWo else 'Wo<=pWo')          # wick<->wick
                rel.append('Bo>pBo' if Bo > pBo else 'Bo<=pBo')          # body<->body
                rel.append('Bo>pWo' if Bo > pWo else 'Bo<=pWo')          # body<->prev wick
                rel.append('Wo>pBo' if Wo > pBo else 'Wo<=pBo')          # wick<->prev body
            tagB = ('Bo>MB' if Bo > MB else ('Bo=MB' if Bo == MB else 'Bo<MB'))
            if newM and Bo <= MB:
                read = 'WOECB' if leg else 'wick-noLeg'
            elif newM and Bo > MB:
                read = 'CEbody'
            elif (not newM) and Co < MB:
                read = 'CEpull'
            else:
                read = ''
            mark = 'X<<<' if k == k0 else ('fut' if k > k0 else '')
            out.append(f"      +{k:>3} O{O:.2f} H{H:.2f} L{L:.2f} C{C:.2f} | {'newM' if newM else 'noM'} "
                       f"{tagB} {'leg' if leg else 'noleg'} {' '.join(rel)} | Cvb{(Co-Eo):+.2f} "
                       f"{'inZone' if zb<=C<=zt else 'out'} [{read}] {mark}")
        pWo, pBo, pBi, pWi = Wo, Bo, Bi, Wi
        if Wo > M: M = Wo
        if Bo > MB: MB = Bo
    return out


t0a = epi.t0_spine_pos.to_numpy(); up_a = (epi.side.values == 'north'); mek = epi.max_end_k.to_numpy()
pos_rows = []; ceb_rows = []
for i in range(len(epi)):
    hz = int(min(mek[i], CAP)) if mek[i] > 0 else CAP
    wo, ceb, cep, occ = classify(int(t0a[i]), bool(up_a[i]), hz)
    if wo >= 0:                                   # WOECB inside the object's live scene
        pos_rows.append((i, wo, len(occ)))
    elif ceb >= 0:
        ceb_rows.append((i, ceb))
    if len(pos_rows) >= 400 and len(ceb_rows) >= 200:
        pass

pos = pd.DataFrame(pos_rows, columns=['i', 'wo', 'n_occ'])
print(f'physical episodes: {len(epi)}. WOECB inside a live object scene: {len(pos)} episodes '
      f'({len(pos)/len(epi):.3f}); earliest-bar p25/50/75='
      f'{[int(pos.wo.quantile(q)) for q in (.25,.5,.75)]}; repeats/epi p50={int(pos.n_occ.median())}')
print(f'nearest CE-body-first episodes (no WOECB first): {len(ceb_rows)}')

print('\n===== 6 POSITIVE physical episodes (earliest WOECB, full geometry) =====')
for i, wo, nocc in pos.sort_values('i').head(6).itertuples(index=False):
    r = epi.iloc[int(i)]
    print(f"\n  t0={int(r.t0_spine_pos)} {r.side} nriz={r.nriz} tf{r.tfs} WOECB@+{wo} occ={nocc} maxLife={int(r.max_end_k)}")
    for ln in geom_window(int(r.t0_spine_pos), r.side == 'north', int(wo), float(r.e), float(r.zt), float(r.zb)):
        print(ln)

print('\n===== 6 NEAREST COUNTEREXAMPLES (CE-body first: new extreme, body ADVANCED front) =====')
for i, ceb in ceb_rows[:6]:
    r = epi.iloc[int(i)]
    print(f"\n  t0={int(r.t0_spine_pos)} {r.side} nriz={r.nriz} tf{r.tfs} CEbody@+{ceb}")
    for ln in geom_window(int(r.t0_spine_pos), r.side == 'north', int(ceb), float(r.e), float(r.zt), float(r.zb)):
        print(ln)
