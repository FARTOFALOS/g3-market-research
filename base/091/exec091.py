#!/usr/bin/env python3
"""091 Economic Completion of the frozen 086 candidate. FREEZE_091.

Reuses base/086/fork086.py geometry() + walk() unchanged. Applies the ONE frozen
execution interpretation: entry = open(q_event+1); continuation target = a; stop = b;
missed if open already beyond a; unknown if same_bar/lost/archive or tape gap at entry.
P&L in points; cost = 1.00 (NQ round trip). No alternatives, no rescue.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('C:/Users/Admin/Claude/g3-market-research')
sys.path.insert(0, str(ROOT / 'base/086'))
import fork086 as F

COST = 1.00
OPEN = np.load(ROOT / 'data/market/NQ/open.npy')
(high, low, close), kind = F.market('NQ')
last = close.size - 1
cells = json.loads((ROOT / 'base/086/cells086.json').read_text(encoding='utf-8'))  # tracked frozen cutpoints
MED = cells['medians']
SEL_CELL = 1  # u+ k- ttc- r-, frozen sign -1 (continuation)

WIN, CANCEL, SAME, GAP_B, GAP_A, LOST, EDGE = 0, 1, 2, 3, 4, 5, 6

def boot_ci(vals, day):
    """widest day-block bootstrap 95% CI of the mean, blocks 1/5/20 (086 protocol)."""
    if len(vals) == 0:
        return [None, None]
    ud, inv = np.unique(day, return_inverse=True); D = ud.size
    s_d = np.bincount(inv, weights=vals, minlength=D); n_d = np.bincount(inv, minlength=D).astype(float)
    widest = None
    for L in (1, 5, 20):
        rng = np.random.default_rng(20260916 + L)
        nb = int(np.ceil(D / L))
        starts = rng.integers(0, max(D - L + 1, 1), size=(2000, nb))
        idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(2000, -1)[:, :D]
        rep = s_d[idx].sum(1) / n_d[idx].sum(1)
        ci = [float(np.quantile(rep, 0.025)), float(np.quantile(rep, 0.975))]
        if widest is None or (ci[1] - ci[0]) > (widest[1] - widest[0]):
            widest = ci
    return [round(widest[0], 4), round(widest[1], 4)]

def run_cell(period):
    g = F.geometry(period)
    ok = np.isfinite(g.ttc.to_numpy())
    g = g[ok].reset_index(drop=True); g.attrs['instrument'] = 'NQ'
    cid = F.cell_ids(g, MED)
    g = g[cid == SEL_CELL].reset_index(drop=True); g.attrs['instrument'] = 'NQ'
    code, kbar = F.walk(g)
    q = g.q.to_numpy(); north = g.north.to_numpy()
    a = g.a.to_numpy(); b = g.b.to_numpy(); c = g.c.to_numpy()
    dirn = np.where(north, 1.0, -1.0)
    day = g.t0_day.to_numpy()

    qn = q + 1
    avail = (qn <= last) & (kind[np.clip(qn, 0, last)] == 0)
    E = np.where(avail, OPEN[np.clip(qn, 0, last)], np.nan)

    # exit level per fork code
    exit_lvl = np.full(len(g), np.nan)
    tgt = np.isin(code, (CANCEL, GAP_A))      # a first  -> favorable
    stp = np.isin(code, (WIN, GAP_B))          # b first  -> adverse
    unk = np.isin(code, (SAME, LOST, EDGE))
    exit_lvl[tgt] = a[tgt]; exit_lvl[stp] = b[stp]

    # classify executable outcome (priority: unavailable > missed > adverse_gap > code)
    cls = np.array(['?'] * len(g), dtype=object)
    missed = avail & (dirn * (E - a) >= 0)                 # open already beyond target
    adverse_gap = avail & (dirn * (E - b) <= 0)            # open already beyond stop (adverse)
    cls[~avail] = 'exec_unavailable'
    cls[avail & unk] = 'unknown'
    cls[avail & tgt & ~missed & ~adverse_gap] = 'target_first'
    cls[avail & stp & ~missed & ~adverse_gap] = 'boundary_first'
    cls[avail & missed] = 'missed'
    cls[avail & adverse_gap & ~missed] = 'adverse_gap'

    gross_E = dirn * (exit_lvl - E)
    gross_c = dirn * (exit_lvl - c)
    give_up = dirn * (E - c)                                # loss recognition->entry (needs only avail)

    # P&L population for point estimate: realized executable outcomes
    real = np.isin(cls, ['target_first', 'boundary_first'])
    gE = gross_E[real]; nE = gE - COST
    ag = cls == 'adverse_gap'
    # per-signal net: realized + adverse_gap(-cost) + missed(0), unknown excluded
    signal_mask = np.isin(cls, ['target_first', 'boundary_first', 'adverse_gap', 'missed'])
    net_signal = np.where(real, gross_E - COST, np.where(ag, -COST, 0.0))[signal_mask]
    day_signal = day[signal_mask]

    from collections import Counter
    cc = Counter(cls.tolist())
    n = len(g)
    res = {
        'period': period, 'N_cell': n, 'days': int(np.unique(day).size),
        'counts': {k: int(cc.get(k, 0)) for k in
                   ['target_first', 'boundary_first', 'unknown', 'missed', 'adverse_gap', 'exec_unavailable']},
        'fork_codes_from_close': {F.NAMES[k]: int((code == k).sum()) for k in F.NAMES},
        'mean_d': round(float(g.d.mean()), 2), 'mean_s': round(float(g.s.mean()), 2),
        'mean_p0': round(float(g.p0.mean()), 4),
        # item 2: give_up (recognition -> entry), over all executable films
        'give_up_mean': round(float(np.nanmean(give_up[avail])), 4),
        'give_up_ci': boot_ci(give_up[avail], day[avail]),
        'frac_open_moved_favorable': round(float((dirn * (E - c) > 0)[avail].mean()), 4),
        # item 4: executable P&L (resolved target/boundary only)
        'resolved_N': int(real.sum()),
        'gross_E_mean': round(float(gE.mean()), 4) if real.any() else None,
        'gross_E_ci': boot_ci(gE, day[real]),
        'net_E_mean': round(float(nE.mean()), 4) if real.any() else None,
        'net_E_ci': boot_ci(nE, day[real]),
        # per-signal net (missed=0, adverse=-cost, unknown excluded)
        'net_per_signal_mean': round(float(net_signal.mean()), 4) if signal_mask.any() else None,
        'net_per_signal_ci': boot_ci(net_signal, day_signal),
        # reference: from close(q_event), same resolved films
        'gross_c_mean': round(float(gross_c[real].mean()), 4) if real.any() else None,
        # item 5: frequency / missed rate
        'target_first_rate_resolved': round(float((cls[real] == 'target_first').mean()), 4) if real.any() else None,
        'missed_rate': round(float((cls == 'missed').mean()), 4),
        'unknown_rate': round(float((cls == 'unknown').mean()), 4),
    }
    return res

if __name__ == '__main__':
    out = {'freeze_sha256': F.sha(ROOT / 'base/091/FREEZE_091.md'),
           'selected_sha256': F.sha(ROOT / 'base/086/selected086.json'),  # tracked frozen sign
           'cost': COST, 'cell': 'u+ k- ttc- r-', 'sign': -1, 'entry': 'open(q_event+1)',
           'periods': {}}
    order = [('NQ_search', 'DISCOVERY/SEARCH'), ('NQ_holdout', 'WEAK 086 HOLDOUT'),
             ('NQ_development', 'context'), ('NQ_evaluation', 'context')]
    for period, role in order:
        r = run_cell(period); r['role'] = role; out['periods'][period] = r
        print(f"\n===== {period} [{role}] : cell u+ k- ttc- r-  N={r['N_cell']} days={r['days']} "
              f"(mean d={r['mean_d']} s={r['mean_s']} p0={r['mean_p0']}) =====")
        print(f"  outcomes: {r['counts']}")
        print(f"  [2] give_up recognition->entry: mean {r['give_up_mean']} pts CI {r['give_up_ci']}  "
              f"(open moved favorable before entry in {r['frac_open_moved_favorable']*100:.1f}% of films)")
        print(f"  [3] resolved target_first {r['counts']['target_first']} / boundary_first {r['counts']['boundary_first']} "
              f"/ unknown {r['counts']['unknown']} / missed {r['counts']['missed']} / adverse_gap {r['counts']['adverse_gap']}")
        print(f"  [4] executable resolved: gross_E {r['gross_E_mean']} CI {r['gross_E_ci']}  "
              f"net_E {r['net_E_mean']} CI {r['net_E_ci']}  (ref from close: gross_c {r['gross_c_mean']})")
        print(f"      per-signal net (missed=0, adverse=-cost, unknown excl): {r['net_per_signal_mean']} CI {r['net_per_signal_ci']}")
        print(f"  [5] target-first rate (resolved) {r['target_first_rate_resolved']}  missed rate {r['missed_rate']}  "
              f"unknown rate {r['unknown_rate']}")
    (ROOT / 'work/091').mkdir(parents=True, exist_ok=True)
    (ROOT / 'work/091/exec091.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"\nsaved {ROOT/'work/091/exec091.json'}")
