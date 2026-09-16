#!/usr/bin/env python3
"""084 — validity diagnostics of candidate event definitions (prefix-only, window +50).

Not new features. Two checks of whether an event keeps one structural meaning:

1. BLOCKED REFERENCE (wick_break, close_break). A break of the reference level
   L[P] without leaving the scene needs a bar with E < L[j] < L[P], so it is
   impossible while L[P] - E <= 1 tick: the scene would end by contact or
   crossing first. At every at-risk bar the reference is classified as blocked
   or open. For each recognition: were there blocked bars before it, and was
   the last update before it marginal (<= 1 tick beyond the previous extreme).

2. PRE-RECOGNITION SCENE EXITS by an uncontacted crossing (price jump) and by a
   tape gap (missing minutes), by epoch and side, for every event and for q=5.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'work/084'
sys.path.insert(0, str(HERE))
from xray import EVENTS, W                                                 # noqa: E402

TICK = {'NQ': 0.25, 'ES': 0.25, 'YM': 1.0}
BINS = ((1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, 50))
TIMING = (('early', 1, 3), ('middle', 4, 10), ('late', 11, 50))     # same groups for every candidate
LAB = {0: '2006-2009', 1: '2010-2013', 2: '2014-2018'}


@njit(cache=True)
def _resume_props(t0s, rs, north, high, low, tick, pause_len, all_tie, any_flat, step_ticks):
    """Pause preceding each resume: bars without update since the last update, all ties?, any flat?"""
    for i in range(t0s.size):
        t0 = t0s[i]; r = rs[i]; up = north[i]
        M = high[t0] if up else -low[t0]
        run = 0; tie_run = True; flat_run = False
        for k in range(1, r + 1):
            j = t0 + k
            Hj = high[j] if up else -low[j]
            if k == r:
                pause_len[i] = run; all_tie[i] = tie_run and run > 0; any_flat[i] = flat_run
                step_ticks[i] = (Hj - M) / tick
                break
            if Hj > M:
                M = Hj; run = 0; tie_run = True; flat_run = False
            else:
                run += 1
                if Hj != M:
                    tie_run = False
                if high[j] == low[j]:
                    flat_run = True


@njit(cache=True)
def _blocked(t0s, rs, north, es, high, low, tick, w, end_k,
             blocked_before, marginal_last, ref_gap_ticks, risk_blk, risk_all,
             blocked_ext_bars, risk_blk_ext):
    for i in range(t0s.size):
        t0 = t0s[i]; r = rs[i]; up = north[i]; e = es[i]
        E = e if up else -e
        M = high[t0] if up else -low[t0]
        P = t0
        last_step = 1e9
        stop = r if r > 0 else end_k[i]          # at-risk bars: 1..r (recognized) or 1..exit/window
        for k in range(1, stop + 1):
            j = t0 + k
            LP = low[P] if up else -high[P]
            blk = (LP - E) <= tick + 1e-9
            Hj = high[j] if up else -low[j]
            risk_all[k] += 1
            if blk:
                risk_blk[k] += 1
                if k < r:
                    blocked_before[i] = True
                if P > t0:                       # blocked by an extreme bar set after T0
                    risk_blk_ext[k] += 1
                    # counts only a bar WITHOUT update: a stall or pullback of the referenced push that
                    # the definition could not register (an update bar can never be a break anyway)
                    if k < r and not (Hj > M):
                        blocked_ext_bars[i] += 1
            if k == r:
                ref_gap_ticks[i] = (LP - E) / tick
                marginal_last[i] = last_step <= tick + 1e-9
                break
            if Hj > M:
                last_step = Hj - M
                M = Hj; P = j


def main(inst='NQ', terr='discovery'):
    d = pd.read_parquet(OUT / f'xray_{inst}_{terr}.parquet')
    f = pd.read_parquet(ROOT / f'work/081a/paths/films_{inst}_{terr}.parquet', columns=['riz_id', 'exit_boundary'])
    d = d.merge(f, on='riz_id', how='left')
    mk = ROOT / 'data/market' / inst
    high = np.load(mk / 'high.npy'); low = np.load(mk / 'low.npy')
    tick = TICK[inst]
    res = {'instrument': inst, 'territory': terr, 'window': W, 'blocked_reference': {}, 'pre_recognition_exits': {}}
    north = (d.side == 'north').to_numpy()
    t0 = d.t0_pos.to_numpy().astype(np.int64); e = d.exit_boundary.to_numpy()
    # end_k for non-recognized films = last bar at risk (exit bar for contact/crossing, window 50;
    # a gap/edge exit at k means bars 1..k-1 were at risk)
    last_risk = np.where(d.end_code.isin([2, 3]).to_numpy(), d.end_k.to_numpy() - 1, d.end_k.to_numpy()).astype(np.int64)
    for name in ('wick_break', 'close_break'):
        r = d[f'r_{name}'].to_numpy().astype(np.int64)
        n = len(d)
        bb = np.zeros(n, np.bool_); ml = np.zeros(n, np.bool_); rg = np.full(n, np.nan)
        rb = np.zeros(W + 1, np.int64); ra = np.zeros(W + 1, np.int64)
        bx = np.zeros(n, np.int64); rbx = np.zeros(W + 1, np.int64)
        _blocked(t0, r, north, e, high, low, tick, W, last_risk, bb, ml, rg, rb, ra, bx, rbx)
        H = json.loads((OUT / f'xray_hist_{inst}_{terr}.json').read_text(encoding='utf-8'))
        ref = np.array(H['risk'])[:, list(EVENTS).index(name)].sum(0)
        assert (ref[1:] == ra[1:]).all(), f'{name}: diagnostic risk set differs from the X-ray scan'
        rec = r > 0
        hits = np.bincount(r[rec], minlength=W + 1)
        rows = []
        for a, b in BINS:
            sel = rec & (r >= a) & (r <= b)
            ua = int(ra[a:b + 1].sum() - rb[a:b + 1].sum())
            rows.append({'bars': f'{a}-{b}' if a != b else str(a),
                         'at_risk': int(ra[a:b + 1].sum()),
                         'blocked_share_of_at_risk': round(float(rb[a:b + 1].sum() / max(ra[a:b + 1].sum(), 1)), 4),
                         'blocked_after_extension_share_of_at_risk':
                             round(float(rbx[a:b + 1].sum() / max(ra[a:b + 1].sum(), 1)), 4),
                         'recognized_after_blocked_extension_bars':
                             round(float((bx[sel] > 0).mean()), 4) if sel.any() else None,
                         'blocked_extension_bars_p50_if_any':
                             float(np.median(bx[sel][bx[sel] > 0])) if (bx[sel] > 0).any() else None,
                         'hazard_all': round(float(hits[a:b + 1].sum() / max(ra[a:b + 1].sum(), 1)), 4),
                         'hazard_open_reference': round(float(hits[a:b + 1].sum() / max(ua, 1)), 4),
                         'recognized': int(sel.sum()),
                         'with_blocked_bars_before': round(float(bb[sel].mean()), 4) if sel.any() else None,
                         'last_update_marginal_1tick': round(float(ml[sel].mean()), 4) if sel.any() else None,
                         'ref_gap_ticks_p50': float(np.median(rg[sel])) if sel.any() else None})
        by_ep = []
        for g in range(3):
            sel = rec & (d.epoch == g).to_numpy()
            by_ep.append({'epoch': g, 'with_blocked_bars_before': round(float(bb[sel].mean()), 4),
                          'last_update_marginal_1tick': round(float(ml[sel].mean()), 4)})
        ep_tim = {}
        for g in range(3):
            for tn, a, b in TIMING:
                sel = rec & (d.epoch == g).to_numpy() & (r >= a) & (r <= b)
                ep_tim[f'{LAB[g]} {tn}'] = [round(float((bx[sel] > 0).mean()), 4) if sel.any() else None, int(sel.sum())]
        res['blocked_reference'][name] = {
            'recognized': int(rec.sum()),
            'with_blocked_bars_before': round(float(bb[rec].mean()), 4),
            'with_blocked_bars_after_extension': round(float((bx[rec] > 0).mean()), 4),
            'last_update_marginal_1tick': round(float(ml[rec].mean()), 4),
            'blocked_before_and_marginal': round(float((bb & ml)[rec].mean()), 4),
            'after_extension_blocking_by_epoch_timing': ep_tim,
            'by_recognition_bar': rows, 'by_epoch': by_ep}
        d[f'blocked_before_{name}'] = bb; d[f'marginal_{name}'] = ml; d[f'blocked_ext_bars_{name}'] = bx
    # resume: the pause it resumes from
    r = d.r_resume.to_numpy().astype(np.int64)
    n = len(d)
    pl = np.zeros(n, np.int64); at = np.zeros(n, np.bool_); af = np.zeros(n, np.bool_); st = np.full(n, np.nan)
    _resume_props(t0, r, north, high, low, tick, pl, at, af, st)
    rec = r > 0
    RZ = {}
    for g in (None, 0, 1, 2):
        for tn, a, b in TIMING:
            sel = rec & (r >= a) & (r <= b) & ((d.epoch == g).to_numpy() if g is not None else True)
            key = f'{LAB[g] if g is not None else "all"} {tn}'
            RZ[key] = {'n': int(sel.sum()),
                       'pause_all_ties': round(float(at[sel].mean()), 4) if sel.any() else None,
                       'pause_has_flat_bar': round(float(af[sel].mean()), 4) if sel.any() else None,
                       'resume_step_1tick': round(float((st[sel] <= 1 + 1e-9).mean()), 4) if sel.any() else None,
                       'pause_len_p50': float(np.median(pl[sel])) if sel.any() else None}
    res['resume_pause_contamination'] = RZ
    d['resume_pause_all_ties'] = at; d['resume_pause_has_flat'] = af
    # pause family: tie / flat of the recognizing bar by epoch and timing group
    PZ = {}
    for name in ('pause1', 'pause2', 'pause3', 'leg_pause'):
        rr = d[f'r_{name}']
        for g in (None, 0, 1, 2):
            for tn, a, b in TIMING:
                sel = (rr >= a) & (rr <= b) & ((d.epoch == g) if g is not None else True)
                key = f'{name} {LAB[g] if g is not None else "all"} {tn}'
                PZ[key] = {'n': int(sel.sum()),
                           'tie': round(float(d.loc[sel, f'tie_{name}'].mean()), 4) if sel.sum() else None,
                           'flat': round(float(d.loc[sel, f'flat_{name}'].mean()), 4) if sel.sum() else None}
    res['pause_tie_flat_by_epoch_timing'] = PZ
    # pre-recognition exits (uncontacted crossing = price jump; gap = missing minutes)
    lab = {0: '2006-2009', 1: '2010-2013', 2: '2014-2018'}
    for name in ('q5',) + EVENTS:
        if name == 'q5':
            cross = (d.end_code == 1) & (d.end_k <= 5); gap = (d.end_code == 2) & (d.end_k <= 5)
            elig = d.end_k > 5
        else:
            nr = d[f'r_{name}'] < 0
            cross = nr & (d.end_code == 1); gap = nr & (d.end_code == 2); elig = ~nr
        out = {'all': {'T0': len(d), 'eligible': int(elig.sum()), 'crossing_before': int(cross.sum()),
                       'gap_before': int(gap.sum()),
                       'crossing_share_T0': round(float(cross.mean()), 5), 'gap_share_T0': round(float(gap.mean()), 5)}}
        for key, mask in [(f'epoch {lab[g]}', d.epoch == g) for g in range(3)] + \
                         [(f'side {s}', d.side == s) for s in ('north', 'south')]:
            out[key] = {'crossing_share_T0': round(float(cross[mask].mean()), 5),
                        'gap_share_T0': round(float(gap[mask].mean()), 5)}
        res['pre_recognition_exits'][name] = out
    p = OUT / f'xray_diag_{inst}_{terr}.json'
    d[['riz_id'] + [c for c in d.columns if c.startswith(('blocked_before_', 'marginal_', 'blocked_ext_bars_',
                                                          'resume_pause_'))]].to_parquet(
        OUT / f'xray_diag_{inst}_{terr}.parquet', index=False)
    for name, B in res['blocked_reference'].items():
        print(f"{name}: recognized {B['recognized']}, blocked bars before {B['with_blocked_bars_before']}, "
              f"blocked after extension {B['with_blocked_bars_after_extension']}, "
              f"marginal last update {B['last_update_marginal_1tick']}")
        print(f"  {'bars':>6}{'at risk':>9}{'blocked':>9}{'blk ext':>9}{'hz all':>8}{'hz open':>9}{'recogn':>8}"
              f"{'aft ext':>9}{'n ext':>7}{'marg':>7}{'gap p50':>9}")
        for x in B['by_recognition_bar']:
            print(f"  {x['bars']:>6}{x['at_risk']:>9}{x['blocked_share_of_at_risk']:>9.3f}"
                  f"{x['blocked_after_extension_share_of_at_risk']:>9.3f}{x['hazard_all']:>8.3f}"
                  f"{x['hazard_open_reference']:>9.3f}{x['recognized']:>8}"
                  f"{(x['recognized_after_blocked_extension_bars'] or 0):>9.3f}"
                  f"{(x['blocked_extension_bars_p50_if_any'] or 0):>7.0f}"
                  f"{(x['last_update_marginal_1tick'] or 0):>7.3f}{(x['ref_gap_ticks_p50'] or 0):>9.1f}")
        print('  post-extension blocking before recognition, by epoch and timing [share, n]:')
        print('   ', '  '.join(f'{k}: {v[0]}/{v[1]}' for k, v in B['after_extension_blocking_by_epoch_timing'].items()))
    print('\nresume: pause made only of equal-extreme bars / pause containing a flat bar / resume by 1 tick / pause p50')
    for k, v in res['resume_pause_contamination'].items():
        print(f"  {k:<20} n {v['n']:>6}  {v['pause_all_ties']}  {v['pause_has_flat_bar']}  {v['resume_step_1tick']}  {v['pause_len_p50']}")
    print('\npause family by epoch and timing: tie / flat of the recognizing bar (n)')
    for name in ('pause1', 'pause2', 'pause3', 'leg_pause'):
        for g in ('all', LAB[0], LAB[1], LAB[2]):
            cells = []
            for tn, _, _ in TIMING:
                v = res['pause_tie_flat_by_epoch_timing'][f'{name} {g} {tn}']
                cells.append(f"{tn}: {v['tie']}/{v['flat']} ({v['n']})" if v['n'] else f'{tn}: -')
            print(f"  {name if g == 'all' else '':<10} {g:>9}  " + '  '.join(cells))
    # pause family: tick-tie and flat content of the recognizing bar by recognition time
    print('\npause family: tie (equal to extreme) / flat share of the recognizing bar by recognition bar')
    for name in ('pause1', 'pause2', 'pause3', 'leg_pause'):
        r = d[f'r_{name}']
        cells = []
        for a, b in BINS:
            sel = (r >= a) & (r <= b)
            if sel.sum() >= 50:
                cells.append(f"{a}-{b}: {d.loc[sel, f'tie_{name}'].mean():.3f}/{d.loc[sel, f'flat_{name}'].mean():.3f} (n {int(sel.sum())})")
        print(f"  {name:<10} " + '  '.join(cells))
        res.setdefault('pause_tie_flat_by_bar', {})[name] = cells
    print('\npre-recognition exits: crossing (price jump) / gap (missing minutes), share of T0 census')
    for name, o in res['pre_recognition_exits'].items():
        cells = '  '.join(f"{k}: {v['crossing_share_T0']:.4f}/{v['gap_share_T0']:.4f}" for k, v in o.items())
        print(f"  {name:<12} eligible {o['all']['eligible']:>6}  {cells}")
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(p)


if __name__ == '__main__':
    main(*(sys.argv[1:3] or ['NQ', 'discovery']))
