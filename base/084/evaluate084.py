#!/usr/bin/env python3
"""FREEZE_084 §5-§7: own vs mirror after close_break -> q_event, with the q = 5 benchmark.

Direction:  B = P(own_first) - P(mirror_first) over N (gap_transition in N, in neither term);
            U_id = same_bar + lost_observability + archive_edge;  [B_lo; B_hi] = (O - M -/+ U_id) / N;
            day-block bootstrap (1/5/20 days, 2000 reps, seed 20260916 + L): [q2.5 B_lo*; q97.5 B_hi*], widest.
Speed:      h = 5, 15, 60: own / mirror / same_bar / gap_transition / unknown / neither by h;
            R_h in [reached <= h ; reached <= h + unknown <= h] / N, reached = own, mirror, same_bar.
Reports:    north / south, q_event inside 03:00 ET -> XNYS close, path weight. Not verdict inputs.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'work/084'
sys.path.insert(0, str(HERE))
import race084                                                               # noqa: E402
sys.path.insert(0, str(ROOT / 'base/081'))
from trading import sessions, bar_session_map                               # noqa: E402

TERR = [('NQ', 'discovery'), ('NQ', 'evaluation'), ('ES', 'evaluation'), ('YM', 'evaluation')]
OWN, MIRROR, SAME, GAP_B, GAP_M, LOST, EDGE = 0, 1, 2, 3, 4, 5, 6
BLOCKS = (1, 5, 20)
NBOOT = 2000
SEED = 20260916
HS = (5, 15, 60)


def freeze_hash():
    return hashlib.sha256((HERE / 'FREEZE_084.md').read_bytes()).hexdigest()


def direction(code, day, w=None):
    w = np.ones(code.size) if w is None else w
    s = (code == OWN).astype(float) - (code == MIRROR)
    u = np.isin(code, (SAME, LOST, EDGE)).astype(float)
    ud, inv = np.unique(day, return_inverse=True)
    D = ud.size
    n_d = np.bincount(inv, weights=w, minlength=D)
    s_d = np.bincount(inv, weights=w * s, minlength=D)
    u_d = np.bincount(inv, weights=w * u, minlength=D)
    N, S, U = n_d.sum(), s_d.sum(), u_d.sum()
    out = {'N_weighted': round(float(N), 2), 'days': int(D),
           'B_lo': round(float((S - U) / N), 5), 'B_hi': round(float((S + U) / N), 5),
           'center_O_minus_M_over_N': round(float(S / N), 5), 'U_id_share': round(float(U / N), 5),
           'outer_set_by_block': {}}
    for L in BLOCKS:
        rng = np.random.default_rng(SEED + L)
        nb = int(np.ceil(D / L))
        starts = rng.integers(0, max(D - L + 1, 1), size=(NBOOT, nb))
        idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(NBOOT, -1)[:, :D]
        nn = n_d[idx].sum(1); ss = s_d[idx].sum(1); uu = u_d[idx].sum(1)
        out['outer_set_by_block'][str(L)] = [round(float(np.quantile((ss - uu) / nn, 0.025)), 5),
                                             round(float(np.quantile((ss + uu) / nn, 0.975)), 5)]
    lo, hi = max(out['outer_set_by_block'].values(), key=lambda x: x[1] - x[0])
    out['outer_set_95'] = [lo, hi]
    if lo > 0:
        out['class'] = 'own first more often: asymmetry established'
    elif hi < 0:
        out['class'] = 'mirror first more often: asymmetry established'
    else:
        out['class'] = 'unresolved'
        out['limiting_cause'] = ('identification' if out['B_lo'] <= 0 <= out['B_hi'] else 'statistical precision')
    return out


def speed(code, kbar):
    N = code.size
    res = {}
    for h in HS:
        by = kbar <= h
        own = int(((code == OWN) & by).sum()); mir = int(((code == MIRROR) & by).sum())
        same = int(((code == SAME) & by).sum()); gap = int((np.isin(code, (GAP_B, GAP_M)) & by).sum())
        unk = int((np.isin(code, (LOST, EDGE)) & by).sum())
        neither = N - own - mir - same - gap - unk
        res[str(h)] = {'own': own, 'mirror': mir, 'same_bar': same, 'gap_transition': gap, 'unknown': unk,
                       'neither': neither,
                       'R_h': [round((own + mir + same) / N, 5), round((own + mir + same + unk) / N, 5)],
                       'gap_transition_share': round(gap / N, 5)}
    return res


def evaluate(inst, terr, cal):
    market, kind = race084.load_market(inst)
    ts = np.load(ROOT / 'data/market' / inst / 'close_ts_utc_ns.npy')
    sess, _ = bar_session_map(ts, *cal)
    T = {}
    frames = []
    for name, p in race084.populations(inst, terr).items():
        north = (p.side == 'north').to_numpy()
        code, kbar = race084.walk(p.q.to_numpy(), p.exit_boundary.to_numpy(), north, market, kind, blind=False)
        day = p.t0_day.to_numpy()
        counts = {race084.FULL[c]: int((code == c).sum()) for c in race084.FULL}
        A = {'N': int(code.size), 'counts': counts,
             'gap_transition': {'total': int(np.isin(code, (GAP_B, GAP_M)).sum()),
                                'share': round(float(np.isin(code, (GAP_B, GAP_M)).mean()), 5),
                                'through_b': counts['gap_own'], 'through_m': counts['gap_mirror']},
             'direction': direction(code, day), 'speed': speed(code, kbar), 'reports': {}}
        for s in ('north', 'south'):
            m = (p.side == s).to_numpy()
            A['reports'][s] = {'N': int(m.sum()), **direction(code[m], day[m])}
        inwin = sess[p.q.to_numpy()] >= 0
        A['reports']['trading_window'] = {'N': int(inwin.sum()), **direction(code[inwin], day[inwin])}
        path = pd.Series(list(zip(p.t0_pos, p.side)))
        w = 1.0 / path.map(path.value_counts()).to_numpy()
        A['reports']['path_weight'] = {'paths': int(path.nunique()), **direction(code, day, w)}
        T[name] = A
        frames.append(pd.DataFrame({'riz_id': p.riz_id, 'anchor': name, 'q': p.q, 'code': code, 'kbar': kbar}))
    pd.concat(frames).to_parquet(OUT / f'race_{inst}_{terr}.parquet', index=False, compression='zstd')
    return T


def show(key, T):
    for name, A in T.items():
        c = A['counts']; D = A['direction']
        print(f"\n{key} | {name}: N {A['N']}  own {c['own_first']}  mirror {c['mirror_first']}  same_bar {c['same_bar']}  "
              f"lost {c['lost_observability']}  edge {c['archive_edge']}  gap_transition {A['gap_transition']['total']} "
              f"(through b {A['gap_transition']['through_b']}, through m {A['gap_transition']['through_m']})")
        print(f"   B identified [{D['B_lo']:+.4f}; {D['B_hi']:+.4f}]  U_id {D['U_id_share']:.4f}  "
              f"outer 95% [{D['outer_set_95'][0]:+.4f}; {D['outer_set_95'][1]:+.4f}]  by block {D['outer_set_by_block']}  "
              f"days {D['days']}  -> {D['class']}{' (' + D['limiting_cause'] + ')' if 'limiting_cause' in D else ''}")
        for rk, R in A['reports'].items():
            print(f"   report {rk:<15} N {R.get('N', R.get('paths'))}: [{R['B_lo']:+.4f}; {R['B_hi']:+.4f}]  "
                  f"outer [{R['outer_set_95'][0]:+.4f}; {R['outer_set_95'][1]:+.4f}]  {R['class']}")
        for h, S in A['speed'].items():
            print(f"   speed h{h:<3} R_h [{S['R_h'][0]:.4f}; {S['R_h'][1]:.4f}]  own {S['own']} mirror {S['mirror']} "
                  f"same {S['same_bar']} gap {S['gap_transition']} unknown {S['unknown']} neither {S['neither']}")


def main():
    fh = freeze_hash()
    cal = sessions()
    res = {'freeze_sha256': fh, 'territories': {}}
    for inst, terr in TERR:
        T = evaluate(inst, terr, cal)
        res['territories'][f'{inst} {terr}'] = T
        show(f'{inst} {terr}', T)
    (OUT / 'evaluation084.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\nfreeze', fh)


if __name__ == '__main__':
    main()
