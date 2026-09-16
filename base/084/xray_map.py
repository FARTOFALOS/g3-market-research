#!/usr/bin/env python3
"""084 — Event X-Ray Map aggregates inside the observation window T0 ... T0+50.

Uses only: the recognition bar of each event (<= +50), bar properties at
recognition, and for films WITHOUT the event the reason their scene left the
risk set before it (contact, crossing, gap, archive edge, window_censored_50).
Nothing after bar +50 and nothing after a film's recognition bar is aggregated.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'work/084'
sys.path.insert(0, str(HERE))
from xray import EVENTS, K_MIN, ENDS, W                                   # noqa: E402

TF_EDGES = [0, 2, 5, 15, 60, 240, 1441]
TF_LAB = ['1-2', '3-5', '6-15', '16-60', '61-240', '241-1440']
BINS = ((11, 15), (16, 20), (21, 30), (31, 40), (41, 50))     # inside the window
Q = (0.10, 0.25, 0.50, 0.75, 0.90)
Q5_EXPECT = {('NQ', 'discovery'): 64849, ('NQ', 'evaluation'): 43955,
             ('ES', 'evaluation'): 35018, ('YM', 'evaluation'): 44829}


def rate(a, b):
    return round(float(a) / float(b), 4) if b else None


def hazard(risk, hits, kmin):
    """Per bar k_min..50 and pooled bins inside the window: (label, hits, at risk, rate)."""
    rows = [(str(k), int(hits[k]), int(risk[k]), rate(hits[k], risk[k])) for k in range(kmin, W + 1)]
    for a, b in BINS:
        h, r = int(hits[a:b + 1].sum()), int(risk[a:b + 1].sum())
        rows.append((f'{a}-{b}', h, r, rate(h, r)))
    return rows


def quant(v):
    return {f'p{int(q * 100)}': (float(np.quantile(v, q)) if v.size else None) for q in Q}


def main(inst='NQ', terr='discovery'):
    d = pd.read_parquet(OUT / f'xray_{inst}_{terr}.parquet')
    H = json.loads((OUT / f'xray_hist_{inst}_{terr}.json').read_text(encoding='utf-8'))
    assert H['window'] == W, 'selection map must be built from the windowed scan'
    risk = np.array(H['risk']); hits = np.array(H['hits']); alive = np.array(H['alive'])
    epochs = H['epochs']
    N = len(d)
    d['tfb'] = pd.cut(d.tf_minutes, TF_EDGES, labels=TF_LAB).astype(str)
    q5 = int((d.end_k > 5).sum())          # no scene exit on bars 1..5 (window-censored films have end_k = 50)
    exp = Q5_EXPECT.get((inst, terr))
    assert exp is None or q5 == exp, f'q=5 population {q5} != prefix check {exp}'
    days_all = d.t0_day.nunique()
    res = {'instrument': inst, 'territory': terr, 'window': W, 'T0_census': N, 'T0_days': int(days_all),
           'T0_physical_paths': int(d.groupby(['t0_pos', 'side']).ngroups), 'epochs': epochs,
           'T0_census_by_epoch': [int((d.epoch == g).sum()) for g in range(len(epochs))],
           'q5_benchmark_population': q5,
           'q5_by_epoch': [int(((d.end_k > 5) & (d.epoch == g)).sum()) for g in range(len(epochs))],
           'film1_observed_at_bar': {str(k): int(alive[:, k].sum()) for k in (1, 2, 3, 5, 10, 20, 30, 40, 50)},
           'scene_exit_within_window': {ENDS[c]: int((d.end_code == c).sum()) for c in range(len(ENDS))},
           'events': {}}
    for x, name in enumerate(EVENTS):
        r = d[f'r_{name}'].to_numpy()
        rec = r >= 0
        kmin = K_MIN[name]
        E = {'k_min': kmin, 'recognized_within_50': int(rec.sum()), 'share_of_T0': rate(rec.sum(), N),
             'not_recognized': {('window_censored_50' if c == 4 else f'ended_{ENDS[c]}'):
                                int(((~rec) & (d.end_code == c)).sum()) for c in range(len(ENDS))},
             'at_risk_at_k_min': int(risk[:, x, kmin].sum())}
        E['recognition_bar'] = quant(r[rec])
        E['share_at_k_min'] = rate((r[rec] == kmin).sum(), rec.sum())
        E['flat_share'] = rate(d.loc[rec, f'flat_{name}'].sum(), rec.sum())
        E['tie_share'] = (rate(d.loc[rec, f'tie_{name}'].sum(), rec.sum())
                          if (name.startswith('pause') or name == 'leg_pause') else None)
        E['day_coverage'] = rate(d.loc[rec, 't0_day'].nunique(), days_all)
        g = d.loc[rec].groupby(['t0_pos', 'side'])[f'r_{name}']
        assert (g.nunique() == 1).all(), f'{name}: one physical path, several recognition bars'
        E['physical_paths_with_event'] = int(g.ngroups)
        E['films_per_path_with_event'] = round(float(rec.sum() / g.ngroups), 3)
        E['hazard_pooled'] = hazard(risk[:, x].sum(0), hits[:, x].sum(0), kmin)
        E['by_epoch'] = []
        for gi, lab in enumerate(epochs):
            m = (d.epoch == gi).to_numpy()
            re = m & rec
            E['by_epoch'].append({
                'epoch': lab, 'T0': int(m.sum()), 'recognized': int(re.sum()),
                'share_of_T0': rate(re.sum(), m.sum()),
                'share_of_q5': rate(re.sum(), ((d.end_k > 5).to_numpy() & m).sum()),
                'recognition_bar': quant(r[re]), 'share_at_k_min': rate((r[re] == kmin).sum(), re.sum()),
                'flat_share': rate(d.loc[re, f'flat_{name}'].sum(), re.sum()),
                'tie_share': E['tie_share'] if E['tie_share'] is None else rate(d.loc[re, f'tie_{name}'].sum(), re.sum()),
                'day_coverage': rate(d.loc[re, 't0_day'].nunique(), d.loc[m, 't0_day'].nunique()),
                'hazard': hazard(risk[gi, x], hits[gi, x], kmin)})
        E['by_side'] = {s: {'share_of_T0': rate(((d.side == s) & rec).sum(), (d.side == s).sum()),
                            'p50': float(np.median(r[(d.side == s).to_numpy() & rec]))}
                        for s in ('north', 'south')}
        E['by_tf_band'] = {b: rate(((d.tfb == b) & rec).sum(), (d.tfb == b).sum()) for b in TF_LAB}
        res['events'][name] = E
    p = OUT / f'xray_map_{inst}_{terr}.json'
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    report(res)
    print(p)


def report(res):
    print(f"{res['instrument']} {res['territory']}, window T0..T0+{res['window']}: T0 census {res['T0_census']}, "
          f"days {res['T0_days']}, physical paths {res['T0_physical_paths']}")
    print(f"q=5 benchmark {res['q5_benchmark_population']} (by epoch {res['q5_by_epoch']} of {res['T0_census_by_epoch']})")
    print('Film-1 observed at bar:', res['film1_observed_at_bar'])
    print('scene exit within window:', res['scene_exit_within_window'])
    print(f"\n{'event':<12}{'k_min':>6}{'recogn.':>9}{'of T0':>7}{'p10':>5}{'p25':>5}{'p50':>5}{'p75':>5}{'p90':>5}"
          f"{'@k_min':>8}{'flat':>7}{'tie':>7}{'days':>7}{'paths':>7}{'f/path':>7}")
    for name, E in res['events'].items():
        q = E['recognition_bar']
        print(f"{name:<12}{E['k_min']:>6}{E['recognized_within_50']:>9}{E['share_of_T0']:>7.3f}"
              f"{q['p10']:>5.0f}{q['p25']:>5.0f}{q['p50']:>5.0f}{q['p75']:>5.0f}{q['p90']:>5.0f}"
              f"{E['share_at_k_min']:>8.3f}{E['flat_share']:>7.3f}"
              f"{(E['tie_share'] if E['tie_share'] is not None else float('nan')):>7.3f}"
              f"{E['day_coverage']:>7.3f}{E['physical_paths_with_event']:>7}{E['films_per_path_with_event']:>7.2f}")
    print('\nnot recognized within the window:')
    for name, E in res['events'].items():
        print(f"  {name:<12} {E['not_recognized']}  at risk at k_min {E['at_risk_at_k_min']}")
    print('\nby epoch: share of T0 | share of q5 population | p50/p75/p90 | @k_min | flat | tie | day coverage')
    for name, E in res['events'].items():
        for b in E['by_epoch']:
            q = b['recognition_bar']
            print(f"  {name if b is E['by_epoch'][0] else '':<12} {b['epoch']:>9}: {b['share_of_T0']:.3f} | "
                  f"{b['share_of_q5']:.3f} | {q['p50']:.0f}/{q['p75']:.0f}/{q['p90']:.0f} | {b['share_at_k_min']:.3f} | "
                  f"{b['flat_share']:.3f} | {(b['tie_share'] if b['tie_share'] is not None else float('nan')):.3f} | "
                  f"{b['day_coverage']:.3f}")
    ks = [str(k) for k in range(1, 11)] + [f'{a}-{b}' for a, b in BINS]
    print('\nrisk-set incidence by bar (hits / at risk), window +50; bins pool bars inside the window:')
    for name, E in res['events'].items():
        def row(hz):
            m = {k: v for k, _, _, v in hz}
            return ' '.join((f"{m[k]:>6.3f}" if m.get(k) is not None else f"{'-':>6}") if k in m else f"{'':>6}"
                            for k in ks)
        print(f"  {name:<12} {'bar:':>9} " + ' '.join(f'{k:>6}' for k in ks))
        print(f"  {'':<12} {'pooled':>9} " + row(E['hazard_pooled']))
        for b in E['by_epoch']:
            print(f"  {'':<12} {b['epoch']:>9} " + row(b['hazard']))
        rk = {k: rr for k, _, rr, _ in E['hazard_pooled']}
        print(f"  {'':<12} {'at risk':>9} " + ' '.join(f"{rk[k]:>6}" if k in rk else f"{'':>6}" for k in ks))
    print('\nby side (share of T0 / p50) and TF band (share of T0):')
    for name, E in res['events'].items():
        s = E['by_side']
        tf = ' '.join(f"{b}:{v:.3f}" for b, v in E['by_tf_band'].items())
        print(f"  {name:<12} N {s['north']['share_of_T0']:.3f}/{s['north']['p50']:.0f}  "
              f"S {s['south']['share_of_T0']:.3f}/{s['south']['p50']:.0f}  | {tf}")


if __name__ == '__main__':
    main(*(sys.argv[1:3] or ['NQ', 'discovery']))
