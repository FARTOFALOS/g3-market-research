#!/usr/bin/env python3
"""084 — one table for the status of every candidate (prefix-only, window +50).

Definition-mechanics share of recognitions, by timing group and epoch:
    pause1, pause2, pause3, leg_pause   recognizing bar is a tie with the extreme or a flat bar
    resume                              the pause it resumes from consisted only of tie bars
    wick_break, close_break             a post-T0 extreme bar sat within 1 tick of the boundary
                                        (break impossible without ending the scene) before recognition
These are diagnostics of the definitions, not features and not selection axes.
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
from xray import EVENTS                                                    # noqa: E402

TIMING = (('early', 1, 3), ('middle', 4, 10), ('late', 11, 50))
LAB = {0: '2006-2009', 1: '2010-2013', 2: '2014-2018'}


def main(inst='NQ', terr='discovery'):
    d = pd.read_parquet(OUT / f'xray_{inst}_{terr}.parquet')
    g = pd.read_parquet(OUT / f'xray_diag_{inst}_{terr}.parquet')
    assert (d.riz_id.to_numpy() == g.riz_id.to_numpy()).all()
    mp = json.loads((OUT / f'xray_map_{inst}_{terr}.json').read_text(encoding='utf-8'))
    contam = {n: (d[f'tie_{n}'] | d[f'flat_{n}']).to_numpy() for n in ('pause1', 'pause2', 'pause3', 'leg_pause')}
    contam['resume'] = g.resume_pause_all_ties.to_numpy()
    # break events: the referenced push was replaced because an earlier failure could not be registered —
    # blocked reference with a stall bar, or an update bar that also went beyond L[P] (outside bar)
    from xray_scenes import _break_props                                    # noqa: E402
    mk = ROOT / 'data/market' / inst
    high = np.load(mk / 'high.npy'); low = np.load(mk / 'low.npy'); close = np.load(mk / 'close.npy')
    for n, by_close in (('wick_break', False), ('close_break', True)):
        r = d[f'r_{n}'].to_numpy().astype(np.int64)
        pt0 = np.zeros(len(d), np.bool_); ob = np.zeros(len(d), np.bool_)
        rr = np.where(r > 0, r, 0)
        _break_props(d.t0_pos.to_numpy().astype(np.int64), rr, (d.side == 'north').to_numpy(),
                     high, low, close, by_close, pt0, ob)
        blk = (g[f'blocked_ext_bars_{n}'] > 0).to_numpy()
        contam[n] = blk | ob
        d[f'_blk_{n}'] = blk; d[f'_ob_{n}'] = ob
    out = {}
    print(f"{'event':<12}{'recogn':>8}{'of T0':>7}{'p50/p75/p90':>13}{'days':>7}  "
          f"{'mechanics share early | middle | late (n)':<52}{'epochs: early / middle / late'}")
    for n in EVENTS:
        r = d[f'r_{n}'].to_numpy(); rec = r >= 0; c = contam[n]
        E = mp['events'][n]; q = E['recognition_bar']
        row = {'recognized': E['recognized_within_50'], 'share_of_T0': E['share_of_T0'],
               'p50_p75_p90': [q['p50'], q['p75'], q['p90']], 'day_coverage': E['day_coverage'], 'timing': {}, 'epochs': {}}
        cells = []
        for tn, a, b in TIMING:
            sel = rec & (r >= a) & (r <= b)
            row['timing'][tn] = [round(float(c[sel].mean()), 3) if sel.any() else None, int(sel.sum())]
            cells.append(f"{row['timing'][tn][0] if sel.any() else '-'} ({int(sel.sum())})")
        ecells = []
        for ep in range(3):
            parts = []
            for tn, a, b in TIMING:
                sel = rec & (r >= a) & (r <= b) & (d.epoch == ep).to_numpy()
                v = round(float(c[sel].mean()), 3) if sel.sum() >= 100 else None
                row['epochs'][f'{LAB[ep]} {tn}'] = [v, int(sel.sum())]
                parts.append('-' if v is None else f'{v:.2f}')
            ecells.append('/'.join(parts))
        out[n] = row
        print(f"{n:<12}{row['recognized']:>8}{row['share_of_T0']:>7.3f}{'/'.join(f'{x:.0f}' for x in row['p50_p75_p90']):>13}"
              f"{row['day_coverage']:>7.3f}  {' | '.join(cells):<52}{'  '.join(ecells)}")
    print('\nbreak events, components by timing (blocked stall / outside bar / union):')
    for n in ('wick_break', 'close_break'):
        r = d[f'r_{n}'].to_numpy()
        cells = []
        for tn, a, b in TIMING:
            sel = (r >= a) & (r <= b)
            cells.append(f"{tn} {d.loc[sel, f'_blk_{n}'].mean():.3f}/{d.loc[sel, f'_ob_{n}'].mean():.3f}/"
                         f"{contam[n][sel].mean():.3f}")
            out[n]['timing'][tn].append({'blocked_stall': round(float(d.loc[sel, f'_blk_{n}'].mean()), 4),
                                         'outside_bar': round(float(d.loc[sel, f'_ob_{n}'].mean()), 4)})
        print(f'  {n:<12} ' + '   '.join(cells))
    p = OUT / f'xray_status_{inst}_{terr}.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(p)


if __name__ == '__main__':
    main(*(sys.argv[1:3] or ['NQ', 'discovery']))
