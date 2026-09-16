#!/usr/bin/env python3
"""084 — real scenes and counterexamples for the Event X-Ray (bars T0 ... recognition bar only).

Scenes per candidate: early (bars 1-3), middle (4-10), late (11-50) — the first film by T0
position in 2014-2018 inside each timing group, without any further filter.
Counterexamples: named, deterministic, first film by T0 position meeting the stated condition.
Also: outside-bar refusal before recognition for the two break events, by timing group.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'work/084'
sys.path.insert(0, str(HERE))
from xray import EVENTS                                                    # noqa: E402

TIMING = (('early', 1, 3), ('middle', 4, 10), ('late', 11, 50))
LAB = {0: '2006-2009', 1: '2010-2013', 2: '2014-2018'}


@njit(cache=True)
def _break_props(t0s, rs, north, high, low, close, by_close, p_is_t0, outside_before):
    """Extreme bar still T0 at recognition; an update bar that also broke L[P] (by wick or by close) before it."""
    for i in range(t0s.size):
        t0 = t0s[i]; r = rs[i]; up = north[i]
        M = high[t0] if up else -low[t0]
        P = t0
        for k in range(1, r):
            j = t0 + k
            Hj = high[j] if up else -low[j]
            Xj = (close[j] if up else -close[j]) if by_close else (low[j] if up else -high[j])
            LP = low[P] if up else -high[P]
            if Hj > M:
                if Xj < LP:
                    outside_before[i] = True
                M = Hj; P = j
        p_is_t0[i] = P == t0


def bars(m, t0, upto, e, side, marks=None):
    op, hi, lo, cl, ts = m
    up = side == 'north'
    M = hi[t0] if up else -lo[t0]
    P = t0; run = 0
    out = []
    for j in range(t0, upto + 1):
        k = j - t0
        tag = ''
        if j > t0:
            Hj = hi[j] if up else -lo[j]
            Lj = lo[j] if up else -hi[j]
            Cj = cl[j] if up else -cl[j]
            LP = lo[P] if up else -hi[P]
            if lo[j] <= e <= hi[j]:
                tag = 'contact e'
            elif Hj < (e if up else -e):
                tag = 'beyond e, no contact'
            elif Hj > M:
                tag = 'update' + ('  (also beyond L[P]: outside bar, not a break)' if Lj < LP else '')
                M = Hj; P = j; run = 0
            else:
                run += 1
                tag = (f'no update (run {run})' + ('  tie' if Hj == M else '') + ('  flat' if hi[j] == lo[j] else '')
                       + ('  wick beyond L[P]' if Lj < LP else '') + ('  close beyond L[P]' if Cj < LP else ''))
            LPn = lo[P] if up else -hi[P]
            if abs(LPn - (e if up else -e)) <= 0.25 + 1e-9 and P != t0:
                tag += '  [reference bar within 1 tick of e: break blocked]'
        if marks and k in marks:
            tag += f'  <<< {marks[k]}'
        out.append(f'  {"T0" if k == 0 else "+" + str(k):>4} {pd.Timestamp(int(ts[j])).strftime("%Y-%m-%d %H:%M")} '
                   f'O {op[j]:>9.2f} H {hi[j]:>9.2f} L {lo[j]:>9.2f} C {cl[j]:>9.2f}  {tag}')
    return out


def main(inst='NQ', terr='discovery'):
    mk = ROOT / 'data/market' / inst
    m = tuple(np.load(mk / f) for f in ('open.npy', 'high.npy', 'low.npy', 'close.npy', 'close_ts_utc_ns.npy'))
    d = pd.read_parquet(OUT / f'xray_{inst}_{terr}.parquet')
    g = pd.read_parquet(OUT / f'xray_diag_{inst}_{terr}.parquet')
    d = pd.concat([d, g.drop(columns='riz_id')], axis=1)
    f = pd.read_parquet(ROOT / f'work/081a/paths/films_{inst}_{terr}.parquet', columns=['riz_id', 'exit_boundary'])
    d = d.merge(f.rename(columns={'exit_boundary': 'e'}), on='riz_id', how='left')
    d = d.sort_values('t0_pos', kind='stable').reset_index(drop=True)
    L = [f'084 Event X-Ray scenes, {inst} {terr}, window T0..T0+50. Bars T0 ... recognition bar only.', '']

    for name, by_close in (('wick_break', False), ('close_break', True)):
        rec = d[d[f'r_{name}'] >= 0]
        r = rec[f'r_{name}'].to_numpy().astype(np.int64)
        pt0 = np.zeros(len(rec), np.bool_); ob = np.zeros(len(rec), np.bool_)
        _break_props(rec.t0_pos.to_numpy().astype(np.int64), r, (rec.side == 'north').to_numpy(),
                     m[1], m[2], m[3], by_close, pt0, ob)
        cells = []
        for tn, a, b in TIMING:
            sel = (r >= a) & (r <= b)
            cells.append(f'{tn} {ob[sel].mean():.3f} (n {int(sel.sum())})')
        L.append(f'{name}: outside bar refused before recognition, by timing: ' + '  '.join(cells)
                 + f';  extreme bar still T0 at recognition {pt0.mean():.4f}')
    L.append('')

    def show(row, name, title):
        r = int(row[f'r_{name}'])
        L.append(f'--- {title} ---')
        L.append(f'  {row.riz_id}  {row.side}  TF {row.tf_minutes}  e {row.e:.2f}  epoch {LAB[int(row.epoch)]}  '
                 f'recognized +{r}')
        L.extend(bars(m, int(row.t0_pos), int(row.t0_pos) + r, row.e, row.side, {r: f'{name} recognized'}))
        L.append('')

    for name in EVENTS:
        L.append(f'=============== {name}: early / middle / late (first film by T0 in 2014-2018) ===============')
        for tn, a, b in TIMING:
            c = d[(d[f'r_{name}'] >= a) & (d[f'r_{name}'] <= b) & (d.epoch == 2)]
            if len(c):
                show(c.iloc[0], name, f'{name} {tn}')
    L.append('=============== counterexamples ===============')
    c = d[(d.blocked_ext_bars_wick_break >= 3) & (d.r_wick_break >= 11) & (d.epoch == 2)]
    show(c.iloc[0], 'wick_break', 'wick_break late after a blocked reference (stall bars could not register a break)')
    c = d[(d.blocked_ext_bars_close_break >= 3) & (d.r_close_break >= 11) & (d.epoch == 2)]
    show(c.iloc[0], 'close_break', 'close_break late after a blocked reference')
    c = d[(d.r_resume >= 1) & (d.r_resume <= 3) & d.resume_pause_all_ties & (d.epoch == 2)]
    show(c.iloc[0], 'resume', 'resume early: the "pause" was only an equal extreme, then +1 tick')
    c = d[(d.r_pause3 == 3) & d.flat_pause3 & (d.epoch == 2)]
    show(c.iloc[0], 'pause3', 'pause3 at the earliest bar completed by a flat bar')
    c = d[(d.r_pause2 == 2) & d.tie_pause2 & (d.epoch == 2)]
    show(c.iloc[0], 'pause2', 'pause2 at the earliest bar completed by a tie')
    c = d[(d.r_wick_break >= 4) & d.flat_wick_break & (d.epoch == 2)]
    show(c.iloc[0], 'wick_break', 'wick_break on a flat bar (thin tape)')
    p = OUT / f'xray_scenes_{inst}_{terr}.txt'
    p.write_text('\n'.join(L), encoding='utf-8')
    print(p, len(L), 'lines')
    print('\n'.join(L[:3]))


if __name__ == '__main__':
    main(*(sys.argv[1:3] or ['NQ', 'discovery']))
