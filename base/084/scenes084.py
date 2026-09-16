#!/usr/bin/env python3
"""FREEZE_084 §8 last step: one real film per terminal of the race after q_event (NQ evaluation).

Deterministic: the first film by q position among close_break films with that terminal.
Illustration of what each code records on the tape, not evidence of the estimate.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'work/084'
sys.path.insert(0, str(HERE))
from race084 import FULL                                                     # noqa: E402

SHOW_HEAD, SHOW_TAIL = 10, 4


def main(inst='NQ', terr='evaluation'):
    mk = ROOT / 'data/market' / inst
    op, hi, lo, cl, ts = (np.load(mk / f) for f in ('open.npy', 'high.npy', 'low.npy', 'close.npy', 'close_ts_utc_ns.npy'))
    r = pd.read_parquet(OUT / f'race_{inst}_{terr}.parquet')
    r = r[r.anchor == 'close_break']
    x = pd.read_parquet(OUT / f'xray_{inst}_{terr}.parquet', columns=['riz_id', 'side', 'tf_minutes', 't0_pos', 'r_close_break'])
    f = pd.read_parquet(ROOT / f'work/081a/paths/films_{inst}_{terr}.parquet', columns=['riz_id', 'exit_boundary'])
    r = r.merge(x, on='riz_id').merge(f, on='riz_id').sort_values('q', kind='stable')
    L = [f'084 race scenes, {inst} {terr}, anchor close_break. b = own exit boundary, m = 2c - b, c = close(q_event).', '']
    groups = [('own_first', [0]), ('mirror_first', [1]), ('same_bar', [2]), ('gap_transition', [3, 4]),
              ('lost_observability', [5])]
    for label, codes in groups:
        sub = r[r.code.isin(codes)]
        if not len(sub):
            continue
        row = sub.iloc[0]
        q = int(row.q); t0 = int(row.t0_pos); k = int(row.kbar); b = float(row.exit_boundary)
        c = float(cl[q]); m = 2 * c - b
        L.append(f'--- {label} ({FULL[int(row.code)]}) --- {row.riz_id}  {row.side}  TF {row.tf_minutes}')
        L.append(f'    T0 {pd.Timestamp(int(ts[t0])).strftime("%Y-%m-%d %H:%M")} UTC, q_event = T0+{int(row.r_close_break)}, '
                 f'b {b:.2f}  c {c:.2f}  d {abs(c - b):.2f}  m {m:.2f}, terminal at q+{k}')
        idx = list(range(t0, q + 1))
        after = list(range(q + 1, q + k + 1))
        if len(after) > SHOW_HEAD + SHOW_TAIL:
            after = after[:SHOW_HEAD] + [None] + after[-SHOW_TAIL:]
        for j in idx + after:
            if j is None:
                L.append('    …')
                continue
            tag = 'T0' if j == t0 else ('q ' if j == q else (f'+{j - t0}' if j < q else f'q+{j - q}'))
            mark = ''
            if j > q:
                if j == q + k and int(row.code) == 5:
                    mark = '  << tape gap before this bar: lost_observability'
                else:
                    hb = lo[j] <= b <= hi[j]; hm = lo[j] <= m <= hi[j]
                    mark = ('  << contains b' if hb else '') + ('  << contains m' if hm else '')
                    if j == q + k and int(row.code) in (3, 4):
                        mark += f'  << entirely beyond {"b" if int(row.code) == 3 else "m"}, no contact'
            L.append(f'    {tag:>5} {pd.Timestamp(int(ts[j])).strftime("%m-%d %H:%M")} '
                     f'O {op[j]:>9.2f} H {hi[j]:>9.2f} L {lo[j]:>9.2f} C {cl[j]:>9.2f}{mark}')
        L.append('')
    p = OUT / f'scenes084_{inst}_{terr}.txt'
    p.write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))
    print(p)


if __name__ == '__main__':
    main(*(sys.argv[1:3] or ['NQ', 'evaluation']))
