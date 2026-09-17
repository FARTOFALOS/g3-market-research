#!/usr/bin/env python3
"""(3) Do WOECB' and joint grammars stay different AFTER the branch bar (its move excluded)?
From the same joint-parent, both branches; from the bar AFTER the branch:
  - post-branch OUTWARD-UPDATE event stream (pauses skipped): j=joint, m=M-only, b=MB-only,
    then absorbing terminal END_B/X/L/#. Early terminals NOT dropped.
  - minute-slice state at +1/+3/+5/+10 bars: terminated(which) or cumulative front move since
    branch (jt/mo/mb/flat).
Compare WOECB' vs joint: offset event distribution + TV; minute slices; suffix-paths.
Falsifier: TV -> ~0 after 1-2 events => local fork; persists => transition candidate.
Dedup by physical act (branch bar, side). NQ discovery.
"""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('C:/Users/Admin/Claude/g3-market-research')
sys.path.insert(0, str(ROOT / 'base/084'))
import race084
TERR = sys.argv[1] if len(sys.argv) > 1 else 'discovery'
CAP = 400
market, kind = race084.load_market('NQ'); high, low, close = market
opn = np.load(ROOT / 'data/market/NQ/open.npy'); last = close.size - 1
films = pd.read_parquet(ROOT / ('work/081a/paths/films_NQ_'+TERR+'.parquet'),
                        columns=['side', 't0_spine_pos', 'exit_boundary'])


def run(t0, up, e):
    sgn = 1 if up else -1; Eo = e if up else -e
    M = high[t0] if up else -low[t0]; MB = max(sgn*opn[t0], sgn*close[t0])
    last_up = None; parent = -1; branch = None; branch_bar = -1
    Mb_br = MBb_br = None
    stream = []; slices = {}; targets = [1, 3, 5, 10]
    term = None
    k = 0
    while k < CAP:
        k += 1
        j = t0 + k
        if j > last or kind[j] != 0:
            term = 'L'; break
        hi = high[j]; lo = low[j]
        if lo <= e <= hi:
            term = 'B'; break
        Ho = hi if up else -lo
        if Ho < Eo:
            term = 'X'; break
        Oo = sgn*opn[j]; Co = sgn*close[j]; Bo = max(Oo, Co)
        Mupd = Ho > M; MBupd = Bo > MB
        cur = 'j' if (Mupd and MBupd) else ('m' if Mupd else ('b' if MBupd else None))
        # record branch = first outward update after parent joint
        if parent >= 0 and branch is None and cur is not None:
            branch = ('WOECB' if (cur == 'm' and Co < Oo) else ('Mout' if cur == 'm'
                      else ('joint' if cur == 'j' else 'MBonly')))
            branch_bar = k; Mb_br = M; MBb_br = MB       # fronts BEFORE this bar's update
        elif branch is not None:
            # post-branch stream (this bar's move counts only after branch bar)
            if cur is not None:
                stream.append(cur)
            for tg in targets:
                if tg not in slices and (k - branch_bar) == tg:
                    dj = M > Mb_br; db = MB > MBb_br     # cumulative front move since branch
                    slices[tg] = 'jt' if (dj and db) else ('mo' if dj else ('mb' if db else 'ff'))
        if parent < 0 and cur == 'j':
            parent = k
        if Ho > M: M = Ho
        if Bo > MB: MB = Bo
    if parent < 0 or branch is None:
        return None
    # fill remaining slices with terminal
    for tg in targets:
        if tg not in slices:
            slices[tg] = 'E' + (term or '#') if (term and (t0+branch_bar+tg) >= t0+k) else 'E?'
    return dict(branch=branch, branch_abs=t0 + branch_bar, up=up,
                stream=''.join(stream) + 'E' + (term or '#'),
                s1=slices[1], s3=slices[3], s5=slices[5], s10=slices[10])


rows = []
t0a = films.t0_spine_pos.to_numpy(); up_a = (films.side.values == 'north'); ea = films.exit_boundary.to_numpy()
for i in range(len(films)):
    r = run(int(t0a[i]), bool(up_a[i]), float(ea[i]))
    if r: rows.append(r)
R = pd.DataFrame(rows).drop_duplicates(subset=['branch_abs', 'up'])
W = R[R.branch == 'WOECB']; J = R[R.branch == 'joint']
print(f'deduped physical acts: WOECB={len(W)} joint={len(J)}\n')


def st(stream, o):
    body = stream[:-2]; term = stream[-1]     # stream ends with 'E'+term
    return body[o] if o < len(body) else 'END_' + term

print('post-branch OUTWARD-UPDATE event by offset (absorbing terminals), WOECB vs joint, + TV:')
for o in range(0, 4):
    dw = Counter(W.stream.map(lambda s: st(s, o))); dj = Counter(J.stream.map(lambda s: st(s, o)))
    nw = sum(dw.values()); nj = sum(dj.values()); keys = set(dw) | set(dj)
    tv = 0.5 * sum(abs(dw.get(x,0)/nw - dj.get(x,0)/nj) for x in keys)
    fw = ','.join(f'{x}{dw[x]*100//nw}' for x in sorted(dw, key=lambda x:-dw[x])[:4])
    fj = ','.join(f'{x}{dj[x]*100//nj}' for x in sorted(dj, key=lambda x:-dj[x])[:4])
    print(f"  offset {o}: TV={tv:.3f}  WOECB[{fw}]  joint[{fj}]")

print('\nminute-slice state after recognition (+1/+3/+5/+10 bars), WOECB vs joint:')
for sl in ['s1', 's3', 's5', 's10']:
    dw = W[sl].value_counts(normalize=True); dj = J[sl].value_counts(normalize=True)
    keys = set(dw.index) | set(dj.index)
    tv = 0.5 * sum(abs(dw.get(x,0) - dj.get(x,0)) for x in keys)
    top = lambda d: ','.join(f'{x}{d.get(x,0)*100:.0f}' for x in sorted(d.index, key=lambda x:-d.get(x,0))[:4])
    print(f"  {sl}: TV={tv:.3f}  WOECB[{top(dw)}]  joint[{top(dj)}]")

print('\ntop post-branch suffix-paths (first 3 events, absorbing), WOECB vs joint:')
sp = lambda s: ' '.join(st(s, i) for i in range(3))
pw = Counter(W.stream.map(sp)); pj = Counter(J.stream.map(sp))
print(' WOECB:', [f'{k}:{v/len(W):.2f}' for k, v in pw.most_common(6)])
print(' joint:', [f'{k}:{v/len(J):.2f}' for k, v in pj.most_common(6)])
