#!/usr/bin/env python3
"""(4) SYMMETRIC residual path after recognition. Frozen only: parent WOECB', event=1st
MB-resume, conservative RECOG=2nd MB-resume. Future after recognition NOT assumed extend.
From the recognition close, measure symmetrically (no favorable/adverse labels):
  path_out  max outward excursion beyond close (sd)
  path_in   max inward excursion below close, toward b and beyond (sd)
  order     which extreme comes first
  t_out,t_in bars to each
  in_before_out  max inward before the outward max (risk-before-reward for an outward bet)
  out_before_in  max outward before the inward max (mirror)
  reached_b, censoring, further M/MB grammar counts
Recognition-forming move is NOT credited (measured strictly from recognition close forward).
Main table = RECOG; EVENT kept alongside. Dedup physical acts. NQ discovery.
"""
from __future__ import annotations
import sys
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


def path_from(anchor, up, e, sig):
    """symmetric path measures from close of bar `anchor` forward to terminal."""
    sgn = 1 if up else -1; Eo = e if up else -e
    c0 = sgn * close[anchor]
    po = pi = 0.0; t_out = t_in = 0; ibo = obi = 0.0
    run_in = 0.0; run_out = 0.0; cause = None; nb = 0; jj = anchor
    nj = nm = nb_ev = 0
    while True:
        jj += 1; nb += 1
        if nb > CAP: cause = 'cap'; break
        if jj > last: cause = 'archive'; break
        if kind[jj] != 0: cause = 'gap'; break
        h2 = high[jj]; l2 = low[jj]
        Ho = h2 if up else -l2; Io = l2 if up else -h2
        out = (Ho - c0) / sig; inn = (c0 - Io) / sig
        if out > po: po = out; t_out = nb; ibo = run_in          # inward seen before this new outward max
        if inn > pi: pi = inn; t_in = nb; obi = run_out
        run_in = max(run_in, inn); run_out = max(run_out, out)
        if l2 <= e <= h2: cause = 'B'; break
        if Ho < Eo: cause = 'X'; break
    return dict(path_out=po, path_in=pi, t_out=t_out, t_in=t_in,
                in_before_out=ibo, out_before_in=obi,
                out_first=(t_out <= t_in), reached_b=(cause == 'B'),
                cens=cause in ('gap', 'cap', 'archive'), cause=cause, bars=nb)


def measure(t0, up, e):
    sgn = 1 if up else -1; Eo = e if up else -e
    M = high[t0] if up else -low[t0]; MB = max(sgn*opn[t0], sgn*close[t0]); last_up = None; parent = -1
    for k in range(1, CAP + 1):
        j = t0 + k
        if j > last or kind[j] != 0: return None
        hi = high[j]; lo = low[j]
        if lo <= e <= hi: return None
        Ho = hi if up else -lo
        if Ho < Eo: return None
        Oo = sgn*opn[j]; Co = sgn*close[j]; Bo = max(Oo, Co)
        cur = 'j' if (Ho > M and Bo > MB) else ('m' if Ho > M else ('b' if Bo > MB else None))
        if parent >= 0 and cur == 'm' and Co < Oo and last_up == 'j':
            reco = j; MBr = MB; sig = (high[t0:j+1] - low[t0:j+1]).mean()
            if sig <= 0: return None
            # find first & second MB-resume bars
            r1 = r2 = -1; jj = reco
            while jj - reco < CAP:
                jj += 1
                if jj > last or kind[jj] != 0: break
                hh = high[jj]; ll = low[jj]
                if ll <= e <= hh: break
                Hx = hh if up else -ll
                if Hx < Eo: break
                Bx = max(sgn*opn[jj], sgn*close[jj])
                if Bx > MBr:
                    if r1 < 0: r1 = jj
                    elif r2 < 0: r2 = jj; break
                    MBr = Bx
            out = dict(reco=reco, up=up, sig=float(sig))
            if r1 > 0:
                for kk, v in path_from(r1, up, e, sig).items(): out['ev_' + kk] = v
            if r2 > 0:
                for kk, v in path_from(r2, up, e, sig).items(): out['rc_' + kk] = v
            out['has_ev'] = r1 > 0; out['has_rc'] = r2 > 0
            return out
        if parent < 0 and cur == 'j': parent = k
        if cur is not None: last_up = cur
        if Ho > M: M = Ho
        if Bo > MB: MB = Bo
    return None


rows = []
t0a = films.t0_spine_pos.to_numpy(); up_a = (films.side.values == 'north'); ea = films.exit_boundary.to_numpy()
for i in range(len(films)):
    r = measure(int(t0a[i]), bool(up_a[i]), float(ea[i]))
    if r: rows.append(r)
R = pd.DataFrame(rows).drop_duplicates(subset=['reco', 'up'])
print(f"WOECB' acts: {len(R)}  with EVENT: {R.has_ev.sum()}  with RECOG(2nd): {R.has_rc.sum()}\n")
q = lambda s: [round(float(np.nanpercentile(s, p)), 2) for p in (25, 50, 75)]

for pre, lab in [('rc_', 'RECOG (2nd resume, main)'), ('ev_', 'EVENT (1st resume)')]:
    g = R[R[pre + 'path_out'].notna()] if (pre + 'path_out') in R else R.iloc[0:0]
    if len(g) == 0: continue
    print(f'=== {lab}  n={len(g)} ===')
    print(f"  path_out (sd)          {q(g[pre+'path_out'])}     path_in (sd) {q(g[pre+'path_in'])}")
    print(f"  out_first fraction     {g[pre+'out_first'].mean():.3f}")
    print(f"  in_before_out (sd)     {q(g[pre+'in_before_out'])}  (risk before an outward extreme)")
    print(f"  out_before_in (sd)     {q(g[pre+'out_before_in'])}  (mirror)")
    print(f"  t_out / t_in (bars)    {q(g[pre+'t_out'])} / {q(g[pre+'t_in'])}")
    print(f"  reached_b {g[pre+'reached_b'].mean():.3f}   censored {g[pre+'cens'].mean():.3f}")
    # symmetric read: is path_out systematically larger than path_in? per-film sign
    po = g[pre+'path_out']; pi = g[pre+'path_in']
    print(f"  per-film path_out>path_in: {(po>pi).mean():.3f}   median(path_out-path_in)={float(np.nanmedian(po-pi)):.2f}sd")
    print()
print('read: answer 1 outward asym | 2 inward asym | 3 order/time asym | 4 symmetric (info exhausted).')
